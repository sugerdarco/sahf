"""
Master Evaluation Suite and Plot Generator for SAHF vs DeePEn Benchmarks.

Runs the CROSS-TOKENIZER pipeline (Stage 8): each agent keeps its own tokenizer,
distributions are reconciled in a shared byte-prefix space, and generation emits
bytes rather than shared token ids. Configured by config_sheaf.yaml.

Runs comprehensive evaluations across GSM8K, MMLU, and ARC-Challenge under both:
1. Clean baseline ensemble (3 honest agents)
2. Poisoned / Adversarial ensemble (1 inverted agent, 2 honest agents)

Saves verbose per-question step data, JSON summaries, and auto-generates benchmark comparison graphs.
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
import time
import torch
import yaml
import matplotlib.pyplot as plt

# This script lives in deepen/, so paths are anchored explicitly rather than
# resolved against the current working directory — it runs the same whether it is
# invoked from the repository root or from inside this folder.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: DeePEn's benchmark data is NOT vendored in this repository. Fetch it from the
#: DeePEn project and place it at deepen/datasets/ (or at the repository root),
#: keeping DeePEn's own layout: GSM/data/, MMLU/dev-jsonl/, ARC-Challenge/.
DATASETS = HERE / "datasets" if (HERE / "datasets").exists() else ROOT / "datasets"

#: Benchmark outputs stay beside the scripts that produce them.
RESULTS = HERE / "results"

#: Per-step run logs keep using the shared out/ tree at the repository root, so
#: they stay in the same format and place as every other run.
OUT_ROOT = ROOT / "out"

from sahf.agents import HFAgent, PoisonedAgentWrapper
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.sheaf import (
    BytePrefixTree,
    SheafOrchestrator,
    UpstreamAgent,
    assert_distinct_tokenizers,
)


def extract_gsm_target_number(answer_str: str) -> str:
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "").replace("$", "")
    match = re.search(r'is\s+([0-9\.\,]+)', answer_str, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").replace("$", "")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_str)
    return numbers[-1] if numbers else answer_str.strip()


def parse_model_gsm_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "").replace("$", "")
    match = re.search(r'(?:answer|result)\s*(?:is|=|:)?\s*\$?([0-9\.\,]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").replace("$", "")
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return numbers[-1].replace(",", "").replace("$", "") if numbers else ""


def parse_model_mcq_answer(text: str) -> str:
    match_start = re.match(r'^\s*(?:option|answer)?\s*[\(\s]*([A-D])[\)\.\s\:]?', text, re.IGNORECASE)
    if match_start:
        return match_start.group(1).upper()
    match = re.search(r'(?:option|answer|choice)?\s*[:\(\s]*([A-D])[\)\.\s\:]', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match_single = re.search(r'\b([A-D])\b', text)
    if match_single:
        return match_single.group(1).upper()
    return ""


def load_dataset_samples(dataset_name: str, category: str = "elementary_mathematics", num_samples: int = 15):
    base_dir = str(DATASETS)
    samples = []

    if dataset_name == "gsm":
        path = os.path.join(base_dir, "GSM", "data", "test.cleand.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                target_num = extract_gsm_target_number(data["answer"])
                prompt = (
                    f"Question: {data['question']}\n"
                    "Calculate step-by-step and write #### followed by the final answer number.\n"
                    "Answer:"
                )
                samples.append({
                    "prompt": prompt,
                    "target_full": data["answer"],
                    "target_parsed": target_num,
                    "type": "gsm"
                })

    elif dataset_name == "mmlu":
        cat = category or "elementary_mathematics"
        path = os.path.join(base_dir, "MMLU", "dev-jsonl", f"{cat}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(base_dir, "MMLU", "test-jsonl", f"{cat}.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                opts = f"A) {data['A']}\nB) {data['B']}\nC) {data['C']}\nD) {data['D']}"
                prompt = (
                    f"Question: {data['question']}\n"
                    f"Options:\n{opts}\n"
                    "Respond ONLY with the single correct option letter (A, B, C, or D).\n"
                    "Answer:"
                )
                samples.append({
                    "prompt": prompt,
                    "target_full": data["answer"],
                    "target_parsed": data["answer"].strip().upper(),
                    "type": "mcq"
                })

    elif dataset_name == "arc":
        path = os.path.join(base_dir, "ARC-Challenge", "test.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                opts = f"A) {data.get('A', '')}\nB) {data.get('B', '')}\nC) {data.get('C', '')}\nD) {data.get('D', '')}"
                prompt = (
                    f"Question: {data['question']}\n"
                    f"Options:\n{opts}\n"
                    "Respond ONLY with the single correct option letter (A, B, C, or D).\n"
                    "Answer:"
                )
                target = data.get("answer", "").strip().upper()
                samples.append({
                    "prompt": prompt,
                    "target_full": target,
                    "target_parsed": target,
                    "type": "mcq"
                })

    return samples[:num_samples]


def run_benchmark_eval(dataset_name: str, category: str, num_samples: int,
                       config_file: str = None, poison: bool = False):
    samples = load_dataset_samples(dataset_name, category, num_samples)
    mode_str = "POISONED" if poison else "CLEAN"
    print(f"\n==================================================================")
    print(f"RUNNING BENCHMARK [{mode_str}]: {dataset_name.upper()} ({len(samples)} samples)")
    print(f"==================================================================")

    config_file = config_file or str(ROOT / "config_sheaf.yaml")
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    setup_app_logging(str(OUT_ROOT))
    dtype = getattr(torch, cfg["dtype"])
    raw_agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    if poison:
        raw_agents[2] = PoisonedAgentWrapper(raw_agents[2], mode="invert")
        print("[INFO] Poisoned Agent 2 (Probability Inversion)")

    # Cross-tokenizer path: each agent keeps its own tokenizer and encodes the
    # shared context itself. UpstreamAgent is the bridge; there is deliberately
    # no shared-vocabulary assertion, because a shared vocabulary is exactly what
    # this configuration does not have.
    agents = [UpstreamAgent(a) for a in raw_agents]
    vocab_info = assert_distinct_tokenizers(agents)
    print(f"[INFO] Vocabularies: {vocab_info}")

    # A mis-detected tokenizer scheme does not raise — it shifts every token by a
    # leading space and silently misaligns the byte tree. Verify once, up front.
    for a in agents:
        if not a.verify(["The capital of France is Paris", "Question: what is 2 + 2?"]):
            raise SystemExit(
                f"Byte round-trip FAILED for {a.name} (scheme={a.vocab_spec().scheme}). "
                "Results would be silently wrong; fix vocabulary extraction first."
            )

    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"],
    )
    sheaf_cfg = cfg.get("sheaf", {})

    tree = None
    tree_path = Path(sheaf_cfg.get("tree_path", "")) if sheaf_cfg.get("tree_path") else None
    if tree_path and tree_path.exists():
        tree = BytePrefixTree.load(tree_path)
        if [v.name for v in tree.vocabs] != cfg["models"]:
            raise SystemExit(
                f"{tree_path} was built for a different model list. Rebuild it with "
                "build_prefix_tree.py, or delete it to build the tree per step."
            )
        print(f"[INFO] Loaded prefix tree: {tree_path} ({tree.n_nodes:,} nodes)")

    correct_count = 0
    total_fast_tokens = 0
    total_fusion_tokens = 0
    total_escalated_tokens = 0
    total_tokens_generated = 0
    records = []

    start_t = time.time()
    for idx, sample in enumerate(samples, 1):
        run_lbl = f"eval_{dataset_name}_{'poison' if poison else 'clean'}_{idx}"
        logger = RunLogger(out_dir=str(OUT_ROOT), run_label=run_lbl)
        orchestrator = SheafOrchestrator(
            agents, thresholds,
            max_new_bytes=sheaf_cfg.get("max_new_bytes", 256),
            mad_multiplier=cfg["gate"]["mad_multiplier"],
            k=sheaf_cfg.get("top_k", 16),
            min_support=sheaf_cfg.get("min_support"),
            byte_entropy_threshold=sheaf_cfg.get("byte_entropy_threshold"),
            tree=tree,
            logger=logger,
        )

        # Text in, text out — with mismatched tokenizers there is no shared id
        # sequence to pass around or decode.
        full_text, history = orchestrator.generate(sample["prompt"])
        gen_text = full_text[len(sample["prompt"]):].strip() \
            if full_text.startswith(sample["prompt"]) else full_text.strip()

        logger.log_final(sample["prompt"], gen_text, history)
        logger.close()

        if sample["type"] == "gsm":
            pred_parsed = parse_model_gsm_answer(gen_text)
        else:
            pred_parsed = parse_model_mcq_answer(gen_text)

        is_correct = (pred_parsed == sample["target_parsed"])
        if is_correct:
            correct_count += 1

        n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
        n_fusion = len(history) - n_fast
        n_escalated = sum(1 for h in history if h.get("escalated"))
        n_bytes = sum(h.get("n_bytes", 0) for h in history)

        total_fast_tokens += n_fast
        total_fusion_tokens += n_fusion
        total_escalated_tokens += n_escalated
        total_tokens_generated += len(history)

        print(f"[{idx}/{len(samples)}] Output: {gen_text[:50]!r} | Pred: {pred_parsed!r} | Target: {sample['target_parsed']!r} | {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

        records.append({
            "idx": idx,
            "prompt": sample["prompt"],
            "generated_text": gen_text,
            "extracted_prediction": pred_parsed,
            "target_ground_truth": sample["target_parsed"],
            "is_correct": is_correct,
            "tokens": len(history),
            "bytes_emitted": n_bytes,
            "fast_tokens": n_fast,
            "fusion_tokens": n_fusion,
            "escalated_tokens": n_escalated,
            "step_history_verbose": history
        })

    elapsed = time.time() - start_t
    acc_pct = (correct_count / len(samples)) * 100.0
    fast_pct = (total_fast_tokens / max(1, total_tokens_generated)) * 100.0
    fusion_pct = (total_fusion_tokens / max(1, total_tokens_generated)) * 100.0
    escalated_pct = (total_escalated_tokens / max(1, total_tokens_generated)) * 100.0

    result_summary = {
        "dataset": dataset_name,
        "category": category,
        "poisoned": poison,
        "num_samples": len(samples),
        "accuracy_percent": acc_pct,
        "correct_count": correct_count,
        "total_time_seconds": elapsed,
        "total_tokens": total_tokens_generated,
        "fast_path_tokens": total_fast_tokens,
        "fast_path_percent": fast_pct,
        "fusion_tokens": total_fusion_tokens,
        "fusion_percent": fusion_pct,
        "escalated_tokens": total_escalated_tokens,
        "escalated_percent": escalated_pct,
        "verbose_records": records
    }

    out_dir = str(RESULTS / "verbose")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{dataset_name}_{'poison' if poison else 'clean'}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    print(f"Results saved to: {out_file}")
    return result_summary


def generate_benchmark_plots(all_results):
    charts_dir = str(RESULTS / "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Chart 1: Accuracy Comparison (Clean vs Poisoned)
    datasets = [r["dataset"].upper() for r in all_results if not r["poisoned"]]
    clean_accs = [r["accuracy_percent"] for r in all_results if not r["poisoned"]]
    poison_accs = [r["accuracy_percent"] for r in all_results if r["poisoned"]]

    x = range(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar([i - width/2 for i in x], clean_accs, width, label='Clean Ensemble (N=3)', color='#2ca02c')
    rects2 = ax.bar([i + width/2 for i in x], poison_accs, width, label='Poisoned Ensemble (1 Inverted)', color='#d62728')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('SAHF Robustness: Accuracy Under Clean vs Poisoned Conditions')
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    chart1_path = os.path.join(charts_dir, "accuracy_comparison.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Generated chart: {chart1_path}")

    # Chart 2: Fast-Path vs Fusion Path Token Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    fast_pcts = [r["fast_path_percent"] for r in all_results if not r["poisoned"]]
    fusion_pcts = [r["fusion_percent"] for r in all_results if not r["poisoned"]]

    ax.bar(datasets, fast_pcts, label='Path A (Fast Passthrough)', color='#1f77b4')
    ax.bar(datasets, fusion_pcts, bottom=fast_pcts, label='Path B (Fusion & Escalation)', color='#ff7f0e')

    ax.set_ylabel('Percentage of Tokens (%)')
    ax.set_title('SAHF Path Allocation Across DeePEn Benchmarks')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    chart2_path = os.path.join(charts_dir, "path_allocation.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Generated chart: {chart2_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Full Evaluation Experiments across DeePEn Benchmarks")
    parser.add_argument("--config", default=str(ROOT / "config_sheaf.yaml"))
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    all_results = []
    datasets_to_test = [
        ("mmlu", "elementary_mathematics"),
        ("gsm", ""),
        ("arc", "")
    ]

    for ds, cat in datasets_to_test:
        # 1. Clean run
        res_clean = run_benchmark_eval(ds, cat, args.num_samples, args.config, poison=False)
        all_results.append(res_clean)

        # 2. Poisoned run
        res_poison = run_benchmark_eval(ds, cat, args.num_samples, args.config, poison=True)
        all_results.append(res_poison)

    generate_benchmark_plots(all_results)
    print("\nAll benchmark evaluation experiments finished successfully!")


if __name__ == "__main__":
    main()
