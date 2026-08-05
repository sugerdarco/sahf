"""
Master Evaluation Suite and Plot Generator for SAHF vs DeePEn Benchmarks.

Runs comprehensive evaluations across GSM8K, MMLU, and ARC-Challenge under both:
1. Clean baseline ensemble (3 honest agents)
2. Poisoned / Adversarial ensemble (1 inverted agent, 2 honest agents)

Saves verbose per-question step data, JSON summaries, and auto-generates benchmark comparison graphs.
"""

import argparse
import json
import os
import re
import time
import torch
import yaml
import matplotlib.pyplot as plt

from sahf.agents import HFAgent, PoisonedAgentWrapper, assert_shared_vocab_size
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.orchestrator import FusionOrchestrator


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
    base_dir = "datasets"
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


def run_benchmark_eval(dataset_name: str, category: str, num_samples: int, config_file: str, poison: bool = False):
    samples = load_dataset_samples(dataset_name, category, num_samples)
    mode_str = "POISONED" if poison else "CLEAN"
    print(f"\n==================================================================")
    print(f"RUNNING BENCHMARK [{mode_str}]: {dataset_name.upper()} ({len(samples)} samples)")
    print(f"==================================================================")

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    setup_app_logging(cfg.get("out_dir", "out"))
    dtype = getattr(torch, cfg["dtype"])
    agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    if poison:
        agents[2] = PoisonedAgentWrapper(agents[2], mode="invert")
        print("[INFO] Poisoned Agent 2 (Probability Inversion)")

    shared_vocab_size = assert_shared_vocab_size(agents)
    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"],
    )

    correct_count = 0
    total_fast_tokens = 0
    total_fusion_tokens = 0
    total_escalated_tokens = 0
    total_tokens_generated = 0
    records = []

    start_t = time.time()
    for idx, sample in enumerate(samples, 1):
        run_lbl = f"eval_{dataset_name}_{'poison' if poison else 'clean'}_{idx}"
        logger = RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=run_lbl)
        orchestrator = FusionOrchestrator(
            agents, thresholds,
            max_new_tokens=cfg["max_new_tokens"],
            mad_multiplier=cfg["gate"]["mad_multiplier"],
            logger=logger,
        )

        tok = agents[0].tokenizer
        prompt_ids = tok(sample["prompt"], return_tensors="pt").input_ids
        output_ids, history = orchestrator.generate(prompt_ids, eos_token_id=tok.eos_token_id)
        raw_output = tok.decode(output_ids[0], skip_special_tokens=True)
        gen_text = raw_output[len(sample["prompt"]):].strip() if raw_output.startswith(sample["prompt"]) else raw_output.strip()

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

    out_dir = "out/evaluation_results/verbose"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{dataset_name}_{'poison' if poison else 'clean'}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    print(f"Results saved to: {out_file}")
    return result_summary


def generate_benchmark_plots(all_results):
    charts_dir = "out/evaluation_results/charts"
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
    parser.add_argument("--config", default="config_local.yaml")
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
