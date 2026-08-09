"""
DeePEn Benchmark Evaluation Script for SAHF (cross-tokenizer / Stage 8).

Evaluates the pipeline on DeePEn dataset benchmarks (GSM8K, MMLU, ARC-Challenge).
Single-dataset control with per-category and poison-index selection; use
run_full_evaluation_experiment.py for the full clean-vs-poisoned sweep with plots.
Configured by config_sheaf.yaml.

Usage:
    python run_deepen_benchmark.py --dataset gsm --max-questions 10
    python run_deepen_benchmark.py --dataset mmlu --category elementary_mathematics --max-questions 5
    python run_deepen_benchmark.py --dataset arc --max-questions 5
"""

import argparse
import json
import os
import time
from pathlib import Path
import torch
import yaml

from sahf.agents import HFAgent, PoisonedAgentWrapper
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.sheaf import (
    BytePrefixTree,
    SheafOrchestrator,
    UpstreamAgent,
    assert_distinct_tokenizers,
)


def load_deepen_samples(dataset_type: str, category: str = None, max_questions: int = 10):
    base_dir = "datasets"
    samples = []

    if dataset_type == "gsm":
        path = os.path.join(base_dir, "GSM", "data", "test.cleand.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                prompt = f"Question: {data['question']}\nAnswer step-by-step:"
                samples.append({"prompt": prompt, "target": data["answer"]})

    elif dataset_type == "mmlu":
        category_name = category or "elementary_mathematics"
        path = os.path.join(base_dir, "MMLU", "dev-jsonl", f"{category_name}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(base_dir, "MMLU", "test-jsonl", f"{category_name}.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                opts = f"A. {data['A']}\nB. {data['B']}\nC. {data['C']}\nD. {data['D']}"
                prompt = f"Question: {data['question']}\nOptions:\n{opts}\nSelect the correct option (A, B, C, or D):"
                samples.append({"prompt": prompt, "target": data["answer"]})

    elif dataset_type == "arc":
        path = os.path.join(base_dir, "ARC-Challenge", "test.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                opts = f"A. {data.get('A', '')}\nB. {data.get('B', '')}\nC. {data.get('C', '')}\nD. {data.get('D', '')}"
                prompt = f"Question: {data['question']}\nOptions:\n{opts}\nSelect the correct option:"
                samples.append({"prompt": prompt, "target": data.get("answer", "")})

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    return samples[:max_questions]


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAHF on DeePEn benchmarks.")
    parser.add_argument("--config", default="config_sheaf.yaml")
    parser.add_argument("--dataset", choices=["gsm", "mmlu", "arc"], default="gsm")
    parser.add_argument("--category", default="elementary_mathematics", help="Category for MMLU benchmark.")
    parser.add_argument("--max-questions", type=int, default=5, help="Number of questions to evaluate.")
    parser.add_argument("--poison-index", type=int, default=None)
    parser.add_argument("--poison-mode", default="invert", choices=PoisonedAgentWrapper.VALID_MODES)
    args = parser.parse_args()

    samples = load_deepen_samples(args.dataset, args.category, args.max_questions)
    print(f"Loaded {len(samples)} samples from DeePEn benchmark dataset: {args.dataset}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    app_log = setup_app_logging(cfg.get("out_dir", "out"))
    app_log.info(f"Loading agents for DeePEn benchmark evaluation: {cfg['models']}")

    dtype = getattr(torch, cfg["dtype"])
    raw_agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    if args.poison_index is not None:
        target = raw_agents[args.poison_index]
        raw_agents[args.poison_index] = PoisonedAgentWrapper(target, mode=args.poison_mode)
        app_log.info(f"POISONING agent {args.poison_index} with mode={args.poison_mode}")

    # Cross-tokenizer path: every agent keeps its own tokenizer and encodes the
    # shared context itself. There is deliberately no shared-vocabulary check —
    # a shared vocabulary is exactly what this configuration does not have.
    agents = [UpstreamAgent(a) for a in raw_agents]
    vocab_info = assert_distinct_tokenizers(agents)
    app_log.info(f"Vocabularies: {vocab_info}")

    # A mis-detected tokenizer scheme does not raise — it shifts every token by a
    # leading space and silently misaligns the byte tree. Verify once, up front.
    for a in agents:
        if not a.verify(["The capital of France is Paris", "Question: what is 2 + 2?"]):
            raise SystemExit(
                f"Byte round-trip FAILED for {a.name} (scheme={a.vocab_spec().scheme}). "
                "Results would be silently wrong; fix vocabulary extraction first."
            )

    sheaf_cfg = cfg.get("sheaf", {})
    tree = None
    _tp = sheaf_cfg.get("tree_path")
    if _tp and Path(_tp).exists():
        tree = BytePrefixTree.load(_tp)
        if [v.name for v in tree.vocabs] != cfg["models"]:
            raise SystemExit(
                f"{_tp} was built for a different model list. Rebuild it with "
                "build_prefix_tree.py, or delete it to build the tree per step."
            )
        app_log.info(f"Loaded prefix tree: {_tp} ({tree.n_nodes:,} nodes)")

    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"],
    )

    batch_start = time.time()
    for idx, sample in enumerate(samples, 1):
        print(f"\n--------------------------------------------------")
        print(f"[{idx}/{len(samples)}] Prompt:\n{sample['prompt']}")
        print(f"--------------------------------------------------")

        logger = RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=f"deepen_{args.dataset}_{idx}")
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
        # sequence to decode.
        output_text, history = orchestrator.generate(sample["prompt"])

        logger.log_final(sample["prompt"], output_text, history)
        logger.close()

        n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
        n_fusion = len(history) - n_fast
        n_escalated = sum(1 for h in history if h.get("escalated"))

        print(f"Output:\n{output_text}")
        print(f"Target/Ground Truth:\n{sample['target']}")
        print(f"Stats: {len(history)} tokens ({n_fast} Fast-Path, {n_fusion} Fusion, {n_escalated} Escalated)")

    print(f"\nBenchmark completed in {time.time() - batch_start:.2f}s")


if __name__ == "__main__":
    main()
