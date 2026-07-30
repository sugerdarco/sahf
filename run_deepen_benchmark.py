"""
DeePEn Benchmark Evaluation Script for SAHF.
Evaluates SAHF pipeline on DeePEn dataset benchmarks (GSM8K, MMLU, ARC-Challenge, etc.).

Usage:
    python run_deepen_benchmark.py --dataset gsm --max-questions 10
    python run_deepen_benchmark.py --dataset mmlu --category elementary_mathematics --max-questions 5
    python run_deepen_benchmark.py --dataset arc --max-questions 5
"""

import argparse
import json
import os
import time
import torch
import yaml

from sahf.agents import HFAgent, PoisonedAgentWrapper, assert_shared_vocab_size
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.orchestrator import FusionOrchestrator


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
    parser.add_argument("--config", default="config_local.yaml")
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
    agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    if args.poison_index is not None:
        target = agents[args.poison_index]
        agents[args.poison_index] = PoisonedAgentWrapper(target, mode=args.poison_mode)
        app_log.info(f"POISONING agent {args.poison_index} with mode={args.poison_mode}")

    shared_vocab_size = assert_shared_vocab_size(agents)
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
        orchestrator = FusionOrchestrator(
            agents, thresholds,
            max_new_tokens=cfg["max_new_tokens"],
            mad_multiplier=cfg["gate"]["mad_multiplier"],
            logger=logger,
        )

        tok = agents[0].tokenizer
        prompt_ids = tok(sample["prompt"], return_tensors="pt").input_ids
        output_ids, history = orchestrator.generate(prompt_ids, eos_token_id=tok.eos_token_id)
        output_text = tok.decode(output_ids[0], skip_special_tokens=True)

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
