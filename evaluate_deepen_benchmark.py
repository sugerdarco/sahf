"""
Comprehensive DeePEn-Compatible Benchmark Evaluator for SAHF.

Evaluates SAHF on DeePEn datasets (GSM8K, MMLU, ARC-Challenge) with automatic answer
extraction, accuracy metrics computation, and direct comparison reports.

Usage:
    python evaluate_deepen_benchmark.py --dataset gsm --num-samples 10
    python evaluate_deepen_benchmark.py --dataset mmlu --category elementary_mathematics --num-samples 10
    python evaluate_deepen_benchmark.py --dataset arc --num-samples 10
"""

import argparse
import json
import os
import re
import time
import torch
import yaml

from sahf.agents import HFAgent, PoisonedAgentWrapper, assert_shared_vocab_size
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.orchestrator import FusionOrchestrator


def extract_gsm_target_number(answer_str: str) -> str:
    """Extracts numerical answer from GSM8K ground truth string."""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "").replace("$", "")
    match = re.search(r'is\s+([0-9\.\,]+)', answer_str, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").replace("$", "")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_str)
    return numbers[-1] if numbers else answer_str.strip()


def parse_model_gsm_answer(text: str) -> str:
    """Parses predicted number from model output."""
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "").replace("$", "")
    
    match = re.search(r'(?:answer|result)\s*(?:is|=|:)?\s*\$?([0-9\.\,]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").replace("$", "")
    
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return numbers[-1].replace(",", "").replace("$", "") if numbers else ""


def parse_model_mcq_answer(text: str) -> str:
    """Parses multiple-choice option (A, B, C, or D) from model output."""
    # First check for standalone option at the start of response e.g. "A" or "A)" or "Option A"
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


def load_deepen_dataset(dataset_name: str, category: str = "elementary_mathematics", num_samples: int = 20):
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
                    "Calculate step-by-step and write #### followed by the final answer.\n"
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

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return samples[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAHF on DeePEn datasets with Accuracy metrics.")
    parser.add_argument("--config", default="config_local.yaml")
    parser.add_argument("--dataset", choices=["gsm", "mmlu", "arc"], default="gsm")
    parser.add_argument("--category", default="elementary_mathematics")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test samples to evaluate.")
    parser.add_argument("--poison-index", type=int, default=None)
    parser.add_argument("--poison-mode", default="invert", choices=PoisonedAgentWrapper.VALID_MODES)
    args = parser.parse_args()

    samples = load_deepen_dataset(args.dataset, args.category, args.num_samples)
    print(f"\n==================================================================")
    print(f"Starting DeePEn Evaluation: Dataset={args.dataset.upper()}, Samples={len(samples)}")
    print(f"==================================================================")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    app_log = setup_app_logging(cfg.get("out_dir", "out"))

    dtype = getattr(torch, cfg["dtype"])
    agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    if args.poison_index is not None:
        target = agents[args.poison_index]
        agents[args.poison_index] = PoisonedAgentWrapper(target, mode=args.poison_mode)
        print(f"[INFO] POISONED Agent {args.poison_index} with mode={args.poison_mode}")

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
    eval_records = []

    start_time = time.time()

    for idx, sample in enumerate(samples, 1):
        logger = RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=f"eval_{args.dataset}_{idx}")
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
        
        if raw_output.startswith(sample["prompt"]):
            gen_text = raw_output[len(sample["prompt"]):].strip()
        else:
            gen_text = raw_output.strip()

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

        print(f"[{idx}/{len(samples)}] Output: {gen_text[:60]!r} | Extracted: {pred_parsed!r} | Target: {sample['target_parsed']!r} | {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

        eval_records.append({
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
        })

    elapsed = time.time() - start_time
    accuracy = (correct_count / len(samples)) * 100.0
    fast_pct = (total_fast_tokens / max(1, total_tokens_generated)) * 100.0
    fusion_pct = (total_fusion_tokens / max(1, total_tokens_generated)) * 100.0
    escalated_pct = (total_escalated_tokens / max(1, total_tokens_generated)) * 100.0

    print(f"\n==================================================================")
    print(f"DEMO BENCHMARK EVALUATION COMPLETE: {args.dataset.upper()}")
    print(f"==================================================================")
    print(f"Accuracy / Exact Match: {accuracy:.2f}% ({correct_count}/{len(samples)})")
    print(f"Total Time Elapsed:     {elapsed:.2f}s")
    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Fast-Path (Path A) %:  {fast_pct:.2f}% ({total_fast_tokens} tokens)")
    print(f"Fusion Path (Path B) %: {fusion_pct:.2f}% ({total_fusion_tokens} tokens)")
    print(f"Escalated (Stage 7) %:  {escalated_pct:.2f}% ({total_escalated_tokens} tokens)")
    print(f"==================================================================\n")

    os.makedirs("out/evaluation_results", exist_ok=True)
    out_json = f"out/evaluation_results/deepen_{args.dataset}_eval_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "category": args.category,
            "num_samples": len(samples),
            "accuracy_percent": accuracy,
            "correct_count": correct_count,
            "total_time_seconds": elapsed,
            "total_tokens": total_tokens_generated,
            "fast_path_tokens": total_fast_tokens,
            "fast_path_percent": fast_pct,
            "fusion_tokens": total_fusion_tokens,
            "fusion_percent": fusion_pct,
            "escalated_tokens": total_escalated_tokens,
            "escalated_percent": escalated_pct,
            "records": eval_records
        }, f, indent=2)
    print(f"Saved complete evaluation report to: {out_json}")


if __name__ == "__main__":
    main()
