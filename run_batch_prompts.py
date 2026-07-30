"""
Batch prompt evaluation script for SAHF.
Loads models ONCE into memory and runs inference sequentially across multiple prompts.

Usage:
    python run_batch_prompts.py --prompts-file prompts.txt
    python run_batch_prompts.py --prompts-file prompts.txt --poison-index 2 --poison-mode invert
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


def main():
    parser = argparse.ArgumentParser(description="Run batch prompts through SAHF pipeline (models loaded once).")
    parser.add_argument("--config", default="config_local.yaml", help="Path to config yaml file.")
    parser.add_argument("--prompts-file", default="prompts.txt", help="Text file containing prompts (one per line).")
    parser.add_argument("--run-label", default="batch_eval", help="Label for batch output folder.")
    parser.add_argument("--poison-index", type=int, default=None,
                        help="0-based index of agent to poison across all prompts.")
    parser.add_argument("--poison-mode", default="invert", choices=PoisonedAgentWrapper.VALID_MODES,
                        help="Poisoning mode: invert, uniform_noise, random_bias.")
    parser.add_argument("--poison-strength", type=float, default=25.0,
                        help="Poison strength for noise/bias modes.")
    args = parser.parse_args()

    prompts_file_path = args.prompts_file
    if not os.path.exists(prompts_file_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file_path}")

    with open(prompts_file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    print(f"Loaded {len(prompts)} prompts from {prompts_file_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    app_log = setup_app_logging(cfg.get("out_dir", "out"))
    app_log.info(f"Loading agents ONCE for batch run: {cfg['models']}")

    dtype = getattr(torch, cfg["dtype"])
    agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    poison_info = None
    if args.poison_index is not None:
        if not (0 <= args.poison_index < len(agents)):
            raise ValueError(f"--poison-index {args.poison_index} out of range for {len(agents)} agents.")
        target = agents[args.poison_index]
        agents[args.poison_index] = PoisonedAgentWrapper(
            target, mode=args.poison_mode, bias_strength=args.poison_strength,
        )
        poison_info = {
            "index": args.poison_index,
            "original_model": cfg["models"][args.poison_index],
            "mode": args.poison_mode,
            "strength": args.poison_strength,
        }
        app_log.info(f"POISONING agent {args.poison_index} ({target.name}) with mode={args.poison_mode}")

    shared_vocab_size = assert_shared_vocab_size(agents)
    app_log.info(f"Shared vocab size confirmed: {shared_vocab_size}")

    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"],
    )

    batch_start_time = time.time()
    batch_results = []

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n==================================================")
        print(f"[{idx}/{len(prompts)}] Processing Prompt: {prompt!r}")
        print(f"==================================================")

        run_label = f"{args.run_label}_p{idx}"
        logger = RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=run_label)
        logger.log_meta({
            "prompt_idx": idx,
            "models": cfg["models"],
            "prompt": prompt,
            "config": cfg,
            "poisoning": poison_info,
        })

        orchestrator = FusionOrchestrator(
            agents, thresholds,
            max_new_tokens=cfg["max_new_tokens"],
            mad_multiplier=cfg["gate"]["mad_multiplier"],
            logger=logger,
        )

        tok = agents[0].tokenizer
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        output_ids, history = orchestrator.generate(prompt_ids, eos_token_id=tok.eos_token_id)
        output_text = tok.decode(output_ids[0], skip_special_tokens=True)

        logger.log_final(prompt, output_text, history)
        logger.close()

        n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
        n_fusion = len(history) - n_fast
        n_escalated = sum(1 for h in history if h.get("escalated"))

        result_summary = {
            "prompt_idx": idx,
            "prompt": prompt,
            "output_text": output_text,
            "total_tokens": len(history),
            "fast_path_tokens": n_fast,
            "fusion_tokens": n_fusion,
            "escalated_tokens": n_escalated,
            "log_dir": logger.run_dir,
        }
        batch_results.append(result_summary)

        print(f"-> Response: {output_text}")
        print(f"-> Stats: {len(history)} tokens generated ({n_fast} Fast-Path, {n_fusion} Fusion, {n_escalated} Escalated)")

    total_duration = time.time() - batch_start_time
    print(f"\n==================================================")
    print(f"Batch evaluation complete! Processed {len(prompts)} prompts in {total_duration:.2f}s")
    print(f"==================================================")


if __name__ == "__main__":
    main()
