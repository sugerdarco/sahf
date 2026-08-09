"""
Batch prompt evaluation script for SAHF (cross-tokenizer / Stage 8).

Loads models ONCE into memory and runs inference sequentially across multiple
prompts. Each agent keeps its own tokenizer; generation emits bytes, which are
reconciled in a shared byte-prefix space. Configured by config_sheaf.yaml.

Usage:
    python run_batch_prompts.py --prompts-file prompts.txt
    python run_batch_prompts.py --prompts-file prompts.txt --poison-index 2 --poison-mode invert
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


def main():
    parser = argparse.ArgumentParser(description="Run batch prompts through SAHF pipeline (models loaded once).")
    parser.add_argument("--config", default="config_sheaf.yaml", help="Path to config yaml file.")
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
    raw_agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    poison_info = None
    if args.poison_index is not None:
        if not (0 <= args.poison_index < len(raw_agents)):
            raise ValueError(f"--poison-index {args.poison_index} out of range for {len(raw_agents)} agents.")
        target = raw_agents[args.poison_index]
        raw_agents[args.poison_index] = PoisonedAgentWrapper(
            target, mode=args.poison_mode, bias_strength=args.poison_strength,
        )
        poison_info = {
            "index": args.poison_index,
            "original_model": cfg["models"][args.poison_index],
            "mode": args.poison_mode,
            "strength": args.poison_strength,
        }
        app_log.info(f"POISONING agent {args.poison_index} ({target.name}) with mode={args.poison_mode}")

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
        output_text, history = orchestrator.generate(prompt)

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
