"""
CLI entry point for a real run with 2 Hugging Face agents sharing one tokenizer.

Usage:
    python run.py --prompt "Explain the water cycle in two sentences."
    python run.py --prompt "..." --config config.yaml
"""

import argparse

import torch
import yaml

from sahf.agents import HFAgent
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.orchestrator import FusionOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-label", default=None, help="Optional label appended to the run folder name.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    app_log = setup_app_logging(cfg.get("out_dir", "out"))
    app_log.info(f"Loading agents: {cfg['models']}")

    dtype = getattr(torch, cfg["dtype"])
    agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    # Shared-tokenizer assumption for the fast version — fail loudly, not silently,
    # if someone points this at two models with different vocabularies.
    vocab_sizes = {len(a.tokenizer) for a in agents}
    if len(vocab_sizes) != 1:
        raise ValueError(
            "Agents report different tokenizer vocab sizes. The fast version assumes a "
            "shared tokenizer (no Stage 8 reconciliation) — pick two models from the same "
            "tokenizer family, e.g. two Qwen2.5 sizes."
        )

    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"],
    )

    logger = RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=args.run_label)
    logger.log_meta({"models": cfg["models"], "prompt": args.prompt, "config": cfg})
    app_log.info(f"Run directory: {logger.run_dir}")

    orchestrator = FusionOrchestrator(
        agents, thresholds,
        max_new_tokens=cfg["max_new_tokens"],
        mad_multiplier=cfg["gate"]["mad_multiplier"],
        logger=logger,
    )

    tok = agents[0].tokenizer
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids
    output_ids, history = orchestrator.generate(prompt_ids, eos_token_id=tok.eos_token_id)
    output_text = tok.decode(output_ids[0], skip_special_tokens=True)

    logger.log_final(args.prompt, output_text, history)
    logger.close()

    n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
    n_fusion = len(history) - n_fast
    n_escalated = sum(1 for h in history if h.get("escalated"))
    app_log.info(
        f"Done: {len(history)} steps ({n_fast} fast-path, {n_fusion} fusion-path, "
        f"{n_escalated} escalated to geometric median)."
    )

    print(output_text)


if __name__ == "__main__":
    main()
