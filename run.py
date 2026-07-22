"""
CLI entry point for a real run with N Hugging Face agents sharing one tokenizer.

Usage:
    python run.py --prompt "Explain the water cycle in two sentences."
    python run.py --prompt "..." --config config.yaml

Testing Stage 6/7 (outlier check + geometric median) on real hardware:
honest real models often just won't disagree enough, on their own, to prove
the escalation tier actually protects you. Rather than hoping, deliberately
poison one configured agent and compare:

    python run.py --prompt "..."                                   # baseline
    python run.py --prompt "..." --poison-index 2 --poison-mode invert
                                                                    # same prompt, one agent corrupted

Compare the two runs' out/runs/*/result.json — with N=3 and only one poisoned
agent, the final text should be largely unaffected (the honest majority
should keep winning at the escalated steps). meta.json always records whether
poisoning was used, and steps.jsonl shows exactly which steps escalated.
"""

import argparse

import torch
import yaml

from sahf.agents import HFAgent, PoisonedAgentWrapper, assert_shared_vocab_size
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.orchestrator import FusionOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-label", default=None, help="Optional label appended to the run folder name.")
    parser.add_argument("--poison-index", type=int, default=None,
                         help="Index (0-based, into config's models list) of the agent to "
                              "deliberately corrupt, for testing Stage 6/7. Omit to run clean.")
    parser.add_argument("--poison-mode", default="invert", choices=PoisonedAgentWrapper.VALID_MODES,
                         help="How to corrupt the poisoned agent's logits. Default: invert "
                              "(most realistic — still well-formed, confidently wrong).")
    parser.add_argument("--poison-strength", type=float, default=25.0,
                         help="Magnitude used by 'uniform_noise' / 'random_bias' poison modes.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    app_log = setup_app_logging(cfg.get("out_dir", "out"))
    app_log.info(f"Loading agents: {cfg['models']}")

    dtype = getattr(torch, cfg["dtype"])
    agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    poison_info = None
    if args.poison_index is not None:
        if not (0 <= args.poison_index < len(agents)):
            raise ValueError(f"--poison-index {args.poison_index} out of range for "
                              f"{len(agents)} configured agents.")
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

    # Shared-tokenizer assumption for the fast version — fail loudly, not silently.
    shared_vocab_size = assert_shared_vocab_size(agents)
    app_log.info(f"Shared vocab size confirmed: {shared_vocab_size}")

    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"],
    )

    logger = RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=args.run_label)
    logger.log_meta({
        "models": cfg["models"],
        "prompt": args.prompt,
        "config": cfg,
        "poisoning": poison_info,  # null if this was a clean run
    })
    app_log.info(f"Run directory: {logger.run_dir}")

    orchestrator = FusionOrchestrator(
        agents, thresholds,
        max_new_tokens=cfg["max_new_tokens"],
        mad_multiplier=cfg["gate"]["mad_multiplier"],
        logger=logger,
    )

    # Tokenizer comes from whichever agent wasn't replaced by a wrapper (a
    # PoisonedAgentWrapper forwards .tokenizer from the agent it wraps, so any
    # agent works here regardless of poisoning).
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
