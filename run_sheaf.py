"""
CLI entry point for a Stage 8 run with N Hugging Face agents that DO NOT share a
tokenizer.

Agents from different model families, whose vocabularies have to be reconciled at
the byte level before they can be fused at all. Their amplitude vectors are
indexed by different token ids and are not even the same length, so there is no
token-space fusion to fall back on.

Usage:
    python run_sheaf.py --prompt "Explain the water cycle in two sentences."
    python run_sheaf.py --prompt "..." --config config_sheaf.yaml

Poisoning is the way to actually exercise Stage 6/7 inside Stage 8, rather than
hoping honest models disagree enough on their own:

    python run_sheaf.py --prompt "..."                                  # baseline
    python run_sheaf.py --prompt "..." --poison-index 2 --poison-mode invert

Compare out/runs/*/steps.jsonl between the two. Records keep the original field
names (path / escalated / entropy / divergence) and add Stage 8 fields —
tree_nodes, fused_nodes, escalated_nodes, coverage, unreachable, stop_mass — so
existing log tooling still reads them.

Verification note: every agent's byte extraction is round-tripped before the run
starts. A tokenizer whose display scheme is mis-detected does not raise, it
shifts every token by one leading space and silently misaligns the whole tree,
so this run aborts rather than producing plausible nonsense. See
docs/STAGE8_AUDIT_REPORT.md (AUDIT-1).
"""

import argparse
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

VERIFY_SAMPLES = [
    "The capital of France is Paris",
    "the cat sat on the mat",
    "def main():\n    return 1",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sheaf.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-label", default=None, help="Optional label appended to the run folder name.")
    parser.add_argument("--poison-index", type=int, default=None,
                        help="Index (0-based, into config's models list) of the agent to "
                             "deliberately corrupt, for testing Stage 6/7. Omit to run clean.")
    parser.add_argument("--poison-mode", default="invert", choices=PoisonedAgentWrapper.VALID_MODES,
                        help="How to corrupt the poisoned agent's logits. Default: invert.")
    parser.add_argument("--poison-strength", type=float, default=25.0,
                        help="Magnitude used by 'uniform_noise' / 'random_bias' poison modes.")
    parser.add_argument("--tree", default=None,
                        help="Prefix-tree artifact from build_prefix_tree.py. Defaults to "
                             "sheaf.tree_path in the config. Pass --no-tree to rebuild per "
                             "step instead (slower; only useful for debugging).")
    parser.add_argument("--no-tree", action="store_true",
                        help="Ignore the prebuilt artifact and rebuild the tree every step.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the byte round-trip check. Not recommended — see module docstring.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    app_log = setup_app_logging(cfg.get("out_dir", "out"))
    app_log.info(f"Loading agents (mismatched tokenizers expected): {cfg['models']}")

    dtype = getattr(torch, cfg["dtype"])
    raw_agents = [HFAgent(name, device=cfg["device"], dtype=dtype) for name in cfg["models"]]

    poison_info = None
    if args.poison_index is not None:
        if not (0 <= args.poison_index < len(raw_agents)):
            raise ValueError(f"--poison-index {args.poison_index} out of range for "
                             f"{len(raw_agents)} configured agents.")
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
        app_log.info(f"POISONING agent {args.poison_index} ({target.name}) mode={args.poison_mode}")

    agents = [UpstreamAgent(a) for a in raw_agents]

    # Flags the case where Stage 8 is unnecessary overhead: if the agents do
    # share a tokenizer, token-level fusion would be cheaper and would not
    # truncate to top-k.
    info = assert_distinct_tokenizers(agents)
    vocab_sizes = {name: size for name, (size, _scheme) in info.items()}
    app_log.info(f"Vocabularies: {info}")

    if not args.skip_verify:
        for a in agents:
            if not a.verify(VERIFY_SAMPLES):
                raise SystemExit(
                    f"Byte round-trip FAILED for {a.name} (scheme="
                    f"{a.vocab_spec().scheme}). Stage 8 would build a silently "
                    "misaligned tree. Fix the vocabulary extraction before running."
                )
        app_log.info("Byte extraction verified for all agents.")

    thresholds = GateThresholds(
        entropy=cfg["gate"]["entropy_threshold"],
        divergence=cfg["gate"]["divergence_threshold"],
    )

    logger = RunLogger(out_dir=cfg.get("out_dir", "out"), run_label=args.run_label)
    logger.log_meta({
        "models": cfg["models"],
        "prompt": args.prompt,
        "config": cfg,
        "poisoning": poison_info,
        "stage8": True,
        "prefix_tree": str(args.tree or sheaf_cfg.get("tree_path")) if tree is not None else None,
        "vocab_sizes": vocab_sizes,
        "schemes": {a.name: a.vocab_spec().scheme for a in agents},
    })
    app_log.info(f"Run directory: {logger.run_dir}")

    sheaf_cfg = cfg.get("sheaf", {})

    tree = None
    if not args.no_tree:
        tree_path = Path(args.tree or sheaf_cfg.get("tree_path", "artifacts/prefix_tree.npz"))
        if not tree_path.exists():
            raise SystemExit(
                f"No prefix tree at {tree_path}. It is built once, separately:\n"
                f"    python build_prefix_tree.py --config {args.config}\n"
                "Or pass --no-tree to rebuild it every step (much slower)."
            )
        tree = BytePrefixTree.load(tree_path)
        if [v.name for v in tree.vocabs] != cfg["models"]:
            raise SystemExit(
                f"{tree_path} was built for {[v.name for v in tree.vocabs]} but the config "
                f"lists {cfg['models']}. Reconciling against the wrong vocabularies would "
                "produce silently wrong output. Rebuild with build_prefix_tree.py."
            )
        app_log.info(f"Loaded prefix tree: {tree_path} ({tree.n_nodes:,} nodes)")

    orchestrator = SheafOrchestrator(
        agents, thresholds,
        max_new_bytes=sheaf_cfg.get("max_new_bytes", 256),
        mad_multiplier=cfg["gate"]["mad_multiplier"],
        k=sheaf_cfg.get("top_k", 16),
        byte_entropy_threshold=sheaf_cfg.get("byte_entropy_threshold"),
        min_support=sheaf_cfg.get("min_support"),
        tree=tree,
        logger=logger,
    )

    output_text, history = orchestrator.generate(args.prompt)

    logger.log_final(args.prompt, output_text, history)
    logger.close()

    n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
    n_fusion = len(history) - n_fast
    n_escalated = sum(1 for h in history if h.get("escalated"))
    total_bytes = sum(h.get("n_bytes", 0) for h in history)
    mean_ms = sum(h.get("elapsed_ms", 0.0) for h in history) / max(len(history), 1)
    app_log.info(
        f"Done: {len(history)} steps ({n_fast} fast-path, {n_fusion} fusion-path, "
        f"{n_escalated} escalated), {total_bytes} bytes emitted, "
        f"{mean_ms:.1f} ms/step mean."
    )

    print(output_text)


if __name__ == "__main__":
    main()
