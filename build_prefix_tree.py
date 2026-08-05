"""
Build the Stage 8 byte-prefix tree ONCE and save it as a reusable artifact.

The tree is a property of the ensemble's vocabularies, not of any prompt. It does
not change between queries, between runs, or between decoding steps, so it is
built here by a separate command and then loaded by run_sheaf.py:

    python build_prefix_tree.py                              # uses config_sheaf.yaml
    python run_sheaf.py --prompt "..."                       # loads the artifact

Only TOKENIZERS are loaded here — no model weights, no GPU. On a machine with the
tokenizers already cached this is a CPU-only job.

What the artifact contains: the union trie over every agent's vocabulary, each
token's precomputed root-to-leaf path (which is what lets a decoding step
propagate only its top-k tokens), and the extracted token->bytes tables. Because
the vocabularies travel with the tree, a run never has to load a tokenizer just
to reconcile bytes.

Rebuild it whenever the model list changes. run_sheaf.py checks the agents
against the artifact and refuses to run on a mismatch rather than silently
reconciling against the wrong vocabulary.
"""

import argparse
import time
from pathlib import Path

import yaml

from sahf.sheaf import BytePrefixTree, VocabSpec

VERIFY_SAMPLES = [
    "The capital of France is Paris",
    "the cat sat on the mat",
    "def main():\n    return 1",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", default="config_sheaf.yaml")
    parser.add_argument("--out", default=None, help="Artifact path (default: from config).")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Truncate tokens longer than this many bytes. Omit for exact.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the byte round-trip check. Not recommended.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = Path(args.out or cfg.get("sheaf", {}).get("tree_path", "artifacts/prefix_tree.npz"))

    from transformers import AutoTokenizer

    specs = []
    for name in cfg["models"]:
        print(f"loading tokenizer: {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        vs = VocabSpec.from_hf(tok, name=name)
        print(f"  scheme={vs.scheme}  vocab={vs.size}")

        if not args.skip_verify:
            # A mis-detected display scheme does not raise -- it shifts every
            # token by one leading space and produces a plausible but silently
            # misaligned tree. Catch it here, once, rather than at inference.
            if not vs.verify(tok, VERIFY_SAMPLES):
                raise SystemExit(
                    f"Byte round-trip FAILED for {name} (scheme={vs.scheme}). "
                    "The tree would be silently misaligned; fix extraction first."
                )
        specs.append(vs)

    print("\nbuilding union byte-prefix tree ...", flush=True)
    t0 = time.perf_counter()
    tree = BytePrefixTree.from_vocabs(specs, max_depth=args.max_depth)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree.save(out)
    save_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    reloaded = BytePrefixTree.load(out)
    load_s = time.perf_counter() - t0
    assert reloaded.n_nodes == tree.n_nodes, "artifact did not round-trip"

    size_mb = out.stat().st_size / 1e6
    print(
        f"\n  nodes      : {tree.n_nodes:,}\n"
        f"  max depth  : {tree.max_depth_seen}\n"
        f"  agents     : {tree.n_agents}\n"
        f"  build      : {build_s:.1f} s\n"
        f"  save       : {save_s:.1f} s\n"
        f"  reload     : {load_s:.1f} s\n"
        f"  artifact   : {out} ({size_mb:.1f} MB)\n"
    )
    print("Run with:  python run_sheaf.py --prompt \"...\"")


if __name__ == "__main__":
    main()
