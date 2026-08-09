"""
Runs the actual SheafOrchestrator (no shortcuts) against agents with three
DIFFERENT tokenizers, using no GPU and no internet. Populates out/ in the same
format as a real run.

MockAgent cannot be used here. It has no tokenizer at all — it emits logits over
an abstract vocabulary of integers — so there is no byte image for its tokens and
nothing for Stage 8 to reconcile. Instead this builds three real vocabularies
over the same target string at different compression ratios (6 bytes / 3 bytes /
1 byte per token), which is precisely the mismatch Stage 8 handles: the coarse
agent sees several bytes ahead per forward pass, the fine one only a single byte.

One agent is deliberately corrupted so Stage 6/7 fire per node and the
honest-majority recovery is visible in the logs, at N=3.

Usage:
    python demo_sheaf_mock_run.py
"""

import numpy as np

from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.sheaf import SheafOrchestrator, StaticAgent, VocabSpec

TARGET = b" Paris is the capital of France"
DECOY = b" Berlin is the capital of Germany"


def covering_vocab(name: str, max_len: int) -> VocabSpec:
    """A vocabulary covering both strings, capped at `max_len` bytes per token."""
    toks = set()
    for text in (TARGET, DECOY):
        for i in range(len(text)):
            for n in range(1, max_len + 1):
                toks.add(text[i : i + n])
    return VocabSpec.from_mapping({i: b for i, b in enumerate(sorted(toks))}, name=name)


class ScriptedAgent(StaticAgent):
    """Predicts its own longest token continuing `goal` from any context."""

    def __init__(self, name, vs, goal: bytes):
        super().__init__(name, vs)
        self._vs = vs
        self._goal = goal

    def next_token_probs(self, context: str) -> np.ndarray:
        produced = context.encode("utf-8")
        remaining = self._goal[len(produced) :] or self._goal
        best, blen = None, 0
        for i, tok in enumerate(self._vs.token_bytes):
            if tok and remaining.startswith(tok) and len(tok) > blen:
                best, blen = i, len(tok)
        p = np.full(self._vs.size, 0.02 / max(self._vs.size - 1, 1))
        if best is not None:
            p[best] = 0.98
        return p / p.sum()

    def stop_probability(self, context: str) -> float:
        return 1.0 if context.encode("utf-8") == TARGET else 0.0


app_log = setup_app_logging("out")
app_log.info("Starting Stage 8 mock demo run (no GPU / internet required).")

specs = [
    covering_vocab("coarse-6B", 6),
    covering_vocab("medium-3B", 3),
    covering_vocab("fine-1B", 1),
]
# 2 honest agents pursuing TARGET, 1 confidently wrong agent pursuing DECOY.
# The equivalent of PoisonedAgentWrapper(mode="invert"): well-formed, and wrong.
agents = [
    ScriptedAgent(specs[0].name, specs[0], TARGET),
    ScriptedAgent(specs[1].name, specs[1], TARGET),
    ScriptedAgent(specs[2].name + "-POISONED", specs[2], DECOY),
]

thresholds = GateThresholds(entropy=2.0, divergence=0.05)
logger = RunLogger(out_dir="out", run_label="sheaf_mock_demo")
logger.log_meta({
    "models": [a.name for a in agents],
    "prompt": "[stage 8 mock demo — three different tokenizers, no real models]",
    "stage8": True,
    "vocab_sizes": {s.name: s.size for s in specs},
    "note": "2 honest agents + 1 pursuing a decoy string, over vocabularies with "
            "6/3/1-byte tokens, to exercise Stage 6/7 per tree node with an "
            "honest MAJORITY (N=3) under genuine tokenizer mismatch.",
})
app_log.info(f"Run directory: {logger.run_dir}")

orchestrator = SheafOrchestrator(
    agents, thresholds, max_new_bytes=len(TARGET), k=32, logger=logger,
)
output_text, history = orchestrator.generate("")
logger.log_final("[stage 8 mock demo]", output_text, history)
logger.close()

n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
n_escalated = sum(1 for h in history if h.get("escalated"))
total_bytes = sum(h.get("n_bytes", 0) for h in history)
correct = sum(1 for i, b in enumerate(output_text.encode()) if i < len(TARGET) and b == TARGET[i])

app_log.info(
    f"Done: {len(history)} steps, {n_fast} fast-path, {n_escalated} escalated, "
    f"{total_bytes} bytes emitted, {correct}/{len(TARGET)} bytes matching the "
    f"honest majority's target despite one poisoned agent."
)
print(f"Output: {output_text!r}")
print(f"Wrote a real example run to: {logger.run_dir}")
