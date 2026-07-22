"""
Runs the actual orchestrator (no shortcuts) against MockAgent instead of real
models, so it needs no GPU and no internet access. Useful to sanity-check the
pipeline on a new machine before spending time downloading real model weights,
and to see the out/ log format populated with a real example.

Uses 3 agents — 2 honest, 1 wrapped with PoisonedAgentWrapper (the exact same
mechanism `run.py --poison-index` uses on real models) — so you can see Stage
6/7 (outlier check -> escalation -> honest-majority recovery) actually fire in
the logs, with N=3 being the minimum where that recovery claim is meaningful
(see ARCHITECTURE.md).

Usage:
    python demo_mock_run.py
"""

import torch

from sahf.agents import MockAgent, PoisonedAgentWrapper
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.orchestrator import FusionOrchestrator

app_log = setup_app_logging("out")
app_log.info("Starting mock demo run (no GPU / internet required).")

honest_1 = MockAgent("honest-agent-1", vocab_size=64, seed=1, bias_token=12, bias_strength=14.0)
honest_2 = MockAgent("honest-agent-2", vocab_size=64, seed=2, bias_token=12, bias_strength=14.0)
honest_3_underlying = MockAgent("honest-agent-3", vocab_size=64, seed=3, bias_token=12, bias_strength=14.0)
poisoned = PoisonedAgentWrapper(honest_3_underlying, mode="invert", seed=42)

agents = [honest_1, honest_2, poisoned]

thresholds = GateThresholds(entropy=2.0, divergence=0.05)
logger = RunLogger(out_dir="out", run_label="mock_demo")
logger.log_meta({
    "models": [a.name for a in agents],
    "prompt": "[mock demo — no real prompt/tokenizer]",
    "note": "2 honest mock agents + 1 agent poisoned via PoisonedAgentWrapper(mode='invert'), "
            "to exercise Stage 6/7 with a real honest MAJORITY (N=3).",
})
app_log.info(f"Run directory: {logger.run_dir}")

orchestrator = FusionOrchestrator(agents, thresholds, max_new_tokens=10, logger=logger)
prompt_ids = torch.tensor([[1, 2, 3]])
output_ids, history = orchestrator.generate(prompt_ids)
logger.log_final("[mock demo]", "[mock demo has no real decoded text]", history)
logger.close()

n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
n_escalated = sum(1 for h in history if h.get("escalated"))
n_recovered = sum(1 for h in history if h.get("token_id") == 12)  # the honest-majority token
app_log.info(
    f"Done: {len(history)} steps, {n_fast} fast-path, {n_escalated} escalated, "
    f"{n_recovered}/{len(history)} steps recovered the honest-majority token despite poisoning."
)
print(f"Wrote a real example run to: {logger.run_dir}")
