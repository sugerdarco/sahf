"""
Runs the actual orchestrator (no shortcuts) against MockAgent instead of real
models, so it needs no GPU and no internet access. Useful to sanity-check the
pipeline on a new machine before spending time downloading real model weights,
and to see the out/ log format populated with a real example.

Includes a 3rd, deliberately "poisoned" mock agent for a few steps so you can
see Stage 6/7 (outlier check -> escalation) actually fire in the logs.

Usage:
    python demo_mock_run.py
"""

import torch

from sahf.agents import MockAgent
from sahf.gate import GateThresholds
from sahf.logger import RunLogger, setup_app_logging
from sahf.orchestrator import FusionOrchestrator

app_log = setup_app_logging("out")
app_log.info("Starting mock demo run (no GPU / internet required).")

agents = [
    MockAgent("honest-agent-1", vocab_size=64, seed=1, bias_token=12, bias_strength=14.0),
    MockAgent("honest-agent-2", vocab_size=64, seed=2, bias_token=12, bias_strength=14.0),
    MockAgent("poisoned-agent", vocab_size=64, seed=3, bias_token=50, bias_strength=20.0),
]

thresholds = GateThresholds(entropy=2.0, divergence=0.05)
logger = RunLogger(out_dir="out", run_label="mock_demo")
logger.log_meta({
    "models": [a.name for a in agents],
    "prompt": "[mock demo — no real prompt/tokenizer]",
    "note": "2 honest mock agents + 1 poisoned mock agent, to exercise Stage 6/7.",
})
app_log.info(f"Run directory: {logger.run_dir}")

orchestrator = FusionOrchestrator(agents, thresholds, max_new_tokens=10, logger=logger)
prompt_ids = torch.tensor([[1, 2, 3]])
output_ids, history = orchestrator.generate(prompt_ids)
logger.log_final("[mock demo]", "[mock demo has no real decoded text]", history)
logger.close()

n_fast = sum(1 for h in history if h["path"] == "A_fast_passthrough")
n_escalated = sum(1 for h in history if h.get("escalated"))
app_log.info(f"Done: {len(history)} steps, {n_fast} fast-path, {n_escalated} escalated.")
print(f"Wrote a real example run to: {logger.run_dir}")
