import json
import shutil
import tempfile
from pathlib import Path

import torch

from sahf.agents import MockAgent
from sahf.gate import GateThresholds
from sahf.logger import RunLogger
from sahf.orchestrator import FusionOrchestrator


def test_two_agreeing_mock_agents_take_fast_path_and_log_correctly():
    tmp_out = tempfile.mkdtemp()
    try:
        agents = [
            MockAgent("mock-a", vocab_size=40, seed=1, bias_token=7, bias_strength=12.0),
            MockAgent("mock-b", vocab_size=40, seed=2, bias_token=7, bias_strength=12.0),
        ]
        thresholds = GateThresholds(entropy=2.0, divergence=0.2)
        logger = RunLogger(out_dir=tmp_out, run_label="test")
        logger.log_meta({"models": ["mock-a", "mock-b"], "prompt": "TEST"})

        orch = FusionOrchestrator(agents, thresholds, max_new_tokens=5, logger=logger)
        prompt_ids = torch.tensor([[1, 2, 3]])
        output_ids, history = orch.generate(prompt_ids)
        logger.log_final("TEST", "n/a", history)
        logger.close()

        assert len(history) == 5
        assert all(h["path"] == "A_fast_passthrough" for h in history)
        assert all(h["token_id"] == 7 for h in history)

        # verify the out/ directory structure was actually written to disk
        steps_path = logger.run_dir / "steps.jsonl"
        result_path = logger.run_dir / "result.json"
        meta_path = logger.run_dir / "meta.json"
        assert steps_path.exists() and result_path.exists() and meta_path.exists()

        lines = steps_path.read_text().strip().split("\n")
        assert len(lines) == 5
        json.loads(lines[0])  # each line must be valid JSON

        result = json.loads(result_path.read_text())
        assert result["num_fast_path"] == 5
        assert result["num_fusion_path"] == 0
    finally:
        shutil.rmtree(tmp_out)


def test_disagreeing_mock_agents_take_fusion_path():
    tmp_out = tempfile.mkdtemp()
    try:
        agents = [
            MockAgent("mock-a", vocab_size=40, seed=1, bias_token=5, bias_strength=15.0),
            MockAgent("mock-b", vocab_size=40, seed=2, bias_token=30, bias_strength=15.0),
        ]
        thresholds = GateThresholds(entropy=2.0, divergence=0.05)
        logger = RunLogger(out_dir=tmp_out, run_label="test2")

        orch = FusionOrchestrator(agents, thresholds, max_new_tokens=3, logger=logger)
        prompt_ids = torch.tensor([[1, 2, 3]])
        _, history = orch.generate(prompt_ids)
        logger.close()

        assert all(h["path"] == "B_fusion_pipeline" for h in history)
    finally:
        shutil.rmtree(tmp_out)


def test_three_agents_one_poisoned_escalates_and_recovers_majority_token():
    """With N=3 there's an honest majority (2 vs 1) — this is the smallest
    config where Stage 7's robustness guarantee is meaningful (see
    ARCHITECTURE.md limitations on N=2)."""
    tmp_out = tempfile.mkdtemp()
    try:
        agents = [
            MockAgent("honest-a", vocab_size=40, seed=1, bias_token=9, bias_strength=15.0),
            MockAgent("honest-b", vocab_size=40, seed=2, bias_token=9, bias_strength=15.0),
            MockAgent("poisoned", vocab_size=40, seed=3, bias_token=33, bias_strength=25.0),
        ]
        thresholds = GateThresholds(entropy=2.0, divergence=0.01)  # force disagreement path
        logger = RunLogger(out_dir=tmp_out, run_label="test3")

        orch = FusionOrchestrator(agents, thresholds, max_new_tokens=1, logger=logger)
        prompt_ids = torch.tensor([[1, 2, 3]])
        _, history = orch.generate(prompt_ids)
        logger.close()

        record = history[0]
        assert record["path"] == "B_fusion_pipeline"
        assert record["escalated"] is True
        assert record["outliers"][2] is True  # the poisoned agent (index 2) flagged
        assert record["token_id"] == 9  # honest majority wins despite the poisoned agent
    finally:
        shutil.rmtree(tmp_out)
