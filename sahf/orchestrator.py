"""
FusionOrchestrator — the per-token decode loop.

At every step: all agents run one forward pass on the same prefix -> Stage 1
(amplitude) -> Stage 2 (gate). Agree -> Path A (fast passthrough, no fusion
computed at all). Disagree -> Path B: Stage 5 (fast mean fusion) -> Stage 6
(outlier check) -> Stage 7 (Weiszfeld geometric median) only if an outlier was
flagged. The chosen token is appended and fed back to *every* agent for the next
step (safe here because both default agents share one tokenizer — no Stage 8
byte-level reconciliation is needed; see ARCHITECTURE.md).
"""

from typing import List, Optional

import torch

from .amplitude import softmax_to_amplitude
from .fusion import fast_mean_fusion
from .gate import GateThresholds, divergence_gate
from .robust import detect_outliers, weiszfeld_geometric_median


class FusionOrchestrator:
    def __init__(self, agents: List, thresholds: GateThresholds,
                 weights: Optional[List[float]] = None, max_new_tokens: int = 64,
                 mad_multiplier: float = 3.0, logger=None):
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents to fuse.")
        self.agents = agents
        self.thresholds = thresholds
        self.weights = weights or [1.0 / len(agents)] * len(agents)
        self.max_new_tokens = max_new_tokens
        self.mad_multiplier = mad_multiplier
        self.logger = logger

    def step(self, input_ids: torch.Tensor, step_idx: int) -> tuple:
        logits_list = [agent.next_logits(input_ids) for agent in self.agents]
        psi_list = [softmax_to_amplitude(logits)[0] for logits in logits_list]

        decision = divergence_gate(psi_list, self.thresholds)
        record = {"step": step_idx, "entropy": decision.entropy, "divergence": decision.divergence}

        if decision.agree:
            mean_probs = sum(psi**2 for psi in psi_list) / len(psi_list)
            token_id = int(torch.argmax(mean_probs).item())
            record.update({"path": "A_fast_passthrough", "token_id": token_id,
                           "escalated": False, "outliers": []})
        else:
            psi_star = fast_mean_fusion(psi_list, self.weights)
            outlier_mask, dists = detect_outliers(psi_list, psi_star, self.mad_multiplier)

            escalated = any(outlier_mask)
            if escalated:
                psi_star, n_iter = weiszfeld_geometric_median(psi_list, self.weights)
                record["weiszfeld_iterations"] = n_iter

            probs = psi_star**2
            token_id = int(torch.argmax(probs).item())
            record.update({"path": "B_fusion_pipeline", "token_id": token_id,
                           "escalated": escalated, "outliers": outlier_mask,
                           "outlier_distances": dists})

        if self.logger:
            self.logger.log_step(record)
        return token_id, record

    def generate(self, prompt_ids: torch.Tensor, eos_token_id: Optional[int] = None) -> tuple:
        input_ids = prompt_ids.clone()
        history = []
        for step_idx in range(self.max_new_tokens):
            token_id, record = self.step(input_ids, step_idx)
            history.append(record)
            next_tok = torch.tensor([[token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            if eos_token_id is not None and token_id == eos_token_id:
                break
        return input_ids, history
