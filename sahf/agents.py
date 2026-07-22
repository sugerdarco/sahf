"""
Agent wrappers. Both expose the same `next_logits(input_ids) -> (1, vocab)` and
`.tokenizer` interface, so the orchestrator doesn't need to know which kind it's
talking to.

HFAgent   — real transformers model, used for actual runs. Requires internet
            access to Hugging Face on first run (to download weights) and a GPU
            for anything beyond toy model sizes.
MockAgent — deterministic-but-noisy fake agent with no internet/GPU dependency,
            used to smoke-test the orchestrator, gate, and escalation logic.
"""

from typing import Optional

import torch


class HFAgent:
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.bfloat16):
        # Imported lazily so MockAgent-only usage (tests, CI) never requires
        # transformers/torch-cuda to be fully set up.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (1, seq_len) full sequence so far.

        v0.1 recomputes the full forward pass every step (no KV-cache) — see
        HISTORY.md: correctness first, caching is the top follow-up for real
        latency. Fine for short generations; O(seq_len) slower per step than a
        cached implementation for long ones.
        """
        out = self.model(input_ids=input_ids.to(self.device))
        return out.logits[:, -1, :].to("cpu")


class MockAgent:
    """
    No internet, no GPU. Produces reproducible pseudo-random logits over a small
    fake vocabulary, with optional strong bias toward one token (simulating a
    confident agent) and an optional fixed "wrong" token (simulating a poisoned /
    adversarial agent, for exercising Stage 6/7).
    """

    def __init__(self, name: str, vocab_size: int = 200, seed: int = 0,
                 bias_token: Optional[int] = None, bias_strength: float = 8.0,
                 noise_scale: float = 1.0):
        self.name = name
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._generator = torch.Generator().manual_seed(seed)
        self.bias_token = bias_token
        self.bias_strength = bias_strength
        self.noise_scale = noise_scale

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.randn(1, self.vocab_size, generator=self._generator) * self.noise_scale
        if self.bias_token is not None:
            logits[0, self.bias_token] += self.bias_strength
        return logits
