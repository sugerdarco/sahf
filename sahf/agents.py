"""
Agent wrappers. All of HFAgent / MockAgent / PoisonedAgentWrapper expose the
same interface — `next_logits(input_ids) -> (1, vocab)`, `.tokenizer`,
`.vocab_size`, `.name` — so the orchestrator never needs to know which kind
it's talking to.

HFAgent              — real transformers model, used for actual runs. Requires
                        internet access to Hugging Face on first run (to
                        download weights) and a GPU for anything beyond toy
                        model sizes.
MockAgent            — deterministic-but-noisy fake agent with no
                        internet/GPU dependency, used to smoke-test the
                        orchestrator, gate, and escalation logic.
PoisonedAgentWrapper — wraps ANY agent (real or mock) and deliberately
                        corrupts its output. This is the tool for actually
                        exercising Stage 6/7 on real hardware: rather than
                        hoping 3 honest real models happen to disagree enough
                        to trigger an escalation, wrap one of them and force
                        the condition Stage 7 exists to survive.
"""

from typing import List, Optional

import torch


class HFAgent:
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.bfloat16):
        # Imported lazily so MockAgent-only usage (tests, CI) never requires
        # transformers/torch-cuda to be fully set up.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
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
    confident agent).
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


class PoisonedAgentWrapper:
    """
    Deliberately corrupts an underlying agent's (real or mock) logits, so
    Stage 6/7's robustness claim can be tested under a controlled, guaranteed
    adversarial condition instead of hoping honest models happen to disagree
    enough. Never used silently — the orchestrator has no special-case for
    this class, it just looks like one more agent whose logits happen to be
    bad; wrapping is something you opt into explicitly (see run.py
    --poison-index) and it is always recorded in the run's meta.json.

    Modes:
      "invert"        — logits = -logits. The most realistic poisoning: still
                        a well-formed distribution derived from the real
                        model, but confidently prefers whatever the honest
                        model finds LEAST likely.
      "uniform_noise" — logits replaced with pure noise, scaled by
                        bias_strength. Simulates a broken / garbage agent.
      "random_bias"   — a fixed random token gets a large additive boost each
                        call. Simulates an agent stuck insisting on one wrong
                        answer.
    """

    VALID_MODES = ("invert", "uniform_noise", "random_bias")

    def __init__(self, wrapped_agent, mode: str = "invert", bias_strength: float = 25.0,
                 seed: int = 0):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown poison mode {mode!r}, expected one of {self.VALID_MODES}")
        self.wrapped = wrapped_agent
        self.name = f"POISONED[{mode}]({wrapped_agent.name})"
        self.tokenizer = wrapped_agent.tokenizer
        self.vocab_size = wrapped_agent.vocab_size
        self.mode = mode
        self.bias_strength = bias_strength
        self._generator = torch.Generator().manual_seed(seed)
        self._fixed_token: Optional[int] = None

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.wrapped.next_logits(input_ids).clone()

        if self.mode == "invert":
            return -logits

        if self.mode == "uniform_noise":
            return torch.randn_like(logits) * self.bias_strength

        if self.mode == "random_bias":
            if self._fixed_token is None:
                self._fixed_token = int(
                    torch.randint(0, logits.shape[-1], (1,), generator=self._generator).item()
                )
            logits[0, self._fixed_token] += self.bias_strength
            return logits

        raise AssertionError("unreachable")  # mode validated in __init__


def assert_shared_vocab_size(agents: List) -> int:
    """
    Fails loudly (not silently) if configured agents don't share a vocabulary —
    this whole prototype's correctness depends on that assumption (see
    ARCHITECTURE.md, "Why no Stage 8?"). Returns the shared vocab size so
    callers can log it.

    Uses each agent's `.vocab_size` attribute rather than `len(agent.tokenizer)`
    directly, since MockAgent (and PoisonedAgentWrapper wrapping one) has no
    real tokenizer object but still has a well-defined vocab size.
    """
    if len(agents) < 2:
        raise ValueError("Need at least 2 agents.")
    sizes = {agent.vocab_size for agent in agents}
    if len(sizes) != 1:
        details = ", ".join(f"{a.name}={a.vocab_size}" for a in agents)
        raise ValueError(
            "Agents do not share a vocabulary size — the fast version assumes a shared "
            f"tokenizer (no Stage 8 reconciliation). Got: {details}"
        )
    return sizes.pop()
