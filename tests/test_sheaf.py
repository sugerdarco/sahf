"""
Unit tests for Stage 8 Sheaf Reconciliation (Multi-Tokenizer Support).
"""

import pytest
import torch
from transformers import AutoTokenizer

from sahf.agents import MockAgent
from sahf.sheaf import SoftVocabularyMapper, SheafReconciler


def test_soft_vocabulary_mapper_basic():
    tok1 = AutoTokenizer.from_pretrained("gpt2")
    tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    mapper = SoftVocabularyMapper(tok1, tok2, device="cpu")
    
    # Create fake probability vector for tok1 (vocab 50257)
    src_probs = torch.zeros((1, len(tok1)))
    src_probs[0, 100] = 1.0  # Option token in GPT-2
    
    proj_probs = mapper.project_probabilities(src_probs)
    
    assert proj_probs.shape == (1, len(tok2))
    assert torch.isclose(proj_probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_sheaf_reconciler_mock_agents():
    agent1 = MockAgent(vocab_size=100, seed=42)
    agent2 = MockAgent(vocab_size=150, seed=43)
    
    # SheafReconciler with mock tokenizers
    agent1.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    agent2.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    reconciler = SheafReconciler([agent1, agent2], reference_agent_index=0)
    
    p1 = torch.softmax(agent1.next_logits(torch.tensor([[1]])), dim=-1)
    p2 = torch.softmax(agent2.next_logits(torch.tensor([[1]])), dim=-1)
    
    c1 = reconciler.project_to_canonical(0, p1)
    c2 = reconciler.project_to_canonical(1, p2)
    
    assert c1.shape == (1, len(agent1.tokenizer))
    assert c2.shape == (1, len(agent1.tokenizer))
    assert torch.isclose(c1.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(c2.sum(), torch.tensor(1.0), atol=1e-4)
