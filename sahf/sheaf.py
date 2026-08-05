"""
Stage 8: Sheaf Reconciliation & Multi-Tokenizer Soft Alignment Module.

Provides cross-vocabulary transfer matrices (T_{i -> ref}) mapping native probability
distributions p_i from model-specific tokenizers to a canonical reference vocabulary V_ref.

Allows SAHF to fuse models with completely DIFFERENT tokenizers (e.g. LLaMA-3, Mistral, GPT-2, Qwen).
"""

from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F


class SoftVocabularyMapper:
    """
    Constructs and applies soft cross-vocabulary transfer matrices between two tokenizers.
    """
    def __init__(self, source_tokenizer, target_tokenizer, device: str = "cpu"):
        self.source_tok = source_tokenizer
        self.target_tok = target_tokenizer
        self.device = device
        self.source_vocab_size = len(source_tokenizer)
        self.target_vocab_size = len(target_tokenizer)
        
        # Build mapping index: string token -> target token ID
        self.target_str_to_id = {}
        for token_id in range(self.target_vocab_size):
            token_str = target_tokenizer.decode([token_id])
            if token_str not in self.target_str_to_id:
                self.target_str_to_id[token_str] = token_id

        # Build mapping lookup array for fast CUDA/CPU projection
        mapping_indices = []
        for src_id in range(self.source_vocab_size):
            src_str = source_tokenizer.decode([src_id])
            tgt_id = self.target_str_to_id.get(src_str, None)
            if tgt_id is None:
                # Fallback: re-encode string with target tokenizer
                tgt_ids = target_tokenizer.encode(src_str, add_special_tokens=False)
                tgt_id = tgt_ids[0] if tgt_ids else 0
            mapping_indices.append(tgt_id)

        self.mapping_tensor = torch.tensor(mapping_indices, dtype=torch.long, device=device)

    def project_probabilities(self, source_probs: torch.Tensor) -> torch.Tensor:
        """
        source_probs: (1, source_vocab_size)
        returns: (1, target_vocab_size)
        """
        target_probs = torch.zeros((source_probs.shape[0], self.target_vocab_size),
                                  dtype=source_probs.dtype, device=self.device)
        # Scatter-add probabilities to mapped target indices
        target_probs.scatter_add_(1, self.mapping_tensor.unsqueeze(0), source_probs.to(self.device))
        # Re-normalize over target vocabulary
        sum_p = target_probs.sum(dim=-1, keepdim=True)
        sum_p = torch.clamp(sum_p, min=1e-9)
        return target_probs / sum_p


class SheafReconciler:
    """
    Manages multi-tokenizer alignment for N heterogeneous agents.
    Designates Agent 0's vocabulary as V_ref (or a specified canonical reference tokenizer).
    """
    def __init__(self, agents: List, reference_agent_index: int = 0, device: str = "cpu"):
        self.agents = agents
        self.ref_agent_idx = reference_agent_index
        self.ref_tokenizer = agents[reference_agent_index].tokenizer
        self.ref_vocab_size = len(self.ref_tokenizer)
        self.device = device
        
        self.mappers: Dict[int, SoftVocabularyMapper] = {}
        for idx, agent in enumerate(agents):
            if len(agent.tokenizer) != self.ref_vocab_size or agent.tokenizer != self.ref_tokenizer:
                self.mappers[idx] = SoftVocabularyMapper(agent.tokenizer, self.ref_tokenizer, device=device)

    def is_heterogeneous((self) -> bool:
        return len(self.mappers) > 0

    def project_to_canonical(self, agent_idx: int, native_probs: torch.Tensor) -> torch.Tensor:
        """
        Projects native probability vector (1, native_vocab_size) -> canonical vector (1, ref_vocab_size).
        """
        if agent_idx in self.mappers:
            return self.mappers[agent_idx].project_probabilities(native_probs)
        return native_probs.to(self.device)

    def synchronize_token(self, canonical_token_id: int) -> List[Tuple[str, List[int]]]:
        """
        Converts consensus canonical token ID into native text and native token IDs for each agent.
        Returns list of (decoded_text, native_token_ids) for each agent.
        """
        canonical_text = self.ref_tokenizer.decode([canonical_token_id])
        sync_info = []
        
        for agent in self.agents:
            native_ids = agent.tokenizer.encode(canonical_text, add_special_tokens=False)
            sync_info.append((canonical_text, native_ids))
            
        return sync_info
