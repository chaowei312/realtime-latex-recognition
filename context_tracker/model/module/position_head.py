"""
Tree-aware Position Module (TPM)

Predicts edit position during AR decoding: (parent_token, relation_type)

Design:
- Query: concat(h_t, h_VC)
  - h_t: AR hidden state (context structure + symbol identity)
  - h_VC: Visual Class token (WHERE - spatial position of strokes)
- Keys: Action embeddings = token_emb + relation_emb
- Multi-head attention with softmax output

Action space: All (parent_token, relation_type) pairs
- N context tokens × R relation types = N×R possible actions
- Relation types: RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE

Named TPM to align with tree-aware literature (TAMER, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .relationship_tensor import RelationType


class TreeAwarePositionModule(nn.Module):
    """
    Tree-aware Position Module (TPM) for predicting edit position during AR decoding.
    
    At each AR step t, predicts WHERE to attach the current token:
    - Output: (parent_token_idx, relation_type)
    - Uses multi-head attention over action embeddings
    
    Query: concat(h_t, h_VC)
    - h_t: current AR hidden state (has context structure via attention)
    - h_VC: Visual Class token (explicit spatial position of strokes)
    
    Keys: action_emb = context_token_emb + relation_emb
    - Context tokens: existing LaTeX tokens
    - Relation embeddings: learned per relation type
    
    Named TPM to align with tree-aware literature (TAMER, DenseBAM, etc.)
    """
    
    def __init__(
        self,
        d_model: int,
        num_relations: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Args:
            d_model: Model dimension (both h_t and h_VC have this dim)
            num_relations: Number of relation types (default 6)
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Softmax temperature for action selection
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Query dimension: concat(h_t, h_VC) = 2 * d_model
        d_concat = 2 * d_model
        
        # Head dimension for query (from concat)
        # Less compression for query (richer signal)
        self.d_head_q = d_concat // num_heads  # e.g., 1024 / 4 = 256
        
        # Head dimension for key (from action embeddings)
        # Can use same or different
        self.d_head_k = d_model // num_heads  # e.g., 512 / 4 = 128
        
        # Project both to common dimension for dot product
        self.d_head_common = self.d_head_q  # Use larger for more capacity
        
        # Learned relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, d_model)
        nn.init.normal_(self.relation_embeddings.weight, std=0.02)
        
        # Multi-head query projections (from concat)
        self.query_projs = nn.ModuleList([
            nn.Linear(d_concat, self.d_head_common) 
            for _ in range(num_heads)
        ])
        
        # Multi-head key projections (from action embeddings)
        self.key_projs = nn.ModuleList([
            nn.Linear(d_model, self.d_head_common) 
            for _ in range(num_heads)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_concat)
    
    def build_action_embeddings(
        self,
        context_embeddings: torch.Tensor,  # [B, N_ctx, d_model]
        generated_embeddings: Optional[torch.Tensor] = None,  # [B, t-1, d_model]
    ) -> Tuple[torch.Tensor, int]:
        """
        Build action embeddings: action = token_emb + relation_emb
        
        Args:
            context_embeddings: Embeddings of existing context tokens
            generated_embeddings: Embeddings of already generated tokens (optional)
        
        Returns:
            action_embeddings: [B, num_actions, d_model]
            num_parents: Number of parent tokens
        """
        B, N_ctx, d = context_embeddings.shape
        R = self.num_relations
        
        # Combine context and generated tokens as potential parents
        if generated_embeddings is not None and generated_embeddings.shape[1] > 0:
            all_parents = torch.cat([context_embeddings, generated_embeddings], dim=1)
        else:
            all_parents = context_embeddings
        
        num_parents = all_parents.shape[1]
        
        # Get relation embeddings: [R, d]
        rel_emb = self.relation_embeddings.weight
        
        # Build all (token, relation) combinations via broadcasting
        # all_parents: [B, N, d] → [B, N, 1, d]
        # rel_emb: [R, d] → [1, 1, R, d]
        # action_emb: [B, N, R, d]
        action_emb = (
            all_parents.unsqueeze(2) +  # [B, N, 1, d]
            rel_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, R, d]
        )
        
        # Flatten to [B, N*R, d]
        action_emb = action_emb.view(B, num_parents * R, d)
        
        return action_emb, num_parents
    
    def forward(
        self,
        h_t: torch.Tensor,              # [B, d_model] - AR hidden state
        h_vc: torch.Tensor,             # [B, d_model] - Visual Class token
        context_embeddings: torch.Tensor,  # [B, N_ctx, d_model]
        generated_embeddings: Optional[torch.Tensor] = None,  # [B, t-1, d_model]
        valid_mask: Optional[torch.Tensor] = None,  # [B, num_actions] - mask invalid actions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict action probabilities.
        
        Args:
            h_t: Current AR hidden state
            h_vc: Visual Class token embedding
            context_embeddings: Embeddings of context tokens
            generated_embeddings: Embeddings of previously generated tokens
            valid_mask: Optional mask for invalid actions (True = valid)
        
        Returns:
            action_probs: [B, num_actions] - softmax probabilities
            action_logits: [B, num_actions] - raw logits (for loss computation)
        """
        B = h_t.shape[0]
        
        # 1. Build query: concat(h_t, h_VC)
        h_concat = torch.cat([h_t, h_vc], dim=-1)  # [B, 2*d_model]
        h_concat = self.layer_norm(h_concat)
        h_concat = self.dropout(h_concat)
        
        # 2. Build action embeddings (keys)
        action_emb, num_parents = self.build_action_embeddings(
            context_embeddings, generated_embeddings
        )
        num_actions = action_emb.shape[1]
        
        # 3. Multi-head attention scores
        head_scores = []
        
        for h in range(self.num_heads):
            # Project query
            q_h = self.query_projs[h](h_concat)  # [B, d_head_common]
            
            # Project keys
            k_h = self.key_projs[h](action_emb)  # [B, num_actions, d_head_common]
            
            # Attention scores: q · k^T
            # [B, d_head_common] @ [B, d_head_common, num_actions] → [B, num_actions]
            scores_h = torch.einsum('bd,bnd->bn', q_h, k_h)
            
            # Scale by sqrt(d_head)
            scores_h = scores_h / (self.d_head_common ** 0.5)
            
            head_scores.append(scores_h)
        
        # 4. Average across heads
        # Stack: [B, num_actions, num_heads] → mean → [B, num_actions]
        avg_scores = torch.stack(head_scores, dim=-1).mean(dim=-1)
        
        # 5. Apply temperature
        action_logits = avg_scores / self.temperature
        
        # 6. Mask invalid actions if provided
        if valid_mask is not None:
            action_logits = action_logits.masked_fill(~valid_mask, float('-inf'))
        
        # 7. Softmax for action selection
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, action_logits
    
    def decode_action(
        self, 
        action_idx: int, 
        num_parents: int
    ) -> Tuple[int, str]:
        """
        Convert action index to (parent_idx, relation_type).
        
        Actions are ordered: 
        [(tok0, rel0), (tok0, rel1), ..., (tok0, relR-1), (tok1, rel0), ...]
        
        Args:
            action_idx: Index in the flattened action space
            num_parents: Number of parent tokens
        
        Returns:
            parent_idx: Index of parent token
            relation_name: Name of relation type
        """
        R = self.num_relations
        parent_idx = action_idx // R
        rel_idx = action_idx % R
        relation_name = RelationType(rel_idx).name
        return parent_idx, relation_name
    
    def encode_action(
        self, 
        parent_idx: int, 
        relation: RelationType
    ) -> int:
        """
        Convert (parent_idx, relation) to action index.
        
        Args:
            parent_idx: Index of parent token
            relation: Relation type
        
        Returns:
            action_idx: Index in the flattened action space
        """
        return parent_idx * self.num_relations + int(relation)


class TPMLoss(nn.Module):
    """
    Loss function for Tree-aware Position Module (TPM) training.
    
    Uses cross-entropy loss on action predictions.
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        action_logits: torch.Tensor,  # [B, num_actions]
        target_actions: torch.Tensor,  # [B] - target action indices
        valid_mask: Optional[torch.Tensor] = None,  # [B] - which samples are valid
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for position prediction.
        
        Args:
            action_logits: Predicted logits from Position Head
            target_actions: Ground truth action indices
            valid_mask: Optional mask for valid samples
        
        Returns:
            loss: Scalar loss value
        """
        # Cross-entropy loss
        loss = F.cross_entropy(
            action_logits,
            target_actions,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        
        # Apply sample mask if provided
        if valid_mask is not None:
            loss = loss * valid_mask.float()
            loss = loss.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


def create_tpm(
    d_model: int,
    num_relations: int = 6,
    num_heads: int = 4,
    dropout: float = 0.1,
    temperature: float = 1.0,
) -> TreeAwarePositionModule:
    """Factory function to create a TreeAwarePositionModule (TPM) instance."""
    return TreeAwarePositionModule(
        d_model=d_model,
        num_relations=num_relations,
        num_heads=num_heads,
        dropout=dropout,
        temperature=temperature,
    )


# Aliases for backwards compatibility
PositionHead = TreeAwarePositionModule
PositionHeadLoss = TPMLoss
create_position_head = create_tpm


# Test code
if __name__ == "__main__":
    # Test the Tree-aware Position Module (TPM)
    B = 4  # Batch size
    d_model = 512
    N_ctx = 6  # Context tokens: frac, a, t, +, b, c
    num_heads = 4
    
    # Create module
    tpm = TreeAwarePositionModule(
        d_model=d_model,
        num_relations=6,
        num_heads=num_heads,
    )
    
    print("Tree-aware Position Module (TPM)")
    print("=" * 50)
    
    # Create dummy inputs
    h_t = torch.randn(B, d_model)  # AR hidden state
    h_vc = torch.randn(B, d_model)  # Visual Class token
    context_emb = torch.randn(B, N_ctx, d_model)  # Context token embeddings
    
    # Forward pass
    action_probs, action_logits = tpm(
        h_t=h_t,
        h_vc=h_vc,
        context_embeddings=context_emb,
    )
    
    print(f"Action space size: {action_probs.shape[1]} (N_ctx={N_ctx} x R=6)")
    print(f"Action probs shape: {action_probs.shape}")
    print(f"Action probs sum: {action_probs.sum(dim=-1)}")  # Should be 1.0
    
    # Decode top action
    top_action = action_probs[0].argmax().item()
    parent_idx, rel_name = tpm.decode_action(top_action, N_ctx)
    print(f"\nTop action for sample 0:")
    print(f"  Action index: {top_action}")
    print(f"  Parent index: {parent_idx}")
    print(f"  Relation: {rel_name}")
    
    # Test loss
    target_actions = torch.randint(0, N_ctx * 6, (B,))
    loss_fn = TPMLoss()
    loss = loss_fn(action_logits, target_actions)
    print(f"\nLoss: {loss.item():.4f}")
    
    # Test with generated embeddings (dynamic action space)
    generated_emb = torch.randn(B, 2, d_model)  # 2 tokens already generated
    action_probs2, _ = tpm(
        h_t=h_t,
        h_vc=h_vc,
        context_embeddings=context_emb,
        generated_embeddings=generated_emb,
    )
    print(f"\nWith generated tokens:")
    print(f"  Action space size: {action_probs2.shape[1]} ((N_ctx + 2) x 6)")

