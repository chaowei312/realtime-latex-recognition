"""
Stroke Modification Module (SMM)

Multi-head stroke selection gate for deciding which strokes to commit.
Uses cross-modal scoring (context+visual query vs patch keys) with
hierarchical aggregation (patch scores → stroke scores).

NOT attention: Uses sigmoid gating (binary decision), not softmax distribution.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class StrokeModificationModule(nn.Module):
    """
    Multi-Head Stroke Selection Gate (SMM)
    
    Decides which strokes to commit based on:
    - Query: concat([INT], [VC]) - context-aware + visual features
    - Keys: patch tokens grouped by stroke
    
    Architecture:
        1. Project query (less compression) and keys (more capacity)
        2. Score each patch: q_h · k_ij
        3. Average scores within each stroke
        4. Average across heads
        5. Sigmoid gating for binary selection
    
    Asymmetric Compression:
        - Query: d_concat → d_head (4× compression if d_head = d_concat/4)
        - Key: d_patch → d_head (2× compression, more capacity preserved)
    """
    
    def __init__(
        self,
        d_int: int,
        d_vc: int,
        d_patch: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            d_int: Dimension of [INT] token (integration/context)
            d_vc: Dimension of [VC] token (visual class)
            d_patch: Dimension of patch tokens
            num_heads: Number of selection heads
            dropout: Dropout on projections (optional regularization)
            temperature: Temperature for sigmoid scaling (default 1.0)
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.temperature = temperature
        d_concat = d_int + d_vc
        
        # d_head based on concat dimension (query has less compression)
        # Query: d_concat → d_head (4× compression)
        # Key: d_patch → d_head (2× compression, more capacity)
        self.d_head = d_concat // num_heads
        
        # Multi-head query projections (from concat([INT], [VC]))
        self.query_projs = nn.ModuleList([
            nn.Linear(d_concat, self.d_head)
            for _ in range(num_heads)
        ])
        
        # Multi-head key projections (from patch tokens)
        self.key_projs = nn.ModuleList([
            nn.Linear(d_patch, self.d_head)
            for _ in range(num_heads)
        ])
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Store dimensions for reference
        self.d_concat = d_concat
        self.d_patch = d_patch
    
    def forward(
        self,
        h_int: torch.Tensor,
        h_vc: torch.Tensor,
        patch_tokens: torch.Tensor,
        stroke_indices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute stroke selection scores.
        
        Args:
            h_int: [B, d_int] - Integration token features
            h_vc: [B, d_vc] - Visual class token features
            patch_tokens: [B, total_patches, d_patch] - All patch token features
            stroke_indices: List of (start, end) tuples defining patch ranges per stroke
        
        Returns:
            selection_mask: [B, num_strokes] - Sigmoid scores (0=keep, 1=commit)
        """
        B = h_int.shape[0]
        num_strokes = len(stroke_indices)
        
        # Concat [INT] and [VC] for query
        h_concat = torch.cat([h_int, h_vc], dim=-1)  # [B, d_concat]
        
        # Collect scores from all heads
        head_stroke_scores = []
        
        for h in range(self.num_heads):
            # Project query and keys for this head
            q_h = self.dropout(self.query_projs[h](h_concat))  # [B, d_head]
            k_h = self.dropout(self.key_projs[h](patch_tokens))  # [B, total_patches, d_head]
            
            # Score EACH patch first: q_h · k_ij
            patch_scores = torch.einsum('bd,bpd->bp', q_h, k_h)  # [B, total_patches]
            
            # Then average within each stroke
            stroke_scores = []
            for (start, end) in stroke_indices:
                if end > start:
                    # Average patch scores within this stroke
                    s_score = patch_scores[:, start:end].mean(dim=1)  # [B]
                else:
                    # Empty stroke (shouldn't happen, but handle gracefully)
                    s_score = torch.zeros(B, device=patch_scores.device)
                stroke_scores.append(s_score)
            
            stroke_scores = torch.stack(stroke_scores, dim=1)  # [B, num_strokes]
            head_stroke_scores.append(stroke_scores)
        
        # Average across heads
        all_head_scores = torch.stack(head_stroke_scores, dim=-1)  # [B, num_strokes, num_heads]
        avg_scores = all_head_scores.mean(dim=-1)  # [B, num_strokes]
        
        # Apply temperature and sigmoid gating
        selection_mask = torch.sigmoid(avg_scores / self.temperature)
        
        return selection_mask
    
    def forward_with_padding(
        self,
        h_int: torch.Tensor,
        h_vc: torch.Tensor,
        patch_tokens: torch.Tensor,
        stroke_lengths: torch.Tensor,
        max_patches_per_stroke: int,
    ) -> torch.Tensor:
        """
        Alternative forward for batched data with padding.
        
        Args:
            h_int: [B, d_int]
            h_vc: [B, d_vc]
            patch_tokens: [B, num_strokes, max_patches, d_patch] - Padded patches
            stroke_lengths: [B, num_strokes] - Actual number of patches per stroke
            max_patches_per_stroke: Maximum patches per stroke (for mask)
        
        Returns:
            selection_mask: [B, num_strokes]
        """
        B, num_strokes, max_patches, d_patch = patch_tokens.shape
        
        # Concat [INT] and [VC]
        h_concat = torch.cat([h_int, h_vc], dim=-1)  # [B, d_concat]
        
        # Create padding mask: [B, num_strokes, max_patches]
        patch_range = torch.arange(max_patches, device=stroke_lengths.device)
        padding_mask = patch_range.unsqueeze(0).unsqueeze(0) < stroke_lengths.unsqueeze(-1)
        # padding_mask: True for valid patches, False for padding
        
        head_stroke_scores = []
        
        for h in range(self.num_heads):
            q_h = self.dropout(self.query_projs[h](h_concat))  # [B, d_head]
            
            # Reshape patches for projection: [B * num_strokes * max_patches, d_patch]
            flat_patches = patch_tokens.view(-1, d_patch)
            k_h = self.dropout(self.key_projs[h](flat_patches))
            k_h = k_h.view(B, num_strokes, max_patches, self.d_head)  # [B, S, P, d_head]
            
            # Score each patch: q_h · k_ij
            # q_h: [B, d_head] → [B, 1, 1, d_head]
            q_h_expanded = q_h.unsqueeze(1).unsqueeze(2)
            patch_scores = (q_h_expanded * k_h).sum(dim=-1)  # [B, num_strokes, max_patches]
            
            # Mask out padding
            patch_scores = patch_scores.masked_fill(~padding_mask, 0.0)
            
            # Average within each stroke (excluding padding)
            valid_counts = stroke_lengths.clamp(min=1)  # Avoid division by zero
            stroke_scores = patch_scores.sum(dim=-1) / valid_counts  # [B, num_strokes]
            
            head_stroke_scores.append(stroke_scores)
        
        # Average across heads
        all_head_scores = torch.stack(head_stroke_scores, dim=-1)
        avg_scores = all_head_scores.mean(dim=-1)
        
        # Sigmoid gating
        selection_mask = torch.sigmoid(avg_scores / self.temperature)
        
        return selection_mask


class StrokeModificationLoss(nn.Module):
    """
    Binary Cross-Entropy loss for stroke selection supervision.
    
    Ground truth: y_stroke[i] = 1 if stroke i should be committed, 0 otherwise.
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        """
        Args:
            pos_weight: Weight for positive class (commit). 
                        Use >1.0 if commits are rare, <1.0 if common.
        """
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self,
        selection_mask: torch.Tensor,
        target: torch.Tensor,
        stroke_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            selection_mask: [B, num_strokes] - Predicted sigmoid scores
            target: [B, num_strokes] - Ground truth (0 or 1)
            stroke_mask: [B, num_strokes] - Optional mask for valid strokes
        
        Returns:
            loss: Scalar BCE loss
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=selection_mask.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
            # Need logits, so undo sigmoid (approximately)
            eps = 1e-7
            logits = torch.log(selection_mask.clamp(eps, 1-eps) / (1 - selection_mask.clamp(eps, 1-eps)))
            loss = loss_fn(logits, target)
        else:
            loss = nn.functional.binary_cross_entropy(
                selection_mask, target, reduction='none'
            )
        
        if stroke_mask is not None:
            loss = loss * stroke_mask
            return loss.sum() / stroke_mask.sum().clamp(min=1)
        else:
            return loss.mean()


def create_stroke_modification_module(
    d_model: int = 512,
    num_heads: int = 4,
    dropout: float = 0.0,
    temperature: float = 1.0,
) -> StrokeModificationModule:
    """
    Factory function for StrokeModificationModule with typical config.
    
    Assumes d_int = d_vc = d_patch = d_model (common case).
    
    Args:
        d_model: Base model dimension
        num_heads: Number of selection heads
        dropout: Dropout rate
        temperature: Sigmoid temperature
    
    Returns:
        StrokeModificationModule instance
    """
    return StrokeModificationModule(
        d_int=d_model,
        d_vc=d_model,
        d_patch=d_model,
        num_heads=num_heads,
        dropout=dropout,
        temperature=temperature,
    )

