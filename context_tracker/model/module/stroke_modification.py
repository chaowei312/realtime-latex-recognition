"""
Stroke Modification Module (SMM)

Multi-head stroke selection gate for deciding which strokes to commit.
Uses cross-modal scoring (context+visual query vs patch keys) with
hierarchical aggregation (patch scores → stroke scores).

NOT attention: Uses sigmoid gating (binary decision), not softmax distribution.

Query Design (same as TPM for consistency):
- Query: concat(h_t, h_VC)
  - h_t: Current AR hidden state (token identity + context)
  - h_VC: Visual Class token (spatial position of strokes)
- This matches TPM's query construction for architectural consistency

Option B Integration (Incremental Stroke Removal):
- SMM is called at EACH AR step, not just once
- Identifies strokes contributing to the CURRENT predicted token
- Previous steps' strokes are already removed (masked out)
- Supports active_mask to handle already-removed strokes

Per-step flow:
    Step t: patches (with removed strokes zeroed) → SMM → strokes for token_t
    → Remove those strokes → Step t+1

Note: INT token is reserved for future teacher distillation with multimodal 
embeddings. Currently, h_t (AR hidden state) is used for both SMM and TPM.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class StrokeModificationModule(nn.Module):
    """
    Multi-Head Stroke Selection Gate (SMM)
    
    Decides which strokes contributed to the CURRENT predicted token.
    Called at each AR step (Option B: incremental removal).
    
    Query: concat(h_t, h_VC) - same as TPM for consistency
      - h_t: Current AR hidden state (token identity + context)
      - h_VC: Visual Class token (spatial stroke position)
    Keys: patch tokens grouped by stroke (already-removed strokes masked)
    
    Architecture:
        1. Project query (less compression) and keys (more capacity)
        2. Score each patch: q_h · k_ij (masked patches contribute 0)
        3. Average scores within each ACTIVE stroke
        4. Average across heads
        5. Sigmoid gating for binary selection
    
    Asymmetric Compression:
        - Query: d_concat → d_head (4× compression if d_head = d_concat/4)
        - Key: d_patch → d_head (2× compression, more capacity preserved)
    
    Option B Usage:
        for t in range(max_steps):
            # patches already have removed strokes zeroed
            scores = smm(h_t, h_vc, masked_patches, stroke_indices, active_strokes)
            # scores only meaningful for active strokes
            contributing = (scores > threshold) & active_strokes
            # remove contributing strokes for next step
            active_strokes = active_strokes & ~contributing
    """
    
    def __init__(
        self,
        d_t: int,
        d_vc: int,
        d_patch: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            d_t: Dimension of h_t (AR hidden state, same as d_model)
            d_vc: Dimension of h_VC (Visual Class token)
            d_patch: Dimension of patch tokens
            num_heads: Number of selection heads
            dropout: Dropout on projections (optional regularization)
            temperature: Temperature for sigmoid scaling (default 1.0)
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.temperature = temperature
        d_concat = d_t + d_vc
        
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
        h_t: torch.Tensor,
        h_vc: torch.Tensor,
        patch_tokens: torch.Tensor,
        stroke_indices: List[Tuple[int, int]],
        active_strokes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute stroke selection scores for CURRENT AR step.
        
        Args:
            h_t: [B, d_t] - AR hidden state (same as TPM query)
            h_vc: [B, d_vc] - Visual class token features
            patch_tokens: [B, total_patches, d_patch] - Patch features (removed strokes zeroed)
            stroke_indices: List of (start, end) tuples defining patch ranges per stroke
            active_strokes: [B, num_strokes] - Optional mask for still-active strokes
                           True = stroke still active, False = already removed
                           If None, all strokes considered active.
        
        Returns:
            logits: [B, num_strokes] - Pre-sigmoid logits for stroke selection
                   Inactive strokes have -inf (sigmoid gives 0)
        """
        B = h_t.shape[0]
        num_strokes = len(stroke_indices)
        device = h_t.device
        
        # Default: all strokes active
        if active_strokes is None:
            active_strokes = torch.ones(B, num_strokes, dtype=torch.bool, device=device)
        
        # Concat h_t and h_VC for query (same as TPM)
        h_concat = torch.cat([h_t, h_vc], dim=-1)  # [B, d_concat]
        
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
            for s_idx, (start, end) in enumerate(stroke_indices):
                if end > start:
                    # Average patch scores within this stroke
                    s_score = patch_scores[:, start:end].mean(dim=1)  # [B]
                else:
                    # Empty stroke
                    s_score = torch.zeros(B, device=device)
                stroke_scores.append(s_score)
            
            stroke_scores = torch.stack(stroke_scores, dim=1)  # [B, num_strokes]
            head_stroke_scores.append(stroke_scores)
        
        # Average across heads
        all_head_scores = torch.stack(head_stroke_scores, dim=-1)  # [B, num_strokes, num_heads]
        avg_scores = all_head_scores.mean(dim=-1)  # [B, num_strokes]
        
        # Apply temperature scaling (return logits, not sigmoid!)
        logits = avg_scores / self.temperature
        
        # Mask inactive strokes with -inf (so sigmoid gives 0)
        logits = logits.masked_fill(~active_strokes, float('-inf'))
        
        return logits
    
    def get_selection_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to selection probabilities (apply sigmoid)."""
        return torch.sigmoid(logits)
    
    def get_contributing_strokes(
        self,
        selection_mask: torch.Tensor,
        active_strokes: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get strokes contributing to current token and update active mask.
        
        Args:
            selection_mask: [B, num_strokes] - SMM output scores
            active_strokes: [B, num_strokes] - Current active mask
            threshold: Selection threshold
            
        Returns:
            contributing: [B, num_strokes] - Bool mask of contributing strokes
            new_active: [B, num_strokes] - Updated active mask (contributing removed)
        """
        # Strokes above threshold AND still active
        contributing = (selection_mask > threshold) & active_strokes
        
        # Remove contributing strokes from active set
        new_active = active_strokes & ~contributing
        
        return contributing, new_active
    
    def remove_strokes_from_patches(
        self,
        patch_tokens: torch.Tensor,
        stroke_indices: List[Tuple[int, int]],
        strokes_to_remove: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out patches belonging to removed strokes.
        
        Args:
            patch_tokens: [B, total_patches, d_patch] - Current patch tokens
            stroke_indices: List of (start, end) for each stroke
            strokes_to_remove: [B, num_strokes] - Bool mask of strokes to remove
            
        Returns:
            masked_patches: [B, total_patches, d_patch] - Patches with removed strokes zeroed
        """
        B, total_patches, d_patch = patch_tokens.shape
        device = patch_tokens.device
        
        # Build patch-level mask from stroke-level mask
        patch_mask = torch.ones(B, total_patches, dtype=torch.bool, device=device)
        
        for s_idx, (start, end) in enumerate(stroke_indices):
            # If stroke is marked for removal, zero its patches
            stroke_removed = strokes_to_remove[:, s_idx]  # [B]
            for p_idx in range(start, end):
                patch_mask[:, p_idx] = patch_mask[:, p_idx] & ~stroke_removed
        
        # Apply mask (keep active patches, zero removed)
        masked_patches = patch_tokens * patch_mask.unsqueeze(-1).float()
        
        return masked_patches
    
    def forward_with_padding(
        self,
        h_t: torch.Tensor,
        h_vc: torch.Tensor,
        patch_tokens: torch.Tensor,
        stroke_lengths: torch.Tensor,
        max_patches_per_stroke: int,
    ) -> torch.Tensor:
        """
        Alternative forward for batched data with padding.
        
        Args:
            h_t: [B, d_t] - AR hidden state
            h_vc: [B, d_vc] - Visual class token
            patch_tokens: [B, num_strokes, max_patches, d_patch] - Padded patches
            stroke_lengths: [B, num_strokes] - Actual number of patches per stroke
            max_patches_per_stroke: Maximum patches per stroke (for mask)
        
        Returns:
            selection_mask: [B, num_strokes]
        """
        B, num_strokes, max_patches, d_patch = patch_tokens.shape
        
        # Concat h_t and h_VC (same as TPM)
        h_concat = torch.cat([h_t, h_vc], dim=-1)  # [B, d_concat]
        
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
    
    Assumes d_t = d_vc = d_patch = d_model (common case).
    
    Args:
        d_model: Base model dimension
        num_heads: Number of selection heads
        dropout: Dropout rate
        temperature: Sigmoid temperature
    
    Returns:
        StrokeModificationModule instance
    """
    return StrokeModificationModule(
        d_t=d_model,
        d_vc=d_model,
        d_patch=d_model,
        num_heads=num_heads,
        dropout=dropout,
        temperature=temperature,
    )

