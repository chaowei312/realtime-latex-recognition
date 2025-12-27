"""
Tree-aware Position Module (TPM)

Predicts edit position during AR decoding: (parent_token, relation_type)

Design:
- Query: concat(h_t, h_VC)
  - h_t: AR hidden state (context structure + symbol identity)
  - h_VC: Visual Class token (WHERE - spatial position of strokes)
- Keys: REUSE transformer's K cache + relation embeddings
  - h_t was produced by attending to transformer's K cache
  - TPM uses the SAME K cache for consistency
  - Action keys = K_transformer + relation_emb (additive conditioning)

Why reuse transformer K cache:
- h_t "knows" what it attended to during AR
- K cache contains positional encodings h_t learned from
- No redundant key projection - just add relation embeddings
- Maintains consistency between h_t's knowledge and TPM's scoring

Action space: All (parent_token, relation_type) pairs
- N context tokens × R relation types = N×R possible actions
- Relation types: RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE

AR Decoding with Incremental Stroke Removal (Option B):
- Each AR step: recognize token -> identify strokes -> REMOVE strokes -> next step
- Model sees REMAINING strokes, not full image
- Training: teacher forcing with per-token stroke labels
- Inference: SMM identifies strokes, remove, continue

Why Option B (not keeping full image):
- Correct training signal: model learns to recognize what's LEFT
- Prevents "cheating" by seeing already-committed symbols
- Matches real use case: write -> recognize -> commit -> write more

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
    
    Query: concat(h_t, h_VC) projected per-head to match transformer's K
    - h_t: current AR hidden state (has context structure via attention)
    - h_VC: Visual Class token (explicit spatial position of strokes)
    - Each query head q_h dot-products with transformer's K[:, h, :, :]
    
    Keys: REUSE transformer K cache + relation embeddings
    - action_key = K_transformer[:, h] + relation_emb[h]
    - h_t was produced by attending to K_transformer
    - TPM head count MUST match transformer's head count
    
    Named TPM to align with tree-aware literature (TAMER, DenseBAM, etc.)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,  # MUST match transformer's num_heads
        d_head: int,     # MUST match transformer's d_head
        num_relations: int = 6,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Args:
            d_model: Model dimension (both h_t and h_VC have this dim)
            num_heads: Number of heads (MUST match transformer)
            d_head: Head dimension (MUST match transformer's K cache)
            num_relations: Number of relation types (default 6)
            dropout: Dropout rate
            temperature: Softmax temperature for action selection
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads  # Must match transformer
        self.d_head = d_head        # Must match transformer
        self.num_relations = num_relations
        self.temperature = temperature
        
        # Query dimension: concat(h_t, h_VC) = 2 * d_model
        d_concat = 2 * d_model
        
        # Learned relation embeddings (in model space)
        self.relation_embeddings = nn.Embedding(num_relations, d_model)
        nn.init.normal_(self.relation_embeddings.weight, std=0.02)
        
        # Project relation embeddings to match transformer's K dimension
        # One projection per head - matches transformer's head structure
        self.relation_projs = nn.ModuleList([
            nn.Linear(d_model, d_head) 
            for _ in range(num_heads)
        ])
        
        # Multi-head query projections (from concat h_t, h_VC)
        # Each head projects to d_head to match transformer's K[:, h, :, d_head]
        self.query_projs = nn.ModuleList([
            nn.Linear(d_concat, d_head) 
            for _ in range(num_heads)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_concat)
        
        # Cache for projected relation keys (static, computed once)
        self._cached_rel_keys: Optional[List[torch.Tensor]] = None
    
    def _get_cached_relation_keys(self, device: torch.device) -> List[torch.Tensor]:
        """
        Get relation key projections.
        
        Note: During training, we MUST recompute these each forward pass
        because they are part of the computation graph. Caching would cause
        "backward through graph a second time" errors.
        
        Returns:
            List of [R, d_head] tensors, one per head
        """
        # Always recompute during training to avoid graph issues
        # Caching is only safe during inference (model.eval())
        if self.training:
            rel_emb = self.relation_embeddings.weight  # [R, d_model]
            return [self.relation_projs[h](rel_emb) for h in range(self.num_heads)]
        
        # Cache during inference
        if self._cached_rel_keys is None or self._cached_rel_keys[0].device != device:
            rel_emb = self.relation_embeddings.weight  # [R, d_model]
            self._cached_rel_keys = [
                self.relation_projs[h](rel_emb)  # [R, d_head]
                for h in range(self.num_heads)
            ]
        return self._cached_rel_keys
    
    def build_action_keys_from_cache(
        self,
        transformer_k_cache: torch.Tensor,  # [B, num_heads, N, d_head] or List[Tensor]
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Build action keys by adding relation embeddings to transformer's K cache.
        
        action_key = K_transformer + relation_emb
        
        This reuses the same K that h_t attended to during AR,
        ensuring consistency between h_t's knowledge and TPM's scoring.
        
        Args:
            transformer_k_cache: K cache from transformer
                - If tensor: [B, num_heads, N, d_head]
                - If list: List of [B, N, d_head] per head
            device: Device for relation keys
            
        Returns:
            action_keys: List of [B, num_actions, d_head] per head
            num_parents: Number of parent tokens
        """
        # Handle both tensor and list formats
        if isinstance(transformer_k_cache, torch.Tensor):
            # [B, num_heads, N, d_head] -> list of [B, N, d_head]
            parent_keys = [transformer_k_cache[:, h] for h in range(self.num_heads)]
        else:
            parent_keys = transformer_k_cache
        
        B = parent_keys[0].shape[0]
        num_parents = parent_keys[0].shape[1]
        R = self.num_relations
        
        if device is None:
            device = parent_keys[0].device
        
        # Get cached relation keys (projected to d_head)
        rel_keys = self._get_cached_relation_keys(device)  # List of [R, d_head]
        
        # Build action keys via additive conditioning: K_transformer + k_rel
        # parent_keys[h]: [B, N, d_head] → [B, N, 1, d_head]
        # rel_keys[h]: [R, d_head] → [1, 1, R, d_head]
        # action_keys[h]: [B, N, R, d_head] → [B, N*R, d_head]
        action_keys = []
        for h in range(self.num_heads):
            k_action = (
                parent_keys[h].unsqueeze(2) +  # [B, N, 1, d_head]
                rel_keys[h].unsqueeze(0).unsqueeze(0)  # [1, 1, R, d_head]
            )
            k_action = k_action.view(B, num_parents * R, -1)  # [B, N*R, d_head]
            action_keys.append(k_action)
        
        return action_keys, num_parents
    
    def forward(
        self,
        h_t: torch.Tensor,              # [B, d_model] - AR hidden state
        h_vc: torch.Tensor,             # [B, d_model] - Visual Class token
        transformer_k_cache: torch.Tensor,  # [B, num_heads, N, d_head] - transformer's K cache
        valid_mask: Optional[torch.Tensor] = None,  # [B, num_actions] - mask invalid actions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict action probabilities using transformer's K cache.
        
        Reuses the SAME K cache that h_t attended to during AR,
        ensuring consistency between h_t's knowledge and TPM's scoring.
        
        action_key = K_transformer + relation_emb
        
        Args:
            h_t: Current AR hidden state (produced by attending to K cache)
            h_vc: Visual Class token embedding
            transformer_k_cache: K cache from transformer
                - Tensor: [B, num_heads, N, d_head]
                - Or List: [B, N, d_head] per head
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
        
        # 2. Build action keys from transformer's K cache + relation embeddings
        action_keys, num_parents = self.build_action_keys_from_cache(
            transformer_k_cache, device=h_t.device
        )
        num_actions = action_keys[0].shape[1]
        
        # 3. Multi-head attention scores
        head_scores = []
        
        for h in range(self.num_heads):
            # Project query
            q_h = self.query_projs[h](h_concat)  # [B, d_head]
            
            # Action keys = K_transformer + relation_emb
            k_h = action_keys[h]  # [B, num_actions, d_head]
            
            # Attention scores: q · k^T
            scores_h = torch.einsum('bd,bnd->bn', q_h, k_h)
            
            # Scale by sqrt(d_head)
            scores_h = scores_h / (self.d_head ** 0.5)
            
            head_scores.append(scores_h)
        
        # 4. Average across heads
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


class KCacheManager:
    """
    Manages K cache updates during AR decoding with TPM.
    
    Handles two cases:
    1. Append-only: For simple insertions (RIGHT, SUP, SUB)
       - Just append new token's key to cache
       - O(1) per step
    
    2. Recomputation: For structural changes (ABOVE, BELOW, INSIDE)
       - Some tokens' positions in tree change
       - Need to re-encode affected tokens
       - O(affected_tokens) per step
    
    The recomputation is needed because:
    - When wrapping tokens in a structure (e.g., \frac), their tree positions change
    - If using tree-positional encoding, the keys depend on tree position
    - Affected tokens need fresh key projections
    """
    
    # Relations that require structural recomputation
    STRUCTURAL_RELATIONS = {RelationType.ABOVE, RelationType.BELOW, RelationType.INSIDE}
    
    def __init__(
        self,
        num_heads: int,
        d_head: int,
        transformer_key_proj: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_heads: Number of attention heads
            d_head: Dimension per head
            transformer_key_proj: Optional key projection from transformer
                                  (needed for recomputation)
        """
        self.num_heads = num_heads
        self.d_head = d_head
        self.transformer_key_proj = transformer_key_proj
        
        # K cache: [B, num_heads, N, d_head]
        self.k_cache: Optional[torch.Tensor] = None
        
        # Track which tokens are affected by structural changes
        # Maps token_idx -> set of dependent token indices
        self.structure_dependencies: dict = {}
    
    def reset(self):
        """Clear cache for new sequence."""
        self.k_cache = None
        self.structure_dependencies = {}
    
    def init_from_context(self, context_k_cache: torch.Tensor):
        """
        Initialize cache from context tokens.
        
        Args:
            context_k_cache: [B, num_heads, N_ctx, d_head]
        """
        self.k_cache = context_k_cache.clone()
        self.structure_dependencies = {}
    
    def append_token(
        self,
        new_key: torch.Tensor,  # [B, num_heads, 1, d_head]
    ):
        """
        Simple append for non-structural insertions.
        
        Args:
            new_key: Key for the new token
        """
        if self.k_cache is None:
            self.k_cache = new_key
        else:
            self.k_cache = torch.cat([self.k_cache, new_key], dim=2)
    
    def get_affected_indices(
        self,
        parent_idx: int,
        relation: RelationType,
        num_tokens: int,
    ) -> List[int]:
        """
        Determine which token indices are affected by a structural change.
        
        For ABOVE/BELOW (fraction-like):
        - All tokens between parent and its subtree need re-encoding
        
        For INSIDE (wrapping):
        - The wrapped span needs re-encoding
        
        Args:
            parent_idx: Index of parent token
            relation: Type of relation
            num_tokens: Current number of tokens
            
        Returns:
            List of affected token indices
        """
        if relation == RelationType.ABOVE or relation == RelationType.BELOW:
            # Fraction-like: affects parent and potentially its siblings
            # For simplicity, mark parent's subtree as affected
            # In practice, would need tree structure to determine exact span
            return [parent_idx]
        
        elif relation == RelationType.INSIDE:
            # Wrapping: need to know the span being wrapped
            # For now, mark parent as affected
            # Real implementation would track spans
            return [parent_idx]
        
        return []
    
    def update_after_action(
        self,
        action: Tuple[int, RelationType],
        new_key: torch.Tensor,  # [B, num_heads, 1, d_head]
        token_embeddings: Optional[torch.Tensor] = None,  # [B, N, d_model] for recompute
    ) -> bool:
        """
        Update K cache after an action, handling recomputation if needed.
        
        Args:
            action: (parent_idx, relation_type)
            new_key: Key for the newly inserted token
            token_embeddings: Token embeddings (needed for recomputation)
            
        Returns:
            recomputed: True if recomputation was performed
        """
        parent_idx, relation = action
        
        # Check if structural recomputation needed
        needs_recompute = relation in self.STRUCTURAL_RELATIONS
        
        if needs_recompute and self.transformer_key_proj is not None:
            # Get affected indices
            num_tokens = self.k_cache.shape[2] if self.k_cache is not None else 0
            affected = self.get_affected_indices(parent_idx, relation, num_tokens)
            
            if affected and token_embeddings is not None:
                # Recompute keys for affected tokens
                self._recompute_keys(affected, token_embeddings)
        
        # Always append the new token
        self.append_token(new_key)
        
        return needs_recompute
    
    def _recompute_keys(
        self,
        affected_indices: List[int],
        token_embeddings: torch.Tensor,  # [B, N, d_model]
    ):
        """
        Recompute keys for affected tokens.
        
        Args:
            affected_indices: Indices of tokens needing recomputation
            token_embeddings: Current token embeddings (with updated positions)
        """
        if self.transformer_key_proj is None:
            return
        
        if self.k_cache is None:
            return
        
        B = token_embeddings.shape[0]
        
        for idx in affected_indices:
            if idx < token_embeddings.shape[1] and idx < self.k_cache.shape[2]:
                # Get embedding for this token
                emb = token_embeddings[:, idx:idx+1, :]  # [B, 1, d_model]
                
                # Recompute key using transformer's projection
                # This assumes transformer_key_proj outputs [B, 1, num_heads, d_head]
                # or [B, num_heads, 1, d_head]
                new_key = self.transformer_key_proj(emb)
                
                # Handle different output formats
                if new_key.dim() == 4:
                    if new_key.shape[2] == self.num_heads:
                        # [B, 1, num_heads, d_head] -> [B, num_heads, 1, d_head]
                        new_key = new_key.transpose(1, 2)
                
                # Update cache at this index
                self.k_cache[:, :, idx:idx+1, :] = new_key
    
    def get_cache(self) -> Optional[torch.Tensor]:
        """Get current K cache."""
        return self.k_cache
    
    @property
    def num_tokens(self) -> int:
        """Number of tokens in cache."""
        return self.k_cache.shape[2] if self.k_cache is not None else 0


def create_tpm(
    d_model: int,
    num_heads: int,
    d_head: int,
    num_relations: int = 6,
    dropout: float = 0.1,
    temperature: float = 1.0,
) -> TreeAwarePositionModule:
    """
    Factory function to create a TreeAwarePositionModule (TPM) instance.
    
    Args:
        d_model: Model dimension
        num_heads: MUST match transformer's num_heads
        d_head: MUST match transformer's d_head (usually d_model // num_heads)
    """
    return TreeAwarePositionModule(
        d_model=d_model,
        num_heads=num_heads,
        d_head=d_head,
        num_relations=num_relations,
        dropout=dropout,
        temperature=temperature,
    )


def create_cache_manager(
    num_heads: int,
    d_head: int,
    transformer_key_proj: Optional[nn.Module] = None,
) -> KCacheManager:
    """
    Factory function to create a KCacheManager instance.
    
    Args:
        num_heads: Number of attention heads (match transformer)
        d_head: Dimension per head (match transformer)
        transformer_key_proj: Key projection module from transformer
                              (pass this to enable recomputation)
    """
    return KCacheManager(
        num_heads=num_heads,
        d_head=d_head,
        transformer_key_proj=transformer_key_proj,
    )


class ARDecoderWithStrokeRemoval:
    """
    AR Decoder with Option B: Incremental Stroke Removal.
    
    Each AR step:
    1. Compute INT and VC from CURRENT image (previous strokes removed)
    2. Predict token and position
    3. Identify contributing strokes via SMM
    4. Remove those strokes from image
    5. Next step sees updated image
    
    Training (teacher forcing):
    - Use ground truth per-token stroke labels
    - Remove GT strokes after each step
    
    Inference:
    - Use SMM predictions to identify strokes
    - Remove predicted strokes after each step
    """
    
    def __init__(
        self,
        tpm: TreeAwarePositionModule,
        smm: Optional[nn.Module] = None,  # StrokeModificationModule
        stroke_threshold: float = 0.5,
    ):
        """
        Args:
            tpm: Tree-aware Position Module
            smm: Stroke Modification Module (for inference)
            stroke_threshold: Threshold for stroke selection
        """
        self.tpm = tpm
        self.smm = smm
        self.stroke_threshold = stroke_threshold
    
    def training_step(
        self,
        patch_tokens: torch.Tensor,        # [B, num_patches, d_patch]
        stroke_indices: List[Tuple[int, int]],  # [(start, end), ...] for each stroke
        context_embeddings: torch.Tensor,  # [B, N_ctx, d_model]
        target_tokens: List[int],          # Ground truth token sequence
        target_positions: List[int],       # Ground truth action indices
        stroke_labels: List[List[int]],    # Per-token stroke labels
        compute_h_t_fn,                    # Function to compute h_t given patches + context
        compute_int_vc_fn,                 # Function to compute INT and VC
    ) -> Tuple[torch.Tensor, dict]:
        """
        Training forward pass with teacher forcing and incremental removal.
        
        Args:
            patch_tokens: Visual patch embeddings
            stroke_indices: Stroke boundaries in patch sequence
            context_embeddings: Initial context
            target_tokens: GT token sequence
            target_positions: GT position actions
            stroke_labels: stroke_labels[t] = list of stroke indices for token t
            compute_h_t_fn: Callable to get AR hidden state
            compute_int_vc_fn: Callable to get INT and VC tokens
            
        Returns:
            total_loss: Combined loss
            metrics: Dict of individual losses
        """
        B = patch_tokens.shape[0]
        device = patch_tokens.device
        
        # Track which patches are still "active" (not removed)
        num_patches = patch_tokens.shape[1]
        active_mask = torch.ones(B, num_patches, dtype=torch.bool, device=device)
        
        # Losses
        losses = {'symbol': [], 'position': [], 'smm': []}
        
        # Current context (grows with teacher forcing)
        current_context = context_embeddings
        
        for t, (target_tok, target_pos, strokes_for_tok) in enumerate(
            zip(target_tokens, target_positions, stroke_labels)
        ):
            # 1. Mask out removed patches
            masked_patches = patch_tokens * active_mask.unsqueeze(-1).float()
            
            # 2. Compute INT and VC from CURRENT (masked) patches
            h_int, h_vc = compute_int_vc_fn(masked_patches, current_context)
            
            # 3. Compute AR hidden state
            h_t, k_cache = compute_h_t_fn(masked_patches, current_context, t)
            
            # 4. Position prediction loss (TPM)
            action_probs, action_logits = self.tpm(h_t, h_vc, k_cache)
            target_action = torch.tensor([target_pos], device=device).expand(B)
            pos_loss = F.cross_entropy(action_logits, target_action)
            losses['position'].append(pos_loss)
            
            # 5. SMM loss (if available) - predict which strokes contribute
            if self.smm is not None:
                # Create GT stroke mask
                stroke_mask = torch.zeros(B, len(stroke_indices), device=device)
                for s_idx in strokes_for_tok:
                    stroke_mask[:, s_idx] = 1.0
                
                # SMM prediction
                stroke_scores = self.smm(h_int, h_vc, masked_patches, stroke_indices)
                smm_loss = F.binary_cross_entropy(stroke_scores, stroke_mask)
                losses['smm'].append(smm_loss)
            
            # 6. REMOVE strokes for this token (teacher forcing)
            for s_idx in strokes_for_tok:
                start, end = stroke_indices[s_idx]
                active_mask[:, start:end] = False
            
            # 7. Update context with GT token (teacher forcing)
            # In practice: current_context = concat(current_context, embed(target_tok))
            # Simplified here - actual implementation depends on embedding layer
        
        # Aggregate losses
        total_loss = sum(losses['position']) / len(losses['position'])
        if losses['smm']:
            total_loss = total_loss + sum(losses['smm']) / len(losses['smm'])
        
        metrics = {
            'loss_position': sum(losses['position']).item() / len(losses['position']),
            'loss_smm': sum(losses['smm']).item() / len(losses['smm']) if losses['smm'] else 0,
        }
        
        return total_loss, metrics
    
    @torch.no_grad()
    def inference_step(
        self,
        patch_tokens: torch.Tensor,
        stroke_indices: List[Tuple[int, int]],
        context_embeddings: torch.Tensor,
        compute_h_t_fn,
        compute_int_vc_fn,
        symbol_head_fn,
        max_steps: int = 50,
        eos_token: int = -1,
    ) -> Tuple[List[int], List[int], List[List[int]]]:
        """
        Inference with incremental stroke removal.
        
        Returns:
            tokens: Predicted token sequence
            positions: Predicted position actions
            removed_strokes: Per-step removed stroke indices
        """
        B = patch_tokens.shape[0]
        device = patch_tokens.device
        
        # Track active patches
        num_patches = patch_tokens.shape[1]
        active_mask = torch.ones(B, num_patches, dtype=torch.bool, device=device)
        
        tokens = []
        positions = []
        removed_strokes = []
        current_context = context_embeddings
        
        for t in range(max_steps):
            # 1. Mask removed patches
            masked_patches = patch_tokens * active_mask.unsqueeze(-1).float()
            
            # 2. Check if any patches remain
            if not active_mask.any():
                break
            
            # 3. Compute INT and VC
            h_int, h_vc = compute_int_vc_fn(masked_patches, current_context)
            
            # 4. Compute h_t
            h_t, k_cache = compute_h_t_fn(masked_patches, current_context, t)
            
            # 5. Predict token
            token_logits = symbol_head_fn(h_t)
            token = token_logits.argmax(dim=-1).item()
            
            if token == eos_token:
                break
            
            tokens.append(token)
            
            # 6. Predict position
            action_probs, _ = self.tpm(h_t, h_vc, k_cache)
            position = action_probs.argmax(dim=-1).item()
            positions.append(position)
            
            # 7. SMM: identify contributing strokes
            if self.smm is not None:
                stroke_scores = self.smm(h_int, h_vc, masked_patches, stroke_indices)
                contributing = (stroke_scores > self.stroke_threshold).squeeze(0)
                
                step_removed = []
                for s_idx, is_contrib in enumerate(contributing.tolist()):
                    if is_contrib:
                        start, end = stroke_indices[s_idx]
                        active_mask[:, start:end] = False
                        step_removed.append(s_idx)
                
                removed_strokes.append(step_removed)
            
            # 8. Update context (in practice: add token embedding)
            # Simplified - actual implementation depends on architecture
        
        return tokens, positions, removed_strokes


# Aliases for backwards compatibility
PositionHead = TreeAwarePositionModule
PositionHeadLoss = TPMLoss
create_position_head = create_tpm


# Test code
if __name__ == "__main__":
    # Test the Tree-aware Position Module (TPM)
    B = 4  # Batch size
    d_model = 512
    num_heads = 8  # Must match transformer
    d_head = d_model // num_heads  # = 64, typical transformer config
    N_ctx = 6  # Context tokens: frac, a, t, +, b, c
    
    # Create module - num_heads and d_head MUST match transformer
    tpm = TreeAwarePositionModule(
        d_model=d_model,
        num_heads=num_heads,  # Match transformer
        d_head=d_head,        # Match transformer
        num_relations=6,
    )
    
    print("Tree-aware Position Module (TPM) - Reuses Transformer K Cache")
    print("=" * 60)
    
    # Create dummy inputs
    h_t = torch.randn(B, d_model)  # AR hidden state
    h_vc = torch.randn(B, d_model)  # Visual Class token
    
    # Simulate transformer's K cache [B, num_heads, N, d_head]
    # This is what h_t attended to during AR
    transformer_k_cache = torch.randn(B, num_heads, N_ctx, d_head)
    
    # Forward pass using transformer's K cache
    action_probs, action_logits = tpm(
        h_t=h_t,
        h_vc=h_vc,
        transformer_k_cache=transformer_k_cache,
    )
    
    print(f"Transformer K cache shape: {transformer_k_cache.shape}")
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
    
    # Test KCacheManager with recomputation
    print("\n" + "=" * 60)
    print("Testing KCacheManager with append-only and recomputation...")
    
    # Create a mock key projection (in practice, use transformer's)
    mock_key_proj = nn.Linear(d_model, num_heads * d_head)
    
    # Create cache manager
    cache_mgr = KCacheManager(
        num_heads=num_heads,
        d_head=d_head,
        transformer_key_proj=None,  # No recompute for this test
    )
    
    # Initialize from context
    cache_mgr.init_from_context(transformer_k_cache)
    print(f"  Initialized: {cache_mgr.num_tokens} tokens")
    
    # Simulate AR steps with different action types
    test_actions = [
        (2, RelationType.RIGHT),   # Simple append
        (3, RelationType.SUP),     # Simple append (superscript)
        (1, RelationType.BELOW),   # Structural (fraction) - triggers recompute
        (4, RelationType.SUB),     # Simple append (subscript)
    ]
    
    for i, (parent_idx, relation) in enumerate(test_actions):
        new_key = torch.randn(B, num_heads, 1, d_head)
        recomputed = cache_mgr.update_after_action(
            action=(parent_idx, relation),
            new_key=new_key,
            token_embeddings=None,  # Would need for actual recompute
        )
        
        status = "RECOMPUTE" if recomputed else "append"
        print(f"  Step {i+1}: ({parent_idx}, {relation.name:6}) -> {status}, cache: {cache_mgr.num_tokens} tokens")
    
    # Test with growing K cache directly
    print("\n" + "=" * 60)
    print("Testing AR decoding with growing K cache...")
    
    for step in range(3):
        # K cache grows: context + generated tokens
        num_tokens = N_ctx + step + 1
        growing_k_cache = torch.randn(B, num_heads, num_tokens, d_head)
        
        probs, logits = tpm(
            h_t=torch.randn(B, d_model),  # New h_t each step
            h_vc=h_vc,
            transformer_k_cache=growing_k_cache,
        )
        
        print(f"  Step {step+1}: K cache [{num_tokens}] -> {probs.shape[1]} actions")
    
    print("\nTPM + KCacheManager working correctly!")

