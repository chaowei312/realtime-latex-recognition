"""
Context Tracker Model

Full model integrating all components for handwritten LaTeX recognition with:
- Visual encoding (VisualPatchEncoder)
- Transformer decoder with causal attention
- [VC] (Visual Class) token for visual aggregation
- [INT] (Integration) token for context-aware decisions
- Head A: Symbol prediction
- Head B (SMM): Stroke selection for incremental removal
- Head C (TPM): Position prediction for tree construction

Architecture:
    [LaTeX context] --causal--> [Patches] --causal--> [VC] --> [INT]
                                    |                   |         |
                                    +----> Head B (SMM) <---------+
                                    |                             |
                                    +----> Head C (TPM) <---------+
                                                                  |
                                    +----> Head A (Symbol) <------+
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import (
    # Visual encoder
    VisualPatchEncoder,
    create_visual_encoder,
    PatchStrokeMapper,
    # Attention and FFN
    FlashAttentionDecoderStack,
    SwiGLUFFN,
    # SMM and TPM
    StrokeModificationModule,
    TreeAwarePositionModule,
    KCacheManager,
    # Relationship parsing
    RelationType,
    LatexRelationshipParser,
)


@dataclass
class ContextTrackerConfig:
    """Configuration for Context Tracker model."""
    
    # Image encoding
    image_size: int = 128
    in_channels: int = 1  # Grayscale
    
    # Model dimensions
    d_model: int = 256
    num_heads: int = 8
    d_head: int = 32  # d_model // num_heads
    
    # Transformer decoder
    num_layers: int = 4
    d_ffn: int = 1024  # 4 * d_model
    dropout: float = 0.1
    
    # Vocabulary
    vocab_size: int = 512  # LaTeX tokens
    max_seq_len: int = 512  # Increased for long MathWriting expressions
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    vc_token_id: int = 3    # [VC] token
    int_token_id: int = 4   # [INT] token
    mask_token_id: int = 5  # [MASK] for edit
    
    # SMM config
    smm_num_heads: int = 4
    smm_temperature: float = 1.0
    
    # TPM config
    num_relations: int = 6  # RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE
    tpm_temperature: float = 1.0
    
    # Visual encoder variant
    visual_encoder_variant: str = 'standard'  # 'tiny', 'standard', 'large'
    
    # Training
    label_smoothing: float = 0.1


class TextEmbedding(nn.Module):
    """Text token embeddings with positional encoding."""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] token indices
            
        Returns:
            embeddings: [B, L, d_model]
        """
        B, L = token_ids.shape
        max_pos = self.position_embedding.num_embeddings
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        # Clamp positions to max_seq_len (for sequences longer than expected)
        positions = positions.clamp(max=max_pos - 1)
        
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class TransformerDecoder(nn.Module):
    """
    Causal transformer decoder with KV caching support.
    
    Processes: [context_tokens, patch_tokens, VC, INT]
    - context_tokens: LaTeX text tokens (causal within themselves)
    - patch_tokens: Visual patches (causal, can attend to context)
    - VC: Attends to patches only
    - INT: Attends to context + VC
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ffn: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.num_layers = num_layers
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ffn, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        """
        Args:
            x: [B, L, d_model] input embeddings
            attention_mask: [B, L, L] or [B, 1, L, L] attention mask
            cache: List of (K, V) tuples per layer for incremental decoding
            use_cache: Whether to return updated cache
            
        Returns:
            hidden_states: [B, L, d_model]
            new_cache: Updated KV cache (if use_cache)
            k_cache_last: K cache from last layer [B, num_heads, L, d_head] for TPM
        """
        new_cache = [] if use_cache else None
        k_cache_last = None
        
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x, layer_new_cache, k = layer(x, attention_mask, layer_cache, use_cache)
            
            if use_cache:
                new_cache.append(layer_new_cache)
            
            # Keep K cache from last layer for TPM
            if i == len(self.layers) - 1:
                k_cache_last = k
        
        x = self.final_norm(x)
        return x, new_cache, k_cache_last


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with pre-norm."""
    
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ffn, dropout=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        B, L, D = x.shape
        
        # Pre-norm
        normed = self.attn_norm(x)
        
        # QKV projections
        q = self.q_proj(normed).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        
        # Apply cache
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L_kv]
        
        if attention_mask is not None:
            # Expand mask if needed
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, L, L_kv]
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, d_head]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)
        
        # Residual + FFN
        x = x + attn_output
        x = x + self.ffn(self.ffn_norm(x))
        
        return x, new_cache, k


class SymbolHead(nn.Module):
    """Head A: Symbol prediction from h_t."""
    
    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )
    
    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: [B, d_model] or [B, L, d_model]
            
        Returns:
            logits: [B, vocab_size] or [B, L, vocab_size]
        """
        return self.proj(h_t)


class ContextTrackerModel(nn.Module):
    """
    Full Context Tracker model for handwritten LaTeX recognition.
    
    Components:
    1. VisualPatchEncoder: image -> patch embeddings + [VC] token
    2. TextEmbedding: LaTeX context -> text embeddings
    3. TransformerDecoder: unified causal decoder
    4. SymbolHead (A): h_t -> symbol prediction
    5. SMM (B): stroke selection for incremental removal
    6. TPM (C): position prediction for tree construction
    """
    
    def __init__(self, config: ContextTrackerConfig):
        super().__init__()
        self.config = config
        
        # 1. Visual encoder
        self.visual_encoder = create_visual_encoder(
            image_size=config.image_size,
            embed_dim=config.d_model,
            variant=config.visual_encoder_variant,
            add_vc_token=True
        )
        self.num_patches = self.visual_encoder.num_patches
        
        # 2. Text embedding
        self.text_embedding = TextEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # 3. [INT] token (learnable)
        self.int_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.int_token, std=0.02)
        
        # 4. Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ffn=config.d_ffn,
            dropout=config.dropout
        )
        
        # 5. Head A: Symbol prediction
        self.symbol_head = SymbolHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            dropout=config.dropout
        )
        
        # 6. Head B: SMM (Stroke Modification Module)
        # Uses same query as TPM: concat(h_t, h_VC)
        self.smm = StrokeModificationModule(
            d_t=config.d_model,
            d_vc=config.d_model,
            d_patch=config.d_model,
            num_heads=config.smm_num_heads,
            dropout=config.dropout,
            temperature=config.smm_temperature
        )
        
        # 7. Head C: TPM (Tree-aware Position Module)
        self.tpm = TreeAwarePositionModule(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_head=config.d_head,
            num_relations=config.num_relations,
            dropout=config.dropout,
            temperature=config.tpm_temperature
        )
        
        # ROOT token for TPM (first symbol attaches here)
        # This is a learned embedding representing the tree root
        self.root_embedding = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.root_embedding, std=0.02)
        
        # Key projection for ROOT (matches transformer's key projection)
        self.root_key_proj = nn.Linear(config.d_model, config.num_heads * config.d_head)
        
        # Patch-to-stroke mapper
        self.patch_mapper = PatchStrokeMapper(
            image_size=config.image_size,
            grid_size=self.visual_encoder.grid_size
        )
        
        # LaTeX parser for relationship tensor
        self.latex_parser = LatexRelationshipParser()
    
    def _create_attention_mask(
        self,
        context_len: int,
        num_patches: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create attention mask for [context, patches, VC, INT].
        
        Rules:
        - context: causal (LaTeX tokens have sequential order)
        - patches: FULL self-attention (strokes exist simultaneously on canvas)
                   + can see all context
        - VC: can see all patches (aggregates visual info)
        - INT: can see context + VC (bridges text and vision)
        
        Why patches need full self-attention:
        - Strokes on canvas have NO causal relationship
        - A stroke at position (x1, y1) doesn't "come before" one at (x2, y2)
        - Each patch needs global context to understand spatial layout
        """
        total_len = context_len + num_patches + 2  # +2 for VC, INT
        mask = torch.zeros(batch_size, total_len, total_len, device=device)
        
        # Context: causal within itself (LaTeX has order)
        for i in range(context_len):
            mask[:, i, :i+1] = 1
        
        # Patches: FULL self-attention + see all context
        patch_start = context_len
        patch_end = context_len + num_patches
        for i in range(num_patches):
            # See all context
            mask[:, patch_start + i, :context_len] = 1
            # See ALL patches (full self-attention, not causal!)
            mask[:, patch_start + i, patch_start:patch_end] = 1
        
        # VC: can see all patches (visual aggregation)
        vc_pos = patch_end
        mask[:, vc_pos, patch_start:patch_end] = 1
        
        # INT: can see context + VC + itself (multimodal integration)
        int_pos = patch_end + 1
        mask[:, int_pos, :context_len] = 1  # Context
        mask[:, int_pos, vc_pos] = 1  # VC
        mask[:, int_pos, int_pos] = 1  # Self (standard practice)
        
        # VC should also see itself
        mask[:, vc_pos, vc_pos] = 1
        
        return mask
    
    def forward(
        self,
        images: torch.Tensor,
        context_ids: torch.Tensor,
        stroke_indices: Optional[List[List[Tuple[int, int]]]] = None,
        active_strokes: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        target_stroke_labels: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: [B, 1, H, W] input images
            context_ids: [B, L_ctx] LaTeX context token IDs
            stroke_indices: List of stroke index ranges per batch item
            active_strokes: [B, num_strokes] mask of active strokes
            target_actions: [B] target action indices for TPM
            target_stroke_labels: [B, num_strokes] binary stroke labels for SMM
            return_all: If True, return all intermediate tensors
            
        Returns:
            Dict with:
            - symbol_logits: [B, vocab_size] symbol predictions
            - stroke_scores: [B, num_strokes] stroke selection scores
            - action_probs: [B, num_actions] position action probabilities
            - losses (if targets provided)
        """
        B = images.shape[0]
        device = images.device
        
        # 1. Visual encoding
        patch_embeds, h_vc = self.visual_encoder(
            images, 
            stroke_mask=None  # Will handle masking separately
        )  # [B, num_patches, d], [B, d]
        
        # 2. Context embedding
        context_embeds = self.text_embedding(context_ids)  # [B, L_ctx, d]
        context_len = context_ids.shape[1]
        
        # 3. Prepare [INT] token
        h_int = self.int_token.expand(B, -1, -1)  # [B, 1, d]
        
        # 4. Concatenate: [context, patches, VC, INT]
        # VC is added as a single token derived from visual encoder
        h_vc_token = h_vc.unsqueeze(1)  # [B, 1, d]
        
        combined = torch.cat([
            context_embeds,  # [B, L_ctx, d]
            patch_embeds,    # [B, num_patches, d]
            h_vc_token,      # [B, 1, d]
            h_int            # [B, 1, d]
        ], dim=1)  # [B, L_ctx + num_patches + 2, d]
        
        # 5. Create attention mask
        attention_mask = self._create_attention_mask(
            context_len, self.num_patches, B, device
        )
        
        # 6. Transformer forward
        hidden_states, _, k_cache = self.decoder(
            combined, 
            attention_mask=attention_mask,
            use_cache=True
        )
        
        # 7. Extract specific hidden states
        # h_t: AR hidden state used for symbol prediction, SMM, and TPM
        # Currently using INT token position; INT is reserved for future distillation
        int_pos = context_len + self.num_patches + 1
        h_t = hidden_states[:, int_pos, :]  # [B, d] - used by Head A, SMM, TPM
        
        # h_VC: Visual Class token (aggregated visual features)
        vc_pos = context_len + self.num_patches
        h_vc_out = hidden_states[:, vc_pos, :]  # [B, d]
        
        # Patch hidden states (for SMM scoring)
        patch_hidden = hidden_states[:, context_len:context_len + self.num_patches, :]  # [B, P, d]
        
        # 8. Head A: Symbol prediction
        symbol_logits = self.symbol_head(h_t)  # [B, vocab_size]
        
        # 9. Head B: SMM stroke selection
        # Uses same query as TPM: concat(h_t, h_VC)
        stroke_scores = None
        if stroke_indices is not None:
            # Use first batch item's stroke indices for now (assume same structure)
            stroke_scores = self.smm(
                h_t=h_t,
                h_vc=h_vc_out,
                patch_tokens=patch_hidden,
                stroke_indices=stroke_indices[0] if isinstance(stroke_indices[0], list) else stroke_indices,
                active_strokes=active_strokes
            )  # [B, num_strokes]
        
        # 10. Head C: TPM position prediction
        # TPM K-cache: [ROOT, context_token_0, context_token_1, ...]
        # 
        # Model learns to match h_VC (stroke position) to the correct parent token
        # Action = parent_idx * num_relations + relation_type
        #
        # Parent indices:
        #   0 = ROOT (for first symbol in new expression)
        #   1 = context[0], 2 = context[1], ...
        #
        # This gives TPM a non-trivial task: find WHERE in context to insert!
        
        # Project ROOT embedding to key space [B, num_heads, 1, d_head]
        root_key = self.root_key_proj(self.root_embedding.expand(B, -1, -1))  # [B, 1, H*d_head]
        root_key = root_key.view(B, 1, self.config.num_heads, self.config.d_head)
        root_key = root_key.transpose(1, 2)  # [B, H, 1, d_head]
        
        # Extract context K-cache from transformer's K-cache
        # k_cache is [B, H, total_len, d_head] where total_len = context + patches + VC + INT
        # We want just the context portion: [:, :, :context_len, :]
        context_k_cache = k_cache[:, :, :context_len, :]  # [B, H, context_len, d_head]
        
        # Concatenate: [ROOT, context_tokens]
        # This gives action space: (1 + context_len) * num_relations
        tpm_k_cache = torch.cat([root_key, context_k_cache], dim=2)  # [B, H, 1+context_len, d_head]
        
        action_probs, action_logits = self.tpm(
            h_t=h_t,
            h_vc=h_vc_out,
            transformer_k_cache=tpm_k_cache
        )
        
        # Store context_len for action decoding
        output_context_len = context_len  # For decoding actions later
        
        # Build output
        output = {
            'symbol_logits': symbol_logits,
            'action_probs': action_probs,
            'action_logits': action_logits,
            'num_parents': 1 + context_len,  # ROOT + context tokens
        }
        
        if stroke_scores is not None:
            output['stroke_scores'] = stroke_scores
        
        if return_all:
            output['h_t'] = h_t  # AR hidden state (used by all heads)
            output['h_vc'] = h_vc_out  # Visual Class token
            output['patch_hidden'] = patch_hidden
            output['k_cache'] = k_cache
        
        return output
    
    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        target_actions: Optional[torch.Tensor] = None,
        target_stroke_labels: Optional[torch.Tensor] = None,
        active_strokes: Optional[torch.Tensor] = None,
        lambda_position: float = 1.0,
        lambda_stroke: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            output: Forward pass output dict
            target_tokens: [B] target token IDs
            target_actions: [B] target action indices
            target_stroke_labels: [B, num_strokes] binary stroke labels
            active_strokes: [B, num_strokes] mask of active strokes
            lambda_position: Weight for position loss
            lambda_stroke: Weight for stroke loss
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # Symbol loss (Head A)
        symbol_loss = F.cross_entropy(
            output['symbol_logits'],
            target_tokens,
            label_smoothing=self.config.label_smoothing
        )
        losses['symbol_loss'] = symbol_loss
        
        total_loss = symbol_loss
        
        # Position loss (Head C / TPM)
        if target_actions is not None and 'action_logits' in output:
            position_loss = F.cross_entropy(
                output['action_logits'],
                target_actions
            )
            losses['position_loss'] = position_loss
            total_loss = total_loss + lambda_position * position_loss
        
        # Stroke selection loss (Head B / SMM)
        # Note: stroke_scores are LOGITS (before sigmoid) for AMP compatibility
        if target_stroke_labels is not None and 'stroke_scores' in output:
            stroke_logits = output['stroke_scores']  # Logits!
            
            # Only compute loss on active strokes using BCEWithLogits (AMP-safe)
            if active_strokes is not None:
                mask = active_strokes.float()
                stroke_loss = F.binary_cross_entropy_with_logits(
                    stroke_logits,
                    target_stroke_labels.float(),
                    reduction='none'
                )
                stroke_loss = (stroke_loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                stroke_loss = F.binary_cross_entropy_with_logits(
                    stroke_logits,
                    target_stroke_labels.float()
                )
            
            losses['stroke_loss'] = stroke_loss
            total_loss = total_loss + lambda_stroke * stroke_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
        stroke_coords: Optional[List[torch.Tensor]] = None,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation with Option B (incremental stroke removal).
        
        Args:
            images: [B, 1, H, W] input images
            context_ids: [B, L_ctx] initial context (optional)
            stroke_coords: List of stroke coordinate tensors for masking
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)
            
        Returns:
            Dict with generated tokens, actions, and intermediate states
        """
        B = images.shape[0]
        device = images.device
        
        # Initialize context if not provided
        if context_ids is None:
            context_ids = torch.full(
                (B, 1), self.config.bos_token_id, 
                dtype=torch.long, device=device
            )
        
        generated_tokens = []
        generated_actions = []
        
        # Initialize active strokes (all active at start)
        num_strokes = len(stroke_coords) if stroke_coords else 0
        active_strokes = torch.ones(B, max(1, num_strokes), dtype=torch.bool, device=device)
        
        # Build stroke indices from coordinates
        stroke_indices = None
        if stroke_coords:
            stroke_indices = self.patch_mapper.build_stroke_to_patch_index(stroke_coords)
        
        # Current image (will be modified as strokes are removed)
        current_images = images.clone()
        
        for step in range(max_length):
            # Forward pass
            output = self.forward(
                images=current_images,
                context_ids=context_ids,
                stroke_indices=[stroke_indices] * B if stroke_indices else None,
                active_strokes=active_strokes if num_strokes > 0 else None
            )
            
            # Sample next token
            logits = output['symbol_logits'] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            generated_tokens.append(next_token)
            
            # Get action (position prediction)
            action_idx = output['action_probs'].argmax(dim=-1)
            generated_actions.append(action_idx)
            
            # Check for EOS
            if (next_token == self.config.eos_token_id).all():
                break
            
            # Update context
            context_ids = torch.cat([context_ids, next_token.unsqueeze(1)], dim=1)
            
            # Option B: Remove strokes based on SMM prediction
            # Note: stroke_scores are logits, apply sigmoid for probabilities
            if num_strokes > 0 and 'stroke_scores' in output:
                # Get contributing strokes (above threshold after sigmoid)
                stroke_probs = torch.sigmoid(output['stroke_scores'])
                contributing = stroke_probs > 0.5
                # Update active strokes (remove contributing ones)
                active_strokes = active_strokes & (~contributing)
                
                # Update image by masking out removed strokes
                stroke_mask = self.patch_mapper.stroke_to_patch_mask(
                    stroke_coords, batch_size=B, active_strokes=active_strokes
                ).to(device)
                
                # Re-encode with updated mask
                # (In practice, would zero out patches instead of re-encoding)
        
        return {
            'tokens': torch.stack(generated_tokens, dim=1),
            'actions': torch.stack(generated_actions, dim=1),
        }


def create_context_tracker(
    config: Optional[ContextTrackerConfig] = None,
    **kwargs
) -> ContextTrackerModel:
    """
    Factory function to create Context Tracker model.
    
    Args:
        config: Model configuration (uses defaults if None)
        **kwargs: Override config values
        
    Returns:
        Configured ContextTrackerModel
    """
    if config is None:
        config = ContextTrackerConfig()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return ContextTrackerModel(config)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Context Tracker Model...")
    print("=" * 60)
    
    # Create model
    config = ContextTrackerConfig(
        image_size=128,
        d_model=256,
        num_heads=8,
        num_layers=4,
        vocab_size=512,
        max_seq_len=64
    )
    model = create_context_tracker(config)
    
    # Print model info
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"Visual encoder patches: {model.num_patches}")
    
    # Test forward pass
    B = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    images = torch.randn(B, 1, 128, 128, device=device)
    context_ids = torch.randint(0, 512, (B, 10), device=device)
    
    # Without stroke info
    print("\n1. Forward pass (no stroke info)...")
    output = model(images, context_ids)
    print(f"   symbol_logits: {output['symbol_logits'].shape}")
    print(f"   action_probs: {output['action_probs'].shape}")
    
    # With stroke info
    print("\n2. Forward pass (with stroke info)...")
    stroke_indices = [(0, 16), (16, 32), (32, 48)]  # 3 strokes
    active_strokes = torch.ones(B, 3, dtype=torch.bool, device=device)
    
    output = model(
        images, context_ids,
        stroke_indices=[stroke_indices, stroke_indices],
        active_strokes=active_strokes,
        return_all=True
    )
    print(f"   symbol_logits: {output['symbol_logits'].shape}")
    print(f"   stroke_scores: {output['stroke_scores'].shape}")
    print(f"   action_probs: {output['action_probs'].shape}")
    print(f"   h_t: {output['h_t'].shape}")
    print(f"   h_vc: {output['h_vc'].shape}")
    
    # Test loss computation
    print("\n3. Loss computation...")
    target_tokens = torch.randint(0, 512, (B,), device=device)
    target_actions = torch.randint(0, 60, (B,), device=device)  # 10 tokens * 6 relations
    target_strokes = torch.zeros(B, 3, device=device)
    target_strokes[:, 0] = 1  # First stroke contributes
    
    losses = model.compute_loss(
        output,
        target_tokens=target_tokens,
        target_actions=target_actions,
        target_stroke_labels=target_strokes,
        active_strokes=active_strokes
    )
    print(f"   symbol_loss: {losses['symbol_loss'].item():.4f}")
    print(f"   position_loss: {losses['position_loss'].item():.4f}")
    print(f"   stroke_loss: {losses['stroke_loss'].item():.4f}")
    print(f"   total_loss: {losses['total_loss'].item():.4f}")
    
    # Test gradient flow
    print("\n4. Gradient flow...")
    losses['total_loss'].backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name.split('.')[0]] = grad_norms.get(name.split('.')[0], 0) + param.grad.norm().item()
    
    for module, norm in sorted(grad_norms.items()):
        print(f"   {module}: {norm:.4f}")
    
    # Test generation (minimal)
    print("\n5. Generation (3 steps)...")
    model.eval()
    gen_output = model.generate(images, max_length=3)
    print(f"   generated tokens: {gen_output['tokens'].shape}")
    print(f"   generated actions: {gen_output['actions'].shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")

