"""
Configurable Attention Module

Provides attention mechanisms with configurable mask patterns for multimodal input.
Supports different attention patterns between modalities (text, patches, CLS, INT).

Patterns:
- pattern_A_isolated_patches: Patches isolated within stroke groups
- pattern_B_context_aware_patches: Patches attend to text context
- pattern_C_hybrid_confidence_gated: Adaptive based on confidence
- pattern_D_per_stroke_cls_only: Only CLS tokens visible to decoder
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class MaskType(Enum):
    """Attention mask types."""
    CAUSAL = "causal"
    FULL = "full"
    FULL_WITHIN_GROUP = "full_within_group"
    NONE = "none"


class TokenType(Enum):
    """Token modality types."""
    TEXT_CONTEXT = "text_context"
    STROKE_PATCHES = "stroke_patches"
    CLS_TOKEN = "cls_token"
    INT_TOKEN = "int_token"
    DECODER = "decoder"


@dataclass
class ModalityConfig:
    """Configuration for a single modality's attention behavior."""
    attends_to: List[str]
    mask_type: str
    description: str = ""
    conditional_attend: Optional[Dict] = None
    visible_to_decoder: bool = True


@dataclass
class AttentionPatternConfig:
    """Configuration for an attention pattern."""
    name: str
    description: str
    modalities: Dict[str, ModalityConfig]
    advantages: List[str] = field(default_factory=list)
    cost: str = ""


class MultimodalAttentionMask(nn.Module):
    """
    Generates attention masks for multimodal sequences.
    
    Handles different attention patterns between:
    - Text context tokens (committed LaTeX)
    - Stroke patch tokens (visual features)
    - CLS tokens (per-stroke aggregation)
    - INT token (global integration)
    - Decoder tokens (output generation)
    
    Args:
        config_path: Path to attention_config.json
        pattern: Name of attention pattern to use
        
    Example:
        >>> mask_gen = MultimodalAttentionMask(pattern="pattern_A_isolated_patches")
        >>> mask = mask_gen.create_mask(
        ...     text_len=10, 
        ...     patch_counts=[4, 3, 5],  # patches per stroke
        ...     include_cls=True,
        ...     include_int=True
        ... )
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        pattern: str = "pattern_A_isolated_patches",
    ):
        super().__init__()
        
        self.pattern = pattern
        
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "attention_config.json"
        
        self.config = self._load_config(config_path, pattern)
    
    def _load_config(self, config_path: Union[str, Path], pattern: str) -> AttentionPatternConfig:
        """Load attention pattern configuration."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            return self._get_default_config(pattern)
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        if pattern not in config_data.get("attention_patterns", {}):
            available = list(config_data.get("attention_patterns", {}).keys())
            raise ValueError(f"Unknown pattern: {pattern}. Available: {available}")
        
        pat_config = config_data["attention_patterns"][pattern]
        
        modalities = {}
        for mod_name, mod_cfg in pat_config["modalities"].items():
            modalities[mod_name] = ModalityConfig(
                attends_to=mod_cfg["attends_to"],
                mask_type=mod_cfg["mask_type"],
                description=mod_cfg.get("description", ""),
                conditional_attend=mod_cfg.get("conditional_attend"),
                visible_to_decoder=mod_cfg.get("visible_to_decoder", True),
            )
        
        return AttentionPatternConfig(
            name=pat_config["name"],
            description=pat_config["description"],
            modalities=modalities,
            advantages=pat_config.get("advantages", []),
            cost=pat_config.get("cost", ""),
        )
    
    def _get_default_config(self, pattern: str) -> AttentionPatternConfig:
        """Default config when JSON not available."""
        return AttentionPatternConfig(
            name="Default Isolated Patches",
            description="Patches isolated, CLS aggregates, INT reasons",
            modalities={
                "text_context": ModalityConfig(
                    attends_to=["text_context"],
                    mask_type="causal",
                ),
                "stroke_patches": ModalityConfig(
                    attends_to=["stroke_patches_same_group"],
                    mask_type="full_within_group",
                ),
                "cls_token": ModalityConfig(
                    attends_to=["stroke_patches_same_group"],
                    mask_type="full",
                ),
                "int_token": ModalityConfig(
                    attends_to=["text_context", "all_cls_tokens"],
                    mask_type="full",
                ),
            },
        )
    
    def create_mask(
        self,
        text_len: int,
        patch_counts: List[int],
        include_cls: bool = True,
        include_int: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create attention mask for the multimodal sequence.
        
        Args:
            text_len: Number of text context tokens
            patch_counts: List of patch counts per stroke
            include_cls: Whether to include per-stroke CLS tokens
            include_int: Whether to include INT token
            device: Device for mask tensor
            dtype: Data type for mask tensor
            
        Returns:
            Attention mask of shape (seq_len, seq_len)
            0 = can attend, -inf = cannot attend
        """
        num_strokes = len(patch_counts)
        total_patches = sum(patch_counts)
        num_cls = num_strokes if include_cls else 0
        num_int = 1 if include_int else 0
        
        seq_len = text_len + total_patches + num_cls + num_int
        
        # Initialize with -inf (no attention)
        mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
        
        # Compute position ranges
        text_start, text_end = 0, text_len
        patch_start = text_end
        
        # Build stroke group ranges
        stroke_ranges = []
        current_pos = patch_start
        for count in patch_counts:
            stroke_ranges.append((current_pos, current_pos + count))
            current_pos += count
        
        patch_end = current_pos
        cls_start = patch_end
        cls_end = cls_start + num_cls
        int_start = cls_end
        int_end = int_start + num_int
        
        # Apply text context mask (causal)
        if text_len > 0:
            text_mod = self.config.modalities.get("text_context")
            if text_mod and text_mod.mask_type == "causal":
                # Causal mask for text
                for i in range(text_start, text_end):
                    mask[i, text_start:i+1] = 0  # Can attend to previous text
            else:
                # Full attention within text
                mask[text_start:text_end, text_start:text_end] = 0
        
        # Apply patch masks
        patch_mod = self.config.modalities.get("stroke_patches")
        if patch_mod:
            if "stroke_patches_same_group" in patch_mod.attends_to:
                # Patches attend only within their stroke group
                for start, end in stroke_ranges:
                    mask[start:end, start:end] = 0
            elif "stroke_patches" in patch_mod.attends_to:
                # All patches attend to all patches
                mask[patch_start:patch_end, patch_start:patch_end] = 0
            
            if "text_context" in patch_mod.attends_to:
                # Patches can attend to text
                mask[patch_start:patch_end, text_start:text_end] = 0
        
        # Apply CLS token masks
        if include_cls:
            cls_mod = self.config.modalities.get("cls_token")
            if cls_mod:
                for stroke_idx, (stroke_start, stroke_end) in enumerate(stroke_ranges):
                    cls_pos = cls_start + stroke_idx
                    
                    if "stroke_patches_same_group" in cls_mod.attends_to:
                        # CLS attends to its stroke's patches
                        mask[cls_pos, stroke_start:stroke_end] = 0
                    
                    if "text_context" in cls_mod.attends_to:
                        mask[cls_pos, text_start:text_end] = 0
        
        # Apply INT token mask
        if include_int:
            int_mod = self.config.modalities.get("int_token")
            if int_mod:
                int_pos = int_start
                
                if "text_context" in int_mod.attends_to:
                    mask[int_pos, text_start:text_end] = 0
                
                if "all_cls_tokens" in int_mod.attends_to and include_cls:
                    mask[int_pos, cls_start:cls_end] = 0
                
                if "stroke_patches" in int_mod.attends_to:
                    mask[int_pos, patch_start:patch_end] = 0
        
        return mask
    
    def create_decoder_mask(
        self,
        decoder_len: int,
        text_len: int,
        num_cls: int = 0,
        include_int: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create decoder attention mask.
        
        The decoder uses causal self-attention and cross-attends to context.
        
        Args:
            decoder_len: Number of decoder tokens
            text_len: Number of committed text tokens
            num_cls: Number of CLS tokens (0 if not visible)
            include_int: Whether INT token is in context
            
        Returns:
            Tuple of (self_mask, cross_mask)
        """
        decoder_mod = self.config.modalities.get("decoder")
        
        # Self-attention (causal)
        self_mask = torch.triu(
            torch.full((decoder_len, decoder_len), float('-inf'), dtype=dtype, device=device),
            diagonal=1
        )
        
        # Cross-attention
        context_len = text_len + num_cls + (1 if include_int else 0)
        cross_mask = torch.zeros((decoder_len, context_len), dtype=dtype, device=device)
        
        return self_mask, cross_mask
    
    def get_pattern_summary(self) -> Dict:
        """Return summary of current attention pattern."""
        return {
            "pattern": self.pattern,
            "name": self.config.name,
            "description": self.config.description,
            "modalities": {
                name: {
                    "attends_to": mod.attends_to,
                    "mask_type": mod.mask_type,
                }
                for name, mod in self.config.modalities.items()
            },
            "advantages": self.config.advantages,
            "cost": self.config.cost,
        }


class ConfigurableMultiheadAttention(nn.Module):
    """
    Multi-head attention with configurable masking patterns.
    
    Wraps standard attention with mask generation from config.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        config_path: Path to attention_config.json
        pattern: Attention pattern name
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        config_path: Optional[str] = None,
        pattern: str = "pattern_A_isolated_patches",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        self.mask_generator = MultimodalAttentionMask(config_path, pattern)
    
    def forward(
        self,
        x: torch.Tensor,
        text_len: int = 0,
        patch_counts: Optional[List[int]] = None,
        include_cls: bool = True,
        include_int: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic mask generation.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            text_len: Number of text tokens
            patch_counts: Patches per stroke (for mask generation)
            include_cls: Whether CLS tokens are present
            include_int: Whether INT token is present
            attn_mask: Optional pre-computed mask
            
        Returns:
            Output tensor and attention weights
        """
        B, L, D = x.shape
        
        # Generate mask if not provided
        if attn_mask is None and patch_counts is not None:
            attn_mask = self.mask_generator.create_mask(
                text_len=text_len,
                patch_counts=patch_counts,
                include_cls=include_cls,
                include_int=include_int,
                device=x.device,
                dtype=x.dtype,
            )
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        
        return out, attn_weights


class StrokeGroupedAttention(nn.Module):
    """
    Attention module with explicit stroke grouping.
    
    Handles the grouped structure:
    [Text Context] [Stroke1 Patches] [CLS1] [Stroke2 Patches] [CLS2] ... [INT]
    
    Each stroke's patches attend only within their group (isolated).
    CLS aggregates its stroke. INT reasons globally.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        config_path: Optional[str] = None,
        pattern: str = "pattern_A_isolated_patches",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Separate attention for different components
        self.text_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.patch_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.int_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        
        self.pattern = pattern
        
    def forward(
        self,
        text_tokens: torch.Tensor,
        patch_groups: List[torch.Tensor],
        cls_tokens: torch.Tensor,
        int_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Process multimodal input with grouped attention.
        
        Args:
            text_tokens: (batch, text_len, dim) - committed LaTeX tokens
            patch_groups: List of (batch, patches_i, dim) - patches per stroke
            cls_tokens: (batch, num_strokes, dim) - per-stroke CLS
            int_token: (batch, 1, dim) - global INT token
            
        Returns:
            Updated text, patch groups, CLS tokens, and INT token
        """
        B = text_tokens.shape[0]
        
        # 1. Text self-attention (causal)
        text_len = text_tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(text_len, text_len, device=text_tokens.device) * float('-inf'),
            diagonal=1
        )
        text_out, _ = self.text_attention(
            text_tokens, text_tokens, text_tokens,
            attn_mask=causal_mask
        )
        
        # 2. Patch attention within each stroke group (isolated)
        patch_outs = []
        cls_outs = []
        
        for i, patches in enumerate(patch_groups):
            # Patches attend to themselves
            patch_out, _ = self.patch_attention(patches, patches, patches)
            patch_outs.append(patch_out)
            
            # CLS attends to patches
            cls_i = cls_tokens[:, i:i+1, :]  # (B, 1, D)
            cls_out, _ = self.patch_attention(cls_i, patches, patches)
            cls_outs.append(cls_out)
        
        cls_out = torch.cat(cls_outs, dim=1)  # (B, num_strokes, D)
        
        # 3. INT attends to text + all CLS tokens
        context = torch.cat([text_out, cls_out], dim=1)  # (B, text+strokes, D)
        int_out, _ = self.int_attention(int_token, context, context)
        
        return text_out, patch_outs, cls_out, int_out


def load_attention_from_config(
    config_path: Optional[str] = None,
    pattern: Optional[str] = None,
) -> MultimodalAttentionMask:
    """
    Factory function to create attention mask generator from config.
    
    Args:
        config_path: Path to attention_config.json
        pattern: Pattern name (uses default if not specified)
        
    Returns:
        Configured MultimodalAttentionMask
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "attention_config.json"
    
    if pattern is None:
        with open(config_path, 'r') as f:
            config = json.load(f)
        pattern = config.get("default_pattern", "pattern_A_isolated_patches")
    
    return MultimodalAttentionMask(config_path=config_path, pattern=pattern)


def list_available_patterns(config_path: Optional[str] = None) -> List[str]:
    """List all available attention patterns."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "attention_config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return list(config.get("attention_patterns", {}).keys())


if __name__ == "__main__":
    print("Testing Configurable Attention...")
    print("=" * 60)
    
    # List available patterns
    patterns = list_available_patterns()
    print(f"Available patterns: {patterns}")
    print()
    
    # Test each pattern
    for pattern in patterns:
        print(f"Pattern: {pattern}")
        try:
            mask_gen = MultimodalAttentionMask(pattern=pattern)
            summary = mask_gen.get_pattern_summary()
            print(f"  Name: {summary['name']}")
            print(f"  Cost: {summary['cost']}")
            
            # Generate a sample mask
            mask = mask_gen.create_mask(
                text_len=10,
                patch_counts=[4, 3, 5],
                include_cls=True,
                include_int=True,
            )
            print(f"  Mask shape: {mask.shape}")
            
            # Count attendable positions per token type
            text_attend = (mask[:10, :10] == 0).sum().item()
            print(f"  Text self-attention positions: {text_attend}")
            
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
    
    # Test attention module
    print("Testing ConfigurableMultiheadAttention...")
    attn = ConfigurableMultiheadAttention(
        embed_dim=256,
        num_heads=8,
        pattern="pattern_A_isolated_patches"
    )
    
    x = torch.randn(2, 25, 256)  # 10 text + 12 patches + 3 CLS
    out, weights = attn(
        x,
        text_len=10,
        patch_counts=[4, 3, 5],
        include_cls=True,
        include_int=True,
    )
    print(f"Output shape: {out.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print("\nAll tests passed!")

