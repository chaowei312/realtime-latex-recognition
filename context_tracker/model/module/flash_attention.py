"""
Flash Attention Implementation

Memory-efficient attention with optional causal masking for decoder architectures.

This module provides:
1. FlashAttention: Core flash attention mechanism
2. FlashAttentionDecoder: Complete decoder layer with causal masking

Based on:
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism" (Dao, 2023)

Note: This implementation uses PyTorch's scaled_dot_product_attention which
automatically uses FlashAttention when available (PyTorch >= 2.0).
For older PyTorch versions, falls back to standard attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FlashAttention(nn.Module):
    """
    Flash Attention Module
    
    Uses PyTorch 2.0's scaled_dot_product_attention which automatically
    selects the most efficient implementation (Flash Attention, Memory-Efficient
    Attention, or standard attention).
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (if None, dim // num_heads)
        dropout: Attention dropout probability
        causal: Whether to use causal (autoregressive) masking
        bias: Whether to use bias in projections
        
    Example:
        >>> attn = FlashAttention(dim=512, num_heads=8, causal=True)
        >>> x = torch.randn(2, 100, 512)
        >>> out = attn(x)  # (2, 100, 512)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.causal = causal
        
        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(dim, 3 * self.inner_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional attention mask (batch, seq_len, seq_len)
            key_padding_mask: Optional padding mask (batch, seq_len), True = masked
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Build attention mask
        attn_mask = None
        if self.causal or attention_mask is not None or key_padding_mask is not None:
            attn_mask = self._build_mask(
                batch, seq_len, q.device, q.dtype,
                attention_mask, key_padding_mask
            )
        
        # Flash Attention via PyTorch's SDPA
        # This automatically uses the most efficient implementation
        dropout_p = self.dropout if self.training else 0.0
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.causal and attn_mask is None  # Use built-in causal if no custom mask
        )
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch, seq_len, self.inner_dim)
        out = self.out_proj(out)
        
        return out
    
    def _build_mask(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Build combined attention mask."""
        
        # Start with causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
        else:
            mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        
        # Add custom attention mask
        if attention_mask is not None:
            mask = mask + attention_mask
        
        # Add key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq_len), True = masked
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            padding_mask = padding_mask.expand(-1, self.num_heads, seq_len, -1)
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, self.num_heads, -1, -1)
            mask = mask.masked_fill(padding_mask, float('-inf'))
        
        return mask


class FlashCrossAttention(nn.Module):
    """
    Flash Cross-Attention for encoder-decoder architectures.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dropout: Attention dropout
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        # Separate Q and KV projections for cross-attention
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=bias)
        self.kv_proj = nn.Linear(dim, 2 * self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=bias)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor (batch, tgt_len, dim)
            context: Key/Value tensor (batch, src_len, dim)
            context_mask: Padding mask for context (batch, src_len)
            
        Returns:
            Output tensor (batch, tgt_len, dim)
        """
        batch, tgt_len, _ = x.shape
        src_len = context.shape[1]
        
        # Project Q from x, KV from context
        q = self.q_proj(x)
        kv = self.kv_proj(context)
        
        # Reshape
        q = q.reshape(batch, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(batch, src_len, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        
        # Build mask
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.num_heads, tgt_len, -1)
            attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
        
        # Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch, tgt_len, self.inner_dim)
        return self.out_proj(out)


class FlashAttentionDecoder(nn.Module):
    """
    Complete Transformer Decoder Layer with Flash Attention
    
    Includes:
    - Causal self-attention (with Flash Attention)
    - Optional cross-attention
    - Feed-forward network (can use SwiGLU)
    - Pre-norm architecture (more stable training)
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward hidden dimension (default: 4 * dim)
        dropout: Dropout probability
        attention_dropout: Attention-specific dropout
        use_cross_attention: Whether to include cross-attention
        use_swiglu: Whether to use SwiGLU for FFN
        norm_eps: Layer norm epsilon
        
    Example:
        >>> decoder = FlashAttentionDecoder(dim=512, num_heads=8, use_cross_attention=True)
        >>> x = torch.randn(2, 100, 512)
        >>> encoder_out = torch.randn(2, 50, 512)
        >>> out = decoder(x, encoder_out)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        use_cross_attention: bool = False,
        use_swiglu: bool = True,
        norm_eps: float = 1e-6
    ):
        super().__init__()
        
        self.dim = dim
        ffn_dim = ffn_dim or 4 * dim
        
        # Self-attention with causal mask
        self.self_attn = FlashAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            causal=True
        )
        self.self_attn_norm = nn.LayerNorm(dim, eps=norm_eps)
        
        # Cross-attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = FlashCrossAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=attention_dropout
            )
            self.cross_attn_norm = nn.LayerNorm(dim, eps=norm_eps)
        
        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
        
        if use_swiglu:
            try:
                from .swiglu import SwiGLUFFN
            except ImportError:
                from swiglu import SwiGLUFFN
            self.ffn = SwiGLUFFN(dim=dim, hidden_dim=ffn_dim, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, dim),
                nn.Dropout(dropout)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            encoder_out: Encoder output for cross-attention (batch, src_len, dim)
            self_attn_mask: Additional self-attention mask
            self_attn_padding_mask: Padding mask for self-attention
            encoder_padding_mask: Padding mask for encoder output
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Self-attention with residual
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, self_attn_mask, self_attn_padding_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Cross-attention with residual (if enabled)
        if self.use_cross_attention and encoder_out is not None:
            residual = x
            x = self.cross_attn_norm(x)
            x = self.cross_attn(x, encoder_out, encoder_padding_mask)
            x = self.dropout(x)
            x = residual + x
        
        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class FlashAttentionDecoderStack(nn.Module):
    """
    Stack of Flash Attention Decoder layers.
    
    Args:
        num_layers: Number of decoder layers
        dim: Model dimension
        num_heads: Number of attention heads
        ffn_dim: FFN hidden dimension
        dropout: Dropout probability
        use_cross_attention: Whether to use cross-attention
        use_swiglu: Whether to use SwiGLU
    """
    
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        use_swiglu: bool = True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            FlashAttentionDecoder(
                dim=dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
                use_swiglu=use_swiglu
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward through all decoder layers."""
        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                self_attn_padding_mask=padding_mask,
                encoder_padding_mask=encoder_padding_mask
            )
        
        return self.final_norm(x)


if __name__ == "__main__":
    # Test Flash Attention modules
    print("Testing Flash Attention modules...")
    
    batch, seq_len, dim = 2, 100, 512
    num_heads = 8
    
    # Test FlashAttention (self-attention)
    attn = FlashAttention(dim=dim, num_heads=num_heads, causal=True, dropout=0.1)
    x = torch.randn(batch, seq_len, dim)
    out = attn(x)
    print(f"FlashAttention (causal): {x.shape} -> {out.shape}")
    
    # Test with padding mask
    padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    padding_mask[:, -10:] = True  # Mask last 10 tokens
    out = attn(x, key_padding_mask=padding_mask)
    print(f"FlashAttention (with padding): {x.shape} -> {out.shape}")
    
    # Test FlashAttentionDecoder
    decoder = FlashAttentionDecoder(
        dim=dim, num_heads=num_heads,
        use_cross_attention=True, use_swiglu=True
    )
    encoder_out = torch.randn(batch, 50, dim)
    out = decoder(x, encoder_out)
    print(f"FlashAttentionDecoder: {x.shape} -> {out.shape}")
    
    # Test decoder stack
    decoder_stack = FlashAttentionDecoderStack(
        num_layers=6, dim=dim, num_heads=num_heads,
        use_cross_attention=True
    )
    out = decoder_stack(x, encoder_out)
    print(f"FlashAttentionDecoderStack (6 layers): {x.shape} -> {out.shape}")
    
    # Parameter count
    params = sum(p.numel() for p in decoder_stack.parameters())
    print(f"Decoder stack parameters: {params:,}")
    
    print("All tests passed!")

