"""
SwiGLU (Swish-Gated Linear Unit)

Implementation of the SwiGLU activation function from:
"GLU Variants Improve Transformer" (Shazeer, 2020)
https://arxiv.org/abs/2002.05202

SwiGLU combines the Swish activation with a gating mechanism,
providing improved performance over standard ReLU/GELU in transformers.

Formula: SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
Where Swish(x) = x * sigmoid(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Module
    
    Implements: output = Swish(x @ W1) * (x @ W2)
    Where Swish(x) = x * sigmoid(x)
    
    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (output of gate and value projections)
        out_features: Output dimension (if None, equals in_features)
        bias: Whether to use bias in linear layers
        
    Example:
        >>> swiglu = SwiGLU(512, 2048)
        >>> x = torch.randn(32, 100, 512)
        >>> out = swiglu(x)  # (32, 100, 512)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        # Gate projection (for Swish)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)
        
        # Value projection
        self.w_value = nn.Linear(in_features, hidden_features, bias=bias)
        
        # Output projection
        self.w_out = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, in_features)
            
        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        # Gate: Swish activation
        gate = F.silu(self.w_gate(x))  # silu = swish = x * sigmoid(x)
        
        # Value
        value = self.w_value(x)
        
        # Gated output
        hidden = gate * value
        
        # Project to output dimension
        return self.w_out(hidden)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network
    
    A complete FFN block using SwiGLU, typically used as a drop-in
    replacement for the standard Transformer FFN.
    
    Includes optional dropout and layer normalization.
    
    Args:
        dim: Model dimension
        hidden_dim: Hidden dimension (if None, 4 * dim * 2/3 rounded to multiple of 256)
        dropout: Dropout probability
        bias: Whether to use bias
        
    Example:
        >>> ffn = SwiGLUFFN(dim=512, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(32, 100, 512)
        >>> out = ffn(x)  # (32, 100, 512)
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        # Default hidden_dim following LLaMA style
        if hidden_dim is None:
            hidden_dim = int(4 * dim * 2 / 3)
            # Round to nearest multiple of 256 for efficiency
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Combined gate and up projection for efficiency
        # Projects to 2 * hidden_dim, then splits
        self.w_gate_up = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        
        # Down projection
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Combined projection and split
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SwiGLU: Swish(gate) * up
        hidden = F.silu(gate) * up
        
        # Down projection with dropout
        return self.dropout(self.w_down(hidden))


def swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Functional SwiGLU activation.
    
    Args:
        gate: Gate tensor (will be passed through Swish)
        value: Value tensor
        
    Returns:
        Swish(gate) * value
    """
    return F.silu(gate) * value


# Alias for compatibility
Swiglu = SwiGLU


if __name__ == "__main__":
    # Test SwiGLU
    print("Testing SwiGLU modules...")
    
    batch, seq_len, dim = 2, 10, 512
    hidden_dim = 2048
    
    x = torch.randn(batch, seq_len, dim)
    
    # Test SwiGLU
    swiglu_module = SwiGLU(dim, hidden_dim)
    out = swiglu_module(x)
    print(f"SwiGLU: {x.shape} -> {out.shape}")
    
    # Test SwiGLUFFN
    ffn = SwiGLUFFN(dim=dim, hidden_dim=hidden_dim, dropout=0.1)
    out = ffn(x)
    print(f"SwiGLUFFN: {x.shape} -> {out.shape}")
    
    # Parameter count
    params = sum(p.numel() for p in ffn.parameters())
    print(f"SwiGLUFFN parameters: {params:,}")
    
    print("All tests passed!")

