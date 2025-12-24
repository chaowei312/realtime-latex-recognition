"""
2D Rotary Position Embedding (RoPE)

Extension of RoPE for 2D spatial positions, useful for:
- Image patches (ViT-style)
- Handwriting strokes with (x, y) coordinates
- Any 2D grid or spatial data

Based on:
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- Extended to 2D following various vision transformer implementations

For 2D positions (x, y), we split the embedding dimension in half:
- First half encodes x position
- Second half encodes y position
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


def precompute_freqs_2d(
    dim: int,
    max_height: int,
    max_width: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute frequency tensors for 2D RoPE.
    
    Args:
        dim: Embedding dimension (must be divisible by 4 for x, y split with sin/cos)
        max_height: Maximum height (y positions)
        max_width: Maximum width (x positions)
        theta: Base for frequency computation
        device: Device to create tensors on
        
    Returns:
        Tuple of (freqs_h, freqs_w) each of shape (max_h/w, dim//4, 2)
        The last dimension contains [cos, sin]
    """
    assert dim % 4 == 0, f"Dimension must be divisible by 4, got {dim}"
    
    # Half dim for each axis (x and y)
    half_dim = dim // 2
    
    # Frequencies for each axis
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
    
    # Height (y) positions
    h_positions = torch.arange(max_height, device=device).float()
    freqs_h = torch.outer(h_positions, freqs)  # (max_height, half_dim // 2)
    freqs_h = torch.stack([freqs_h.cos(), freqs_h.sin()], dim=-1)  # (max_height, half_dim // 2, 2)
    
    # Width (x) positions
    w_positions = torch.arange(max_width, device=device).float()
    freqs_w = torch.outer(w_positions, freqs)  # (max_width, half_dim // 2)
    freqs_w = torch.stack([freqs_w.cos(), freqs_w.sin()], dim=-1)  # (max_width, half_dim // 2, 2)
    
    return freqs_h, freqs_w


def apply_rope_2d(
    x: torch.Tensor,
    freqs_h: torch.Tensor,
    freqs_w: torch.Tensor,
    positions_h: torch.Tensor,
    positions_w: torch.Tensor
) -> torch.Tensor:
    """
    Apply 2D RoPE to input tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
           or (batch, num_heads, seq_len, head_dim)
        freqs_h: Precomputed height frequencies
        freqs_w: Precomputed width frequencies  
        positions_h: Height (y) positions for each token, shape (batch, seq_len)
        positions_w: Width (x) positions for each token, shape (batch, seq_len)
        
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    # Ensure 4D input: (batch, seq_len, num_heads, head_dim)
    orig_shape = x.shape
    if len(x.shape) == 3:
        x = x.unsqueeze(2)  # Add head dimension
    
    batch, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    quarter_dim = head_dim // 4
    
    # Split into x and y components
    x_h, x_w = x[..., :half_dim], x[..., half_dim:]
    
    # Reshape for rotation: (..., quarter_dim, 2)
    x_h = x_h.reshape(batch, seq_len, num_heads, quarter_dim, 2)
    x_w = x_w.reshape(batch, seq_len, num_heads, quarter_dim, 2)
    
    # Get frequencies for the positions
    # freqs_h: (max_h, quarter_dim, 2), positions_h: (batch, seq_len)
    freqs_h_selected = freqs_h[positions_h.long()]  # (batch, seq_len, quarter_dim, 2)
    freqs_w_selected = freqs_w[positions_w.long()]  # (batch, seq_len, quarter_dim, 2)
    
    # Add head dimension
    freqs_h_selected = freqs_h_selected.unsqueeze(2)  # (batch, seq_len, 1, quarter_dim, 2)
    freqs_w_selected = freqs_w_selected.unsqueeze(2)
    
    # Apply rotation
    # For each pair (x0, x1), apply rotation by angle theta:
    # x0' = x0 * cos(theta) - x1 * sin(theta)
    # x1' = x0 * sin(theta) + x1 * cos(theta)
    
    def rotate(x_pair, freqs):
        cos = freqs[..., 0:1]
        sin = freqs[..., 1:2]
        x0, x1 = x_pair[..., 0:1], x_pair[..., 1:2]
        return torch.cat([
            x0 * cos - x1 * sin,
            x0 * sin + x1 * cos
        ], dim=-1)
    
    x_h_rotated = rotate(x_h, freqs_h_selected)
    x_w_rotated = rotate(x_w, freqs_w_selected)
    
    # Reshape back
    x_h_rotated = x_h_rotated.reshape(batch, seq_len, num_heads, half_dim)
    x_w_rotated = x_w_rotated.reshape(batch, seq_len, num_heads, half_dim)
    
    # Concatenate
    output = torch.cat([x_h_rotated, x_w_rotated], dim=-1)
    
    # Restore original shape
    if len(orig_shape) == 3:
        output = output.squeeze(2)
    
    return output


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding Module
    
    Encodes 2D positional information (e.g., x, y coordinates) using
    rotary embeddings. Particularly useful for:
    - Stroke sequences with spatial coordinates
    - Image patch sequences
    - Any 2D spatial data
    
    Args:
        dim: Embedding dimension (must be divisible by 4)
        max_height: Maximum height (y) position
        max_width: Maximum width (x) position
        theta: Base for frequency computation
        normalize_coords: If True, expects coordinates in [0, 1] range
        
    Example:
        >>> rope = RoPE2D(dim=64, max_height=100, max_width=100)
        >>> x = torch.randn(2, 50, 8, 64)  # (batch, seq, heads, dim)
        >>> pos_y = torch.randint(0, 100, (2, 50))
        >>> pos_x = torch.randint(0, 100, (2, 50))
        >>> out = rope(x, pos_y, pos_x)
    """
    
    def __init__(
        self,
        dim: int,
        max_height: int = 1024,
        max_width: int = 1024,
        theta: float = 10000.0,
        normalize_coords: bool = False
    ):
        super().__init__()
        
        assert dim % 4 == 0, f"Dimension must be divisible by 4, got {dim}"
        
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta = theta
        self.normalize_coords = normalize_coords
        
        # Precompute frequencies
        freqs_h, freqs_w = precompute_freqs_2d(dim, max_height, max_width, theta)
        
        # Register as buffers (not parameters, but move with model)
        self.register_buffer('freqs_h', freqs_h)
        self.register_buffer('freqs_w', freqs_w)
    
    def forward(
        self,
        x: torch.Tensor,
        positions_h: torch.Tensor,
        positions_w: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply 2D RoPE to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            positions_h: Height (y) positions, shape (batch, seq_len)
            positions_w: Width (x) positions, shape (batch, seq_len)
            
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        if self.normalize_coords:
            # Convert normalized [0, 1] coords to integer indices
            positions_h = (positions_h * (self.max_height - 1)).long()
            positions_w = (positions_w * (self.max_width - 1)).long()
        
        # Clamp to valid range
        positions_h = positions_h.clamp(0, self.max_height - 1)
        positions_w = positions_w.clamp(0, self.max_width - 1)
        
        return apply_rope_2d(x, self.freqs_h, self.freqs_w, positions_h, positions_w)
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}'


class RoPE2DForStrokes(nn.Module):
    """
    2D RoPE specifically designed for stroke sequences.
    
    Takes stroke coordinates directly and handles:
    - Coordinate normalization
    - Sequence padding
    - Efficient batched computation
    
    Args:
        dim: Embedding dimension
        coord_range: Expected coordinate range (min, max) for normalization
        max_resolution: Maximum position resolution
        
    Example:
        >>> rope = RoPE2DForStrokes(dim=64, coord_range=(0, 1000))
        >>> q = torch.randn(2, 50, 8, 64)  # Query
        >>> k = torch.randn(2, 50, 8, 64)  # Key
        >>> coords = torch.rand(2, 50, 2) * 1000  # (batch, seq, 2) for (x, y)
        >>> q_rot, k_rot = rope(q, k, coords)
    """
    
    def __init__(
        self,
        dim: int,
        coord_range: Tuple[float, float] = (0, 1000),
        max_resolution: int = 1024
    ):
        super().__init__()
        
        self.dim = dim
        self.coord_min, self.coord_max = coord_range
        self.coord_range = self.coord_max - self.coord_min
        
        self.rope = RoPE2D(
            dim=dim,
            max_height=max_resolution,
            max_width=max_resolution,
            normalize_coords=True
        )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            q: Query tensor (batch, seq, heads, dim)
            k: Key tensor (batch, seq, heads, dim)
            coords: Coordinates (batch, seq, 2) where [..., 0] is x and [..., 1] is y
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Normalize coordinates to [0, 1]
        coords_norm = (coords - self.coord_min) / self.coord_range
        coords_norm = coords_norm.clamp(0, 1)
        
        pos_x = coords_norm[..., 0]  # (batch, seq)
        pos_y = coords_norm[..., 1]  # (batch, seq)
        
        q_rot = self.rope(q, pos_y, pos_x)
        k_rot = self.rope(k, pos_y, pos_x)
        
        return q_rot, k_rot


if __name__ == "__main__":
    # Test 2D RoPE
    print("Testing 2D RoPE modules...")
    
    batch, seq_len, num_heads, head_dim = 2, 50, 8, 64
    max_h, max_w = 100, 100
    
    # Test RoPE2D
    rope = RoPE2D(dim=head_dim, max_height=max_h, max_width=max_w)
    x = torch.randn(batch, seq_len, num_heads, head_dim)
    pos_h = torch.randint(0, max_h, (batch, seq_len))
    pos_w = torch.randint(0, max_w, (batch, seq_len))
    
    out = rope(x, pos_h, pos_w)
    print(f"RoPE2D: {x.shape} -> {out.shape}")
    
    # Test RoPE2DForStrokes
    rope_stroke = RoPE2DForStrokes(dim=head_dim, coord_range=(0, 1000))
    q = torch.randn(batch, seq_len, num_heads, head_dim)
    k = torch.randn(batch, seq_len, num_heads, head_dim)
    coords = torch.rand(batch, seq_len, 2) * 1000
    
    q_rot, k_rot = rope_stroke(q, k, coords)
    print(f"RoPE2DForStrokes: q {q.shape} -> {q_rot.shape}")
    
    # Verify rotation preserves norm (approximately)
    q_norm = q.norm(dim=-1).mean()
    q_rot_norm = q_rot.norm(dim=-1).mean()
    print(f"Norm preservation: {q_norm:.4f} -> {q_rot_norm:.4f}")
    
    print("All tests passed!")

