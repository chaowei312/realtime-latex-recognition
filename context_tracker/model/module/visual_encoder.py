"""
Visual Patch Encoder

Stacked convolutional encoder for processing full images into patch embeddings.
Designed for 128x128 input images → 64 (8×8) patch embeddings.

Architecture:
- 3 layers of 3x3 Conv with stride 2 (progressive downsampling)
- 2D sinusoidal positional embeddings for spatial awareness
- Supports stroke masking for incremental removal during AR decoding

Key Design Choices:
- Stacked 3x3 kernels outperform single large kernel (e.g., 16x16) for ViT-like tasks
- Each conv layer has receptive field growth: 3→7→15 effective RF
- Final 8×8 grid provides 64 patch tokens, matching typical transformer sequence lengths
- 2D positional embeddings preserve spatial relationships for position prediction (TPM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class StackedConvStem(nn.Module):
    """
    Stacked 3x3 Convolutional Stem
    
    Progressive downsampling with small kernels:
    - Layer 1: 128×128 → 64×64 (stride 2)
    - Layer 2: 64×64 → 32×32 (stride 2)
    - Layer 3: 32×32 → 16×16 (stride 2)
    - Layer 4: 16×16 → 8×8 (stride 2)
    
    Final: 64 patches in 8×8 grid
    
    Args:
        in_channels: Input image channels (1 for grayscale, 3 for RGB)
        embed_dim: Output embedding dimension per patch
        num_layers: Number of conv layers (default: 4 for 128→8 = 16x downsampling)
        hidden_dims: Hidden channel dimensions for intermediate layers
        norm: Normalization type ('batch', 'layer', 'group', None)
        activation: Activation function ('gelu', 'relu', 'silu')
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_layers: int = 4,
        hidden_dims: Optional[List[int]] = None,
        norm: str = 'batch',
        activation: str = 'gelu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Default hidden dims: progressive expansion
        if hidden_dims is None:
            # e.g., for embed_dim=256: [32, 64, 128, 256]
            hidden_dims = [embed_dim // (2 ** (num_layers - i - 1)) for i in range(num_layers)]
            # Ensure minimum of 16 channels
            hidden_dims = [max(16, d) for d in hidden_dims]
        
        assert len(hidden_dims) == num_layers, f"hidden_dims length must match num_layers"
        
        self.hidden_dims = hidden_dims
        
        # Build conv layers
        layers = []
        current_channels = in_channels
        
        for i, out_channels in enumerate(hidden_dims):
            layers.append(self._make_conv_block(
                current_channels, out_channels,
                kernel_size=3, stride=2, padding=1,
                norm=norm, activation=activation, dropout=dropout,
                is_last=(i == num_layers - 1)
            ))
            current_channels = out_channels
        
        self.layers = nn.ModuleList(layers)
        
        # Ensure final output matches embed_dim
        if hidden_dims[-1] != embed_dim:
            self.output_proj = nn.Conv2d(hidden_dims[-1], embed_dim, kernel_size=1)
        else:
            self.output_proj = nn.Identity()
    
    def _make_conv_block(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm: str,
        activation: str,
        dropout: float,
        is_last: bool = False
    ) -> nn.Sequential:
        """Create a single conv block with norm and activation."""
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=(norm is None))
        ]
        
        # Normalization
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == 'layer':
            layers.append(nn.GroupNorm(1, out_ch))  # LayerNorm equivalent
        elif norm == 'group':
            num_groups = min(32, out_ch)
            while out_ch % num_groups != 0:
                num_groups -= 1
            layers.append(nn.GroupNorm(num_groups, out_ch))
        
        # Activation
        if activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'silu':
            layers.append(nn.SiLU(inplace=True))
        
        # Dropout (not on last layer)
        if dropout > 0 and not is_last:
            layers.append(nn.Dropout2d(dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, H, W] (e.g., [B, 1, 128, 128])
            
        Returns:
            Patch features [B, embed_dim, H', W'] (e.g., [B, 256, 8, 8])
        """
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x


class PositionalEmbedding2D(nn.Module):
    """
    2D Sinusoidal Positional Embeddings
    
    Creates fixed or learnable positional embeddings for a 2D grid of patches.
    Uses sinusoidal encoding similar to ViT but extended to 2D.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Size of the 2D grid (H', W') or int for square grid
        learnable: If True, use learnable embeddings instead of sinusoidal
        temperature: Temperature for sinusoidal encoding (higher = lower freq)
    """
    
    def __init__(
        self,
        embed_dim: int,
        grid_size: Union[int, Tuple[int, int]] = 8,
        learnable: bool = False,
        temperature: float = 10000.0
    ):
        super().__init__()
        
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.learnable = learnable
        self.num_patches = grid_size[0] * grid_size[1]
        
        if learnable:
            # Learnable positional embeddings
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Fixed sinusoidal embeddings
            pos_embed = self._create_sinusoidal_2d(grid_size, embed_dim, temperature)
            self.register_buffer('pos_embed', pos_embed)
    
    def _create_sinusoidal_2d(
        self,
        grid_size: Tuple[int, int],
        embed_dim: int,
        temperature: float
    ) -> torch.Tensor:
        """Create 2D sinusoidal positional embeddings."""
        h, w = grid_size
        
        # Split embed_dim between y and x coordinates
        dim_per_coord = embed_dim // 2
        
        # Create position indices
        y_pos = torch.arange(h).float().unsqueeze(1)  # [H, 1]
        x_pos = torch.arange(w).float().unsqueeze(1)  # [W, 1]
        
        # Create frequency bands
        dim_idx = torch.arange(dim_per_coord).float()
        freq = 1.0 / (temperature ** (2 * (dim_idx // 2) / dim_per_coord))
        
        # Compute sinusoidal embeddings
        y_embed = torch.zeros(h, dim_per_coord)
        x_embed = torch.zeros(w, dim_per_coord)
        
        y_embed[:, 0::2] = torch.sin(y_pos * freq[0::2])
        y_embed[:, 1::2] = torch.cos(y_pos * freq[1::2])
        
        x_embed[:, 0::2] = torch.sin(x_pos * freq[0::2])
        x_embed[:, 1::2] = torch.cos(x_pos * freq[1::2])
        
        # Combine y and x embeddings for each grid position
        pos_embed = torch.zeros(h * w, embed_dim)
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                pos_embed[idx, :dim_per_coord] = y_embed[i]
                pos_embed[idx, dim_per_coord:2*dim_per_coord] = x_embed[j]
        
        # Handle odd embed_dim (pad with zeros)
        # Already handled by slicing
        
        return pos_embed.unsqueeze(0)  # [1, H*W, embed_dim]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch tokens [B, N, D] where N = H' * W'
            
        Returns:
            Positionally-encoded patch tokens [B, N, D]
        """
        return x + self.pos_embed
    
    def get_embedding(self) -> torch.Tensor:
        """Get the raw positional embeddings [1, N, D]."""
        return self.pos_embed


class VisualPatchEncoder(nn.Module):
    """
    Visual Patch Encoder
    
    Complete encoder that converts input images to patch embeddings with positional encoding.
    
    Architecture:
    1. Stacked 3x3 Conv stem: 128×128 → 8×8 grid (64 patches)
    2. Flatten: [B, C, 8, 8] → [B, 64, C]
    3. 2D Positional Embeddings: Add spatial information
    4. Optional: [VC] (Visual Class) token prepended
    
    Args:
        image_size: Input image size (int for square, tuple for rectangular)
        in_channels: Input image channels (1 for grayscale)
        embed_dim: Output embedding dimension
        num_conv_layers: Number of conv layers in stem
        hidden_dims: Hidden dimensions for conv layers
        add_vc_token: Whether to add [VC] token
        pos_embed_type: 'sinusoidal' or 'learnable'
        norm: Normalization type
        activation: Activation function
        dropout: Dropout probability
        
    Example:
        >>> encoder = VisualPatchEncoder(image_size=128, embed_dim=256)
        >>> x = torch.randn(2, 1, 128, 128)
        >>> patches, vc = encoder(x)
        >>> print(patches.shape)  # [2, 64, 256]
        >>> print(vc.shape)       # [2, 256]
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 128,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_conv_layers: int = 4,
        hidden_dims: Optional[List[int]] = None,
        add_vc_token: bool = True,
        pos_embed_type: str = 'sinusoidal',
        norm: str = 'batch',
        activation: str = 'gelu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.add_vc_token = add_vc_token
        
        # Compute output grid size: each conv layer halves spatial dims
        self.grid_size = (
            image_size[0] // (2 ** num_conv_layers),
            image_size[1] // (2 ** num_conv_layers)
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Convolutional stem
        self.conv_stem = StackedConvStem(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_conv_layers,
            hidden_dims=hidden_dims,
            norm=norm,
            activation=activation,
            dropout=dropout
        )
        
        # Positional embeddings
        self.pos_embed = PositionalEmbedding2D(
            embed_dim=embed_dim,
            grid_size=self.grid_size,
            learnable=(pos_embed_type == 'learnable')
        )
        
        # [VC] (Visual Class) token
        if add_vc_token:
            self.vc_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.vc_token, std=0.02)
        
        # Layer norm after positional embedding (stabilizes training)
        self.post_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        stroke_mask: Optional[torch.Tensor] = None,
        return_grid: bool = False
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], 
               Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        """
        Args:
            x: Input image [B, C, H, W]
            stroke_mask: Optional mask [B, H', W'] indicating active patches
                        (1 = active, 0 = removed/masked)
            return_grid: If True, also return the 2D feature grid
            
        Returns:
            patches: Patch embeddings [B, N, D] where N = num_patches (or N+1 with VC)
            vc_token: [VC] token embedding [B, D] if add_vc_token, else None
            grid (optional): 2D feature grid [B, D, H', W'] before flattening
        """
        B = x.shape[0]
        
        # 1. Conv stem: [B, C, H, W] -> [B, D, H', W']
        grid = self.conv_stem(x)
        
        # 2. Flatten to patches: [B, D, H', W'] -> [B, N, D]
        patches = grid.flatten(2).transpose(1, 2)  # [B, H'*W', D]
        
        # 3. Normalize stroke_mask if provided
        mask_flat = None
        if stroke_mask is not None:
            # stroke_mask: [B, H', W'] or [B, N]
            if stroke_mask.dim() == 3:
                mask_flat = stroke_mask.flatten(1)  # [B, N]
            else:
                mask_flat = stroke_mask
        
        # 4. Add positional embeddings
        patches = self.pos_embed(patches)
        
        # 5. Layer norm
        patches = self.post_norm(patches)
        
        # 6. Apply stroke mask AFTER pos embed and norm (zero out removed patches)
        # This ensures masked patches have exactly zero contribution
        if mask_flat is not None:
            patches = patches * mask_flat.unsqueeze(-1)
        
        # 7. [VC] token
        vc_token = None
        if self.add_vc_token:
            # Pool patches to create VC token (mean pooling)
            if mask_flat is not None:
                # Masked mean pooling (only active patches)
                mask_sum = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
                vc_token = patches.sum(dim=1) / mask_sum
            else:
                vc_token = patches.mean(dim=1)  # [B, D]
            
            # Alternative: use learnable VC token directly (prepend to sequence)
            # For this design, we compute VC from patches but also have a learnable base
            vc_token = vc_token + self.vc_token.squeeze(1).expand(B, -1)
        
        if return_grid:
            return patches, vc_token, grid
        return patches, vc_token
    
    def get_num_patches(self) -> int:
        """Get number of output patches."""
        return self.num_patches
    
    def get_grid_size(self) -> Tuple[int, int]:
        """Get output grid size (H', W')."""
        return self.grid_size
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.embed_dim


class PatchStrokeMapper(nn.Module):
    """
    Maps stroke indices to patch indices and vice versa.
    
    Used for:
    - Converting stroke-level masks to patch-level masks for masking
    - Computing which strokes contributed to which patches
    
    This module handles the spatial relationship between:
    - Original image coordinates (where strokes are drawn)
    - Patch grid coordinates (output of VisualPatchEncoder)
    
    Args:
        image_size: Original image size
        grid_size: Output patch grid size
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        grid_size: Union[int, Tuple[int, int]]
    ):
        super().__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        
        self.image_size = image_size
        self.grid_size = grid_size
        
        # Compute patch size in image coordinates
        self.patch_size = (
            image_size[0] // grid_size[0],
            image_size[1] // grid_size[1]
        )
    
    def stroke_to_patches(
        self,
        stroke_coords: torch.Tensor,  # [N_points, 2] or list of [N_i, 2]
        stroke_ids: Optional[torch.Tensor] = None  # [N_points] stroke index per point
    ) -> torch.Tensor:
        """
        Convert stroke coordinates to patch indices.
        
        Args:
            stroke_coords: Stroke point coordinates [N, 2] in image space (x, y)
            stroke_ids: Optional stroke ID for each point [N]
            
        Returns:
            patch_indices: [N] index of patch containing each point (flat index)
        """
        # Clamp to valid range
        x = stroke_coords[:, 0].clamp(0, self.image_size[1] - 1)
        y = stroke_coords[:, 1].clamp(0, self.image_size[0] - 1)
        
        # Convert to patch grid coordinates
        patch_x = (x / self.patch_size[1]).long().clamp(0, self.grid_size[1] - 1)
        patch_y = (y / self.patch_size[0]).long().clamp(0, self.grid_size[0] - 1)
        
        # Flat patch index
        patch_indices = patch_y * self.grid_size[1] + patch_x
        
        return patch_indices
    
    def stroke_to_patch_mask(
        self,
        stroke_coords: List[torch.Tensor],  # List of [N_i, 2] per stroke
        batch_size: int = 1,
        active_strokes: Optional[torch.Tensor] = None  # [B, num_strokes]
    ) -> torch.Tensor:
        """
        Convert list of strokes to a patch mask.
        
        Args:
            stroke_coords: List of stroke coordinate tensors
            batch_size: Batch size for output mask
            active_strokes: Boolean mask [B, num_strokes] indicating active strokes
            
        Returns:
            patch_mask: [B, H', W'] binary mask (1 = has stroke, 0 = empty)
        """
        num_patches = self.grid_size[0] * self.grid_size[1]
        patch_mask = torch.zeros(batch_size, num_patches, dtype=torch.float32)
        
        for stroke_idx, coords in enumerate(stroke_coords):
            if coords.numel() == 0:
                continue
                
            patch_indices = self.stroke_to_patches(coords)
            
            for b in range(batch_size):
                if active_strokes is not None and not active_strokes[b, stroke_idx]:
                    continue
                patch_mask[b, patch_indices.unique()] = 1.0
        
        return patch_mask.view(batch_size, *self.grid_size)
    
    def build_stroke_to_patch_index(
        self,
        stroke_coords: List[torch.Tensor]  # List of [N_i, 2] per stroke
    ) -> List[Tuple[int, int]]:
        """
        Build stroke index mapping: stroke_indices[i] = (start_patch, end_patch).
        
        Returns list of (start, end) patch ranges for each stroke.
        Note: Multiple strokes may share patches, so ranges may overlap.
        """
        stroke_indices = []
        
        for coords in stroke_coords:
            if coords.numel() == 0:
                stroke_indices.append((0, 0))
                continue
            
            patch_indices = self.stroke_to_patches(coords)
            unique_patches = patch_indices.unique().tolist()
            
            if len(unique_patches) == 0:
                stroke_indices.append((0, 0))
            else:
                stroke_indices.append((min(unique_patches), max(unique_patches) + 1))
        
        return stroke_indices


def create_visual_encoder(
    image_size: int = 128,
    embed_dim: int = 256,
    variant: str = 'standard',
    **kwargs
) -> VisualPatchEncoder:
    """
    Factory function to create visual encoder with common configurations.
    
    Args:
        image_size: Input image size
        embed_dim: Output embedding dimension
        variant: Configuration variant
            - 'tiny': 4 layers, small hidden dims (for fast prototyping)
            - 'standard': 4 layers, balanced hidden dims (default)
            - 'large': 4 layers, larger hidden dims (for better accuracy)
        **kwargs: Override any VisualPatchEncoder argument
        
    Returns:
        Configured VisualPatchEncoder
    """
    configs = {
        'tiny': {
            'num_conv_layers': 4,
            'hidden_dims': [16, 32, 64, embed_dim],
        },
        'standard': {
            'num_conv_layers': 4,
            'hidden_dims': [32, 64, 128, embed_dim],
        },
        'large': {
            'num_conv_layers': 4,
            'hidden_dims': [64, 128, 256, embed_dim],
        },
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    config = configs[variant].copy()
    config.update(kwargs)
    
    return VisualPatchEncoder(
        image_size=image_size,
        embed_dim=embed_dim,
        **config
    )


if __name__ == "__main__":
    print("Testing Visual Patch Encoder...")
    
    # Test standard configuration
    encoder = create_visual_encoder(
        image_size=128,
        embed_dim=256,
        variant='standard',
        add_vc_token=True
    )
    
    print(f"\nEncoder configuration:")
    print(f"  Image size: {encoder.image_size}")
    print(f"  Grid size: {encoder.grid_size}")
    print(f"  Num patches: {encoder.num_patches}")
    print(f"  Embed dim: {encoder.embed_dim}")
    
    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    patches, vc_token = encoder(x)
    
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Patches: {patches.shape}")
    print(f"  VC token: {vc_token.shape if vc_token is not None else None}")
    
    # Test with stroke mask
    stroke_mask = torch.ones(2, 8, 8)
    stroke_mask[:, 4:, 4:] = 0  # Mask out bottom-right quadrant
    
    patches_masked, vc_masked = encoder(x, stroke_mask=stroke_mask)
    print(f"\nWith stroke mask:")
    print(f"  Masked patches: {patches_masked.shape}")
    print(f"  VC token: {vc_masked.shape if vc_masked is not None else None}")
    
    # Verify masked patches are zero
    mask_flat = stroke_mask.flatten(1).unsqueeze(-1)
    masked_norm = (patches_masked * (1 - mask_flat)).norm()
    print(f"  Masked region norm: {masked_norm:.6f} (should be ~0)")
    
    # Test with return_grid
    patches, vc, grid = encoder(x, return_grid=True)
    print(f"\nWith return_grid:")
    print(f"  Grid: {grid.shape}")
    
    # Test PatchStrokeMapper
    mapper = PatchStrokeMapper(image_size=128, grid_size=8)
    
    # Simulate stroke coordinates
    stroke1 = torch.tensor([[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]])  # Stroke in top-left
    stroke2 = torch.tensor([[100.0, 100.0], [110.0, 110.0]])  # Stroke in bottom-right
    
    patch_indices = mapper.stroke_to_patches(stroke1)
    print(f"\nStroke-to-patch mapping:")
    print(f"  Stroke 1 coords: {stroke1.shape} -> patches: {patch_indices.tolist()}")
    
    patch_mask = mapper.stroke_to_patch_mask([stroke1, stroke2], batch_size=2)
    print(f"  Patch mask shape: {patch_mask.shape}")
    print(f"  Active patches: {patch_mask.sum().item()}")
    
    # Parameter count
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    # Test gradient flow
    loss = patches.sum() + (vc_token.sum() if vc_token is not None else 0)
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
    print(f"Gradient norm: {grad_norm:.4f}")
    
    print("\nAll tests passed!")

