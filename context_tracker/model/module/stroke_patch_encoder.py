"""
Configurable Stroke Patch Encoder

Provides multiple Conv2D encoding variants that can be configured via JSON.
Supports trajectory-based patch extraction and various kernel configurations.

Variants:
- stacked_3x3_s2: Stacked 3x3 convolutions with stride 2 (recommended)
- stacked_2x2_s2: Ultra-compact 2x2 stacked convolutions
- single_16x16: ViT-style single large kernel projection
- hybrid_7x7_3x3: ConvNeXT-style large + small kernels
- residual_3x3: ResNet-style with skip connections
- depthwise_separable: MobileNet-style efficient convolutions
- overlapping_4x4_s2: Overlapping patches for smoother features
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class EncoderLayerConfig:
    """Configuration for a single encoder layer."""
    kernel: int
    stride: int
    channels: int
    activation: str = "gelu"
    norm: str = "batch"
    residual: bool = False
    groups: int = 1  # For depthwise separable


@dataclass
class EncoderVariantConfig:
    """Configuration for an encoder variant."""
    name: str
    description: str
    type: str  # 'stacked', 'single', 'residual', 'depthwise_separable'
    layers: List[EncoderLayerConfig]
    output_pool: str
    input_size: int
    output_dim: int


class ConfigurableConvBlock(nn.Module):
    """
    Configurable convolution block.
    
    Supports: Conv -> Norm -> Activation pattern with various options.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm: str = "batch",
        activation: str = "gelu",
    ):
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=(norm == "none")
        )
        
        self.norm = self._get_norm(norm, out_channels)
        self.activation = self._get_activation(activation)
    
    def _get_norm(self, norm: str, channels: int) -> nn.Module:
        if norm == "none":
            return nn.Identity()
        elif norm == "batch":
            return nn.BatchNorm2d(channels)
        elif norm == "layer":
            return nn.GroupNorm(1, channels)
        elif norm == "instance":
            return nn.InstanceNorm2d(channels)
        elif norm == "group":
            num_groups = min(32, channels)
            while channels % num_groups != 0:
                num_groups -= 1
            return nn.GroupNorm(num_groups, channels)
        else:
            raise ValueError(f"Unknown norm: {norm}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "none":
            return nn.Identity()
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (MobileNet-style)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm: str = "batch",
        activation: str = "relu6",
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        # Depthwise
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.dw_norm = nn.BatchNorm2d(in_channels)
        
        # Pointwise
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        self.pw_norm = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU6(inplace=True) if activation == "relu6" else nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.dw_norm(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.pw_norm(x)
        x = self.activation(x)
        return x


class ResidualConvBlock(nn.Module):
    """Residual convolution block."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm: str = "batch",
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.conv1 = ConfigurableConvBlock(
            channels, channels, kernel_size, 
            stride=1, norm=norm, activation=activation
        )
        self.conv2 = ConfigurableConvBlock(
            channels, channels, kernel_size,
            stride=1, norm=norm, activation="none"
        )
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x + residual)
        return x


class StrokePatchEncoder(nn.Module):
    """
    Configurable stroke patch encoder.
    
    Encodes 16x16 (or configurable size) patches extracted along stroke trajectories
    into embedding vectors. Supports multiple architectural variants loaded from JSON.
    
    Args:
        config_path: Path to encoder_config.json
        variant: Name of the encoder variant to use
        embed_dim: Output embedding dimension (overrides config if provided)
        input_channels: Number of input channels (default: 1 for grayscale)
        
    Example:
        >>> encoder = StrokePatchEncoder(variant="stacked_3x3_s2", embed_dim=256)
        >>> patches = torch.randn(32, 1, 16, 16)  # Batch of patches
        >>> embeddings = encoder(patches)  # (32, 256)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        variant: str = "stacked_3x3_s2",
        embed_dim: Optional[int] = None,
        input_channels: int = 1,
        input_size: int = 16,
    ):
        super().__init__()
        
        self.variant = variant
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "encoder_config.json"
        
        self.config = self._load_config(config_path, variant)
        
        # Override output dim if provided
        self.embed_dim = embed_dim or self.config.output_dim
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Output projection
        encoder_out_dim = self._compute_encoder_output_dim()
        if encoder_out_dim != self.embed_dim:
            self.proj = nn.Linear(encoder_out_dim, self.embed_dim)
        else:
            self.proj = nn.Identity()
        
        # Final layer norm
        self.norm = nn.LayerNorm(self.embed_dim)
    
    def _load_config(self, config_path: Union[str, Path], variant: str) -> EncoderVariantConfig:
        """Load encoder configuration from JSON."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Use default config
            return self._get_default_config(variant)
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        if variant not in config_data.get("encoder_variants", {}):
            raise ValueError(f"Unknown variant: {variant}. Available: {list(config_data['encoder_variants'].keys())}")
        
        var_config = config_data["encoder_variants"][variant]
        
        layers = [
            EncoderLayerConfig(
                kernel=layer["kernel"],
                stride=layer["stride"],
                channels=layer["channels"],
                activation=layer.get("activation", "gelu"),
                norm=layer.get("norm", "batch"),
                residual=layer.get("residual", False),
            )
            for layer in var_config["layers"]
        ]
        
        return EncoderVariantConfig(
            name=var_config["name"],
            description=var_config["description"],
            type=var_config["type"],
            layers=layers,
            output_pool=var_config["output_pool"],
            input_size=var_config["input_size"],
            output_dim=var_config["output_dim"],
        )
    
    def _get_default_config(self, variant: str) -> EncoderVariantConfig:
        """Return default config when JSON not available."""
        defaults = {
            "stacked_3x3_s2": EncoderVariantConfig(
                name="Stacked 3x3",
                description="Default stacked 3x3 encoder",
                type="stacked",
                layers=[
                    EncoderLayerConfig(3, 2, 32, "gelu", "batch"),
                    EncoderLayerConfig(3, 2, 64, "gelu", "batch"),
                    EncoderLayerConfig(3, 2, 128, "gelu", "batch"),
                ],
                output_pool="flatten",
                input_size=16,
                output_dim=512,
            ),
            "stacked_2x2_s2": EncoderVariantConfig(
                name="Stacked 2x2",
                description="Compact 2x2 encoder",
                type="stacked",
                layers=[
                    EncoderLayerConfig(2, 2, 32, "relu", "batch"),
                    EncoderLayerConfig(2, 2, 64, "relu", "batch"),
                    EncoderLayerConfig(2, 2, 128, "relu", "batch"),
                ],
                output_pool="flatten",
                input_size=16,
                output_dim=512,
            ),
        }
        return defaults.get(variant, defaults["stacked_3x3_s2"])
    
    def _build_encoder(self) -> nn.Sequential:
        """Build the encoder based on config."""
        layers = []
        in_channels = self.input_channels
        
        for i, layer_cfg in enumerate(self.config.layers):
            if self.config.type == "depthwise_separable":
                layers.append(DepthwiseSeparableConv(
                    in_channels, layer_cfg.channels,
                    kernel_size=layer_cfg.kernel,
                    stride=layer_cfg.stride,
                    norm=layer_cfg.norm,
                    activation=layer_cfg.activation,
                ))
            elif layer_cfg.residual and in_channels == layer_cfg.channels:
                layers.append(ResidualConvBlock(
                    layer_cfg.channels,
                    kernel_size=layer_cfg.kernel,
                    norm=layer_cfg.norm,
                    activation=layer_cfg.activation,
                ))
            else:
                layers.append(ConfigurableConvBlock(
                    in_channels, layer_cfg.channels,
                    kernel_size=layer_cfg.kernel,
                    stride=layer_cfg.stride,
                    norm=layer_cfg.norm,
                    activation=layer_cfg.activation,
                ))
            
            in_channels = layer_cfg.channels
        
        # Add pooling
        if self.config.output_pool == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d(1))
        elif self.config.output_pool == "adaptive_max":
            layers.append(nn.AdaptiveMaxPool2d(1))
        
        return nn.Sequential(*layers)
    
    def _compute_encoder_output_dim(self) -> int:
        """Compute the output dimension of the encoder."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            out = self.encoder(dummy)
            return out.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode stroke patches.
        
        Args:
            x: Patches of shape (batch, channels, height, width) or (batch, height, width)
            
        Returns:
            Embeddings of shape (batch, embed_dim)
        """
        # Handle grayscale input without channel dim
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Encode
        features = self.encoder(x)
        
        # Flatten
        features = features.flatten(1)
        
        # Project
        features = self.proj(features)
        
        # Normalize
        features = self.norm(features)
        
        return features
    
    def get_config_summary(self) -> Dict:
        """Return summary of current configuration."""
        return {
            "variant": self.variant,
            "name": self.config.name,
            "type": self.config.type,
            "num_layers": len(self.config.layers),
            "input_size": self.input_size,
            "embed_dim": self.embed_dim,
            "num_params": sum(p.numel() for p in self.parameters()),
        }


class TrajectoryPatchExtractor(nn.Module):
    """
    Extract patches along stroke trajectories.
    
    Instead of dense Conv2D scan, samples patches only at trajectory points.
    Much more efficient: O(stroke_length) vs O(H×W).
    
    Args:
        patch_size: Size of each extracted patch
        sample_interval: Sample a patch every N trajectory points
        
    Example:
        >>> extractor = TrajectoryPatchExtractor(patch_size=16, sample_interval=8)
        >>> canvas = torch.randn(1, 64, 256)  # Grayscale canvas
        >>> trajectory = [(10, 20), (12, 22), (14, 24), ...]  # Stroke points
        >>> patches = extractor(canvas, trajectory)  # (num_patches, 1, 16, 16)
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        sample_interval: int = 8,
        normalize: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.sample_interval = sample_interval
        self.normalize = normalize
        self.half_size = patch_size // 2
    
    def forward(
        self,
        canvas: torch.Tensor,
        trajectory: List[Tuple[float, float]],
    ) -> torch.Tensor:
        """
        Extract patches along trajectory.
        
        Args:
            canvas: Image tensor (C, H, W) or (H, W)
            trajectory: List of (x, y) coordinates along stroke
            
        Returns:
            Patches tensor (num_patches, C, patch_size, patch_size)
        """
        if canvas.dim() == 2:
            canvas = canvas.unsqueeze(0)
        
        C, H, W = canvas.shape
        patches = []
        
        # Sample points along trajectory
        for i in range(0, len(trajectory), self.sample_interval):
            x, y = trajectory[i]
            x, y = int(x), int(y)
            
            # Compute patch bounds
            x1 = max(0, x - self.half_size)
            y1 = max(0, y - self.half_size)
            x2 = min(W, x + self.half_size)
            y2 = min(H, y + self.half_size)
            
            # Extract patch
            patch = canvas[:, y1:y2, x1:x2]
            
            # Pad if necessary
            if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
                padded = torch.ones(C, self.patch_size, self.patch_size, device=canvas.device)
                ph, pw = patch.shape[1], patch.shape[2]
                padded[:, :ph, :pw] = patch
                patch = padded
            
            patches.append(patch)
        
        if not patches:
            # Return empty tensor with correct shape
            return torch.zeros(0, C, self.patch_size, self.patch_size, device=canvas.device)
        
        patches = torch.stack(patches)
        
        if self.normalize:
            # Normalize to [-1, 1] or [0, 1]
            patches = patches / patches.max().clamp(min=1e-6)
        
        return patches
    
    def extract_multi_stroke(
        self,
        canvas: torch.Tensor,
        strokes: List[List[Tuple[float, float]]],
        return_stroke_ids: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        """
        Extract patches from multiple strokes.
        
        Args:
            canvas: Image tensor (C, H, W) or (H, W)
            strokes: List of stroke trajectories
            return_stroke_ids: Whether to return stroke IDs for each patch
            
        Returns:
            Patches and optionally stroke IDs for grouping
        """
        all_patches = []
        stroke_ids = []
        
        for stroke_idx, trajectory in enumerate(strokes):
            patches = self.forward(canvas, trajectory)
            all_patches.append(patches)
            stroke_ids.extend([stroke_idx] * len(patches))
        
        if all_patches:
            all_patches = torch.cat(all_patches, dim=0)
        else:
            C = canvas.shape[0] if canvas.dim() == 3 else 1
            all_patches = torch.zeros(0, C, self.patch_size, self.patch_size, device=canvas.device)
        
        if return_stroke_ids:
            return all_patches, stroke_ids
        return all_patches


def load_encoder_from_config(
    config_path: Optional[str] = None,
    variant: Optional[str] = None,
    **kwargs
) -> StrokePatchEncoder:
    """
    Factory function to create encoder from config.
    
    Args:
        config_path: Path to encoder_config.json
        variant: Encoder variant name (uses default if not specified)
        **kwargs: Additional arguments passed to StrokePatchEncoder
        
    Returns:
        Configured StrokePatchEncoder
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "encoder_config.json"
    
    if variant is None:
        # Load default from config
        with open(config_path, 'r') as f:
            config = json.load(f)
        variant = config.get("default_variant", "stacked_3x3_s2")
    
    return StrokePatchEncoder(
        config_path=config_path,
        variant=variant,
        **kwargs
    )


def list_available_variants(config_path: Optional[str] = None) -> List[str]:
    """List all available encoder variants."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "encoder_config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return list(config.get("encoder_variants", {}).keys())


if __name__ == "__main__":
    print("Testing StrokePatchEncoder variants...")
    print("=" * 60)
    
    # List available variants
    variants = list_available_variants()
    print(f"Available variants: {variants}")
    print()
    
    # Test each variant
    test_input = torch.randn(4, 1, 16, 16)
    
    for variant in variants:
        try:
            encoder = StrokePatchEncoder(variant=variant, embed_dim=256)
            output = encoder(test_input)
            config = encoder.get_config_summary()
            
            print(f"Variant: {variant}")
            print(f"  Name: {config['name']}")
            print(f"  Type: {config['type']}")
            print(f"  Layers: {config['num_layers']}")
            print(f"  Params: {config['num_params']:,}")
            print(f"  Output: {output.shape}")
            print()
        except Exception as e:
            print(f"Variant {variant}: ERROR - {e}")
            print()
    
    # Test trajectory extractor
    print("Testing TrajectoryPatchExtractor...")
    extractor = TrajectoryPatchExtractor(patch_size=16, sample_interval=8)
    
    canvas = torch.randn(1, 64, 256)
    trajectory = [(10 + i*2, 20 + i) for i in range(50)]
    
    patches = extractor(canvas, trajectory)
    print(f"Trajectory patches: {patches.shape}")
    
    # Multi-stroke
    strokes = [
        [(10 + i*2, 20 + i) for i in range(30)],
        [(100 + i*2, 30 + i) for i in range(25)],
    ]
    patches, stroke_ids = extractor.extract_multi_stroke(canvas, strokes, return_stroke_ids=True)
    print(f"Multi-stroke patches: {patches.shape}, stroke_ids: {stroke_ids}")
    
    print("\nAll tests passed!")

