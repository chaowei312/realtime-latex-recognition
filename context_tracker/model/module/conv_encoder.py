"""
Convolutional Encoder

CNN-based feature encoder for processing:
- Stroke images
- Line images
- Raw stroke sequences (1D convolutions)

Provides various convolutional building blocks:
- ConvBlock: Basic convolution + norm + activation
- ResidualConvBlock: Residual connection wrapper
- ConvEncoder: Complete encoder network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union


class ConvBlock(nn.Module):
    """
    Basic Convolution Block
    
    Conv -> Norm -> Activation -> Dropout
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding (default: auto for same size)
        dilation: Convolution dilation
        groups: Convolution groups
        bias: Whether to use bias (usually False with BatchNorm)
        norm: Normalization type ('batch', 'layer', 'instance', 'group', None)
        activation: Activation function ('relu', 'gelu', 'silu', 'none')
        dropout: Dropout probability
        conv_type: '1d' or '2d'
        
    Example:
        >>> block = ConvBlock(64, 128, kernel_size=3, stride=2)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = block(x)  # (2, 128, 16, 16)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        norm: str = 'batch',
        activation: str = 'gelu',
        dropout: float = 0.0,
        conv_type: str = '2d'
    ):
        super().__init__()
        
        # Auto-compute padding for 'same' output size (when stride=1)
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        # Bias default: False if using normalization
        if bias is None:
            bias = norm is None or norm == 'none'
        
        # Select conv type
        Conv = nn.Conv2d if conv_type == '2d' else nn.Conv1d
        
        self.conv = Conv(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Normalization
        self.norm = self._get_norm(norm, out_channels, conv_type)
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _get_norm(self, norm: str, channels: int, conv_type: str) -> nn.Module:
        if norm is None or norm == 'none':
            return nn.Identity()
        elif norm == 'batch':
            return nn.BatchNorm2d(channels) if conv_type == '2d' else nn.BatchNorm1d(channels)
        elif norm == 'layer':
            return nn.GroupNorm(1, channels)  # LayerNorm equivalent for conv
        elif norm == 'instance':
            return nn.InstanceNorm2d(channels) if conv_type == '2d' else nn.InstanceNorm1d(channels)
        elif norm == 'group':
            num_groups = min(32, channels)
            while channels % num_groups != 0:
                num_groups -= 1
            return nn.GroupNorm(num_groups, channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu' or activation == 'swish':
            return nn.SiLU(inplace=True)
        elif activation == 'none' or activation is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualConvBlock(nn.Module):
    """
    Residual Convolution Block
    
    Applies convolutions with a skip connection:
    output = activation(x + conv_block(x))
    
    Args:
        channels: Number of channels (in == out for residual)
        kernel_size: Convolution kernel size
        expansion: Channel expansion ratio in bottleneck
        norm: Normalization type
        activation: Activation function
        dropout: Dropout probability
        conv_type: '1d' or '2d'
        
    Example:
        >>> block = ResidualConvBlock(128, kernel_size=3)
        >>> x = torch.randn(2, 128, 32, 32)
        >>> out = block(x)  # (2, 128, 32, 32)
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        expansion: float = 1.0,
        norm: str = 'batch',
        activation: str = 'gelu',
        dropout: float = 0.0,
        conv_type: str = '2d'
    ):
        super().__init__()
        
        hidden_channels = int(channels * expansion)
        
        self.conv1 = ConvBlock(
            channels, hidden_channels,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
            conv_type=conv_type
        )
        
        self.conv2 = ConvBlock(
            hidden_channels, channels,
            kernel_size=kernel_size,
            norm=norm,
            activation='none',  # No activation before residual add
            dropout=dropout,
            conv_type=conv_type
        )
        
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x + residual)
        return x


class DownsampleBlock(nn.Module):
    """
    Downsampling block that reduces spatial dimensions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        factor: Downsampling factor (2 = half size)
        mode: 'conv' (strided conv) or 'pool' (pooling + conv)
        conv_type: '1d' or '2d'
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        mode: str = 'conv',
        conv_type: str = '2d'
    ):
        super().__init__()
        
        if mode == 'conv':
            self.down = ConvBlock(
                in_channels, out_channels,
                kernel_size=factor,
                stride=factor,
                padding=0,
                conv_type=conv_type
            )
        else:
            Pool = nn.MaxPool2d if conv_type == '2d' else nn.MaxPool1d
            self.down = nn.Sequential(
                Pool(kernel_size=factor, stride=factor),
                ConvBlock(
                    in_channels, out_channels,
                    kernel_size=1,
                    conv_type=conv_type
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder Network
    
    Hierarchical CNN encoder that progressively:
    - Increases channels
    - Decreases spatial resolution
    - Builds rich feature representations
    
    Args:
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        base_channels: Base channel count (doubled at each stage)
        num_stages: Number of downsampling stages
        blocks_per_stage: Number of residual blocks per stage
        output_channels: Output channels (if None, uses final stage channels)
        kernel_size: Convolution kernel size
        norm: Normalization type
        activation: Activation function
        dropout: Dropout probability
        conv_type: '1d' or '2d'
        pool_type: Final pooling ('adaptive_avg', 'adaptive_max', 'flatten', 'none')
        
    Example:
        >>> encoder = ConvEncoder(in_channels=1, base_channels=64, num_stages=4)
        >>> x = torch.randn(2, 1, 64, 256)  # Line image
        >>> features = encoder(x)  # (2, 512) or (2, 512, H', W')
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_stages: int = 4,
        blocks_per_stage: int = 2,
        output_channels: Optional[int] = None,
        kernel_size: int = 3,
        norm: str = 'batch',
        activation: str = 'gelu',
        dropout: float = 0.0,
        conv_type: str = '2d',
        pool_type: str = 'adaptive_avg'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.conv_type = conv_type
        self.pool_type = pool_type
        
        # Stem (initial feature extraction)
        self.stem = ConvBlock(
            in_channels, base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            norm=norm,
            activation=activation,
            conv_type=conv_type
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for stage_idx in range(num_stages):
            stage_channels = base_channels * (2 ** stage_idx)
            
            # First block may have channel change
            blocks = []
            
            # Downsample at start of each stage (except first)
            if stage_idx > 0:
                blocks.append(DownsampleBlock(
                    current_channels, stage_channels,
                    factor=2,
                    conv_type=conv_type
                ))
                current_channels = stage_channels
            
            # Residual blocks
            for _ in range(blocks_per_stage):
                # Ensure channels match for residual
                if current_channels != stage_channels:
                    blocks.append(ConvBlock(
                        current_channels, stage_channels,
                        kernel_size=1,
                        conv_type=conv_type
                    ))
                    current_channels = stage_channels
                
                blocks.append(ResidualConvBlock(
                    stage_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    conv_type=conv_type
                ))
            
            self.stages.append(nn.Sequential(*blocks))
            current_channels = stage_channels
        
        self.final_channels = current_channels
        
        # Output projection
        if output_channels is not None and output_channels != current_channels:
            self.output_proj = ConvBlock(
                current_channels, output_channels,
                kernel_size=1,
                norm=norm,
                activation=activation,
                conv_type=conv_type
            )
            self.final_channels = output_channels
        else:
            self.output_proj = nn.Identity()
        
        # Pooling
        self.pool = self._get_pool(pool_type, conv_type)
    
    def _get_pool(self, pool_type: str, conv_type: str) -> nn.Module:
        if pool_type == 'adaptive_avg':
            return nn.AdaptiveAvgPool2d(1) if conv_type == '2d' else nn.AdaptiveAvgPool1d(1)
        elif pool_type == 'adaptive_max':
            return nn.AdaptiveMaxPool2d(1) if conv_type == '2d' else nn.AdaptiveMaxPool1d(1)
        elif pool_type == 'flatten':
            return nn.Flatten(1)
        elif pool_type == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, channels, height, width) for 2D
               or (batch, channels, length) for 1D
            return_features: If True, also return intermediate features
            
        Returns:
            If return_features=False: Final features
            If return_features=True: (final_features, list of stage features)
        """
        features = []
        
        # Stem
        x = self.stem(x)
        features.append(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Pooling
        if self.pool_type != 'none':
            x = self.pool(x)
            if self.pool_type in ['adaptive_avg', 'adaptive_max']:
                x = x.flatten(1)  # Remove spatial dims
        
        if return_features:
            return x, features
        return x
    
    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.final_channels


class Conv1DEncoder(ConvEncoder):
    """
    1D Convolutional Encoder for sequence data.
    
    Useful for encoding stroke sequences (x, y, t) directly.
    
    Args:
        in_channels: Number of input features per timestep
        **kwargs: Other arguments passed to ConvEncoder
        
    Example:
        >>> encoder = Conv1DEncoder(in_channels=3)  # (x, y, t)
        >>> x = torch.randn(2, 3, 100)  # (batch, features, seq_len)
        >>> out = encoder(x)  # (2, 512)
    """
    
    def __init__(self, in_channels: int = 3, **kwargs):
        kwargs['conv_type'] = '1d'
        super().__init__(in_channels=in_channels, **kwargs)


class Conv2DEncoder(ConvEncoder):
    """
    2D Convolutional Encoder for image data.
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        **kwargs: Other arguments passed to ConvEncoder
        
    Example:
        >>> encoder = Conv2DEncoder(in_channels=1)
        >>> x = torch.randn(2, 1, 64, 256)  # Line image
        >>> out = encoder(x)  # (2, 512)
    """
    
    def __init__(self, in_channels: int = 1, **kwargs):
        kwargs['conv_type'] = '2d'
        super().__init__(in_channels=in_channels, **kwargs)


if __name__ == "__main__":
    # Test Conv modules
    print("Testing Convolutional Encoder modules...")
    
    # Test 2D encoder
    encoder_2d = Conv2DEncoder(
        in_channels=1,
        base_channels=32,
        num_stages=4,
        blocks_per_stage=2
    )
    
    x_2d = torch.randn(2, 1, 64, 256)  # Line image
    out_2d = encoder_2d(x_2d)
    print(f"Conv2DEncoder: {x_2d.shape} -> {out_2d.shape}")
    
    # With intermediate features
    out_2d, features = encoder_2d(x_2d, return_features=True)
    print(f"Intermediate feature shapes: {[f.shape for f in features]}")
    
    # Test 1D encoder
    encoder_1d = Conv1DEncoder(
        in_channels=3,  # (x, y, t)
        base_channels=32,
        num_stages=3,
        blocks_per_stage=2
    )
    
    x_1d = torch.randn(2, 3, 100)  # Stroke sequence
    out_1d = encoder_1d(x_1d)
    print(f"Conv1DEncoder: {x_1d.shape} -> {out_1d.shape}")
    
    # Parameter counts
    params_2d = sum(p.numel() for p in encoder_2d.parameters())
    params_1d = sum(p.numel() for p in encoder_1d.parameters())
    print(f"Conv2DEncoder parameters: {params_2d:,}")
    print(f"Conv1DEncoder parameters: {params_1d:,}")
    
    print("All tests passed!")

