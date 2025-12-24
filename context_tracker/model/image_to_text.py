"""
Image-to-Text Recognition Models

This module provides two architectures:

1. ImageToTextModel (Encoder-Decoder with Cross-Attention):
   Image → ConvEncoder → [CLS] → Cross-Attention Decoder → Text
   
2. DecoderOnlyImageToText (Decoder-Only with Visual Prefix):
   Image → ConvEncoder → Visual Tokens → [Visual | Text] Self-Attention → Text
   
The decoder-only approach is simpler and follows modern VLM architectures
like LLaVA, Flamingo, and Donut.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from .module import (
    Conv2DEncoder,
    Conv1DEncoder,
    FlashAttention,
    FlashAttentionDecoder,
    FlashAttentionDecoderStack,
    SwiGLUFFN,
    RoPE2D,
)


@dataclass
class ImageToTextConfig:
    """Configuration for Image-to-Text model."""
    
    # Image encoder
    image_channels: int = 1  # Grayscale
    encoder_base_channels: int = 32
    encoder_num_stages: int = 4
    encoder_blocks_per_stage: int = 2
    
    # Model dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    head_dim: int = 32  # hidden_dim // num_heads
    
    # Decoder
    num_decoder_layers: int = 4
    ffn_dim: int = 1024  # 4 * hidden_dim
    dropout: float = 0.1
    
    # Vocabulary
    vocab_size: int = 128  # Extended ASCII
    max_seq_len: int = 64
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Position encoding
    use_2d_rope: bool = False
    max_height: int = 64
    max_width: int = 256


class PatchEmbedding(nn.Module):
    """
    Convert 2D feature maps to patch sequences.
    
    Takes ConvEncoder output (B, C, H', W') and produces
    patch embeddings (B, num_patches, hidden_dim).
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: int = 1,  # 1x1 means each spatial position is a patch
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.projection = nn.Linear(in_channels * patch_size * patch_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Feature map (batch, channels, height, width)
            
        Returns:
            patches: (batch, num_patches, hidden_dim)
            spatial_shape: (height, width) of patch grid
        """
        B, C, H, W = x.shape
        
        if self.patch_size > 1:
            # Reshape into patches
            x = x.unfold(2, self.patch_size, self.patch_size)
            x = x.unfold(3, self.patch_size, self.patch_size)
            H, W = x.shape[2], x.shape[3]
            x = x.contiguous().view(B, C, H, W, -1)
            x = x.permute(0, 2, 3, 1, 4).reshape(B, H * W, -1)
        else:
            # Each spatial position is a patch
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        patches = self.projection(x)
        patches = self.norm(patches)
        
        return patches, (H, W)


class CLSAggregator(nn.Module):
    """
    [CLS] Token Aggregation Module
    
    Uses attention to aggregate patch embeddings into a single [CLS] feature.
    The [CLS] token attends to all patches, producing a global representation.
    
    Based on experiments.md design:
    - [CLS] attends to patches only (isolated from text context)
    - Produces local symbol recognition features
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_swiglu: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Position embedding for patches (learned)
        self.max_patches = 1024
        self.patch_pos_embed = nn.Parameter(torch.randn(1, self.max_patches, hidden_dim) * 0.02)
        
        # Self-attention layers for patches + CLS
        self.layers = nn.ModuleList([
            CLSAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_swiglu=use_swiglu,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        patches: torch.Tensor,
        patch_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patches: Patch embeddings (batch, num_patches, hidden_dim)
            patch_positions: Optional 2D positions (batch, num_patches, 2)
            
        Returns:
            cls_feature: Aggregated feature (batch, hidden_dim)
        """
        B, N, D = patches.shape
        
        # Add position embeddings to patches
        if patch_positions is None:
            patches = patches + self.patch_pos_embed[:, :N, :]
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 1 + N, D)
        
        # Process through attention layers
        for layer in self.layers:
            x = layer(x)
        
        # Extract [CLS] output
        cls_feature = self.final_norm(x[:, 0, :])
        
        return cls_feature


class CLSAttentionLayer(nn.Module):
    """Single attention layer for CLS aggregation."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_swiglu: bool = True,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = FlashAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=False,  # CLS can attend to all patches
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        if use_swiglu:
            self.ffn = SwiGLUFFN(dim=hidden_dim, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TextEmbedding(nn.Module):
    """Text token embedding with positional encoding."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
            
        Returns:
            embeddings: (batch, seq_len, hidden_dim)
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class ImageToTextModel(nn.Module):
    """
    Complete Image-to-Text Recognition Model
    
    Architecture:
    1. ConvEncoder: Image → Feature maps
    2. PatchEmbedding: Feature maps → Patch sequence
    3. CLSAggregator: Patches → [CLS] feature
    4. Decoder: [CLS] + text tokens → next token prediction
    
    For classification (single letter), use forward_classify().
    For sequence generation (word/sentence), use forward() or generate().
    """
    
    def __init__(self, config: ImageToTextConfig):
        super().__init__()
        
        self.config = config
        
        # Image encoder
        self.image_encoder = Conv2DEncoder(
            in_channels=config.image_channels,
            base_channels=config.encoder_base_channels,
            num_stages=config.encoder_num_stages,
            blocks_per_stage=config.encoder_blocks_per_stage,
            pool_type='none',  # Keep spatial dimensions
        )
        
        # Get encoder output channels
        encoder_out_channels = config.encoder_base_channels * (2 ** (config.encoder_num_stages - 1))
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=encoder_out_channels,
            hidden_dim=config.hidden_dim,
        )
        
        # CLS aggregator
        self.cls_aggregator = CLSAggregator(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=2,
            dropout=config.dropout,
        )
        
        # Text embedding
        self.text_embed = TextEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pad_token_id=config.pad_token_id,
        )
        
        # Decoder (autoregressive with causal mask)
        self.decoder = FlashAttentionDecoderStack(
            num_layers=config.num_decoder_layers,
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            use_cross_attention=True,  # Cross-attend to [CLS]
            use_swiglu=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Classification head (for single-letter classification)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vocab_size),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to [CLS] features.
        
        Args:
            images: (batch, channels, height, width)
            
        Returns:
            cls_features: (batch, hidden_dim)
        """
        # CNN encoding
        features = self.image_encoder(images)  # (B, C, H', W')
        
        # Patch embedding
        patches, spatial_shape = self.patch_embed(features)  # (B, N, D)
        
        # CLS aggregation
        cls_features = self.cls_aggregator(patches)  # (B, D)
        
        return cls_features
    
    def forward(
        self,
        images: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            images: Input images (batch, channels, height, width)
            target_ids: Target token IDs (batch, seq_len), including BOS
            target_mask: Optional padding mask (batch, seq_len)
            
        Returns:
            Dict with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar if targets provided
        """
        B = images.shape[0]
        
        # Encode images to [CLS]
        cls_features = self.encode_image(images)  # (B, D)
        
        # Expand [CLS] for cross-attention: (B, 1, D)
        encoder_out = cls_features.unsqueeze(1)
        
        # Embed target tokens (shifted right for teacher forcing)
        text_embeds = self.text_embed(target_ids)  # (B, L, D)
        
        # Decode with cross-attention to [CLS]
        decoder_out = self.decoder(
            text_embeds,
            encoder_out=encoder_out,
            padding_mask=target_mask,
        )  # (B, L, D)
        
        # Project to vocabulary
        logits = self.output_proj(decoder_out)  # (B, L, V)
        
        return {"logits": logits}
    
    def forward_classify(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-letter classification.
        
        Args:
            images: Input images (batch, channels, height, width)
            labels: Optional class labels (batch,)
            
        Returns:
            Dict with:
                - logits: (batch, vocab_size)
                - loss: scalar if labels provided
        """
        # Encode images to [CLS]
        cls_features = self.encode_image(images)  # (B, D)
        
        # Classify
        logits = self.classifier(cls_features)  # (B, V)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text from images autoregressively.
        
        Args:
            images: Input images (batch, channels, height, width)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling (None = disabled)
            top_p: Nucleus sampling (None = disabled)
            
        Returns:
            generated_ids: (batch, seq_len) generated token IDs
        """
        self.eval()
        B = images.shape[0]
        device = images.device
        
        # Encode images
        cls_features = self.encode_image(images)
        encoder_out = cls_features.unsqueeze(1)  # (B, 1, D)
        
        # Start with BOS token
        generated = torch.full(
            (B, 1), 
            self.config.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Generate tokens autoregressively
        for _ in range(max_length - 1):
            # Embed current sequence
            text_embeds = self.text_embed(generated)
            
            # Decode
            decoder_out = self.decoder(text_embeds, encoder_out=encoder_out)
            
            # Get logits for last position
            logits = self.output_proj(decoder_out[:, -1, :])  # (B, V)
            
            # Greedy decoding if temperature <= 0
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply nucleus (top-p) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    for b in range(B):
                        logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated


class StrokeClassifier(nn.Module):
    """
    Simplified model for single-stroke letter classification.
    
    Uses only ConvEncoder + [CLS] + classifier head.
    Lighter weight than full ImageToTextModel.
    """
    
    def __init__(
        self,
        num_classes: int = 55,
        hidden_dim: int = 256,
        encoder_base_channels: int = 32,
        encoder_num_stages: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Image encoder
        self.encoder = Conv2DEncoder(
            in_channels=1,
            base_channels=encoder_base_channels,
            num_stages=encoder_num_stages,
            blocks_per_stage=2,
            pool_type='adaptive_avg',  # Global average pool
        )
        
        encoder_out_dim = encoder_base_channels * (2 ** (encoder_num_stages - 1))
        
        # Project to hidden dim
        self.projection = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (batch, 1, height, width)
            labels: (batch,) class indices
            
        Returns:
            Dict with logits and optionally loss
        """
        # Encode
        features = self.encoder(images)  # (B, encoder_out_dim)
        features = self.projection(features)  # (B, hidden_dim)
        
        # Classify
        logits = self.classifier(features)  # (B, num_classes)
        
        result = {"logits": logits, "features": features}
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss
        
        return result
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        self.eval()
        logits = self.forward(images)["logits"]
        return logits.argmax(dim=-1)


def create_model_from_config(config_dict: Dict[str, Any]) -> ImageToTextModel:
    """Create model from config dictionary."""
    config = ImageToTextConfig(**config_dict)
    return ImageToTextModel(config)


def create_stroke_classifier(
    num_classes: int = 55,
    hidden_dim: int = 256,
    **kwargs
) -> StrokeClassifier:
    """Create a simple stroke classifier."""
    return StrokeClassifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        **kwargs
    )


# =============================================================================
# DECODER-ONLY IMAGE-TO-TEXT MODEL
# =============================================================================

@dataclass
class DecoderOnlyConfig:
    """Configuration for Decoder-Only Image-to-Text model."""
    
    # Image encoder
    image_channels: int = 1
    encoder_base_channels: int = 32
    encoder_num_stages: int = 4
    encoder_blocks_per_stage: int = 2
    
    # Model dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    
    # Vocabulary
    vocab_size: int = 128
    max_seq_len: int = 128
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Visual tokens
    max_visual_tokens: int = 256


class VisualTokenizer(nn.Module):
    """
    Convert CNN feature maps to visual tokens.
    
    Image → ConvEncoder → Feature Map → Flatten → Project → Visual Tokens
    """
    
    def __init__(
        self,
        image_channels: int = 1,
        encoder_base_channels: int = 32,
        encoder_num_stages: int = 4,
        encoder_blocks_per_stage: int = 2,
        hidden_dim: int = 256,
        max_visual_tokens: int = 256,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_visual_tokens = max_visual_tokens
        
        # CNN encoder
        self.encoder = Conv2DEncoder(
            in_channels=image_channels,
            base_channels=encoder_base_channels,
            num_stages=encoder_num_stages,
            blocks_per_stage=encoder_blocks_per_stage,
            pool_type='none',
        )
        
        # Get output channels
        encoder_out_channels = encoder_base_channels * (2 ** (encoder_num_stages - 1))
        
        # Project to hidden dim
        self.projection = nn.Linear(encoder_out_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Learnable position embeddings for visual tokens
        self.visual_pos_embed = nn.Parameter(
            torch.randn(1, max_visual_tokens, hidden_dim) * 0.02
        )
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            images: (batch, channels, height, width)
            
        Returns:
            visual_tokens: (batch, num_visual_tokens, hidden_dim)
            num_visual_tokens: Number of visual tokens
        """
        # CNN encoding: (B, C, H, W) -> (B, C', H', W')
        features = self.encoder(images)
        
        B, C, H, W = features.shape
        num_tokens = H * W
        
        # Flatten spatial dimensions: (B, C', H', W') -> (B, H'*W', C')
        features = features.permute(0, 2, 3, 1).reshape(B, num_tokens, C)
        
        # Project to hidden dim
        visual_tokens = self.projection(features)
        visual_tokens = self.norm(visual_tokens)
        
        # Add position embeddings
        visual_tokens = visual_tokens + self.visual_pos_embed[:, :num_tokens, :]
        
        return visual_tokens, num_tokens


class DecoderOnlyBlock(nn.Module):
    """
    Single transformer block for decoder-only architecture.
    
    Uses self-attention only (no cross-attention).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_swiglu: bool = True,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = FlashAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=False,  # We'll handle masking manually
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        if use_swiglu:
            self.ffn = SwiGLUFFN(dim=hidden_dim, hidden_dim=ffn_dim, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout),
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with custom mask
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask=attention_mask))
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderOnlyImageToText(nn.Module):
    """
    Decoder-Only Image-to-Text Model
    
    Architecture:
        Image → VisualTokenizer → [v₁, v₂, ..., vₙ]
                                        ↓
        [v₁...vₙ | BOS, t₁, t₂, ...] → Self-Attention Decoder → Next Token
        
    Attention Mask:
        - Visual tokens: bidirectional among themselves
        - Text tokens: causal (can see visual + previous text)
        
    This follows modern VLM architectures (LLaVA, Flamingo, Donut).
    """
    
    def __init__(self, config: DecoderOnlyConfig):
        super().__init__()
        
        self.config = config
        
        # Visual tokenizer
        self.visual_tokenizer = VisualTokenizer(
            image_channels=config.image_channels,
            encoder_base_channels=config.encoder_base_channels,
            encoder_num_stages=config.encoder_num_stages,
            encoder_blocks_per_stage=config.encoder_blocks_per_stage,
            hidden_dim=config.hidden_dim,
            max_visual_tokens=config.max_visual_tokens,
        )
        
        # Text embedding
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.text_pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.text_embed.weight, std=0.02)
        nn.init.normal_(self.text_pos_embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def _create_attention_mask(
        self,
        num_visual: int,
        num_text: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create attention mask for visual-text sequence.
        
        Visual tokens: bidirectional (can attend to each other)
        Text tokens: causal + can attend to all visual tokens
        
        Returns:
            mask: (1, 1, seq_len, seq_len) where 0 = attend, -inf = block
        """
        total_len = num_visual + num_text
        
        # Start with all blocked
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        
        # Visual tokens attend to each other (bidirectional)
        mask[:num_visual, :num_visual] = 0
        
        # Text tokens attend to all visual tokens
        mask[num_visual:, :num_visual] = 0
        
        # Text tokens attend to previous text (causal)
        for i in range(num_text):
            text_idx = num_visual + i
            mask[text_idx, num_visual:text_idx + 1] = 0
        
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        return mask
    
    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            images: (batch, channels, height, width)
            text_ids: (batch, seq_len) - includes BOS, targets shifted
            text_mask: Optional padding mask
            
        Returns:
            Dict with logits and optionally loss
        """
        B = images.shape[0]
        device = images.device
        
        # Get visual tokens
        visual_tokens, num_visual = self.visual_tokenizer(images)
        
        # Embed text tokens
        text_len = text_ids.shape[1]
        text_positions = torch.arange(text_len, device=device).unsqueeze(0).expand(B, -1)
        text_tokens = self.text_embed(text_ids) + self.text_pos_embed(text_positions)
        
        # Concatenate: [visual | text]
        sequence = torch.cat([visual_tokens, text_tokens], dim=1)
        
        # Create attention mask
        attn_mask = self._create_attention_mask(num_visual, text_len, device)
        
        # Apply decoder layers
        x = sequence
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)
        
        x = self.final_norm(x)
        
        # Get logits for text positions only
        text_features = x[:, num_visual:, :]  # (B, text_len, D)
        logits = self.output_proj(text_features)  # (B, text_len, V)
        
        return {"logits": logits, "num_visual_tokens": num_visual}
    
    def compute_loss(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.
        
        Args:
            images: Input images
            text_ids: Input text (with BOS, without last token)
            target_ids: Target text (without BOS, with EOS)
            ignore_index: Index to ignore in loss computation
            
        Returns:
            Dict with loss and logits
        """
        output = self.forward(images, text_ids)
        logits = output["logits"]
        
        # Flatten for cross entropy
        B, L, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1),
            ignore_index=ignore_index,
        )
        
        output["loss"] = loss
        return output
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            images: Input images (batch, channels, height, width)
            max_length: Maximum text length
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            generated_ids: (batch, seq_len)
        """
        self.eval()
        B = images.shape[0]
        device = images.device
        
        # Get visual tokens (computed once)
        visual_tokens, num_visual = self.visual_tokenizer(images)
        
        # Start with BOS
        generated = torch.full(
            (B, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        
        for _ in range(max_length - 1):
            text_len = generated.shape[1]
            
            # Embed current text
            text_positions = torch.arange(text_len, device=device).unsqueeze(0).expand(B, -1)
            text_tokens = self.text_embed(generated) + self.text_pos_embed(text_positions)
            
            # Concatenate with visual
            sequence = torch.cat([visual_tokens, text_tokens], dim=1)
            
            # Create mask
            attn_mask = self._create_attention_mask(num_visual, text_len, device)
            
            # Forward through decoder
            x = sequence
            for layer in self.layers:
                x = layer(x, attention_mask=attn_mask)
            
            x = self.final_norm(x)
            
            # Get logits for last text position
            last_features = x[:, -1, :]
            logits = self.output_proj(last_features)  # (B, V)
            
            # Greedy decoding if temperature <= 0
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated


def create_decoder_only_model(
    vocab_size: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 6,
    **kwargs
) -> DecoderOnlyImageToText:
    """Create a decoder-only image-to-text model."""
    config = DecoderOnlyConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
    )
    return DecoderOnlyImageToText(config)


if __name__ == "__main__":
    print("Testing Image-to-Text Models...")
    
    # Test StrokeClassifier
    print("\n1. Testing StrokeClassifier...")
    classifier = StrokeClassifier(num_classes=55, hidden_dim=256)
    
    images = torch.randn(4, 1, 64, 64)  # 4 grayscale stroke images
    labels = torch.randint(0, 55, (4,))
    
    output = classifier(images, labels)
    print(f"   Input: {images.shape}")
    print(f"   Logits: {output['logits'].shape}")
    print(f"   Loss: {output['loss'].item():.4f}")
    
    preds = classifier.predict(images)
    print(f"   Predictions: {preds}")
    
    params = sum(p.numel() for p in classifier.parameters())
    print(f"   Parameters: {params:,}")
    
    # Test ImageToTextModel (Encoder-Decoder)
    print("\n2. Testing ImageToTextModel (Encoder-Decoder)...")
    config = ImageToTextConfig(
        vocab_size=64,
        hidden_dim=256,
        num_decoder_layers=2,
    )
    model = ImageToTextModel(config)
    
    images = torch.randn(2, 1, 64, 128)  # Line images
    target_ids = torch.randint(3, 64, (2, 10))  # Target sequences
    target_ids[:, 0] = config.bos_token_id  # Start with BOS
    
    output = model(images, target_ids)
    print(f"   Image shape: {images.shape}")
    print(f"   Target shape: {target_ids.shape}")
    print(f"   Logits shape: {output['logits'].shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    
    # Test DecoderOnlyImageToText
    print("\n3. Testing DecoderOnlyImageToText...")
    decoder_config = DecoderOnlyConfig(
        vocab_size=64,
        hidden_dim=256,
        num_layers=4,
    )
    decoder_model = DecoderOnlyImageToText(decoder_config)
    
    images = torch.randn(2, 1, 64, 256)  # Line images (wider)
    text_ids = torch.randint(3, 64, (2, 15))  # Input text
    text_ids[:, 0] = decoder_config.bos_token_id
    target_ids = torch.randint(3, 64, (2, 15))  # Target text
    
    output = decoder_model(images, text_ids)
    print(f"   Image shape: {images.shape}")
    print(f"   Text shape: {text_ids.shape}")
    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Num visual tokens: {output['num_visual_tokens']}")
    
    # Test loss computation
    print("\n4. Testing loss computation...")
    loss_output = decoder_model.compute_loss(images, text_ids, target_ids)
    print(f"   Loss: {loss_output['loss'].item():.4f}")
    
    # Test generation
    print("\n5. Testing generation...")
    generated = decoder_model.generate(images, max_length=20, temperature=0.8)
    print(f"   Generated shape: {generated.shape}")
    print(f"   Generated tokens: {generated[0].tolist()}")
    
    params = sum(p.numel() for p in decoder_model.parameters())
    print(f"\n   DecoderOnly parameters: {params:,}")
    
    print("\nAll tests passed!")

