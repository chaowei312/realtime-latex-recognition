"""
Dataset for Context Tracker Training

Provides EditDataset that wraps MathWritingAtomic with:
- Compositional augmentation (context + edit chunk)
- Stroke-level labels for SMM training
- Action labels for TPM training (using spatial relation inference)
- Image rendering with optional artifacts
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

import torch
from torch.utils.data import Dataset, DataLoader

from ..data.mathwriting_atomic import (
    MathWritingAtomic,
    MathWritingChunk,
    ChunkRenderer,
    CompositionalAugmentor,
    SymbolMaskingAugmentor,
    Stroke,
    StrokeBBox,
)
from ..model.tree_to_latex import (
    RelationType,
    SpatialRelationInferrer,
    tokens_and_actions_to_latex,
)
from ..model.module.relationship_tensor import LatexRelationshipParser


@dataclass
class EditSample:
    """A single training sample for edit operation."""
    context: str                    # Text context with [MASK]
    context_ids: torch.Tensor       # Tokenized context
    image: torch.Tensor             # [1, H, W] rendered chunk
    target_token: int               # Target token ID for this step
    target_action: int              # Target action index (parent_idx * num_relations + relation)
    stroke_labels: torch.Tensor     # [num_strokes] binary labels
    active_strokes: torch.Tensor    # [num_strokes] which strokes are still active
    stroke_indices: List[Tuple[int, int]]  # Patch ranges per stroke


class SimpleTokenizer:
    """
    Simple tokenizer for LaTeX tokens.
    
    For prototyping - maps symbols to IDs.
    In production, use a proper BPE/SentencePiece tokenizer.
    """
    
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.vc_token = '<VC>'
        self.int_token = '<INT>'
        self.mask_token = '<MASK>'
        self.unk_token = '<UNK>'
        
        # Build vocabulary
        self.special_tokens = [
            self.pad_token, self.bos_token, self.eos_token,
            self.vc_token, self.int_token, self.mask_token, self.unk_token
        ]
        
        # Common LaTeX symbols
        self.symbols = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.symbols += list('0123456789')
        self.symbols += list('+-*/=()[]{}.,;:!?')
        self.symbols += ['\\frac', '\\sqrt', '\\sum', '\\int', '\\prod', '\\lim']
        self.symbols += ['\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon']
        self.symbols += ['\\theta', '\\lambda', '\\mu', '\\pi', '\\sigma', '\\omega']
        self.symbols += ['\\sin', '\\cos', '\\tan', '\\log', '\\exp', '\\ln']
        self.symbols += ['\\infty', '\\partial', '\\nabla', '\\pm', '\\cdot']
        self.symbols += ['_', '^', ' ', '\\', '{', '}']
        
        # Token to ID
        self.token2id = {tok: i for i, tok in enumerate(self.special_tokens)}
        for sym in self.symbols:
            if sym not in self.token2id:
                self.token2id[sym] = len(self.token2id)
        
        # ID to token
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        # Special token IDs
        self.pad_id = self.token2id[self.pad_token]
        self.bos_id = self.token2id[self.bos_token]
        self.eos_id = self.token2id[self.eos_token]
        self.unk_id = self.token2id[self.unk_token]
        self.mask_id = self.token2id[self.mask_token]
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        ids = [self.token2id.get(t, self.unk_id) for t in tokens]
        
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.id2token.get(i, self.unk_token) for i in ids]
        # Filter special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        return ''.join(tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        tokens = []
        i = 0
        while i < len(text):
            # Check for LaTeX commands
            if text[i] == '\\':
                # Find end of command
                j = i + 1
                while j < len(text) and text[j].isalpha():
                    j += 1
                tokens.append(text[i:j])
                i = j
            else:
                tokens.append(text[i])
                i += 1
        return tokens
    
    def __len__(self) -> int:
        return len(self.token2id)


class EditDataset(Dataset):
    """
    Dataset for training Context Tracker with compositional augmentation.
    
    Each sample contains:
    - Context (text) with [MASK] placeholder
    - Edit chunk rendered as image
    - Per-step targets (token, action, stroke_labels)
    
    For Option B training, we provide the full sequence and let the
    training loop handle incremental stroke removal.
    """
    
    def __init__(
        self,
        mathwriting: MathWritingAtomic,
        tokenizer: SimpleTokenizer,
        image_size: int = 128,
        max_context_len: int = 64,
        min_chunks: int = 2,
        max_chunks: int = 4,
        artifact_ratio: float = 0.3,
        num_relations: int = 6
    ):
        """
        Args:
            mathwriting: MathWriting dataset
            tokenizer: Tokenizer for text
            image_size: Output image size
            max_context_len: Maximum context length
            min_chunks: Minimum chunks in context
            max_chunks: Maximum chunks in context
            artifact_ratio: Probability of adding stroke artifacts
            num_relations: Number of relation types for TPM
        """
        self.mathwriting = mathwriting
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_context_len = max_context_len
        self.num_relations = num_relations
        
        # Renderer
        self.renderer = ChunkRenderer(image_size=image_size)
        
        # PRIMARY: Symbol-level masking (preserves real 2D positions!)
        # Masks symbols WITHIN a single expression, uses real bboxes
        self.symbol_augmentor = SymbolMaskingAugmentor(
            mathwriting,
            min_mask_symbols=1,
            max_mask_symbols=3,
            artifact_ratio=artifact_ratio,
            require_bboxes=True
        )
        
        # FALLBACK: Chunk-level augmentation (when symbol masking fails)
        self.chunk_augmentor = CompositionalAugmentor(
            mathwriting,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
            artifact_ratio=artifact_ratio
        )
        
        # LaTeX parser for ground truth relations
        self.latex_parser = LatexRelationshipParser()
        
        # Spatial relation inferrer (fallback)
        self.relation_inferrer = SpatialRelationInferrer()
    
    def __len__(self) -> int:
        # Virtual size (samples are generated on-the-fly)
        return len(self.mathwriting) * 10
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with proper per-symbol stroke grouping.
        
        Uses SymbolMaskingAugmentor to preserve REAL 2D positions:
        - Masks symbols WITHIN a single expression
        - Context has [MASK] where masked symbols were
        - Image shows ONLY masked symbol strokes
        - SMM learns which strokes belong to current token
        
        Image contains:
        - Target symbol strokes (from MathWriting per-symbol bboxes)
        - Artifact strokes (incomplete, should be ignored by SMM)
        
        SMM learns to distinguish complete symbols vs incomplete artifacts.
        
        Returns:
            Dict with:
            - context_ids: [L] tokenized context
            - image: [1, H, W] rendered edit chunk with artifacts
            - target_tokens: [T] target token sequence
            - target_actions: [T] target action indices
            - stroke_labels: [T, num_groups] per-step stroke group labels
            - stroke_indices: List of (start, end) patch ranges per group
            - num_stroke_groups: Number of stroke groups (target + artifact)
            - is_target_group: [num_groups] 1 if target, 0 if artifact
            - masked_bboxes: [T] bboxes for 2D position info (if available)
        """
        # Try symbol-level masking first (preserves real 2D positions)
        # NO [MASK] in context - model must infer position from stroke positions!
        masked_bboxes = None
        insertion_idx = -1
        try:
            if self.symbol_augmentor.valid_indices:
                context, edit_chunk, target, masked_bboxes, insertion_idx, artifacts = \
                    self.symbol_augmentor.create_edit_sample()
            else:
                raise ValueError("No valid samples")
        except Exception:
            # Fall back to chunk-level augmentation
            context, edit_chunk, target, artifacts = self.chunk_augmentor.create_edit_sample()
            insertion_idx = -1  # Insert at end for chunk-level
        
        # Tokenize context
        context_ids = self.tokenizer.encode(context, add_special=True)
        context_ids = context_ids[:self.max_context_len]
        context_ids = torch.tensor(context_ids, dtype=torch.long)
        
        # Tokenize target (each symbol becomes a token)
        target_tokens = self.tokenizer.encode(target, add_special=False)
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)
        T = len(target_tokens)
        
        # Render image (target + artifacts)
        image = self.renderer.render_to_tensor(edit_chunk, artifacts)
        
        # ============================================================
        # Build stroke groups from per-symbol bboxes
        # ============================================================
        # 
        # Group 1: Target symbol strokes (from edit_chunk.symbol_bboxes)
        # Group 2+: Artifact strokes (from artifacts list)
        #
        # Each group gets a patch range based on bbox position
        # ============================================================
        
        stroke_groups = []      # List of (start_patch, end_patch)
        is_target_group = []    # 1 if target symbol, 0 if artifact
        group_to_symbol = []    # Which symbol (token index) this group belongs to
        
        # Grid size for patch mapping (assuming 8x8 grid for 128x128 image)
        grid_h, grid_w = 8, 8
        patch_h = self.image_size // grid_h
        patch_w = self.image_size // grid_w
        
        # Get global bounding box for normalization
        x_min, y_min, x_max, y_max = edit_chunk.bbox
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
        
        # Add target symbol groups (from per-symbol bboxes)
        if edit_chunk.symbol_bboxes:
            for sym_idx, bbox in enumerate(edit_chunk.symbol_bboxes):
                # Normalize bbox to image coordinates
                norm_x_min = (bbox.x_min - x_min) / (x_max - x_min)
                norm_x_max = (bbox.x_max - x_min) / (x_max - x_min)
                norm_y_min = (bbox.y_min - y_min) / (y_max - y_min)
                norm_y_max = (bbox.y_max - y_min) / (y_max - y_min)
                
                # Convert to patch indices
                patch_x_start = int(norm_x_min * grid_w)
                patch_x_end = int(norm_x_max * grid_w) + 1
                patch_y_start = int(norm_y_min * grid_h)
                patch_y_end = int(norm_y_max * grid_h) + 1
                
                # Clamp to valid range
                patch_x_start = max(0, min(grid_w - 1, patch_x_start))
                patch_x_end = max(patch_x_start + 1, min(grid_w, patch_x_end))
                patch_y_start = max(0, min(grid_h - 1, patch_y_start))
                patch_y_end = max(patch_y_start + 1, min(grid_h, patch_y_end))
                
                # Convert 2D patch range to 1D (row-major)
                start_patch = patch_y_start * grid_w + patch_x_start
                end_patch = patch_y_end * grid_w  # Approximate - covers the row range
                
                stroke_groups.append((start_patch, min(end_patch, grid_h * grid_w)))
                is_target_group.append(1)
                group_to_symbol.append(min(sym_idx, T - 1))  # Map to token index
        else:
            # Fallback: treat all patches as one target group
            stroke_groups.append((0, grid_h * grid_w))
            is_target_group.append(1)
            group_to_symbol.append(0)
        
        # Add artifact groups (incomplete strokes - should NOT be selected)
        num_artifact_groups = 0
        if artifacts:
            for art_idx, artifact in enumerate(artifacts):
                if artifact.num_points > 0:
                    # Compute artifact bbox in normalized coords
                    art_x_min, art_y_min, art_x_max, art_y_max = artifact.bbox
                    
                    # Normalize to edit chunk space (approximate)
                    norm_x = (art_x_min - x_min) / (x_max - x_min) if x_max > x_min else 0.5
                    norm_y = (art_y_min - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                    
                    # Assign small patch range for artifact
                    patch_x = int(norm_x * grid_w)
                    patch_y = int(norm_y * grid_h)
                    patch_x = max(0, min(grid_w - 1, patch_x))
                    patch_y = max(0, min(grid_h - 1, patch_y))
                    
                    start_patch = patch_y * grid_w + patch_x
                    end_patch = min(start_patch + 2, grid_h * grid_w)  # Small range for artifact
                    
                    stroke_groups.append((start_patch, end_patch))
                    is_target_group.append(0)  # NOT a target
                    group_to_symbol.append(-1)  # No symbol
                    num_artifact_groups += 1
        
        num_groups = len(stroke_groups)
        
        # ============================================================
        # Build stroke labels: which groups contribute to which tokens
        # ============================================================
        # 
        # For AR decoding step t:
        #   - stroke_labels[t, g] = 1 if group g is for token t (target)
        #   - stroke_labels[t, g] = 0 if group g is artifact or other token
        #
        # SMM learns: output high score for target groups, low for artifacts
        # ============================================================
        
        stroke_labels = torch.zeros(T, num_groups)
        
        for g, (is_target, sym_idx) in enumerate(zip(is_target_group, group_to_symbol)):
            if is_target and 0 <= sym_idx < T:
                # This group contributes to token at sym_idx
                stroke_labels[sym_idx, g] = 1.0
        
        # Convert to tensors
        stroke_indices = stroke_groups
        is_target_tensor = torch.tensor(is_target_group, dtype=torch.float)
        
        # ============================================================
        # Infer action labels from spatial positions (bboxes)
        # ============================================================
        # 
        # Uses SpatialRelationInferrer to determine:
        # - Which token is the parent
        # - What relation (RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE)
        #
        # This enables proper fraction, subscript, superscript handling
        # ============================================================
        
        target_actions = self._get_actions_from_latex(edit_chunk, T, insertion_idx)
        
        result = {
            'context_ids': context_ids,
            'image': image,
            'target_tokens': target_tokens,
            'target_actions': target_actions,
            'stroke_labels': stroke_labels,
            'stroke_indices': stroke_indices,
            'num_strokes': num_groups,
            'is_target_group': is_target_tensor,
            'num_target_groups': sum(is_target_group),
            'num_artifact_groups': num_artifact_groups,
        }
        
        # Include 2D position info from masked bboxes (if symbol-level masking used)
        # Model uses h_VC (spatial position) to figure out insertion point
        if masked_bboxes:
            # Store normalized bbox centers for position learning
            bbox_centers = []
            for bbox in masked_bboxes:
                cx = (bbox.x_min + bbox.x_max) / 2
                cy = (bbox.y_min + bbox.y_max) / 2
                bbox_centers.append((cx, cy))
            result['bbox_centers'] = bbox_centers
            result['masked_bboxes'] = masked_bboxes
        
        # Ground truth insertion position (for TPM training)
        # Model must learn to predict this from stroke positions, not from [MASK]!
        result['insertion_idx'] = insertion_idx
        
        return result
    
    def _get_actions_from_latex(
        self, 
        chunk: MathWritingChunk, 
        num_tokens: int,
        insertion_idx: int = -1
    ) -> torch.Tensor:
        """
        Get ground truth actions for TPM.
        
        Action Space Design:
        =====================
        TPM K-cache contains: [ROOT, context[0], context[1], ...]
        Action = parent_idx * num_relations + relation_type
        
        Parent indices:
          0 = ROOT (first symbol of new expression, or insert at beginning)
          1 = context[0]
          2 = context[1]
          ... etc
        
        insertion_idx mapping:
          -1 → parent_idx = 0 (ROOT, insert at very beginning)
           0 → parent_idx = 1 (after context[0])
           N → parent_idx = N+1 (after context[N])
        
        Ground Truth Sources:
          1. insertion_idx: WHERE in context to insert (from augmentor)
          2. LaTeX parsing: WHAT relation type (SUP/SUB/RIGHT/...)
        
        Args:
            chunk: MathWriting chunk with latex and symbol_bboxes
            num_tokens: Number of target tokens
            insertion_idx: Position in context to insert after (-1 = beginning)
            
        Returns:
            Tensor of action indices [T]
        """
        target_actions = torch.zeros(num_tokens, dtype=torch.long)
        
        # Map insertion_idx to parent_idx in TPM action space
        # insertion_idx: -1 = beginning, 0 = after context[0], N = after context[N]
        # parent_idx:     0 = ROOT,      1 = context[0],       N+1 = context[N]
        parent_idx = insertion_idx + 1  # Shift by 1 to account for ROOT at index 0
        parent_idx = max(0, parent_idx)  # Clamp to valid range (ROOT is always valid)
        
        # ============================================================
        # Get RELATION from LaTeX parsing (or spatial inference)
        # ============================================================
        relation = RelationType.RIGHT  # Default
        
        # Try parsing LaTeX for first token's relation
        try:
            parsed = self.latex_parser.parse(chunk.latex)
            if parsed.relations and len(parsed.relations) > 0:
                # First relation tells us how first token connects
                _, _, first_relation = parsed.relations[0]
                relation = first_relation
        except Exception:
            pass  # Fall back to spatial inference below
        
        # If LaTeX parsing didn't give a relation, try spatial inference
        if relation == RelationType.RIGHT and chunk.symbol_bboxes:
            bboxes = []
            for bbox in chunk.symbol_bboxes:
                bboxes.append((bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max))
            
            if len(bboxes) > 0:
                # Compute expression center
                all_y_min = min(b[1] for b in bboxes)
                all_y_max = max(b[3] for b in bboxes)
                center_y = (all_y_min + all_y_max) / 2
                expr_height = max(all_y_max - all_y_min, 1)
                
                # Check first symbol's position
                bbox = bboxes[0]
                sym_center_y = (bbox[1] + bbox[3]) / 2
                sym_height = bbox[3] - bbox[1]
                
                y_offset = sym_center_y - center_y
                relative_offset = y_offset / expr_height if expr_height > 0 else 0
                
                if relative_offset < -0.25 and sym_height < expr_height * 0.5:
                    relation = RelationType.SUP
                elif relative_offset > 0.25 and sym_height < expr_height * 0.5:
                    relation = RelationType.SUB
                elif relative_offset < -0.3:
                    relation = RelationType.ABOVE
                elif relative_offset > 0.3:
                    relation = RelationType.BELOW
                # else: keep RIGHT
        
        # Encode action: parent_idx * num_relations + relation
        action = parent_idx * self.num_relations + relation.value
        
        # All target tokens share the same action (they're being inserted together)
        for t in range(num_tokens):
            target_actions[t] = action
        
        return target_actions


def collate_edit_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for EditDataset.
    
    Pads sequences and stacks tensors.
    Handles variable number of stroke groups (target + artifact).
    """
    # Find max lengths
    max_context = max(b['context_ids'].shape[0] for b in batch)
    max_target = max(b['target_tokens'].shape[0] for b in batch)
    max_groups = max(b['num_strokes'] for b in batch)  # num_strokes = num_groups
    
    B = len(batch)
    
    # Pad and stack
    context_ids = torch.zeros(B, max_context, dtype=torch.long)
    images = torch.stack([b['image'] for b in batch])
    target_tokens = torch.zeros(B, max_target, dtype=torch.long)
    target_actions = torch.zeros(B, max_target, dtype=torch.long)
    stroke_labels = torch.zeros(B, max_target, max_groups)
    active_strokes = torch.zeros(B, max_groups, dtype=torch.bool)
    is_target_group = torch.zeros(B, max_groups)  # 1 if target, 0 if artifact
    
    stroke_indices_list = []
    
    for i, b in enumerate(batch):
        L = b['context_ids'].shape[0]
        T = b['target_tokens'].shape[0]
        G = b['num_strokes']  # Number of groups
        
        context_ids[i, :L] = b['context_ids']
        target_tokens[i, :T] = b['target_tokens']
        target_actions[i, :T] = b['target_actions']
        stroke_labels[i, :T, :G] = b['stroke_labels']
        active_strokes[i, :G] = True
        
        # Track which groups are targets vs artifacts
        if 'is_target_group' in b:
            is_target_group[i, :G] = b['is_target_group']
        else:
            is_target_group[i, :G] = 1  # Assume all target if not specified
        
        # Pad stroke indices
        padded_indices = list(b['stroke_indices'])
        while len(padded_indices) < max_groups:
            padded_indices.append((0, 0))
        stroke_indices_list.append(padded_indices)
    
    return {
        'context_ids': context_ids,
        'images': images,
        'target_tokens': target_tokens,
        'target_actions': target_actions,
        'stroke_labels': stroke_labels,
        'active_strokes': active_strokes,
        'is_target_group': is_target_group,  # For debugging/analysis
        'stroke_indices': stroke_indices_list,
        'seq_lens': torch.tensor([b['target_tokens'].shape[0] for b in batch]),
    }


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 128,
    vocab_size: int = 512
) -> Tuple[DataLoader, DataLoader, SimpleTokenizer]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to MathWriting data
        batch_size: Batch size
        num_workers: DataLoader workers
        image_size: Image size
        vocab_size: Vocabulary size
        
    Returns:
        (train_loader, val_loader, tokenizer)
    """
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    # Load datasets
    train_mw = MathWritingAtomic(data_dir, split='train', lazy_load=True)
    val_mw = MathWritingAtomic(data_dir, split='valid', lazy_load=True)
    
    # Create edit datasets
    train_ds = EditDataset(train_mw, tokenizer, image_size=image_size)
    val_ds = EditDataset(val_mw, tokenizer, image_size=image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_edit_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_edit_batch,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing EditDataset...")
    
    # Test tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")
    
    test_text = "x^2 + \\frac{1}{2}"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    print("\nAll tests passed!")

