"""
MathWriting Atomic Data Loader

Loads MathWriting dataset with per-symbol atomic extraction for edit operations.
Supports:
- Full expression chunks (synthetic/)
- Per-symbol bounding boxes (synthetic-bboxes.jsonl)
- Individual symbol strokes (symbols/)
- Compositional augmentation for edit scenarios
- Stroke rendering to images with artifacts

MathWriting Dataset Structure:
```
mathwriting/
├── synthetic/              396K expressions with strokes (InkML)
├── synthetic-bboxes.jsonl  Per-symbol bounding boxes
└── symbols/                6,423 individual symbol strokes (InkML)
```

Usage:
    from context_tracker.data import MathWritingAtomic, load_mathwriting_splits
    
    # Load dataset
    train_ds, val_ds, test_ds = load_mathwriting_splits("path/to/mathwriting")
    
    # Get a sample
    chunk = train_ds[0]
    print(chunk.latex)           # "x^2 + 1"
    print(chunk.strokes)         # List of stroke arrays
    print(chunk.symbol_bboxes)   # Per-symbol bounding boxes
    
    # Render to image
    renderer = ChunkRenderer(image_size=128)
    image = renderer.render(chunk)
    
    # Compositional augmentation
    augmentor = CompositionalAugmentor(train_ds)
    context, edit_chunk, target = augmentor.create_edit_sample()
"""

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StrokeBBox:
    """Bounding box for a symbol's strokes."""
    symbol: str              # LaTeX symbol (e.g., "x", "\\frac", "2")
    stroke_indices: List[int]  # Which strokes belong to this symbol
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    def contains_point(self, x: float, y: float, margin: float = 0) -> bool:
        return (self.x_min - margin <= x <= self.x_max + margin and
                self.y_min - margin <= y <= self.y_max + margin)


@dataclass
class Stroke:
    """A single stroke (pen down to pen up)."""
    points: np.ndarray  # [N, 2] or [N, 3] for (x, y) or (x, y, t)
    
    @property
    def num_points(self) -> int:
        return len(self.points)
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max)."""
        if len(self.points) == 0:
            return (0, 0, 0, 0)
        x_min, y_min = self.points[:, :2].min(axis=0)
        x_max, y_max = self.points[:, :2].max(axis=0)
        return (x_min, y_min, x_max, y_max)
    
    def get_partial(self, ratio: float) -> 'Stroke':
        """Get first `ratio` portion of stroke (for incomplete artifacts)."""
        n = max(1, int(len(self.points) * ratio))
        return Stroke(points=self.points[:n].copy())


@dataclass 
class MathWritingChunk:
    """A complete expression chunk from MathWriting."""
    latex: str                           # LaTeX string
    strokes: List[Stroke]                # List of strokes
    symbol_bboxes: List[StrokeBBox] = field(default_factory=list)  # Per-symbol boxes
    sample_id: str = ""                  # Original sample ID
    
    @property
    def num_strokes(self) -> int:
        return len(self.strokes)
    
    @property
    def total_points(self) -> int:
        return sum(s.num_points for s in self.strokes)
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Global bounding box of all strokes."""
        if not self.strokes:
            return (0, 0, 0, 0)
        all_points = np.vstack([s.points[:, :2] for s in self.strokes if s.num_points > 0])
        if len(all_points) == 0:
            return (0, 0, 0, 0)
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        return (float(x_min), float(y_min), float(x_max), float(y_max))
    
    def get_symbol_strokes(self, symbol_idx: int) -> List[Stroke]:
        """Get strokes for a specific symbol by index."""
        if symbol_idx >= len(self.symbol_bboxes):
            return []
        bbox = self.symbol_bboxes[symbol_idx]
        return [self.strokes[i] for i in bbox.stroke_indices if i < len(self.strokes)]
    
    def extract_atomic(self, symbol_idx: int) -> 'MathWritingChunk':
        """Extract a single symbol as a new chunk."""
        if symbol_idx >= len(self.symbol_bboxes):
            raise IndexError(f"Symbol index {symbol_idx} out of range")
        
        bbox = self.symbol_bboxes[symbol_idx]
        strokes = self.get_symbol_strokes(symbol_idx)
        
        return MathWritingChunk(
            latex=bbox.symbol,
            strokes=strokes,
            symbol_bboxes=[StrokeBBox(
                symbol=bbox.symbol,
                stroke_indices=list(range(len(strokes))),
                x_min=bbox.x_min,
                y_min=bbox.y_min,
                x_max=bbox.x_max,
                y_max=bbox.y_max
            )],
            sample_id=f"{self.sample_id}_sym{symbol_idx}"
        )


# =============================================================================
# InkML Parser
# =============================================================================

class InkMLParser:
    """Parse InkML files from MathWriting dataset."""
    
    INKML_NS = {'inkml': 'http://www.w3.org/2003/InkML'}
    
    @classmethod
    def parse_file(cls, filepath: Union[str, Path]) -> MathWritingChunk:
        """Parse an InkML file into a MathWritingChunk."""
        filepath = Path(filepath)
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse {filepath}: {e}")
        
        # Extract LaTeX (try multiple annotation types)
        latex = cls._extract_latex(root)
        
        # Extract strokes
        strokes = cls._extract_strokes(root)
        
        # Extract symbol bboxes if available (from traceGroup structure)
        symbol_bboxes = cls._extract_symbol_bboxes(root, strokes)
        
        return MathWritingChunk(
            latex=latex,
            strokes=strokes,
            symbol_bboxes=symbol_bboxes,
            sample_id=filepath.stem
        )
    
    @classmethod
    def _extract_latex(cls, root: ET.Element) -> str:
        """Extract LaTeX annotation from InkML."""
        # Try different annotation types
        for ann_type in ['truth', 'normalizedLabel', 'label', 'transcription']:
            # With namespace
            for ann in root.findall('.//inkml:annotation', cls.INKML_NS):
                if ann.get('type') == ann_type and ann.text:
                    return ann.text.strip()
            # Without namespace
            for ann in root.findall('.//annotation'):
                if ann.get('type') == ann_type and ann.text:
                    return ann.text.strip()
        
        # Fallback: any annotation with text
        for ann in root.iter():
            if 'annotation' in ann.tag.lower() and ann.text:
                return ann.text.strip()
        
        return ""
    
    @classmethod
    def _extract_strokes(cls, root: ET.Element) -> List[Stroke]:
        """Extract stroke data from InkML."""
        strokes = []
        
        # Find all trace elements
        traces = (root.findall('.//inkml:trace', cls.INKML_NS) or 
                  root.findall('.//trace'))
        
        for trace in traces:
            points = cls._parse_trace_points(trace.text or "")
            if len(points) > 0:
                strokes.append(Stroke(points=points))
        
        return strokes
    
    @classmethod
    def _parse_trace_points(cls, trace_text: str) -> np.ndarray:
        """Parse trace text into point array."""
        if not trace_text.strip():
            return np.array([]).reshape(0, 2)
        
        points = []
        for point_str in trace_text.strip().split(','):
            parts = point_str.strip().split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    t = float(parts[2]) if len(parts) > 2 else 0.0
                    points.append([x, y, t] if len(parts) > 2 else [x, y])
                except ValueError:
                    continue
        
        if not points:
            return np.array([]).reshape(0, 2)
        
        return np.array(points, dtype=np.float32)
    
    @classmethod
    def _extract_symbol_bboxes(
        cls, 
        root: ET.Element, 
        strokes: List[Stroke]
    ) -> List[StrokeBBox]:
        """Extract per-symbol bounding boxes from traceGroup structure."""
        bboxes = []
        
        # Find traceGroup elements with symbol annotations
        trace_groups = (root.findall('.//inkml:traceGroup', cls.INKML_NS) or
                        root.findall('.//traceGroup'))
        
        for group in trace_groups:
            # Get symbol annotation
            symbol = None
            for ann in (group.findall('.//inkml:annotation', cls.INKML_NS) or
                        group.findall('.//annotation')):
                if ann.get('type') in ['truth', 'symbol', 'label'] and ann.text:
                    symbol = ann.text.strip()
                    break
            
            if not symbol:
                continue
            
            # Get trace references
            trace_refs = (group.findall('.//inkml:traceView', cls.INKML_NS) or
                          group.findall('.//traceView'))
            
            stroke_indices = []
            for ref in trace_refs:
                trace_ref = ref.get('traceDataRef', '')
                # Parse trace reference (e.g., "#trace0" -> 0)
                if trace_ref.startswith('#trace'):
                    try:
                        idx = int(trace_ref[6:])
                        stroke_indices.append(idx)
                    except ValueError:
                        pass
            
            if not stroke_indices:
                continue
            
            # Compute bounding box from strokes
            all_points = []
            for idx in stroke_indices:
                if idx < len(strokes) and strokes[idx].num_points > 0:
                    all_points.append(strokes[idx].points[:, :2])
            
            if not all_points:
                continue
            
            all_points = np.vstack(all_points)
            x_min, y_min = all_points.min(axis=0)
            x_max, y_max = all_points.max(axis=0)
            
            bboxes.append(StrokeBBox(
                symbol=symbol,
                stroke_indices=stroke_indices,
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max)
            ))
        
        return bboxes


# =============================================================================
# Bounding Box JSON Parser (for synthetic-bboxes.jsonl)
# =============================================================================

class BBoxJSONParser:
    """Parse per-symbol bounding boxes from synthetic-bboxes.jsonl.
    
    MathWriting JSONL format:
    {
        "label": "\\frac{0}{u}",
        "normalizedLabel": "\\frac{0}{u}",
        "bboxes": [
            {"token": "0", "xMin": 102.39, "yMin": -865.7, "xMax": 430.07, "yMax": -443.36},
            {"token": "\\frac", "xMin": 78.64, "yMin": -176.95, "xMax": 453.81, "yMax": -150.73},
            {"token": "u", "xMin": 78.64, "yMin": 167.38, "xMax": 453.81, "yMax": 449.54}
        ]
    }
    
    Note: We index by normalizedLabel (ground truth used for training).
    """
    
    @classmethod
    def load_bboxes(cls, jsonl_path: Union[str, Path]) -> Dict[str, List[StrokeBBox]]:
        """Load all bboxes indexed by normalizedLabel (the ground truth).
        
        Returns:
            Dict mapping normalizedLabel -> List[StrokeBBox]
        """
        bboxes_by_label = {}
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Use normalizedLabel as the key (this is the ground truth)
                    label = data.get('normalizedLabel', data.get('label', ''))
                    bbox_list = data.get('bboxes', [])
                    
                    if not label or not bbox_list:
                        continue
                    
                    bboxes = []
                    for i, bbox in enumerate(bbox_list):
                        bboxes.append(StrokeBBox(
                            symbol=bbox.get('token', ''),
                            stroke_indices=[i],  # Approximate: one bbox = one stroke group
                            x_min=bbox.get('xMin', 0.0),
                            y_min=bbox.get('yMin', 0.0),
                            x_max=bbox.get('xMax', 0.0),
                            y_max=bbox.get('yMax', 0.0)
                        ))
                    
                    if bboxes:
                        bboxes_by_label[label] = bboxes
                        
                except json.JSONDecodeError:
                    continue
        
        return bboxes_by_label


# =============================================================================
# Chunk Renderer
# =============================================================================

class ChunkRenderer:
    """Render strokes to images."""
    
    def __init__(
        self,
        image_size: int = 128,
        line_width: int = 2,
        padding: float = 0.1,
        background: int = 255,
        foreground: int = 0,
        normalize: bool = True
    ):
        """
        Args:
            image_size: Output image size (square)
            line_width: Stroke line width in pixels
            padding: Padding ratio around content
            background: Background color (0-255)
            foreground: Foreground/stroke color (0-255)
            normalize: If True, normalize strokes to fit image
        """
        self.image_size = image_size
        self.line_width = line_width
        self.padding = padding
        self.background = background
        self.foreground = foreground
        self.normalize = normalize
    
    def render(
        self,
        chunk: MathWritingChunk,
        artifact_strokes: Optional[List[Stroke]] = None,
        return_numpy: bool = True
    ) -> Union[np.ndarray, 'Image.Image']:
        """
        Render chunk strokes to image.
        
        Args:
            chunk: MathWritingChunk to render
            artifact_strokes: Optional incomplete strokes to overlay
            return_numpy: If True, return numpy array, else PIL Image
            
        Returns:
            Grayscale image [H, W] as numpy or PIL
        """
        if not HAS_PIL:
            raise ImportError("PIL required for rendering. Install with: pip install Pillow")
        
        # Collect all strokes
        all_strokes = list(chunk.strokes)
        if artifact_strokes:
            all_strokes.extend(artifact_strokes)
        
        if not all_strokes:
            # Empty image
            img = Image.new('L', (self.image_size, self.image_size), self.background)
            return np.array(img) if return_numpy else img
        
        # Compute transform
        transform = self._compute_transform(all_strokes)
        
        # Create image
        img = Image.new('L', (self.image_size, self.image_size), self.background)
        draw = ImageDraw.Draw(img)
        
        # Draw strokes
        for stroke in all_strokes:
            self._draw_stroke(draw, stroke, transform)
        
        return np.array(img) if return_numpy else img
    
    def render_symbol(
        self,
        chunk: MathWritingChunk,
        symbol_idx: int,
        return_numpy: bool = True
    ) -> Union[np.ndarray, 'Image.Image']:
        """Render only a specific symbol's strokes."""
        atomic = chunk.extract_atomic(symbol_idx)
        return self.render(atomic, return_numpy=return_numpy)
    
    def _compute_transform(
        self, 
        strokes: List[Stroke]
    ) -> Tuple[float, float, float]:
        """Compute scale and offset to fit strokes in image."""
        all_points = np.vstack([s.points[:, :2] for s in strokes if s.num_points > 0])
        
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width < 1e-6:
            width = 1.0
        if height < 1e-6:
            height = 1.0
        
        # Scale to fit with padding
        usable_size = self.image_size * (1 - 2 * self.padding)
        scale = min(usable_size / width, usable_size / height)
        
        # Center offset
        offset_x = self.image_size / 2 - (x_min + width / 2) * scale
        offset_y = self.image_size / 2 - (y_min + height / 2) * scale
        
        return (scale, offset_x, offset_y)
    
    def _draw_stroke(
        self,
        draw: 'ImageDraw.ImageDraw',
        stroke: Stroke,
        transform: Tuple[float, float, float]
    ):
        """Draw a single stroke."""
        if stroke.num_points < 2:
            return
        
        scale, offset_x, offset_y = transform
        
        # Transform points
        points = stroke.points[:, :2] * scale
        points[:, 0] += offset_x
        points[:, 1] += offset_y
        
        # Draw as connected lines
        point_list = [(p[0], p[1]) for p in points]
        draw.line(point_list, fill=self.foreground, width=self.line_width)
    
    def render_to_tensor(
        self,
        chunk: MathWritingChunk,
        artifact_strokes: Optional[List[Stroke]] = None
    ) -> 'torch.Tensor':
        """Render to PyTorch tensor [1, H, W] normalized to [0, 1]."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")
        
        img = self.render(chunk, artifact_strokes, return_numpy=True)
        tensor = torch.from_numpy(img).float() / 255.0
        return tensor.unsqueeze(0)  # Add channel dim


# =============================================================================
# Compositional Augmentor
# =============================================================================

class SymbolMaskingAugmentor:
    """
    Create edit samples by masking symbols WITHIN a single MathWriting sample.
    
    This preserves REAL spatial relationships from the original expression!
    
    Strategy:
    1. Take a MathWriting sample with per-symbol bboxes
    2. Mask out 1-3 consecutive symbols (pretend never written)
    3. Context = remaining symbols (text with 2D positions preserved)
    4. Edit = masked symbol strokes cropped from original
    5. Model learns to predict masked symbols at correct positions
    
    Example:
        Original: "\\frac{a+b}{c}" with bboxes
        Mask "b" → Context: "\\frac{a+[MASK]}{c}"
                   Edit: strokes of "b" only
                   Target: "b" with action (a, RIGHT)
    """
    
    def __init__(
        self,
        dataset: 'MathWritingAtomic',
        min_mask_symbols: int = 1,
        max_mask_symbols: int = 3,
        artifact_ratio: float = 0.3,
        require_bboxes: bool = True
    ):
        """
        Args:
            dataset: MathWritingAtomic with per-symbol bboxes
            min_mask_symbols: Minimum symbols to mask
            max_mask_symbols: Maximum symbols to mask
            artifact_ratio: Probability of adding incomplete strokes
            require_bboxes: Only use samples with bounding boxes
        """
        self.dataset = dataset
        self.min_mask = min_mask_symbols
        self.max_mask = max_mask_symbols
        self.artifact_ratio = artifact_ratio
        self.require_bboxes = require_bboxes
        
        # Pre-filter samples with bboxes if required
        if require_bboxes:
            self.valid_indices = [
                i for i in range(len(dataset))
                if dataset[i].symbol_bboxes and len(dataset[i].symbol_bboxes) >= 2
            ]
        else:
            self.valid_indices = list(range(len(dataset)))
    
    def create_edit_sample(self) -> Tuple[str, MathWritingChunk, str, List[StrokeBBox], int, Optional[List[Stroke]]]:
        """
        Create an edit sample by masking symbols within a single expression.
        
        NO [MASK] in context! Model must infer insertion position from stroke positions.
        This is realistic: user writes strokes, model figures out where they go.
        
        Returns:
            context: LaTeX WITHOUT mask - just the committed expression
            edit_chunk: Chunk containing ONLY the masked symbol strokes
            target: LaTeX of masked symbols
            masked_bboxes: Bboxes of masked symbols (for 2D position info)
            insertion_idx: Index in context AFTER which to insert (-1 = beginning)
            artifacts: Optional incomplete strokes
        """
        if not self.valid_indices:
            raise ValueError("No valid samples with bboxes found")
        
        # Get a sample with bboxes
        sample_idx = random.choice(self.valid_indices)
        sample = self.dataset[sample_idx]
        
        num_symbols = len(sample.symbol_bboxes)
        
        # Choose how many consecutive symbols to mask
        num_to_mask = min(
            random.randint(self.min_mask, self.max_mask),
            num_symbols - 1  # Keep at least 1 symbol in context
        )
        
        # Choose starting position for mask
        max_start = num_symbols - num_to_mask
        mask_start = random.randint(0, max_start)
        mask_end = mask_start + num_to_mask
        
        # Split symbols into context and masked
        context_bboxes = sample.symbol_bboxes[:mask_start] + sample.symbol_bboxes[mask_end:]
        masked_bboxes = sample.symbol_bboxes[mask_start:mask_end]
        
        # Build context string WITHOUT [MASK] - model must infer position from strokes!
        # Context shows the committed expression, new strokes indicate where user is writing
        context_tokens = [bbox.symbol for bbox in sample.symbol_bboxes[:mask_start]]
        context_tokens.extend([bbox.symbol for bbox in sample.symbol_bboxes[mask_end:]])
        context = " ".join(context_tokens)
        
        # Store insertion position: index in context where masked symbols should go
        # This is the ground truth for TPM - insert AFTER this context token index
        # -1 means insert at beginning
        insertion_idx = mask_start - 1
        
        # Target is the masked symbols
        target = " ".join([bbox.symbol for bbox in masked_bboxes])
        
        # Extract strokes for masked region
        # Use stroke_indices from bboxes to get relevant strokes
        masked_stroke_indices = set()
        for bbox in masked_bboxes:
            masked_stroke_indices.update(bbox.stroke_indices)
        
        # Create edit chunk with only masked strokes
        masked_strokes = []
        for idx in sorted(masked_stroke_indices):
            if idx < len(sample.strokes):
                masked_strokes.append(sample.strokes[idx])
        
        # If no stroke indices available, use approximate bbox-based extraction
        if not masked_strokes and masked_bboxes:
            # Get bounding box of masked region
            min_x = min(b.x_min for b in masked_bboxes)
            max_x = max(b.x_max for b in masked_bboxes)
            min_y = min(b.y_min for b in masked_bboxes)
            max_y = max(b.y_max for b in masked_bboxes)
            
            # Find strokes that fall within this bbox
            for stroke in sample.strokes:
                sx_min, sy_min, sx_max, sy_max = stroke.bbox
                # Check overlap
                if (sx_max >= min_x and sx_min <= max_x and
                    sy_max >= min_y and sy_min <= max_y):
                    masked_strokes.append(stroke)
        
        # Create edit chunk
        edit_chunk = MathWritingChunk(
            latex=target,
            strokes=masked_strokes if masked_strokes else sample.strokes[:1],
            symbol_bboxes=masked_bboxes,
            sample_id=f"{sample.sample_id}_masked"
        )
        
        # Optional artifacts
        artifacts = None
        if random.random() < self.artifact_ratio:
            artifacts = self._create_artifacts_from_context(sample, context_bboxes)
        
        return context, edit_chunk, target, masked_bboxes, insertion_idx, artifacts
    
    def _create_artifacts_from_context(
        self, 
        sample: MathWritingChunk,
        context_bboxes: List[StrokeBBox]
    ) -> Optional[List[Stroke]]:
        """Create incomplete strokes from context symbols (things being written)."""
        if not context_bboxes:
            return None
        
        artifacts = []
        # Pick 1-2 random context symbols to create partial strokes
        num_artifacts = min(random.randint(1, 2), len(context_bboxes))
        
        for bbox in random.sample(context_bboxes, num_artifacts):
            # Get strokes for this symbol
            for idx in bbox.stroke_indices:
                if idx < len(sample.strokes):
                    stroke = sample.strokes[idx]
                    # Take first 20-40% (incomplete)
                    ratio = random.uniform(0.2, 0.4)
                    partial = stroke.get_partial(ratio)
                    if partial.num_points > 1:
                        artifacts.append(partial)
                    break  # One partial stroke per symbol
        
        return artifacts if artifacts else None


class CompositionalAugmentor:
    """
    Create compositional training samples for edit scenarios.
    
    Strategy (chunk-level, for simpler cases):
    1. COMPOSE: Concatenate 2-4 MathWriting chunks -> context
    2. COMMIT: Context becomes TEXT tokens (not patches)
    3. EDIT: Replace one chunk with different MathWriting sample
    
    For symbol-level augmentation with real 2D positions,
    use SymbolMaskingAugmentor instead.
    """
    
    def __init__(
        self,
        dataset: 'MathWritingAtomic',
        min_chunks: int = 2,
        max_chunks: int = 4,
        artifact_ratio: float = 0.5,
        incomplete_ratio: float = 0.3
    ):
        """
        Args:
            dataset: MathWritingAtomic dataset to sample from
            min_chunks: Minimum chunks in context
            max_chunks: Maximum chunks in context
            artifact_ratio: Probability of adding stroke artifacts
            incomplete_ratio: How much of artifact stroke to include (0.3 = first 30%)
        """
        self.dataset = dataset
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.artifact_ratio = artifact_ratio
        self.incomplete_ratio = incomplete_ratio
    
    def create_edit_sample(
        self
    ) -> Tuple[str, MathWritingChunk, str, Optional[List[Stroke]]]:
        """
        Create an edit training sample.
        
        Returns:
            context: Text context with [MASK] placeholder
            edit_chunk: The chunk to recognize from image
            target: Target LaTeX for edit_chunk
            artifacts: Optional incomplete strokes for distraction
        """
        # Sample chunks for context
        num_chunks = random.randint(self.min_chunks, self.max_chunks)
        chunk_indices = random.sample(range(len(self.dataset)), num_chunks + 1)
        
        # One chunk will be the edit target
        edit_idx = random.randint(0, num_chunks - 1)
        
        # Build context with [MASK]
        context_parts = []
        for i, idx in enumerate(chunk_indices[:-1]):
            chunk = self.dataset[idx]
            if i == edit_idx:
                context_parts.append("[MASK]")
            else:
                context_parts.append(chunk.latex)
        
        context = ", ".join(context_parts)
        
        # Edit chunk is a different sample
        edit_chunk = self.dataset[chunk_indices[-1]]
        target = edit_chunk.latex
        
        # Optional artifacts
        artifacts = None
        if random.random() < self.artifact_ratio:
            artifacts = self._create_artifacts()
        
        return context, edit_chunk, target, artifacts
    
    def _create_artifacts(self) -> List[Stroke]:
        """Create incomplete stroke artifacts."""
        artifacts = []
        
        # Sample 1-2 incomplete strokes from random symbols
        num_artifacts = random.randint(1, 2)
        
        for _ in range(num_artifacts):
            # Random chunk
            chunk = self.dataset[random.randint(0, len(self.dataset) - 1)]
            
            if chunk.num_strokes > 0:
                # Random stroke
                stroke = random.choice(chunk.strokes)
                # Take incomplete portion
                partial = stroke.get_partial(self.incomplete_ratio)
                if partial.num_points > 1:
                    artifacts.append(partial)
        
        return artifacts if artifacts else None
    
    def create_batch(
        self,
        batch_size: int,
        renderer: Optional[ChunkRenderer] = None
    ) -> Dict:
        """
        Create a batch of edit samples.
        
        Returns:
            Dict with keys: contexts, images, targets, artifacts_present
        """
        contexts = []
        chunks = []
        targets = []
        artifacts_list = []
        
        for _ in range(batch_size):
            context, edit_chunk, target, artifacts = self.create_edit_sample()
            contexts.append(context)
            chunks.append(edit_chunk)
            targets.append(target)
            artifacts_list.append(artifacts)
        
        result = {
            'contexts': contexts,
            'chunks': chunks,
            'targets': targets,
            'artifacts': artifacts_list,
        }
        
        # Optionally render to images
        if renderer is not None:
            if HAS_TORCH:
                images = torch.stack([
                    renderer.render_to_tensor(chunk, artifacts)
                    for chunk, artifacts in zip(chunks, artifacts_list)
                ])
                result['images'] = images
            else:
                result['images'] = [
                    renderer.render(chunk, artifacts, return_numpy=True)
                    for chunk, artifacts in zip(chunks, artifacts_list)
                ]
        
        return result


# =============================================================================
# Main Dataset Class
# =============================================================================

class MathWritingAtomic(Dataset if HAS_TORCH else object):
    """
    MathWriting dataset with atomic symbol support.
    
    Loads:
    - synthetic/ expressions (InkML)
    - symbols/ individual symbols (InkML)
    - synthetic-bboxes.jsonl per-symbol bounding boxes
    
    Provides:
    - Full expression chunks
    - Atomic symbol extraction
    - Compositional augmentation
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        load_symbols: bool = True,
        load_bboxes: bool = True,
        lazy_load: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Path to MathWriting root directory
            split: 'train', 'valid', or 'test'
            load_symbols: Whether to include symbols/ directory
            load_bboxes: Whether to load per-symbol bounding boxes
            lazy_load: If True, parse InkML on access (saves memory)
            max_samples: Limit number of samples (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.lazy_load = lazy_load
        self.load_symbols = load_symbols
        
        # Collect sample paths
        self.sample_paths: List[Path] = []
        self._cache: Dict[int, MathWritingChunk] = {}
        
        # Load synthetic expressions
        synthetic_dir = self.data_dir / 'synthetic'
        if synthetic_dir.exists():
            self._add_inkml_files(synthetic_dir, split)
        
        # Load individual symbols
        if load_symbols:
            symbols_dir = self.data_dir / 'symbols'
            if symbols_dir.exists():
                self._add_inkml_files(symbols_dir)
        
        # Apply max_samples limit
        if max_samples is not None:
            self.sample_paths = self.sample_paths[:max_samples]
        
        # Load bounding boxes (indexed by normalizedLabel/latex)
        self.bboxes_by_label: Dict[str, List[StrokeBBox]] = {}
        if load_bboxes:
            bbox_file = self.data_dir / 'synthetic-bboxes.jsonl'
            if bbox_file.exists():
                self.bboxes_by_label = BBoxJSONParser.load_bboxes(bbox_file)
        
        # Preload if not lazy
        if not lazy_load:
            self._preload_all()
    
    def _add_inkml_files(
        self, 
        directory: Path, 
        split: Optional[str] = None
    ):
        """Add InkML files from directory."""
        # Check for split subdirectory
        if split:
            split_dir = directory / split
            if split_dir.exists():
                directory = split_dir
        
        # Collect .inkml files
        inkml_files = list(directory.glob('**/*.inkml'))
        self.sample_paths.extend(sorted(inkml_files))
    
    def _preload_all(self):
        """Preload all samples into cache."""
        for idx in range(len(self.sample_paths)):
            self._load_sample(idx)
    
    def _load_sample(self, idx: int) -> MathWritingChunk:
        """Load a sample, using cache if available."""
        if idx in self._cache:
            return self._cache[idx]
        
        path = self.sample_paths[idx]
        chunk = InkMLParser.parse_file(path)
        
        # Add bboxes if available (indexed by latex label)
        if chunk.latex in self.bboxes_by_label:
            chunk.symbol_bboxes = self.bboxes_by_label[chunk.latex]
        
        if not self.lazy_load:
            self._cache[idx] = chunk
        
        return chunk
    
    def __len__(self) -> int:
        return len(self.sample_paths)
    
    def __getitem__(self, idx: int) -> MathWritingChunk:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        return self._load_sample(idx)
    
    def get_sample_by_id(self, sample_id: str) -> Optional[MathWritingChunk]:
        """Get sample by its ID (filename stem)."""
        for idx, path in enumerate(self.sample_paths):
            if path.stem == sample_id:
                return self._load_sample(idx)
        return None
    
    def get_atomic_samples(self) -> List[MathWritingChunk]:
        """Get all samples that have symbol-level bboxes."""
        atomic = []
        for idx in range(len(self)):
            chunk = self._load_sample(idx)
            if chunk.symbol_bboxes:
                for sym_idx in range(len(chunk.symbol_bboxes)):
                    atomic.append(chunk.extract_atomic(sym_idx))
        return atomic
    
    def filter_by_latex_pattern(self, pattern: str) -> List[MathWritingChunk]:
        """Filter samples by regex pattern on LaTeX."""
        import re
        regex = re.compile(pattern)
        matches = []
        for idx in range(len(self)):
            chunk = self._load_sample(idx)
            if regex.search(chunk.latex):
                matches.append(chunk)
        return matches


# =============================================================================
# Factory Functions
# =============================================================================

def load_mathwriting_splits(
    data_dir: Union[str, Path],
    **kwargs
) -> Tuple[MathWritingAtomic, MathWritingAtomic, MathWritingAtomic]:
    """
    Load train/valid/test splits.
    
    Args:
        data_dir: Path to MathWriting root
        **kwargs: Passed to MathWritingAtomic
        
    Returns:
        (train_dataset, valid_dataset, test_dataset)
    """
    train = MathWritingAtomic(data_dir, split='train', **kwargs)
    valid = MathWritingAtomic(data_dir, split='valid', **kwargs)
    test = MathWritingAtomic(data_dir, split='test', **kwargs)
    return train, valid, test


def create_edit_dataloader(
    dataset: MathWritingAtomic,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 0,
    **kwargs
) -> 'torch.utils.data.DataLoader':
    """
    Create a DataLoader for edit training.
    
    Returns batches with:
    - contexts: List[str] text contexts with [MASK]
    - images: [B, 1, H, W] rendered edit chunks
    - targets: List[str] target LaTeX
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")
    
    renderer = ChunkRenderer(image_size=image_size)
    augmentor = CompositionalAugmentor(dataset)
    
    class EditDataset(Dataset):
        def __init__(self, aug, rend, size):
            self.augmentor = aug
            self.renderer = rend
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            context, chunk, target, artifacts = self.augmentor.create_edit_sample()
            image = self.renderer.render_to_tensor(chunk, artifacts)
            return {
                'context': context,
                'image': image,
                'target': target
            }
    
    edit_ds = EditDataset(augmentor, renderer, len(dataset) * 10)  # 10x augmentation
    
    def collate_fn(batch):
        return {
            'contexts': [b['context'] for b in batch],
            'images': torch.stack([b['image'] for b in batch]),
            'targets': [b['target'] for b in batch]
        }
    
    return torch.utils.data.DataLoader(
        edit_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    print("MathWriting Atomic Data Loader")
    print("=" * 50)
    
    # Test data structures
    print("\n1. Testing data structures...")
    
    stroke = Stroke(points=np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32))
    print(f"   Stroke: {stroke.num_points} points, bbox={stroke.bbox}")
    
    partial = stroke.get_partial(0.5)
    print(f"   Partial (50%): {partial.num_points} points")
    
    bbox = StrokeBBox(
        symbol="x", 
        stroke_indices=[0, 1],
        x_min=0, y_min=0, x_max=10, y_max=10
    )
    print(f"   BBox: {bbox.symbol}, size={bbox.width}x{bbox.height}")
    
    chunk = MathWritingChunk(
        latex="x^2",
        strokes=[stroke],
        symbol_bboxes=[bbox],
        sample_id="test_001"
    )
    print(f"   Chunk: '{chunk.latex}', {chunk.num_strokes} strokes")
    
    # Test renderer (if PIL available)
    if HAS_PIL:
        print("\n2. Testing renderer...")
        renderer = ChunkRenderer(image_size=64)
        img = renderer.render(chunk, return_numpy=True)
        print(f"   Rendered image: {img.shape}, dtype={img.dtype}")
        print(f"   Value range: [{img.min()}, {img.max()}]")
    else:
        print("\n2. Skipping renderer test (PIL not installed)")
    
    print("\n3. Testing parser on mock data...")
    # Create mock InkML
    mock_inkml = """<?xml version="1.0" encoding="UTF-8"?>
    <ink xmlns="http://www.w3.org/2003/InkML">
        <annotation type="truth">x^2</annotation>
        <trace id="trace0">0 0, 10 0, 10 10, 0 10</trace>
        <trace id="trace1">15 0, 25 0, 25 10</trace>
    </ink>
    """
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.inkml', delete=False) as f:
        f.write(mock_inkml)
        temp_path = f.name
    
    parsed = InkMLParser.parse_file(temp_path)
    print(f"   Parsed: latex='{parsed.latex}', {parsed.num_strokes} strokes")
    
    import os
    os.unlink(temp_path)
    
    print("\nAll tests passed!")

