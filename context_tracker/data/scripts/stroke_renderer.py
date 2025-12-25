"""
GPU-Accelerated Stroke Renderer using Captured Stroke Data

This module renders handwritten symbols using the captured stroke variations
from stroke_data_merged.json with combinatorial augmentation and GPU acceleration.

Features:
    - Load captured stroke data (236 symbols, 2257 variations)
    - Combinatorial variation mixing (e.g., 3×3×3 = 27 combinations for 3-stroke symbol)
    - GPU batch rendering with PyTorch
    - Augmentation: jitter, rotation, scale, thickness variation
    - Anti-aliased rendering using signed distance fields

Usage:
    from scripts.stroke_renderer import StrokeRenderer, GPUStrokeRenderer
    
    # CPU rendering
    renderer = StrokeRenderer()
    canvas = renderer.render_symbol("α", variation_indices=[0, 1, 2])
    
    # GPU batch rendering
    gpu_renderer = GPUStrokeRenderer(device='cuda')
    canvases = gpu_renderer.render_batch(["α", "β", "x"], batch_size=32)
"""

import json
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np

# Optional GPU support
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. GPU rendering disabled.")

# Optional PIL for image export
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StrokePoint:
    """A point in a stroke with normalized coordinates."""
    x: float  # 0-1 normalized
    y: float  # 0-1 normalized
    
    def scale(self, width: int, height: int) -> Tuple[float, float]:
        """Scale to pixel coordinates."""
        return (self.x * width, self.y * height)
    
    def jitter(self, amount: float) -> 'StrokePoint':
        """Add random jitter."""
        return StrokePoint(
            x=self.x + random.uniform(-amount, amount),
            y=self.y + random.uniform(-amount, amount)
        )


@dataclass
class StrokeData:
    """A single stroke consisting of points."""
    points: List[StrokePoint]
    
    @classmethod
    def from_list(cls, point_list: List[List[float]]) -> 'StrokeData':
        """Create from [[x, y], [x, y], ...] format."""
        return cls(points=[StrokePoint(p[0], p[1]) for p in point_list])
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [N, 2]."""
        return np.array([[p.x, p.y] for p in self.points])
    
    def interpolate(self, num_points: int = 50) -> 'StrokeData':
        """Interpolate to have more points for smoother rendering."""
        if len(self.points) < 2:
            return self
        
        arr = self.to_array()
        t_old = np.linspace(0, 1, len(arr))
        t_new = np.linspace(0, 1, num_points)
        
        new_points = []
        for t in t_new:
            x = np.interp(t, t_old, arr[:, 0])
            y = np.interp(t, t_old, arr[:, 1])
            new_points.append(StrokePoint(x, y))
        
        return StrokeData(points=new_points)
    
    def apply_transform(self, 
                       rotation: float = 0,
                       scale: float = 1.0,
                       translate: Tuple[float, float] = (0, 0),
                       center: Tuple[float, float] = (0.5, 0.5)) -> 'StrokeData':
        """Apply affine transformation."""
        cos_r = math.cos(math.radians(rotation))
        sin_r = math.sin(math.radians(rotation))
        cx, cy = center
        tx, ty = translate
        
        new_points = []
        for p in self.points:
            # Translate to origin
            x = p.x - cx
            y = p.y - cy
            # Rotate and scale
            x_new = (x * cos_r - y * sin_r) * scale + cx + tx
            y_new = (x * sin_r + y * cos_r) * scale + cy + ty
            new_points.append(StrokePoint(x_new, y_new))
        
        return StrokeData(points=new_points)
    
    def add_jitter(self, amount: float = 0.01) -> 'StrokeData':
        """Add random jitter to all points."""
        return StrokeData(points=[p.jitter(amount) for p in self.points])


@dataclass
class SymbolData:
    """Complete symbol with multiple stroke variations."""
    symbol: str
    latex: str
    strokes: Dict[str, List[StrokeData]]  # stroke_name -> list of variations
    
    @property
    def num_strokes(self) -> int:
        return len(self.strokes)
    
    @property
    def total_variations(self) -> int:
        """Total number of variation combinations."""
        if not self.strokes:
            return 0
        result = 1
        for variations in self.strokes.values():
            result *= len(variations)
        return result
    
    def get_variation_count(self, stroke_name: str) -> int:
        """Get number of variations for a specific stroke."""
        return len(self.strokes.get(stroke_name, []))
    
    def get_stroke_variation(self, stroke_name: str, var_idx: int) -> Optional[StrokeData]:
        """Get specific stroke variation."""
        variations = self.strokes.get(stroke_name)
        if variations and 0 <= var_idx < len(variations):
            return variations[var_idx]
        return None


@dataclass
class AugmentationConfig:
    """Configuration for rendering augmentations."""
    # Jitter
    jitter_amount: float = 0.01  # Normalized coords
    
    # Rotation
    rotation_range: Tuple[float, float] = (-10, 10)  # Degrees
    
    # Scale
    scale_range: Tuple[float, float] = (0.9, 1.1)
    
    # Line thickness
    thickness_range: Tuple[float, float] = (2, 4)  # Pixels
    
    # Translation
    translate_range: Tuple[float, float] = (-0.05, 0.05)  # Normalized
    
    # Interpolation
    interpolate_points: int = 50  # Points per stroke for smooth rendering
    
    def sample(self) -> Dict[str, Any]:
        """Sample random augmentation parameters."""
        return {
            'jitter': random.uniform(0, self.jitter_amount),
            'rotation': random.uniform(*self.rotation_range),
            'scale': random.uniform(*self.scale_range),
            'thickness': random.uniform(*self.thickness_range),
            'translate': (
                random.uniform(*self.translate_range),
                random.uniform(*self.translate_range)
            ),
        }


# =============================================================================
# Stroke Data Loader
# =============================================================================

class StrokeDataLoader:
    """Load and manage captured stroke data."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize loader with stroke data file.
        
        Args:
            data_path: Path to stroke_data_merged.json. If None, uses default location.
        """
        if data_path is None:
            # Default path relative to this file
            data_path = Path(__file__).parent.parent / "stroke_corpus" / "annotations" / "stroke_data_merged.json"
        
        self.data_path = Path(data_path)
        self.symbols: Dict[str, SymbolData] = {}
        self._load_data()
    
    def _load_data(self):
        """Load stroke data from JSON file."""
        if not self.data_path.exists():
            print(f"Warning: Stroke data not found at {self.data_path}")
            return
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        for symbol_key, symbol_info in raw_data.items():
            strokes = {}
            for stroke_name, stroke_data in symbol_info.get('strokes', {}).items():
                variations = []
                for var_points in stroke_data.get('variations', []):
                    stroke = StrokeData.from_list(var_points)
                    variations.append(stroke)
                strokes[stroke_name] = variations
            
            self.symbols[symbol_key] = SymbolData(
                symbol=symbol_info.get('symbol', symbol_key),
                latex=symbol_info.get('latex', symbol_key),
                strokes=strokes
            )
    
    def get_symbol(self, symbol: str) -> Optional[SymbolData]:
        """Get symbol data by character or key."""
        return self.symbols.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all available symbols."""
        return list(self.symbols.keys())
    
    def get_latex_mapping(self) -> Dict[str, str]:
        """Get symbol -> latex mapping."""
        return {k: v.latex for k, v in self.symbols.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_strokes = 0
        total_variations = 0
        
        for sym_data in self.symbols.values():
            for stroke_vars in sym_data.strokes.values():
                total_strokes += 1
                total_variations += len(stroke_vars)
        
        return {
            'num_symbols': len(self.symbols),
            'total_strokes': total_strokes,
            'total_variations': total_variations,
            'avg_strokes_per_symbol': total_strokes / max(1, len(self.symbols)),
            'avg_variations_per_stroke': total_variations / max(1, total_strokes),
        }


# =============================================================================
# CPU Stroke Renderer
# =============================================================================

class StrokeRenderer:
    """CPU-based stroke renderer using PIL."""
    
    def __init__(self, 
                 data_loader: Optional[StrokeDataLoader] = None,
                 canvas_size: int = 256,
                 augmentation: Optional[AugmentationConfig] = None,
                 margin: float = 0.1):
        """
        Initialize renderer.
        
        Args:
            data_loader: StrokeDataLoader instance. Creates new one if None.
            canvas_size: Size of output canvas (square).
            augmentation: Augmentation configuration. Uses defaults if None.
            margin: Margin as fraction of canvas (0.1 = strokes use 80% of canvas,
                    leaving 10% on each side for line thickness). Default 0.1.
        """
        self.loader = data_loader or StrokeDataLoader()
        self.canvas_size = canvas_size
        self.augmentation = augmentation or AugmentationConfig()
        self.margin = margin
    
    def render_symbol(self,
                      symbol: str,
                      variation_indices: Optional[List[int]] = None,
                      augment: bool = True,
                      return_points: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[float, float]]]]:
        """
        Render a symbol to canvas.
        
        Args:
            symbol: Symbol character or key
            variation_indices: Which variation to use for each stroke. Random if None.
            augment: Apply random augmentation
            return_points: Also return stroke points for conv encoding
            
        Returns:
            Canvas as numpy array [H, W], optionally with points list
        """
        sym_data = self.loader.get_symbol(symbol)
        if sym_data is None:
            # Return empty canvas for unknown symbols
            canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
            if return_points:
                return canvas, []
            return canvas
        
        # Select variations
        stroke_names = sorted(sym_data.strokes.keys())
        if variation_indices is None:
            variation_indices = [
                random.randint(0, len(sym_data.strokes[name]) - 1)
                for name in stroke_names
            ]
        
        # Get augmentation params
        aug_params = self.augmentation.sample() if augment else {
            'jitter': 0, 'rotation': 0, 'scale': 1.0, 
            'thickness': 3.0, 'translate': (0, 0)
        }
        
        # Collect all stroke points
        all_strokes: List[StrokeData] = []
        
        # Compute margin transform: scale strokes to fit within margin
        # Original strokes are in [0,1] range, transform to [margin, 1-margin]
        margin_scale = 1.0 - 2 * self.margin
        margin_offset = self.margin
        
        for i, stroke_name in enumerate(stroke_names):
            var_idx = variation_indices[i] if i < len(variation_indices) else 0
            var_idx = var_idx % len(sym_data.strokes[stroke_name])
            
            stroke = sym_data.strokes[stroke_name][var_idx]
            
            # Apply augmentations
            stroke = stroke.interpolate(self.augmentation.interpolate_points)
            
            # Apply margin first (before augmentation rotation/scale)
            stroke = stroke.apply_transform(
                scale=margin_scale,
                translate=(margin_offset, margin_offset),
                center=(0, 0)
            )
            
            # Then apply user augmentation (rotation, scale around center)
            stroke = stroke.apply_transform(
                rotation=aug_params['rotation'],
                scale=aug_params['scale'],
                translate=aug_params['translate']
            )
            if aug_params['jitter'] > 0:
                stroke = stroke.add_jitter(aug_params['jitter'])
            
            all_strokes.append(stroke)
        
        # Render to canvas
        canvas = self._render_strokes(all_strokes, aug_params['thickness'])
        
        if return_points:
            all_points = []
            for stroke in all_strokes:
                for p in stroke.points:
                    all_points.append((p.x * self.canvas_size, p.y * self.canvas_size))
            return canvas, all_points
        
        return canvas
    
    def _render_strokes(self, strokes: List[StrokeData], thickness: float) -> np.ndarray:
        """Render strokes to numpy canvas using PIL."""
        if not HAS_PIL:
            return self._render_strokes_numpy(strokes, thickness)
        
        # Create PIL image
        img = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        draw = ImageDraw.Draw(img)
        
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue
            
            # Convert to pixel coords
            points = [p.scale(self.canvas_size, self.canvas_size) for p in stroke.points]
            
            # Draw line segments
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=255, width=int(thickness))
        
        return np.array(img, dtype=np.float32) / 255.0
    
    def _render_strokes_numpy(self, strokes: List[StrokeData], thickness: float) -> np.ndarray:
        """Fallback numpy-based rendering (anti-aliased but slower)."""
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue
            
            for i in range(len(stroke.points) - 1):
                p1 = stroke.points[i]
                p2 = stroke.points[i + 1]
                self._draw_line_aa(canvas, p1, p2, thickness)
        
        return np.clip(canvas, 0, 1)
    
    def _draw_line_aa(self, canvas: np.ndarray, p1: StrokePoint, p2: StrokePoint, thickness: float):
        """Draw anti-aliased line using Bresenham with thickness."""
        x0, y0 = int(p1.x * self.canvas_size), int(p1.y * self.canvas_size)
        x1, y1 = int(p2.x * self.canvas_size), int(p2.y * self.canvas_size)
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # Draw thick point
            r = int(thickness / 2)
            for dy_t in range(-r, r + 1):
                for dx_t in range(-r, r + 1):
                    px, py = x0 + dx_t, y0 + dy_t
                    if 0 <= px < self.canvas_size and 0 <= py < self.canvas_size:
                        dist = math.sqrt(dx_t**2 + dy_t**2)
                        intensity = max(0, 1 - dist / (thickness / 2))
                        canvas[py, px] = max(canvas[py, px], intensity)
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def render_to_size(self,
                       symbol: str,
                       target_size: Tuple[int, int],
                       num_strokes: Optional[int] = None,
                       augment: bool = True,
                       variation_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Render a symbol to a specific target size with high-quality scaling.
        
        This is the recommended method for rendering strokes to match a bounding box.
        It renders at high resolution internally then scales down for quality.
        
        Args:
            symbol: Symbol character to render
            target_size: (width, height) tuple for output size
            num_strokes: Number of strokes to render (None = all, for complete symbol)
            augment: Apply random augmentation
            variation_indices: Which variation for each stroke (random if None)
            
        Returns:
            Canvas as numpy array [H, W] with values 0-255 (uint8)
        """
        sym_data = self.loader.get_symbol(symbol)
        if sym_data is None:
            return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        
        # Render at higher resolution for quality
        render_size = max(target_size[0], target_size[1], 64) * 4
        
        # Temporarily set canvas size
        original_size = self.canvas_size
        self.canvas_size = render_size
        
        # Get strokes to render
        stroke_names = sorted(sym_data.strokes.keys())
        total_strokes = len(stroke_names)
        if num_strokes is None:
            num_strokes = total_strokes
        num_strokes = min(num_strokes, total_strokes)
        
        # Select variations
        if variation_indices is None:
            variation_indices = [
                random.randint(0, len(sym_data.strokes[name]) - 1)
                for name in stroke_names
            ]
        
        # Get augmentation params
        aug_params = self.augmentation.sample() if augment else {
            'jitter': 0, 'rotation': 0, 'scale': 1.0,
            'thickness': 3.0, 'translate': (0, 0)
        }
        
        # Collect strokes (only num_strokes)
        all_strokes: List[StrokeData] = []
        margin_scale = 1.0 - 2 * self.margin
        margin_offset = self.margin
        
        for i, stroke_name in enumerate(stroke_names[:num_strokes]):
            var_idx = variation_indices[i] if i < len(variation_indices) else 0
            var_idx = var_idx % len(sym_data.strokes[stroke_name])
            
            stroke = sym_data.strokes[stroke_name][var_idx]
            stroke = stroke.interpolate(self.augmentation.interpolate_points)
            
            # Apply margin
            stroke = stroke.apply_transform(
                scale=margin_scale,
                translate=(margin_offset, margin_offset),
                center=(0, 0)
            )
            
            # Apply augmentation
            stroke = stroke.apply_transform(
                rotation=aug_params['rotation'],
                scale=aug_params['scale'],
                translate=aug_params['translate']
            )
            if aug_params['jitter'] > 0:
                stroke = stroke.add_jitter(aug_params['jitter'])
            
            all_strokes.append(stroke)
        
        # Render
        canvas = self._render_strokes(all_strokes, aug_params['thickness'])
        
        # Restore canvas size
        self.canvas_size = original_size
        
        # Scale down to target size
        if HAS_PIL:
            img = Image.fromarray((canvas * 255).astype(np.uint8), mode='L')
            img = img.resize((target_size[0], target_size[1]), Image.LANCZOS)
            return np.array(img)
        else:
            # Fallback: simple resize
            from scipy.ndimage import zoom
            scale_y = target_size[1] / canvas.shape[0]
            scale_x = target_size[0] / canvas.shape[1]
            return (zoom(canvas, (scale_y, scale_x)) * 255).astype(np.uint8)
    
    def render_all_variations(self, symbol: str, augment: bool = False) -> List[np.ndarray]:
        """Render all variation combinations for a symbol."""
        sym_data = self.loader.get_symbol(symbol)
        if sym_data is None:
            return []
        
        stroke_names = sorted(sym_data.strokes.keys())
        variation_counts = [len(sym_data.strokes[name]) for name in stroke_names]
        
        # Generate all combinations
        from itertools import product
        all_indices = list(product(*[range(c) for c in variation_counts]))
        
        canvases = []
        for indices in all_indices:
            canvas = self.render_symbol(symbol, list(indices), augment=augment)
            canvases.append(canvas)
        
        return canvases
    
    def render_symbol_group(self,
                            symbols: List[str],
                            spacing: float = 0.15,
                            augment: bool = True,
                            variation_indices_per_symbol: Optional[List[List[int]]] = None,
                            return_metadata: bool = True
                            ) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]]:
        """
        Render multiple symbols side-by-side in a single canvas.
        
        This is useful for rendering edited regions where multiple symbols
        are written in one stroke sequence (e.g., "x+y" written together).
        
        Args:
            symbols: List of symbol characters to render ['a', '+', 'b']
            spacing: Spacing between symbols as fraction of symbol width (0.15 = 15%)
            augment: Apply random augmentation to each symbol
            variation_indices_per_symbol: Optional variation indices for each symbol
            return_metadata: Return per-symbol bounding box metadata
            
        Returns:
            If return_metadata=False: Canvas as numpy array [H, W]
            If return_metadata=True: Tuple of (canvas, metadata_list)
                where metadata_list contains per-symbol info:
                [{'symbol': 'a', 'bbox': [x0, y0, x1, y1], 'latex': '...'}, ...]
        """
        if not symbols:
            empty = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
            return (empty, []) if return_metadata else empty
        
        # Render each symbol to individual canvases
        individual_canvases = []
        individual_bboxes = []  # Tight bounding boxes for each symbol
        
        for i, sym in enumerate(symbols):
            var_indices = None
            if variation_indices_per_symbol and i < len(variation_indices_per_symbol):
                var_indices = variation_indices_per_symbol[i]
            
            canvas = self.render_symbol(sym, var_indices, augment=augment)
            individual_canvases.append(canvas)
            
            # Find tight bounding box of the rendered symbol
            # (where the actual strokes are, not the full canvas)
            rows = np.any(canvas > 0.1, axis=1)
            cols = np.any(canvas > 0.1, axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                individual_bboxes.append((x_min, y_min, x_max + 1, y_max + 1))
            else:
                # Empty symbol - use center region
                c = self.canvas_size // 2
                individual_bboxes.append((c - 10, c - 10, c + 10, c + 10))
        
        # Calculate layout - place symbols side by side with spacing
        # First, get the cropped size of each symbol
        cropped_widths = []
        cropped_heights = []
        for bbox in individual_bboxes:
            cropped_widths.append(bbox[2] - bbox[0])
            cropped_heights.append(bbox[3] - bbox[1])
        
        max_height = max(cropped_heights)
        spacing_px = int(spacing * np.mean(cropped_widths))
        
        # Calculate total width needed
        total_width = sum(cropped_widths) + spacing_px * (len(symbols) - 1)
        
        # Determine output canvas size (maintain aspect ratio, fit content)
        # Scale to fit in canvas_size while preserving aspect ratio
        scale = min(self.canvas_size / total_width, self.canvas_size / max_height) * 0.85
        
        output_width = int(total_width * scale)
        output_height = int(max_height * scale)
        
        # Create output canvas (centered)
        output = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        
        # Offset to center the group
        x_offset = (self.canvas_size - output_width) // 2
        y_offset = (self.canvas_size - output_height) // 2
        
        # Place each symbol
        metadata = []
        current_x = x_offset
        
        for i, (canvas, bbox, sym) in enumerate(zip(individual_canvases, individual_bboxes, symbols)):
            # Extract the symbol region from its canvas
            x0, y0, x1, y1 = bbox
            cropped = canvas[y0:y1, x0:x1]
            
            # Scale to fit
            new_w = int(cropped_widths[i] * scale)
            new_h = int(cropped_heights[i] * scale)
            
            if new_w > 0 and new_h > 0:
                # Resize using simple interpolation
                from PIL import Image
                if HAS_PIL:
                    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
                    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    resized = np.array(resized_img, dtype=np.float32) / 255.0
                else:
                    # Fallback: simple nearest-neighbor resize
                    resized = np.zeros((new_h, new_w), dtype=np.float32)
                    for py in range(new_h):
                        for px in range(new_w):
                            src_y = int(py / new_h * cropped.shape[0])
                            src_x = int(px / new_w * cropped.shape[1])
                            resized[py, px] = cropped[src_y, src_x]
                
                # Calculate placement position (vertically centered)
                place_y = y_offset + (output_height - new_h) // 2
                place_x = current_x
                
                # Ensure we don't go out of bounds
                end_y = min(place_y + new_h, self.canvas_size)
                end_x = min(place_x + new_w, self.canvas_size)
                
                # Calculate actual paste dimensions
                paste_h = end_y - place_y
                paste_w = end_x - place_x
                
                # Only paste if there's actually space
                if paste_h > 0 and paste_w > 0:
                    # Paste with max blending (in case of overlap)
                    output[place_y:end_y, place_x:end_x] = np.maximum(
                        output[place_y:end_y, place_x:end_x],
                        resized[:paste_h, :paste_w]
                    )
                
                # Record metadata (even if partially clipped)
                sym_data = self.loader.get_symbol(sym)
                metadata.append({
                    'symbol': sym,
                    'latex': sym_data.latex if sym_data else sym,
                    'bbox': [max(0, place_x), max(0, place_y), 
                            min(end_x, self.canvas_size), min(end_y, self.canvas_size)],
                    'index': i
                })
                
                current_x += new_w + spacing_px
            else:
                # Symbol too small, skip but record metadata
                sym_data = self.loader.get_symbol(sym)
                metadata.append({
                    'symbol': sym,
                    'latex': sym_data.latex if sym_data else sym,
                    'bbox': [current_x, y_offset, min(current_x + 10, self.canvas_size), 
                            min(y_offset + 10, self.canvas_size)],
                    'index': i
                })
                current_x += spacing_px
        
        if return_metadata:
            return output, metadata
        return output


# =============================================================================
# GPU Stroke Renderer
# =============================================================================

class GPUStrokeRenderer:
    """GPU-accelerated stroke renderer using PyTorch."""
    
    def __init__(self,
                 data_loader: Optional[StrokeDataLoader] = None,
                 canvas_size: int = 256,
                 augmentation: Optional[AugmentationConfig] = None,
                 device: str = 'cuda'):
        """
        Initialize GPU renderer.
        
        Args:
            data_loader: StrokeDataLoader instance
            canvas_size: Size of output canvas
            augmentation: Augmentation configuration
            device: PyTorch device ('cuda' or 'cpu')
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU rendering")
        
        self.loader = data_loader or StrokeDataLoader()
        self.canvas_size = canvas_size
        self.augmentation = augmentation or AugmentationConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def render_symbol(self,
                      symbol: str,
                      variation_indices: Optional[List[int]] = None,
                      augment: bool = True) -> torch.Tensor:
        """
        Render a single symbol on GPU.
        
        Args:
            symbol: Symbol character
            variation_indices: Variation indices per stroke
            augment: Apply augmentation
            
        Returns:
            Canvas tensor [1, H, W]
        """
        sym_data = self.loader.get_symbol(symbol)
        if sym_data is None:
            return torch.zeros(1, self.canvas_size, self.canvas_size, device=self.device)
        
        stroke_names = sorted(sym_data.strokes.keys())
        if variation_indices is None:
            variation_indices = [
                random.randint(0, len(sym_data.strokes[name]) - 1)
                for name in stroke_names
            ]
        
        aug_params = self.augmentation.sample() if augment else {
            'jitter': 0, 'rotation': 0, 'scale': 1.0,
            'thickness': 3.0, 'translate': (0, 0)
        }
        
        # Collect points
        all_points = []
        for i, stroke_name in enumerate(stroke_names):
            var_idx = variation_indices[i] if i < len(variation_indices) else 0
            var_idx = var_idx % len(sym_data.strokes[stroke_name])
            
            stroke = sym_data.strokes[stroke_name][var_idx]
            stroke = stroke.interpolate(self.augmentation.interpolate_points)
            stroke = stroke.apply_transform(
                rotation=aug_params['rotation'],
                scale=aug_params['scale'],
                translate=aug_params['translate']
            )
            if aug_params['jitter'] > 0:
                stroke = stroke.add_jitter(aug_params['jitter'])
            
            points = torch.tensor(
                [[p.x, p.y] for p in stroke.points],
                dtype=torch.float32, device=self.device
            )
            all_points.append(points)
        
        # Render using distance field
        canvas = self._render_distance_field(all_points, aug_params['thickness'])
        return canvas.unsqueeze(0)
    
    def render_batch(self,
                     symbols: List[str],
                     augment: bool = True) -> torch.Tensor:
        """
        Render batch of symbols.
        
        Args:
            symbols: List of symbol characters
            augment: Apply augmentation
            
        Returns:
            Batch tensor [B, 1, H, W]
        """
        canvases = []
        for symbol in symbols:
            canvas = self.render_symbol(symbol, augment=augment)
            canvases.append(canvas)
        
        return torch.stack(canvases, dim=0)
    
    def _render_distance_field(self, 
                               stroke_points: List[torch.Tensor],
                               thickness: float) -> torch.Tensor:
        """
        Render strokes using signed distance field for anti-aliasing.
        
        Args:
            stroke_points: List of point tensors per stroke [N_i, 2]
            thickness: Line thickness in pixels
            
        Returns:
            Canvas tensor [H, W]
        """
        # Create pixel grid
        y, x = torch.meshgrid(
            torch.linspace(0, 1, self.canvas_size, device=self.device),
            torch.linspace(0, 1, self.canvas_size, device=self.device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1)  # [H, W, 2]
        
        # Compute minimum distance to all stroke segments
        min_dist = torch.full((self.canvas_size, self.canvas_size), 
                             float('inf'), device=self.device)
        
        for points in stroke_points:
            if len(points) < 2:
                continue
            
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                dist = self._point_to_segment_distance(grid, p1, p2)
                min_dist = torch.minimum(min_dist, dist)
        
        # Convert distance to intensity (thickness in normalized coords)
        thickness_norm = thickness / self.canvas_size
        intensity = torch.clamp(1.0 - min_dist / thickness_norm, 0, 1)
        
        return intensity
    
    def _point_to_segment_distance(self,
                                   grid: torch.Tensor,
                                   p1: torch.Tensor,
                                   p2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from each grid point to line segment p1-p2.
        
        Args:
            grid: [H, W, 2] grid of points
            p1, p2: [2] segment endpoints
            
        Returns:
            Distance tensor [H, W]
        """
        v = p2 - p1  # Segment vector
        w = grid - p1  # Vector from p1 to each grid point
        
        # Project w onto v
        c1 = (w * v).sum(dim=-1)  # dot(w, v)
        c2 = (v * v).sum()  # dot(v, v)
        
        # Handle degenerate segment
        if c2 < 1e-10:
            return torch.norm(w, dim=-1)
        
        # Clamp projection to segment
        t = torch.clamp(c1 / c2, 0, 1)
        
        # Closest point on segment
        closest = p1 + t.unsqueeze(-1) * v
        
        # Distance from grid to closest point
        return torch.norm(grid - closest, dim=-1)


# =============================================================================
# Convenience Functions
# =============================================================================

# Global loader instance (lazy initialization)
_default_loader: Optional[StrokeDataLoader] = None
_default_renderer: Optional[StrokeRenderer] = None


def _get_default_loader() -> StrokeDataLoader:
    """Get or create default loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = StrokeDataLoader()
    return _default_loader


def _get_default_renderer() -> StrokeRenderer:
    """Get or create default renderer."""
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = StrokeRenderer(_get_default_loader())
    return _default_renderer


def render_symbol(symbol: str,
                  variation_indices: Optional[List[int]] = None,
                  augment: bool = True,
                  canvas_size: int = 256) -> np.ndarray:
    """
    Render a symbol to canvas (convenience function).
    
    Args:
        symbol: Symbol character
        variation_indices: Which variation for each stroke
        augment: Apply random augmentation
        canvas_size: Output canvas size
        
    Returns:
        Canvas as numpy array [H, W]
    """
    renderer = _get_default_renderer()
    if renderer.canvas_size != canvas_size:
        renderer = StrokeRenderer(_get_default_loader(), canvas_size=canvas_size)
    return renderer.render_symbol(symbol, variation_indices, augment)


def render_batch(symbols: List[str],
                 augment: bool = True,
                 canvas_size: int = 256,
                 device: str = 'cuda') -> 'torch.Tensor':
    """
    Render batch of symbols on GPU (convenience function).
    
    Args:
        symbols: List of symbol characters
        augment: Apply augmentation
        canvas_size: Canvas size
        device: PyTorch device
        
    Returns:
        Batch tensor [B, 1, H, W]
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for batch rendering")
    
    renderer = GPUStrokeRenderer(
        _get_default_loader(),
        canvas_size=canvas_size,
        device=device
    )
    return renderer.render_batch(symbols, augment)


def get_available_symbols() -> List[str]:
    """Get list of all available symbols."""
    return _get_default_loader().get_all_symbols()


def get_symbol_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Get information about a symbol."""
    sym_data = _get_default_loader().get_symbol(symbol)
    if sym_data is None:
        return None
    
    return {
        'symbol': sym_data.symbol,
        'latex': sym_data.latex,
        'num_strokes': sym_data.num_strokes,
        'total_variations': sym_data.total_variations,
        'strokes': {
            name: len(vars) for name, vars in sym_data.strokes.items()
        }
    }


def render_symbol_group(symbols: List[str],
                        spacing: float = 0.15,
                        augment: bool = True,
                        canvas_size: int = 256,
                        return_metadata: bool = True
                        ) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]]:
    """
    Render multiple symbols side-by-side (convenience function).
    
    Useful for rendering edited regions where multiple symbols are
    written together (e.g., "x+y" in one editing action).
    
    Args:
        symbols: List of symbol characters ['a', '+', 'b']
        spacing: Spacing between symbols (fraction of symbol width)
        augment: Apply random augmentation
        canvas_size: Output canvas size
        return_metadata: Return per-symbol bbox metadata
        
    Returns:
        If return_metadata=False: Canvas [H, W]
        If return_metadata=True: (canvas, metadata_list)
            metadata_list: [{'symbol': 'a', 'bbox': [x0,y0,x1,y1], 'latex': '...'}, ...]
    
    Example:
        >>> canvas, meta = render_symbol_group(['x', '+', 'y'])
        >>> print(meta[0])  # {'symbol': 'x', 'bbox': [...], 'latex': 'x', 'index': 0}
    """
    renderer = _get_default_renderer()
    if renderer.canvas_size != canvas_size:
        renderer = StrokeRenderer(_get_default_loader(), canvas_size=canvas_size)
    return renderer.render_symbol_group(symbols, spacing, augment, 
                                        return_metadata=return_metadata)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo and test the stroke renderer."""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("Stroke Renderer - Using Captured Stroke Data")
    print("=" * 60)
    
    # Load data
    loader = StrokeDataLoader()
    stats = loader.get_statistics()
    
    print(f"\nLoaded stroke data:")
    print(f"  Symbols: {stats['num_symbols']}")
    print(f"  Total strokes: {stats['total_strokes']}")
    print(f"  Total variations: {stats['total_variations']}")
    print(f"  Avg strokes/symbol: {stats['avg_strokes_per_symbol']:.1f}")
    print(f"  Avg variations/stroke: {stats['avg_variations_per_stroke']:.1f}")
    
    # Test rendering
    print("\n" + "-" * 40)
    print("Testing CPU Renderer")
    print("-" * 40)
    
    renderer = StrokeRenderer(loader, canvas_size=128)
    
    # Render some symbols
    test_symbols = ['A', 'x', '+', '0']
    for sym in test_symbols:
        info = get_symbol_info(sym)
        if info:
            canvas = renderer.render_symbol(sym, augment=True)
            print(f"  {sym} ({info['latex']}): {info['num_strokes']} strokes, "
                  f"{info['total_variations']} combinations, "
                  f"rendered {canvas.shape}")
        else:
            print(f"  {sym}: Not found in dataset")
    
    # Test GPU rendering if available
    if HAS_TORCH and torch.cuda.is_available():
        print("\n" + "-" * 40)
        print("Testing GPU Renderer")
        print("-" * 40)
        
        gpu_renderer = GPUStrokeRenderer(loader, canvas_size=128, device='cuda')
        batch = gpu_renderer.render_batch(test_symbols[:3], augment=True)
        print(f"  Batch rendered: {batch.shape}")
    else:
        print("\n  (GPU rendering skipped - CUDA not available)")
    
    print("\n" + "=" * 60)
    print("Renderer test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

