"""
Comprehensive Pipeline Test for LaTeX Edit Recognition

Generates demo outputs including:
1. Commutative diagrams (tikzcd)
2. Atomic samples at various depths with metadata (symbol bboxes)
3. Target symbols with complete vs incomplete strokes
4. Single and multi-symbol stroke rendering
5. Compositional augmentation with context
6. Positional embeddings from stroke center offset from LaTeX bbox
7. Distractor strokes (random incomplete strokes around target)

Output structure:
    demos/pipeline_test/
    ├── commutative_diagrams/     # tikzcd samples
    ├── atomic_samples/           # Math expressions at each depth
    ├── stroke_progression/       # Complete vs incomplete strokes
    ├── multi_symbol/             # Multiple symbols in one edit
    ├── compositional/            # Context composition examples
    ├── positional_embedding/     # Stroke position vs LaTeX bbox
    ├── distractor_strokes/       # Target with noise strokes
    └── metadata.json             # All metadata
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Setup path
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DATA_DIR))

# Imports
from synthetic_case_generator import CaseGenerator, analyze_latex_complexity
from augmentation import ChunkPool, ContextComposer
from scripts.stroke_dataset import StrokeDatasetGenerator
from scripts.stroke_renderer import StrokeDataLoader, StrokeRenderer, AugmentationConfig

# Optional imports
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available, some visualizations disabled")

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("Warning: PyMuPDF not available, PDF extraction disabled")


# =============================================================================
# Stroke Center and Position Embedding
# =============================================================================

@dataclass
class StrokePositionInfo:
    """Position information for rendered strokes."""
    symbol: str
    latex: str
    
    # Stroke-based position (from handwriting)
    stroke_center: Tuple[float, float]  # (x, y) normalized 0-1
    stroke_bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    
    # LaTeX-based position (from rendering)
    latex_center: Optional[Tuple[float, float]] = None
    latex_bbox: Optional[Tuple[float, float, float, float]] = None
    
    # Offset from LaTeX to stroke (for positional embedding)
    position_offset: Optional[Tuple[float, float]] = None
    
    # Metadata
    num_strokes: int = 1
    strokes_drawn: int = 1
    is_complete: bool = True
    variation_indices: List[int] = field(default_factory=list)


def calculate_stroke_center(stroke_points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate center of mass for stroke points."""
    if not stroke_points:
        return (0.5, 0.5)
    
    x_coords = [p[0] for p in stroke_points]
    y_coords = [p[1] for p in stroke_points]
    
    return (np.mean(x_coords), np.mean(y_coords))


def calculate_stroke_bbox(stroke_points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Calculate bounding box for stroke points."""
    if not stroke_points:
        return (0.4, 0.4, 0.6, 0.6)
    
    x_coords = [p[0] for p in stroke_points]
    y_coords = [p[1] for p in stroke_points]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


def get_strokes_with_position(
    loader: StrokeDataLoader,
    symbol: str,
    num_strokes: Optional[int] = None,
    canvas_size: int = 256,
    offset: Tuple[float, float] = (0, 0)
) -> Tuple[np.ndarray, StrokePositionInfo]:
    """
    Render strokes and calculate position embedding.
    
    Args:
        loader: Stroke data loader
        symbol: Symbol to render
        num_strokes: Number of strokes (None = all = complete)
        canvas_size: Output canvas size
        offset: Position offset to apply (simulating handwriting position)
        
    Returns:
        (canvas, position_info)
    """
    sym_data = loader.get_symbol(symbol)
    if sym_data is None:
        # Return empty
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        return canvas, StrokePositionInfo(
            symbol=symbol, latex=symbol,
            stroke_center=(0.5, 0.5), stroke_bbox=(0.4, 0.4, 0.6, 0.6),
            is_complete=False
        )
    
    stroke_names = sorted(sym_data.strokes.keys())
    total_strokes = len(stroke_names)
    
    if num_strokes is None:
        num_strokes = total_strokes
    num_strokes = min(num_strokes, total_strokes)
    
    # Select random variations
    variation_indices = []
    all_points = []
    
    for i, stroke_name in enumerate(stroke_names[:num_strokes]):
        variations = sym_data.strokes[stroke_name]
        var_idx = random.randint(0, len(variations) - 1)
        variation_indices.append(var_idx)
        
        stroke = variations[var_idx]
        for p in stroke.points:
            # Apply offset
            all_points.append((p.x + offset[0], p.y + offset[1]))
    
    # Calculate position info
    stroke_center = calculate_stroke_center(all_points)
    stroke_bbox = calculate_stroke_bbox(all_points)
    
    # Render to canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    thickness = random.uniform(2, 4)
    
    for i, stroke_name in enumerate(stroke_names[:num_strokes]):
        stroke = sym_data.strokes[stroke_name][variation_indices[i]]
        
        # Interpolate for smooth rendering
        interpolated = stroke.interpolate(50)
        points = [(int((p.x + offset[0]) * canvas_size), 
                   int((p.y + offset[1]) * canvas_size)) 
                  for p in interpolated.points]
        
        # Draw stroke on canvas
        for j in range(len(points) - 1):
            x0, y0 = points[j]
            x1, y1 = points[j + 1]
            _draw_line(canvas, x0, y0, x1, y1, thickness)
    
    position_info = StrokePositionInfo(
        symbol=sym_data.symbol,
        latex=sym_data.latex,
        stroke_center=stroke_center,
        stroke_bbox=stroke_bbox,
        num_strokes=total_strokes,
        strokes_drawn=num_strokes,
        is_complete=(num_strokes >= total_strokes),
        variation_indices=variation_indices
    )
    
    return canvas, position_info


def _draw_line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: float):
    """Draw anti-aliased line on canvas using Bresenham."""
    h, w = canvas.shape
    
    # Simple line drawing
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steps = max(dx, dy, 1)
    
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        x = int(x0 + t * (x1 - x0))
        y = int(y0 + t * (y1 - y0))
        
        # Draw with thickness
        r = int(thickness / 2)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px, py = x + dx, y + dy
                if 0 <= px < w and 0 <= py < h:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= thickness / 2:
                        canvas[py, px] = min(1.0, canvas[py, px] + (1 - dist / (thickness / 2 + 0.5)))


def add_distractor_strokes(
    canvas: np.ndarray,
    loader: StrokeDataLoader,
    target_center: Tuple[float, float],
    num_distractors: int = 3,
    max_strokes_per_distractor: int = 2,
    distance_range: Tuple[float, float] = (0.15, 0.35)
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Add random incomplete strokes around the target symbol.
    
    Args:
        canvas: Existing canvas with target symbol
        loader: Stroke data loader
        target_center: Center of target symbol (normalized 0-1)
        num_distractors: Number of distractor symbols to add
        max_strokes_per_distractor: Max strokes per distractor (always incomplete)
        distance_range: Distance from target center (normalized)
        
    Returns:
        (modified_canvas, distractor_metadata)
    """
    canvas = canvas.copy()
    h, w = canvas.shape
    available_symbols = loader.get_all_symbols()
    distractor_info = []
    
    for i in range(num_distractors):
        # Pick random symbol
        symbol = random.choice(available_symbols)
        sym_data = loader.get_symbol(symbol)
        if sym_data is None or sym_data.num_strokes < 1:
            continue
        
        # Random position around target
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(*distance_range)
        offset_x = target_center[0] + distance * np.cos(angle) - 0.5
        offset_y = target_center[1] + distance * np.sin(angle) - 0.5
        
        # Clamp to valid range
        offset_x = max(-0.3, min(0.3, offset_x))
        offset_y = max(-0.3, min(0.3, offset_y))
        
        # Draw partial strokes (incomplete)
        num_strokes = random.randint(1, min(max_strokes_per_distractor, sym_data.num_strokes - 1))
        if num_strokes == sym_data.num_strokes:
            num_strokes = max(1, num_strokes - 1)  # Ensure incomplete
        
        stroke_names = sorted(sym_data.strokes.keys())[:num_strokes]
        all_points = []
        
        for stroke_name in stroke_names:
            variations = sym_data.strokes[stroke_name]
            stroke = random.choice(variations)
            
            for p in stroke.interpolate(30).points:
                px = int((p.x + offset_x) * w)
                py = int((p.y + offset_y) * h)
                
                if 0 <= px < w and 0 <= py < h:
                    all_points.append((px, py))
        
        # Draw with thin lines (distractor style)
        for j in range(len(all_points) - 1):
            x0, y0 = all_points[j]
            x1, y1 = all_points[j + 1]
            _draw_line(canvas, x0, y0, x1, y1, thickness=1.5)
        
        distractor_info.append({
            'symbol': symbol,
            'latex': sym_data.latex,
            'offset': (offset_x, offset_y),
            'strokes_drawn': num_strokes,
            'total_strokes': sym_data.num_strokes,
            'is_complete': False,
            'type': 'distractor'
        })
    
    return canvas, distractor_info


# =============================================================================
# LaTeX Rendering Helpers
# =============================================================================

def render_latex_to_image(latex: str, output_path: str, dpi: int = 150) -> Optional[Dict]:
    """
    Render LaTeX to image and extract symbol bboxes.
    
    Returns:
        Dict with image_path, symbols, and metadata
    """
    if not HAS_PIL:
        return None
    
    try:
        import subprocess
        import tempfile
        
        # Create LaTeX document
        doc = f"""\\documentclass[preview,border=5pt]{{standalone}}
\\usepackage{{amsmath,amssymb,tikz-cd}}
\\begin{{document}}
${latex}$
\\end{{document}}
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, 'expr.tex')
            pdf_path = os.path.join(tmpdir, 'expr.pdf')
            
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(doc)
            
            # Compile LaTeX
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_path],
                capture_output=True, timeout=30
            )
            
            if not os.path.exists(pdf_path):
                return None
            
            # Convert PDF to image
            if HAS_FITZ:
                doc = fitz.open(pdf_path)
                page = doc[0]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Extract symbol bboxes
                symbols = []
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                bbox = span["bbox"]
                                # Scale to image coords
                                scale = dpi / 72
                                symbols.append({
                                    'text': span["text"],
                                    'bbox': [b * scale for b in bbox],
                                    'center': ((bbox[0]+bbox[2])/2 * scale, (bbox[1]+bbox[3])/2 * scale)
                                })
                doc.close()
            else:
                # Fallback: use pdf2image
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(pdf_path, dpi=dpi)
                    img = images[0]
                    symbols = []
                except ImportError:
                    return None
            
            # Save image
            img.save(output_path)
            
            return {
                'image_path': output_path,
                'latex': latex,
                'symbols': symbols,
                'width': img.width,
                'height': img.height
            }
            
    except Exception as e:
        print(f"Error rendering LaTeX: {e}")
        return None


# =============================================================================
# Pipeline Test Generator
# =============================================================================

class PipelineTestGenerator:
    """Generate comprehensive pipeline test outputs."""
    
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize generators
        self.case_gen = CaseGenerator(seed=seed)
        self.stroke_gen = StrokeDatasetGenerator(seed=seed)
        self.loader = self.stroke_gen.loader
        
        # Metadata collection
        self.metadata = {
            'generated_at': datetime.now().isoformat(),
            'seed': seed,
            'tests': {}
        }
    
    def generate_all(self):
        """Generate all test outputs."""
        print("=" * 60)
        print("Pipeline Test Generator")
        print("=" * 60)
        
        self.generate_commutative_diagrams()
        self.generate_atomic_samples()
        self.generate_stroke_progression()
        self.generate_multi_symbol()
        self.generate_compositional()
        self.generate_positional_embedding()
        self.generate_distractor_strokes()
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print()
        print("=" * 60)
        print(f"Pipeline test outputs saved to: {self.output_dir}")
        print("=" * 60)
    
    def generate_commutative_diagrams(self):
        """Generate commutative diagram samples (tikzcd)."""
        print("\n1. Commutative Diagrams")
        print("-" * 40)
        
        out_dir = self.output_dir / 'commutative_diagrams'
        out_dir.mkdir(exist_ok=True)
        
        # Sample tikzcd expressions
        diagrams = [
            # Simple arrow
            r"\begin{tikzcd} A \arrow[r] & B \end{tikzcd}",
            # Triangle
            r"\begin{tikzcd} A \arrow[r] \arrow[dr] & B \arrow[d] \\ & C \end{tikzcd}",
            # Square
            r"\begin{tikzcd} A \arrow[r] \arrow[d] & B \arrow[d] \\ C \arrow[r] & D \end{tikzcd}",
            # With labels
            r"\begin{tikzcd} X \arrow[r, \"f\"] & Y \arrow[r, \"g\"] & Z \end{tikzcd}",
            # Complex
            r"\begin{tikzcd} A \arrow[r, \"f\"] \arrow[d, \"h\"'] & B \arrow[d, \"k\"] \\ C \arrow[r, \"g\"'] & D \end{tikzcd}",
        ]
        
        diagram_data = []
        for i, diag in enumerate(diagrams):
            # Try to render
            img_path = out_dir / f'diagram_{i:02d}.png'
            result = render_latex_to_image(diag, str(img_path))
            
            if result:
                print(f"  Generated: diagram_{i:02d}.png")
                diagram_data.append({
                    'id': f'diagram_{i:02d}',
                    'latex': diag,
                    'image_path': str(img_path.relative_to(self.output_dir)),
                    'symbols': result.get('symbols', []),
                    'size': (result.get('width'), result.get('height'))
                })
            else:
                print(f"  [SKIP] diagram_{i:02d} (render failed)")
        
        self.metadata['tests']['commutative_diagrams'] = diagram_data
        print(f"  Total: {len(diagram_data)} diagrams")
    
    def generate_atomic_samples(self):
        """Generate atomic samples at various depths with symbol bboxes."""
        print("\n2. Atomic Samples (by depth)")
        print("-" * 40)
        
        out_dir = self.output_dir / 'atomic_samples'
        out_dir.mkdir(exist_ok=True)
        
        atomic_data = []
        
        for depth in [1, 2, 3, 4]:
            depth_dir = out_dir / f'depth_{depth}'
            depth_dir.mkdir(exist_ok=True)
            
            # Generate cases at this depth
            cases = self.case_gen.generate_at_depth(target_depth=depth, count=5)
            
            for i, case in enumerate(cases):
                case_id = f'd{depth}_{i:02d}'
                
                # Render before expression
                img_path = depth_dir / f'{case_id}_before.png'
                result = render_latex_to_image(case.before_latex, str(img_path))
                
                # Analyze complexity
                complexity = analyze_latex_complexity(case.after_latex)
                
                sample_data = {
                    'id': case_id,
                    'depth': depth,
                    'operation': case.operation,
                    'before_latex': case.before_latex,
                    'after_latex': case.after_latex,
                    'category': case.category,
                    'complexity': {
                        'depth': complexity.depth,
                        'num_fractions': complexity.num_fractions,
                        'num_scripts': complexity.num_scripts,
                        'total_tokens': complexity.total_tokens
                    }
                }
                
                if result:
                    sample_data['image_path'] = str(img_path.relative_to(self.output_dir))
                    sample_data['symbols_bbox'] = result.get('symbols', [])
                
                atomic_data.append(sample_data)
            
            print(f"  Depth {depth}: {len(cases)} samples")
        
        self.metadata['tests']['atomic_samples'] = atomic_data
    
    def generate_stroke_progression(self):
        """Generate complete vs incomplete stroke examples."""
        print("\n3. Stroke Progression (complete vs incomplete)")
        print("-" * 40)
        
        out_dir = self.output_dir / 'stroke_progression'
        out_dir.mkdir(exist_ok=True)
        
        progression_data = []
        
        # Find symbols with multiple strokes
        multi_stroke_symbols = []
        for sym in self.loader.get_all_symbols()[:50]:
            sym_data = self.loader.get_symbol(sym)
            if sym_data and sym_data.num_strokes >= 2:
                multi_stroke_symbols.append((sym, sym_data.num_strokes))
        
        # Generate progressions for top 5
        for sym, total_strokes in multi_stroke_symbols[:5]:
            sym_dir = out_dir / f'symbol_{sym.replace("/", "_")}'
            sym_dir.mkdir(exist_ok=True)
            
            sym_data = self.loader.get_symbol(sym)
            
            for num_strokes in range(1, total_strokes + 1):
                is_complete = (num_strokes >= total_strokes)
                canvas, pos_info = get_strokes_with_position(
                    self.loader, sym, num_strokes=num_strokes
                )
                
                # Save canvas as image
                img_path = sym_dir / f'strokes_{num_strokes}of{total_strokes}.png'
                if HAS_PIL:
                    img = Image.fromarray((canvas * 255).astype(np.uint8), mode='L')
                    img.save(img_path)
                
                status = 'COMPLETE' if is_complete else 'INCOMPLETE'
                expected_output = sym_data.latex if is_complete else '<WAIT>'
                
                progression_data.append({
                    'symbol': sym,
                    'latex': sym_data.latex,
                    'strokes_drawn': num_strokes,
                    'total_strokes': total_strokes,
                    'is_complete': is_complete,
                    'expected_output': expected_output,
                    'image_path': str(img_path.relative_to(self.output_dir)),
                    'stroke_center': pos_info.stroke_center,
                    'stroke_bbox': pos_info.stroke_bbox
                })
            
            print(f"  Symbol '{sym}': {total_strokes} stroke stages")
        
        self.metadata['tests']['stroke_progression'] = progression_data
    
    def generate_multi_symbol(self):
        """Generate multi-symbol stroke examples (user writes multiple at once)."""
        print("\n4. Multi-Symbol Strokes")
        print("-" * 40)
        
        out_dir = self.output_dir / 'multi_symbol'
        out_dir.mkdir(exist_ok=True)
        
        multi_data = []
        
        # Symbol groups to render together
        symbol_groups = [
            ['x', '+', 'y'],
            ['a', 'b', 'c'],
            ['1', '2', '3'],
        ]
        
        available = set(self.loader.get_all_symbols())
        
        for group in symbol_groups:
            # Filter to available symbols
            group = [s for s in group if s in available]
            if len(group) < 2:
                continue
            
            group_id = '_'.join(group)
            
            # Render group with metadata
            canvas, metadata = self.stroke_gen.renderer.render_symbol_group(
                group, augment=True, return_metadata=True
            )
            
            # Save image
            img_path = out_dir / f'group_{group_id}.png'
            if HAS_PIL:
                img = Image.fromarray((canvas * 255).astype(np.uint8), mode='L')
                img.save(img_path)
            
            multi_data.append({
                'id': group_id,
                'symbols': group,
                'image_path': str(img_path.relative_to(self.output_dir)),
                'symbol_bboxes': metadata,
                'target_latex': ' '.join([m.get('latex', s) for s, m in zip(group, metadata)])
            })
            
            print(f"  Group: {group}")
        
        self.metadata['tests']['multi_symbol'] = multi_data
    
    def generate_compositional(self):
        """Generate compositional augmentation examples with context."""
        print("\n5. Compositional Augmentation")
        print("-" * 40)
        
        out_dir = self.output_dir / 'compositional'
        out_dir.mkdir(exist_ok=True)
        
        comp_data = []
        
        # Build chunk pool
        pool = ChunkPool.from_case_generator(self.case_gen, chunks_per_depth=20, max_depth=4)
        composer = ContextComposer(pool)
        
        # Generate composed examples
        examples = composer.compose_batch(num_examples=10, target_depth=2)
        
        for i, ex in enumerate(examples):
            ex_id = f'comp_{i:03d}'
            
            # Render full context
            img_path = out_dir / f'{ex_id}_context.png'
            result = render_latex_to_image(ex.full_context_before, str(img_path))
            
            comp_item = {
                'id': ex_id,
                'full_context_before': ex.full_context_before,
                'full_context_after': ex.full_context_after,
                'edit_start_pos': ex.edit_start_pos,
                'edit_end_pos': ex.edit_end_pos,
                'edit_old_content': ex.edit_old_content,
                'edit_new_content': ex.edit_new_content,
                'context_depths': ex.context_depths,
                'target_depth': ex.target_depth,
                'num_context_chunks': len(ex.context_chunks)
            }
            
            if result:
                comp_item['image_path'] = str(img_path.relative_to(self.output_dir))
                comp_item['symbols_bbox'] = result.get('symbols', [])
            
            comp_data.append(comp_item)
        
        print(f"  Generated: {len(comp_data)} composed examples")
        self.metadata['tests']['compositional'] = comp_data
    
    def generate_positional_embedding(self):
        """Generate examples showing stroke position vs LaTeX bbox position."""
        print("\n6. Positional Embedding (stroke center vs LaTeX bbox)")
        print("-" * 40)
        
        out_dir = self.output_dir / 'positional_embedding'
        out_dir.mkdir(exist_ok=True)
        
        pos_data = []
        
        # Test symbols
        test_symbols = ['x', 'y', '+', '=']
        available = set(self.loader.get_all_symbols())
        test_symbols = [s for s in test_symbols if s in available]
        
        for sym in test_symbols[:4]:
            sym_data = self.loader.get_symbol(sym)
            if not sym_data:
                continue
            
            # Generate multiple variations with different offsets
            for var_idx in range(3):
                # Random offset simulating handwriting position variation
                offset = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
                
                canvas, pos_info = get_strokes_with_position(
                    self.loader, sym, offset=offset
                )
                
                # Simulate LaTeX bbox (centered, normalized)
                latex_center = (0.5, 0.5)
                latex_bbox = (0.3, 0.3, 0.7, 0.7)
                
                # Calculate position offset (what model should learn)
                position_offset = (
                    pos_info.stroke_center[0] - latex_center[0],
                    pos_info.stroke_center[1] - latex_center[1]
                )
                
                # Save image
                img_id = f'{sym}_var{var_idx}'
                img_path = out_dir / f'{img_id}.png'
                if HAS_PIL:
                    img = Image.fromarray((canvas * 255).astype(np.uint8), mode='L')
                    img.save(img_path)
                
                pos_data.append({
                    'id': img_id,
                    'symbol': sym,
                    'latex': sym_data.latex,
                    'image_path': str(img_path.relative_to(self.output_dir)),
                    'stroke_center': pos_info.stroke_center,
                    'stroke_bbox': pos_info.stroke_bbox,
                    'latex_center': latex_center,
                    'latex_bbox': latex_bbox,
                    'position_offset': position_offset,
                    'applied_offset': offset,
                    'variation_indices': pos_info.variation_indices
                })
            
            print(f"  Symbol '{sym}': 3 position variations")
        
        self.metadata['tests']['positional_embedding'] = pos_data
    
    def generate_distractor_strokes(self):
        """Generate examples with target symbol + distractor noise strokes."""
        print("\n7. Distractor Strokes (target + noise)")
        print("-" * 40)
        
        out_dir = self.output_dir / 'distractor_strokes'
        out_dir.mkdir(exist_ok=True)
        
        distractor_data = []
        
        # Test cases: target symbol with distractors around it
        test_targets = ['x', 'y', '+']
        available = set(self.loader.get_all_symbols())
        test_targets = [s for s in test_targets if s in available]
        
        for target in test_targets[:3]:
            target_data = self.loader.get_symbol(target)
            if not target_data:
                continue
            
            for case_idx in range(2):
                # Render complete target
                canvas, pos_info = get_strokes_with_position(
                    self.loader, target, num_strokes=None  # Complete
                )
                
                # Add distractor strokes
                num_distractors = random.randint(2, 4)
                canvas_with_distractors, distractor_info = add_distractor_strokes(
                    canvas, self.loader, pos_info.stroke_center,
                    num_distractors=num_distractors
                )
                
                # Save both versions
                case_id = f'{target}_case{case_idx}'
                
                # Clean version (target only)
                clean_path = out_dir / f'{case_id}_clean.png'
                if HAS_PIL:
                    img = Image.fromarray((canvas * 255).astype(np.uint8), mode='L')
                    img.save(clean_path)
                
                # Noisy version (with distractors)
                noisy_path = out_dir / f'{case_id}_noisy.png'
                if HAS_PIL:
                    img = Image.fromarray((canvas_with_distractors * 255).astype(np.uint8), mode='L')
                    img.save(noisy_path)
                
                distractor_data.append({
                    'id': case_id,
                    'target_symbol': target,
                    'target_latex': target_data.latex,
                    'target_complete': True,
                    'expected_output': target_data.latex,
                    'clean_image': str(clean_path.relative_to(self.output_dir)),
                    'noisy_image': str(noisy_path.relative_to(self.output_dir)),
                    'target_bbox': pos_info.stroke_bbox,
                    'target_center': pos_info.stroke_center,
                    'distractors': distractor_info,
                    'num_distractors': len(distractor_info)
                })
            
            print(f"  Target '{target}': 2 cases with distractors")
        
        self.metadata['tests']['distractor_strokes'] = distractor_data


# =============================================================================
# Main
# =============================================================================

def main():
    """Run pipeline test generation."""
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    output_dir = SCRIPT_DIR / 'pipeline_test'
    
    generator = PipelineTestGenerator(str(output_dir), seed=42)
    generator.generate_all()


if __name__ == '__main__':
    main()

