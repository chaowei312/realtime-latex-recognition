"""
Demo: Visualize Full Context with Stroke Replacement

Shows:
1. Full composed context (LaTeX + commutative diagram)
2. Edit symbol replaced with handwriting strokes
3. Red dots showing position jittering (where model sees each symbol)

Uses only the interface code from generate_training_data.py
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

# Setup
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DATA_DIR))

# Interface imports only
from generate_training_data import (
    generate_atomic_cases,
    generate_commutative_diagrams,
    generate_stroke_examples,
    CaseGenerator,
)
from scripts.stroke_renderer import StrokeDataLoader, StrokeRenderer, AugmentationConfig
from scripts.stroke_dataset import StrokeDatasetGenerator

try:
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Need: PIL, matplotlib")

try:
    import subprocess
    import tempfile
    import fitz  # PyMuPDF
    HAS_RENDER = True
except ImportError:
    HAS_RENDER = False
    print("Need: PyMuPDF for LaTeX rendering")


def _compute_ink_bbox(page, span_bbox, scale: int = 10) -> tuple:
    """
    Compute actual ink bounding box by rendering and finding non-white pixels.
    
    PyMuPDF span_bbox includes font ascender/descender, which is much larger
    than the actual visible ink. This computes the tight bbox around visible pixels.
    
    IMPORTANT: Clip exactly to span_bbox (no padding) to avoid picking up
    pixels from neighboring characters in dense expressions like x^2.
    """
    try:
        import numpy as np
        
        # Clip EXACTLY to span_bbox - no padding to avoid neighboring chars
        clip = fitz.Rect(
            max(0, span_bbox[0]),
            max(0, span_bbox[1]),
            min(page.rect.width, span_bbox[2]),
            min(page.rect.height, span_bbox[3])
        )
        
        if clip.is_empty or clip.width < 1 or clip.height < 1:
            return span_bbox
        
        # Render at high resolution
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        
        # Find non-white pixels (ink)
        gray = arr.mean(axis=2)
        ink_pixels = np.where(gray < 250)
        
        if len(ink_pixels[0]) == 0:
            return span_bbox
        
        y_min, y_max = ink_pixels[0].min(), ink_pixels[0].max()
        x_min, x_max = ink_pixels[1].min(), ink_pixels[1].max()
        
        # Convert back to PDF points
        return (
            clip.x0 + x_min / scale,
            clip.y0 + y_min / scale,
            clip.x0 + (x_max + 1) / scale,
            clip.y0 + (y_max + 1) / scale,
        )
    except Exception:
        return span_bbox


def render_latex_to_pil(latex: str, dpi: int = 150) -> tuple:
    """
    Render LaTeX to PIL Image and extract symbol positions.
    
    Returns:
        (PIL.Image, list of symbol dicts with bbox)
    """
    if not HAS_RENDER:
        return None, []
    
    doc_template = r"""
\documentclass[preview,border=10pt]{standalone}
\usepackage{amsmath,amssymb,tikz-cd}
\begin{document}
$%s$
\end{document}
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, 'expr.tex')
        pdf_path = os.path.join(tmpdir, 'expr.pdf')
        
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(doc_template % latex)
        
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_path],
                capture_output=True, timeout=30
            )
        except Exception as e:
            print(f"LaTeX error: {e}")
            return None, []
        
        if not os.path.exists(pdf_path):
            return None, []
        
        # Convert to image and extract symbols
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Render to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Extract symbol positions with accurate ink bboxes
        symbols = []
        scale = dpi / 72
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_bbox = span["bbox"]
                        text = span["text"].strip()
                        
                        # Compute ink bbox for single characters (more accurate)
                        if len(text) == 1:
                            ink_bbox = _compute_ink_bbox(page, span_bbox, scale=10)
                        else:
                            ink_bbox = span_bbox
                        
                        # Scale to image coordinates
                        bbox = [b * scale for b in ink_bbox]
                        
                        symbols.append({
                            'text': span["text"],
                            'bbox': bbox,
                            'span_bbox': [b * scale for b in span_bbox],
                            'center': ((ink_bbox[0]+ink_bbox[2])/2 * scale, 
                                      (ink_bbox[1]+ink_bbox[3])/2 * scale),
                        })
        
        doc.close()
        return img, symbols


def render_strokes_to_array(
    loader: StrokeDataLoader,
    symbol: str,
    size: tuple = (64, 64),
    num_strokes: int = None,  # None = complete
) -> np.ndarray:
    """
    Render strokes to numpy array using the core StrokeRenderer.
    
    This is a thin wrapper around StrokeRenderer.render_to_size().
    Uses anti-aliased PIL rendering for smooth strokes.
    
    Args:
        loader: StrokeDataLoader instance
        symbol: Symbol character to render
        size: (width, height) target size
        num_strokes: Number of strokes (None = complete symbol)
        
    Returns:
        numpy array [H, W] with values 0-255 (uint8)
    """
    # Use the core renderer's render_to_size method
    renderer = StrokeRenderer(loader, margin=0.15)
    return renderer.render_to_size(
        symbol=symbol,
        target_size=size,
        num_strokes=num_strokes,
        augment=True
    )


def compose_context_with_diagram(seed: int = 42) -> dict:
    """
    Compose a context that includes:
    - At least one commutative diagram
    - Additional atomic LaTeX expressions
    - Edit target info
    """
    random.seed(seed)
    
    # Generate components
    diagrams = generate_commutative_diagrams(count=3, complexity_range=(2, 3), seed=seed)
    cases = generate_atomic_cases(min_depth=1, max_depth=3, count=5, seed=seed)
    
    # Pick one diagram and one atomic case
    diagram = diagrams[0]
    atomic = random.choice(cases[:3])
    
    # Compose context (no "," - these are all atomic, just space separated for display)
    context_parts = [
        atomic.before_latex,
        diagram['latex'],
    ]
    
    # Full context
    full_context = " \\quad ".join(context_parts)
    
    # Edit info - get actual symbol from metadata (NOT string diff which includes LaTeX syntax)
    edit_target = None
    if hasattr(atomic, 'metadata') and atomic.metadata:
        edit_target = atomic.metadata.get('edit_symbol') or atomic.metadata.get('edit_symbols', [None])[0]
    
    # Fallback: try to extract a writable symbol from the diff
    if not edit_target:
        import re
        diff = atomic.after_latex.replace(atomic.before_latex, "").strip()
        
        # Look for Greek letters first (they're the most likely edit in math)
        greek_match = re.search(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega|iota|'
                                r'Gamma|Delta|Theta|Lambda|Sigma|Phi|Psi|Omega|Pi)', diff)
        if greek_match:
            # Map to Unicode for display
            greek_map = {
                'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
                'theta': 'θ', 'lambda': 'λ', 'mu': 'μ', 'sigma': 'σ', 'phi': 'φ',
                'psi': 'ψ', 'omega': 'ω', 'iota': 'ι', 'Gamma': 'Γ', 'Delta': 'Δ',
                'Theta': 'Θ', 'Lambda': 'Λ', 'Sigma': 'Σ', 'Phi': 'Φ', 'Psi': 'Ψ',
                'Omega': 'Ω', 'Pi': 'Π'
            }
            edit_target = greek_map.get(greek_match.group(1), greek_match.group(1))
        else:
            # Look for single letters/digits (actual writable symbols)
            char_match = re.search(r'(?<![a-zA-Z])([a-zA-Z0-9])(?![a-zA-Z])', diff)
            if char_match:
                edit_target = char_match.group(1)
    
    if not edit_target:
        edit_target = "x"  # Final fallback
    
    # Symbol positions with jitter (small - within half bbox width)
    positions = []
    jitter_range = 0.015  # ~1.5% - stays well within symbol bounds
    
    # Atomic part position
    positions.append({
        'latex': atomic.before_latex,
        'type': 'atomic',
        'latex_center': (0.15, 0.5),
        'jitter': (random.uniform(-jitter_range, jitter_range), 
                   random.uniform(-jitter_range, jitter_range)),
    })
    
    # Diagram node positions
    for node_pos in diagram['node_positions']:
        # Shift to right side of context
        shifted_center = (0.5 + node_pos['latex_center'][0] * 0.4, node_pos['latex_center'][1])
        positions.append({
            'latex': node_pos['symbol'],
            'type': 'diagram_node',
            'latex_center': shifted_center,
            'jitter': node_pos['position_offset'],
        })
    
    return {
        'full_context': full_context,
        'atomic_part': atomic.before_latex,
        'diagram_part': diagram['latex'],
        'diagram_nodes': diagram['nodes'],
        'edit_target': edit_target,
        'edit_operation': atomic.operation,
        'positions': positions,
    }


def visualize_context_with_strokes(
    context_data: dict,
    output_path: str,
    show: bool = True,
    seed: int = 42,
):
    """
    Visualize full context with:
    1. Rendered LaTeX
    2. Edit symbol replaced with handwriting strokes
    3. Red dots showing position jittering
    """
    random.seed(seed)
    
    if not HAS_DEPS:
        print("Missing dependencies")
        return
    
    # Render the LaTeX context
    print(f"Rendering: {context_data['full_context'][:60]}...")
    latex_img, symbols = render_latex_to_pil(context_data['full_context'], dpi=150)
    
    if latex_img is None:
        print("LaTeX rendering failed, creating placeholder")
        latex_img = Image.new('RGB', (800, 200), 'white')
        symbols = []
    
    # Load stroke renderer
    loader = StrokeDataLoader()
    available_symbols = set(loader.get_all_symbols())
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # === Panel 1: Original LaTeX ===
    ax1 = axes[0, 0]
    ax1.imshow(latex_img)
    ax1.set_title("1. Original LaTeX Context", fontsize=12)
    ax1.axis('off')
    
    # === Panel 2: With position jitter dots ===
    ax2 = axes[0, 1]
    ax2.imshow(latex_img)
    ax2.set_title("2. Position Jittering (red = where model sees)", fontsize=12)
    
    # Draw jitter dots on symbols
    img_w, img_h = latex_img.size
    for sym in symbols:
        cx, cy = sym['center']
        # Small jitter (within ~1.5% of image, stays inside symbol)
        bbox = sym.get('bbox', [cx-10, cy-10, cx+10, cy+10])
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        max_jitter = min(bbox_w, bbox_h) * 0.3  # Max 30% of smaller bbox dimension
        jx = random.uniform(-max_jitter, max_jitter)
        jy = random.uniform(-max_jitter, max_jitter)
        
        # Original position (blue)
        ax2.plot(cx, cy, 'bo', markersize=6, alpha=0.5)
        # Jittered position (red) - where model sees it
        ax2.plot(cx + jx, cy + jy, 'ro', markersize=8)
        # Arrow showing offset
        ax2.annotate('', xy=(cx + jx, cy + jy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.7))
    
    ax2.axis('off')
    
    # === Panel 3: Edit target replaced with strokes ===
    ax3 = axes[1, 0]
    
    # Make a copy and overlay strokes
    img_with_strokes = latex_img.copy()
    draw = ImageDraw.Draw(img_with_strokes)
    
    # Find a symbol to replace with strokes
    # The edit_target should be a writable symbol (letter, Greek, digit) - NOT LaTeX syntax
    edit_symbol = context_data['edit_target']
    
    # Find this symbol (or a suitable one) in rendered image that we can replace with strokes
    target_sym = None
    replaced_char = None
    
    # Filter and sort symbols: prefer SMALL single-character symbols
    # (superscripts/subscripts are small, full expressions are large)
    def symbol_size(sym):
        bbox = sym['bbox']
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # area
    
    # Get single-character symbols that are in stroke corpus, sorted by size (smallest first)
    single_char_symbols = []
    for sym in symbols:
        sym_text = sym['text'].strip()
        # Only consider single characters (not groups like "1γ")
        if len(sym_text) == 1 and sym_text in available_symbols:
            single_char_symbols.append((sym, sym_text))
    
    # Sort by size (prefer smaller = superscripts/subscripts)
    single_char_symbols.sort(key=lambda x: symbol_size(x[0]))
    
    # First try: find a small symbol matching the edit target
    for sym, char in single_char_symbols:
        if char == edit_symbol:
            target_sym = sym
            replaced_char = char
            break
    
    # Second try: find any small single-character symbol in stroke corpus
    # Prefer the SMALLEST one (likely a superscript/subscript)
    if not target_sym and single_char_symbols:
        target_sym, replaced_char = single_char_symbols[0]  # Smallest first
    
    if target_sym and replaced_char:
        bbox = target_sym['bbox']
        x0, y0, x1, y1 = [int(b) for b in bbox]
        bbox_w = x1 - x0
        bbox_h = y1 - y0
        
        # Render stroke to fit EXACTLY within the bbox (no minimum - use actual size)
        # Make it square based on the larger dimension to preserve aspect ratio
        stroke_size = max(bbox_w, bbox_h)
        
        # Render at higher resolution then scale down for quality
        render_size = max(stroke_size * 4, 64)  # At least 64px for quality
        stroke_arr = render_strokes_to_array(loader, replaced_char, size=(render_size, render_size))
        
        # Scale down to actual bbox size
        from PIL import Image as PILImage
        stroke_img = PILImage.fromarray(stroke_arr, mode='L')
        stroke_img = stroke_img.resize((stroke_size, stroke_size), PILImage.LANCZOS)
        stroke_arr = np.array(stroke_img)
        
        # White out just the original symbol area (tight)
        draw.rectangle([x0-1, y0-1, x1+1, y1+1], fill='white')
        
        # Center the stroke on the symbol bbox
        paste_x = x0 + (bbox_w - stroke_size) // 2
        paste_y = y0 + (bbox_h - stroke_size) // 2
        
        # Small jitter (very small - just 1-2 pixels)
        max_jitter = max(1, min(bbox_w, bbox_h) // 6)
        jitter_x = random.randint(-max_jitter, max_jitter)
        jitter_y = random.randint(-max_jitter, max_jitter)
        paste_x += jitter_x
        paste_y += jitter_y
        
        # Convert stroke to RGBA for pasting
        stroke_rgba = Image.new('RGBA', stroke_img.size, (255, 255, 255, 0))
        for px in range(stroke_img.width):
            for py in range(stroke_img.height):
                v = stroke_arr[py, px]
                if v > 20:
                    stroke_rgba.putpixel((px, py), (0, 0, 0, min(255, int(v * 1.5))))
        
        img_with_strokes.paste(stroke_rgba, (paste_x, paste_y), stroke_rgba)
        
        # Mark the replacement area - TIGHT around original symbol bbox
        box_pad = 1
        draw.rectangle([x0 - box_pad, y0 - box_pad, x1 + box_pad, y1 + box_pad], 
                      outline='red', width=2)
    
    ax3.imshow(img_with_strokes)
    # Show what was actually replaced (the symbol we drew strokes for)
    display_symbol = replaced_char if replaced_char else edit_symbol
    ax3.set_title(f"3. Edit Symbol Replaced with Strokes ('{display_symbol}')", fontsize=12)
    ax3.axis('off')
    
    # === Panel 4: Stroke examples with WAIT states ===
    ax4 = axes[1, 1]
    
    # Generate stroke progression
    stroke_examples = generate_stroke_examples(
        symbols=['x', 'y', '+'],
        count_per_symbol=1,
        include_incomplete=True,
        add_distractors=False,
        seed=seed,
    )
    
    # Create grid of stroke states
    grid_img = Image.new('RGB', (400, 300), 'white')
    grid_draw = ImageDraw.Draw(grid_img)
    
    col, row = 0, 0
    for ex in stroke_examples[:9]:
        if ex['symbol'] not in available_symbols:
            continue
        
        # Render this stroke state
        stroke_arr = render_strokes_to_array(
            loader, ex['symbol'], 
            size=(80, 80),
            num_strokes=ex['strokes_drawn']
        )
        
        # Paste into grid
        x = 10 + col * 130
        y = 10 + row * 100
        
        stroke_pil = Image.fromarray(stroke_arr, mode='L').convert('RGB')
        grid_img.paste(stroke_pil, (x, y))
        
        # Label
        status = "✓" if ex['is_complete'] else "WAIT"
        label = f"{ex['symbol']} {ex['strokes_drawn']}/{ex['total_strokes']} {status}"
        grid_draw.text((x, y + 82), label, fill='black')
        
        # Red dot showing jittered position
        jx = ex['position_offset'][0] * 80 + 40
        jy = ex['position_offset'][1] * 80 + 40
        grid_draw.ellipse([x + jx - 4, y + jy - 4, x + jx + 4, y + jy + 4], fill='red')
        
        col += 1
        if col >= 3:
            col = 0
            row += 1
    
    ax4.imshow(grid_img)
    ax4.set_title("4. Stroke Progressions (complete vs WAIT) + Position Jitter (red dots)", fontsize=12)
    ax4.axis('off')
    
    # Add overall title
    plt.suptitle(
        f"Context: {context_data['atomic_part'][:30]}... + diagram with nodes {context_data['diagram_nodes'][:3]}",
        fontsize=11, y=0.98
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Run the demo."""
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("Context Visualization Demo")
    print("=" * 60)
    
    # Compose context with diagram
    print("\n1. Composing context (atomic + diagram)...")
    context = compose_context_with_diagram(seed=42)
    
    print(f"   Atomic: {context['atomic_part'][:40]}...")
    print(f"   Diagram nodes: {context['diagram_nodes']}")
    print(f"   Edit target: {context['edit_target']}")
    print(f"   Operation: {context['edit_operation']}")
    
    # Visualize
    print("\n2. Generating visualization...")
    output_path = SCRIPT_DIR / 'context_visualization.png'
    
    visualize_context_with_strokes(
        context,
        output_path=str(output_path),
        show=False,
        seed=42,
    )
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

