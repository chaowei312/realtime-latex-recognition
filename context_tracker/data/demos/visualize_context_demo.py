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
        
        # Extract symbol positions
        symbols = []
        scale = dpi / 72
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = span["bbox"]
                        symbols.append({
                            'text': span["text"],
                            'bbox': [b * scale for b in bbox],
                            'center': ((bbox[0]+bbox[2])/2 * scale, (bbox[1]+bbox[3])/2 * scale),
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
    Render strokes to numpy array using the proper StrokeRenderer.
    
    Uses anti-aliased PIL rendering for smooth strokes like in all_symbols_grid.png
    """
    sym_data = loader.get_symbol(symbol)
    if sym_data is None:
        # Fallback: render a simple mark
        arr = np.zeros(size, dtype=np.uint8)
        cx, cy = size[0] // 2, size[1] // 2
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if 0 <= cx+dx < size[0] and 0 <= cy+dy < size[1]:
                    arr[cy+dy, cx+dx] = 200
        return arr
    
    # Use proper renderer with good canvas size
    canvas_size = max(size[0], size[1], 128)
    renderer = StrokeRenderer(loader, canvas_size=canvas_size)
    
    stroke_names = sorted(sym_data.strokes.keys())
    total = len(stroke_names)
    if num_strokes is None:
        num_strokes = total
    num_strokes = min(num_strokes, total)
    
    # Select random variations for each stroke
    selected_strokes = []
    for stroke_name in stroke_names[:num_strokes]:
        variations = sym_data.strokes[stroke_name]
        stroke = random.choice(variations)
        # Interpolate for smooth curves
        selected_strokes.append(stroke.interpolate(50))
    
    # Render using PIL (anti-aliased)
    thickness = random.uniform(2.5, 3.5)
    canvas = renderer._render_strokes(selected_strokes, thickness)
    
    # Resize to target size
    from PIL import Image as PILImage
    img = PILImage.fromarray((canvas * 255).astype(np.uint8), mode='L')
    img = img.resize(size, PILImage.LANCZOS)
    return np.array(img)


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
    
    # Edit info (from atomic case)
    edit_target = atomic.after_latex.replace(atomic.before_latex, "").strip()
    if not edit_target:
        edit_target = "x"  # Fallback
    
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
    edit_symbol = context_data['edit_target']
    if len(edit_symbol) > 1:
        edit_symbol = edit_symbol[0]  # Just first char
    
    # Find symbol in rendered image
    target_sym = None
    for sym in symbols:
        if sym['text'].strip() and sym['text'].strip()[0] in available_symbols:
            target_sym = sym
            break
    
    if target_sym:
        # Get the symbol to replace
        sym_char = target_sym['text'].strip()[0]
        bbox = target_sym['bbox']
        
        # Render stroke for this symbol
        stroke_w = int(bbox[2] - bbox[0]) + 20
        stroke_h = int(bbox[3] - bbox[1]) + 20
        stroke_size = max(stroke_w, stroke_h, 40)
        
        stroke_arr = render_strokes_to_array(loader, sym_char, size=(stroke_size, stroke_size))
        stroke_img = Image.fromarray(stroke_arr, mode='L')
        
        # White out original symbol area
        x0, y0, x1, y1 = [int(b) for b in bbox]
        draw.rectangle([x0-5, y0-5, x1+5, y1+5], fill='white')
        
        # Paste stroke (with position jitter)
        jitter_x = random.randint(-8, 8)
        jitter_y = random.randint(-8, 8)
        paste_x = int(bbox[0] - 10 + jitter_x)
        paste_y = int(bbox[1] - 10 + jitter_y)
        
        # Convert stroke to RGBA for pasting
        stroke_rgba = Image.new('RGBA', stroke_img.size, (255, 255, 255, 0))
        for px in range(stroke_img.width):
            for py in range(stroke_img.height):
                v = stroke_arr[py, px]
                if v > 20:
                    stroke_rgba.putpixel((px, py), (0, 0, 0, min(255, int(v * 1.5))))
        
        img_with_strokes.paste(stroke_rgba, (paste_x, paste_y), stroke_rgba)
        
        # Mark the replacement area
        draw.rectangle([paste_x, paste_y, paste_x + stroke_size, paste_y + stroke_size], 
                      outline='red', width=2)
    
    ax3.imshow(img_with_strokes)
    ax3.set_title(f"3. Edit Symbol Replaced with Strokes (target: '{edit_symbol}')", fontsize=12)
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

