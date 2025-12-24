r"""
LaTeX to Image Renderer

Renders LaTeX expressions (including TikZ-CD diagrams) to images.
Used for:
- Ground truth expression images
- Edit region visualization
- Training data generation

Requirements:
    - LaTeX installation (MiKTeX, TeX Live)
    - PyMuPDF (pip install pymupdf) or pdf2image + Poppler

Usage:
    from scripts.latex_renderer import render_latex, render_latex_batch
    
    # Single expression
    img = render_latex(r"x^2 + y^2 = r^2")
    
    # TikZ-CD diagram
    img = render_latex(r"\begin{tikzcd} A \arrow[r] & B \end{tikzcd}")
"""

import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# PDF converter detection
HAS_PYMUPDF = False
HAS_PDF2IMAGE = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    pass

if not HAS_PYMUPDF:
    try:
        from pdf2image import convert_from_path
        HAS_PDF2IMAGE = True
    except ImportError:
        pass

# PIL for image handling
try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# LaTeX Templates
# =============================================================================

LATEX_PREAMBLE = r"""
\documentclass[preview,border=10pt]{standalone}
\usepackage{tikz-cd}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{esint}
\begin{document}
"""

LATEX_POSTAMBLE = r"""
\end{document}
"""


# =============================================================================
# Path Detection
# =============================================================================

def find_pdflatex() -> Optional[str]:
    """Find pdflatex executable."""
    # Try PATH first
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        return pdflatex
    
    # Common Windows paths
    common_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\MiKTeX\miktex\bin\x64\pdflatex.exe"),
        r"C:\Users\chaow\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        # TeX Live
        os.path.expandvars(r"%PROGRAMFILES%\texlive\2024\bin\win64\pdflatex.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\texlive\2023\bin\win64\pdflatex.exe"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def find_poppler_path() -> Optional[str]:
    """Find Poppler installation for pdf2image."""
    if shutil.which("pdftoppm"):
        return None  # Use system PATH
    
    common_paths = [
        os.path.expandvars(r"%USERPROFILE%\miniconda3\Library\bin"),
        os.path.expandvars(r"%USERPROFILE%\anaconda3\Library\bin"),
        r"C:\Users\chaow\miniconda3\Library\bin",
    ]
    
    for path in common_paths:
        if os.path.exists(os.path.join(path, "pdftoppm.exe")):
            return path
    
    return None


# Cache paths
_PDFLATEX_PATH = find_pdflatex()
_POPPLER_PATH = find_poppler_path()


# =============================================================================
# Core Rendering
# =============================================================================

def render_latex_to_pdf(latex: str, output_path: str, 
                        pdflatex_path: Optional[str] = None) -> bool:
    """
    Render LaTeX expression to PDF.
    
    Args:
        latex: LaTeX expression or full document
        output_path: Output PDF path
        pdflatex_path: Path to pdflatex (auto-detect if None)
        
    Returns:
        True if successful
    """
    if pdflatex_path is None:
        pdflatex_path = _PDFLATEX_PATH
    
    if pdflatex_path is None:
        print("Error: pdflatex not found. Install MiKTeX or TeX Live.")
        return False
    
    # Wrap in document if needed
    if r'\begin{document}' not in latex:
        if r'\begin{tikzcd}' in latex:
            full_doc = LATEX_PREAMBLE + latex + LATEX_POSTAMBLE
        else:
            # Regular math expression
            full_doc = LATEX_PREAMBLE + f"$\\displaystyle {latex}$" + LATEX_POSTAMBLE
    else:
        full_doc = latex
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "expr.tex")
        pdf_file = os.path.join(tmpdir, "expr.pdf")
        
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(full_doc)
        
        try:
            result = subprocess.run(
                [pdflatex_path, "-interaction=batchmode", "-output-directory", tmpdir, tex_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if os.path.exists(pdf_file):
                shutil.copy(pdf_file, output_path)
                return True
            return False
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def autocrop_image(image: 'Image.Image', padding: int = 10) -> 'Image.Image':
    """Auto-crop whitespace from image."""
    if not HAS_PIL:
        return image
    
    gray = image.convert('L')
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()
    
    if bbox:
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(image.width, bbox[2] + padding)
        bottom = min(image.height, bbox[3] + padding)
        return image.crop((left, top, right, bottom))
    
    return image


def render_latex(latex: str, 
                 dpi: int = 200,
                 autocrop: bool = True,
                 padding: int = 10) -> Optional[np.ndarray]:
    """
    Render LaTeX expression to image.
    
    Args:
        latex: LaTeX expression (e.g., r"x^2 + y^2")
        dpi: Resolution
        autocrop: Remove whitespace borders
        padding: Padding when autocropping
        
    Returns:
        Image as numpy array [H, W, 3] or None if failed
    """
    if not latex or not latex.strip():
        return None
    
    if not HAS_PYMUPDF and not HAS_PDF2IMAGE:
        print("Error: No PDF converter. Install: pip install pymupdf")
        return None
    
    # Render to temp PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        pdf_path = tmp.name
    
    try:
        if not render_latex_to_pdf(latex, pdf_path):
            return None
        
        img = None
        
        if HAS_PYMUPDF:
            import fitz
            from PIL import Image
            import io
            
            doc = fitz.open(pdf_path)
            page = doc[0]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            doc.close()
        
        elif HAS_PDF2IMAGE:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=dpi, poppler_path=_POPPLER_PATH)
            if images:
                img = images[0]
        
        if img is None:
            return None
        
        if autocrop:
            img = autocrop_image(img, padding)
        
        return np.array(img)
        
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def render_latex_batch(expressions: List[str],
                       dpi: int = 150,
                       autocrop: bool = True) -> List[Optional[np.ndarray]]:
    """
    Render multiple LaTeX expressions.
    
    Args:
        expressions: List of LaTeX strings
        dpi: Resolution
        autocrop: Auto-crop whitespace
        
    Returns:
        List of images (None for failed renders)
    """
    return [render_latex(expr, dpi=dpi, autocrop=autocrop) for expr in expressions]


# =============================================================================
# TikZ-CD Diagram Templates
# =============================================================================

class DiagramTemplates:
    """Common commutative diagram templates."""
    
    @staticmethod
    def simple_arrow(A: str = "A", B: str = "B", f: str = "f") -> str:
        """A --f--> B"""
        return rf"\begin{{tikzcd}} {A} \arrow[r, \"{f}\"] & {B} \end{{tikzcd}}"
    
    @staticmethod
    def triangle(A: str = "A", B: str = "B", C: str = "C",
                 f: str = "f", g: str = "g", h: str = "h") -> str:
        """Commutative triangle."""
        return rf"""\begin{{tikzcd}}
{A} \arrow[r, "{f}"] \arrow[dr, "{h}"'] & {B} \arrow[d, "{g}"] \\
& {C}
\end{{tikzcd}}"""
    
    @staticmethod
    def square(A: str = "A", B: str = "B", C: str = "C", D: str = "D",
               f: str = "f", g: str = "g", h: str = "h", k: str = "k") -> str:
        """Commutative square."""
        return rf"""\begin{{tikzcd}}
{A} \arrow[r, "{f}"] \arrow[d, "{g}"'] & {B} \arrow[d, "{h}"] \\
{C} \arrow[r, "{k}"'] & {D}
\end{{tikzcd}}"""
    
    @staticmethod
    def pullback(P: str = "P", X: str = "X", Y: str = "Y", Z: str = "Z") -> str:
        """Pullback diagram with corner marker."""
        return rf"""\begin{{tikzcd}}
{P} \arrow[r] \arrow[d] \arrow[dr, phantom, "\lrcorner", very near start] & {X} \arrow[d] \\
{Y} \arrow[r] & {Z}
\end{{tikzcd}}"""
    
    @staticmethod
    def exact_sequence(labels: List[str] = None) -> str:
        """Short exact sequence: 0 → A → B → C → 0"""
        if labels is None:
            labels = ["0", "A", "B", "C", "0"]
        arrows = " \\arrow[r] & ".join(labels)
        return rf"\begin{{tikzcd}} {arrows} \end{{tikzcd}}"


# =============================================================================
# Mathematical Expression Templates
# =============================================================================

class MathTemplates:
    """Common mathematical expression templates."""
    
    @staticmethod
    def integral_definite(f: str = "f(x)", a: str = "a", b: str = "b") -> str:
        return rf"\int_{{{a}}}^{{{b}}} {f} \, dx"
    
    @staticmethod
    def integral_double() -> str:
        return r"\iint_D f(x,y) \, dA"
    
    @staticmethod
    def sum_series(expr: str = r"\frac{1}{n^2}", 
                   lower: str = "n=1", upper: str = r"\infty") -> str:
        return rf"\sum_{{{lower}}}^{{{upper}}} {expr}"
    
    @staticmethod
    def limit(expr: str = r"\frac{f(x)}{g(x)}", 
              var: str = "x", to: str = "0") -> str:
        return rf"\lim_{{{var} \to {to}}} {expr}"
    
    @staticmethod
    def derivative(f: str = "f", var: str = "x") -> str:
        return rf"\frac{{d{f}}}{{d{var}}}"
    
    @staticmethod
    def partial_derivative(f: str = "f", var: str = "x") -> str:
        return rf"\frac{{\partial {f}}}{{\partial {var}}}"
    
    @staticmethod
    def matrix_2x2(a: str = "a", b: str = "b", 
                   c: str = "c", d: str = "d") -> str:
        return rf"\begin{{pmatrix}} {a} & {b} \\ {c} & {d} \end{{pmatrix}}"
    
    @staticmethod
    def fraction(num: str = "a", den: str = "b") -> str:
        return rf"\frac{{{num}}}{{{den}}}"
    
    @staticmethod
    def binomial(n: str = "n", k: str = "k") -> str:
        return rf"\binom{{{n}}}{{{k}}}"
    
    @staticmethod
    def sqrt(expr: str = "x", n: Optional[str] = None) -> str:
        if n:
            return rf"\sqrt[{n}]{{{expr}}}"
        return rf"\sqrt{{{expr}}}"


# =============================================================================
# Utility Functions
# =============================================================================

def check_latex_available() -> bool:
    """Check if LaTeX is available for rendering."""
    return _PDFLATEX_PATH is not None


def check_pdf_converter_available() -> bool:
    """Check if a PDF converter is available."""
    return HAS_PYMUPDF or HAS_PDF2IMAGE


def get_render_capabilities() -> Dict[str, Any]:
    """Get information about rendering capabilities."""
    return {
        'pdflatex_available': _PDFLATEX_PATH is not None,
        'pdflatex_path': _PDFLATEX_PATH,
        'pymupdf_available': HAS_PYMUPDF,
        'pdf2image_available': HAS_PDF2IMAGE,
        'poppler_path': _POPPLER_PATH,
        'can_render': (_PDFLATEX_PATH is not None) and (HAS_PYMUPDF or HAS_PDF2IMAGE),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test LaTeX rendering."""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("LaTeX Renderer")
    print("=" * 60)
    
    # Check capabilities
    caps = get_render_capabilities()
    print("\nCapabilities:")
    print(f"  pdflatex: {'✓' if caps['pdflatex_available'] else '✗'}")
    print(f"  PyMuPDF: {'✓' if caps['pymupdf_available'] else '✗'}")
    print(f"  pdf2image: {'✓' if caps['pdf2image_available'] else '✗'}")
    print(f"  Can render: {'✓' if caps['can_render'] else '✗'}")
    
    if not caps['can_render']:
        print("\nCannot render. Install: MiKTeX + pymupdf")
        return
    
    # Test renders
    print("\n" + "-" * 40)
    print("Test Renders")
    print("-" * 40)
    
    test_expressions = [
        r"x^2 + y^2 = r^2",
        r"\frac{1}{2}",
        r"\int_0^\infty e^{-x} dx",
        r"\sum_{n=1}^{\infty} \frac{1}{n^2}",
    ]
    
    for expr in test_expressions:
        img = render_latex(expr, dpi=100)
        if img is not None:
            print(f"  ✓ {expr[:30]:30} -> {img.shape}")
        else:
            print(f"  ✗ {expr[:30]:30} -> FAILED")
    
    # Test diagram
    print("\n" + "-" * 40)
    print("Test Diagram")
    print("-" * 40)
    
    diagram = DiagramTemplates.square()
    img = render_latex(diagram, dpi=100)
    if img is not None:
        print(f"  ✓ Commutative square -> {img.shape}")
    else:
        print(f"  ✗ Commutative square -> FAILED")
    
    print("\n" + "=" * 60)
    print("LaTeX renderer test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

