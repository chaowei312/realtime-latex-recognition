"""
Symbol Position Extraction

Extract exact symbol bounding boxes from rendered LaTeX PDFs using PyMuPDF.

Functions:
    - extract_symbol_positions_from_pdf: Get all symbol bboxes from a PDF
    - find_symbol_bbox_in_latex: Find specific symbol's bbox
    - get_all_symbol_bboxes: Render LaTeX and extract all bboxes
"""

import os
import tempfile
from typing import List, Dict, Optional, Tuple, Any


def extract_symbol_positions_from_pdf(pdf_path: str, use_ink_bbox: bool = True) -> List[Dict]:
    """
    Extract symbol positions from a rendered LaTeX PDF using PyMuPDF.
    
    IMPORTANT: PyMuPDF span bboxes include font metrics (ascender/descender),
    which are much larger than the actual visible ink. When use_ink_bbox=True,
    we compute the actual ink bbox from rendered pixels, which is more accurate
    for matching with handwriting stroke bboxes.
    
    Args:
        pdf_path: Path to rendered PDF
        use_ink_bbox: If True, compute actual ink bbox (recommended)
        
    Returns:
        List of {text, bbox, center, ink_bbox} dicts for each symbol
    """
    try:
        import fitz
        import numpy as np
    except ImportError:
        return []
    
    symbols = []
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        span_bbox = span["bbox"]  # Font metrics bbox (larger)
                        
                        # Compute actual ink bbox if requested
                        if use_ink_bbox and len(text.strip()) == 1:
                            ink_bbox = _compute_ink_bbox_for_span(
                                doc, page, span_bbox, render_scale=10
                            )
                        else:
                            ink_bbox = span_bbox
                        
                        # Use ink bbox as the primary bbox
                        bbox = ink_bbox if ink_bbox else span_bbox
                        
                        symbols.append({
                            "text": text,
                            "bbox": bbox,
                            "span_bbox": span_bbox,  # Keep original for reference
                            "ink_bbox": ink_bbox,
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "center": ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2),
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                        })
        doc.close()
    except Exception as e:
        print(f"Error extracting symbols: {e}")
    
    return symbols


def _compute_ink_bbox_for_span(doc, page, span_bbox, render_scale: int = 10) -> tuple:
    """
    Compute actual ink bounding box by rendering and finding non-white pixels.
    
    This is more accurate than span_bbox which includes font ascender/descender.
    
    Args:
        doc: PyMuPDF document
        page: Page object
        span_bbox: Original span bounding box
        render_scale: Render scale for precision (higher = more accurate but slower)
        
    Returns:
        (x0, y0, x1, y1) in PDF points, or None if failed
    """
    try:
        import fitz
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    
    try:
        # Expand span_bbox slightly to ensure we capture all ink
        pad = 2  # PDF points
        clip_rect = fitz.Rect(
            max(0, span_bbox[0] - pad),
            max(0, span_bbox[1] - pad),
            min(page.rect.width, span_bbox[2] + pad),
            min(page.rect.height, span_bbox[3] + pad)
        )
        
        # Render clip region at high resolution
        mat = fitz.Matrix(render_scale, render_scale)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr = np.array(img)
        
        # Find non-white pixels (ink)
        gray = arr.mean(axis=2)
        ink_mask = gray < 250  # Threshold for "not white"
        ink_pixels = np.where(ink_mask)
        
        if len(ink_pixels[0]) == 0:
            return span_bbox  # No ink found, return original
        
        # Get bounding box of ink pixels
        y_min, y_max = ink_pixels[0].min(), ink_pixels[0].max()
        x_min, x_max = ink_pixels[1].min(), ink_pixels[1].max()
        
        # Convert back to PDF points (accounting for clip offset)
        ink_bbox = (
            clip_rect.x0 + x_min / render_scale,
            clip_rect.y0 + y_min / render_scale,
            clip_rect.x0 + (x_max + 1) / render_scale,
            clip_rect.y0 + (y_max + 1) / render_scale,
        )
        
        return ink_bbox
        
    except Exception as e:
        # Fall back to span bbox on any error
        return span_bbox


def find_symbol_bbox_in_latex(latex: str, target_symbol: str, 
                               pdf_path: Optional[str] = None) -> Optional[Dict]:
    """
    Find the bounding box for a specific symbol in rendered LaTeX.
    
    Args:
        latex: LaTeX expression
        target_symbol: Symbol to find (e.g., 'α', 'x', '+')
        pdf_path: Optional existing PDF path (will render if not provided)
        
    Returns:
        Dict with bbox info, or None if not found
    """
    # Render if needed
    should_cleanup = False
    if pdf_path is None:
        try:
            from generate_tikzcd import render_tikzcd_to_pdf
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                pdf_path = f.name
            success = render_tikzcd_to_pdf(latex, pdf_path)
            if not success:
                return None
            should_cleanup = True
        except ImportError:
            return None
    
    try:
        symbols = extract_symbol_positions_from_pdf(pdf_path)
        
        # Find matching symbol
        for sym in symbols:
            if sym["text"] == target_symbol:
                return sym
        
        return None
    finally:
        if should_cleanup and os.path.exists(pdf_path):
            os.unlink(pdf_path)


def get_all_symbol_bboxes(latex: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Render LaTeX and extract all symbol bounding boxes.
    
    Returns:
        (list of symbol dicts, path to temp PDF/image)
        Caller should clean up the temp file.
    """
    try:
        from generate_tikzcd import render_tikzcd_to_pdf
    except ImportError:
        return [], None
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        pdf_path = f.name
    
    success = render_tikzcd_to_pdf(latex, pdf_path)
    if not success:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        return [], None
    
    symbols = extract_symbol_positions_from_pdf(pdf_path)
    return symbols, pdf_path


def create_symbol_crop(
    pdf_path: str,
    target_symbol: str,
    padding_pct: float = 0.2,
    output_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create a cropped image around a specific symbol from a rendered PDF.
    
    Args:
        pdf_path: Path to rendered PDF
        target_symbol: Symbol to crop around
        padding_pct: Extra padding as fraction of bbox size
        output_path: If provided, save crop to this path
        
    Returns:
        PIL Image of crop (or None if failed)
    """
    try:
        from PIL import Image
        import fitz
    except ImportError:
        return None
    
    # Extract symbol positions
    symbols = extract_symbol_positions_from_pdf(pdf_path)
    
    # Find target symbol
    target_bbox = None
    for sym in symbols:
        sym_text = sym["text"].strip()
        if sym_text == target_symbol or sym_text == target_symbol.strip():
            target_bbox = sym
            break
    
    # Try partial match if not found
    if not target_bbox:
        for sym in symbols:
            if target_symbol in sym["text"] or sym["text"].strip() in target_symbol:
                target_bbox = sym
                break
    
    if not target_bbox:
        return None
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        page_rect = page.rect
        
        # Calculate crop region with padding
        pad_x = target_bbox["width"] * padding_pct
        pad_y = target_bbox["height"] * padding_pct
        
        crop_rect = fitz.Rect(
            max(0, target_bbox["x0"] - pad_x),
            max(0, target_bbox["y0"] - pad_y),
            min(page_rect.width, target_bbox["x1"] + pad_x),
            min(page_rect.height, target_bbox["y1"] + pad_y)
        )
        
        # Render cropped region at high resolution
        mat = fitz.Matrix(3, 3)  # 3x zoom for quality
        pix = page.get_pixmap(matrix=mat, clip=crop_rect)
        
        # Convert to PIL
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        
        if output_path:
            img.save(output_path)
        
        return img
        
    except Exception as e:
        print(f"Error creating crop: {e}")
        return None

