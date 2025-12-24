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


def extract_symbol_positions_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract symbol positions from a rendered LaTeX PDF using PyMuPDF.
    
    This gives us EXACT bounding boxes for each symbol/character in the PDF.
    
    Args:
        pdf_path: Path to rendered PDF
        
    Returns:
        List of {text, bbox, center} dicts for each symbol
    """
    try:
        import fitz
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
                        bbox = span["bbox"]  # (x0, y0, x1, y1) in PDF points
                        symbols.append({
                            "text": text,
                            "bbox": bbox,
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

