"""
Mechanical Edit Manager for LaTeX Context

This module handles all position-based operations WITHOUT model learning:
- Find affected tokens by bbox overlap
- Re-recognize affected regions
- Update LaTeX at positions
- Recalculate token positions after edits

The MODEL only does recognition. All position logic is here.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box with overlap detection."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, 
                (self.y_min + self.y_max) / 2)
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def overlaps(self, other: 'BoundingBox', threshold: float = 0.3) -> bool:
        """Check if overlaps with another bbox above threshold."""
        x_overlap = max(0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        y_overlap = max(0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        intersection = x_overlap * y_overlap
        
        min_area = min(self.area, other.area)
        if min_area == 0:
            return False
        
        return (intersection / min_area) >= threshold
    
    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        return BoundingBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max)
        )


class EditType(Enum):
    """Edit types detected mechanically."""
    WRITE_NEW = "write_new"      # Writing in empty area
    WRITE_OVER = "write_over"    # Writing over existing token
    ERASE = "erase"              # Scribble pattern detected
    MODIFY = "modify"            # Partial edit of existing


@dataclass
class StrokeRecord:
    """Record of a single stroke."""
    stroke_id: int
    points: np.ndarray  # (N, 2) array of (x, y) points
    bbox: BoundingBox
    timestamp: float
    consumed: bool = False      # True if recognized as part of a token
    token_id: Optional[int] = None  # Which token consumed it
    
    @classmethod
    def from_points(cls, stroke_id: int, points: np.ndarray, timestamp: float = 0.0):
        """Create from point array."""
        if len(points) == 0:
            bbox = BoundingBox(0, 0, 0, 0)
        else:
            bbox = BoundingBox(
                x_min=float(points[:, 0].min()),
                y_min=float(points[:, 1].min()),
                x_max=float(points[:, 0].max()),
                y_max=float(points[:, 1].max())
            )
        return cls(stroke_id, points, bbox, timestamp)


@dataclass
class TokenRecord:
    """Record of a recognized token."""
    token: str              # The LaTeX token (e.g., "x", "\\alpha")
    token_id: int           # Vocabulary ID
    latex_index: int        # Position in LaTeX sequence
    bbox: BoundingBox       # Visual bounding box
    stroke_ids: List[int]   # Which strokes form this token
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'token': self.token,
            'token_id': self.token_id,
            'latex_index': self.latex_index,
            'bbox': {'x_min': self.bbox.x_min, 'y_min': self.bbox.y_min,
                     'x_max': self.bbox.x_max, 'y_max': self.bbox.y_max},
            'stroke_ids': self.stroke_ids,
            'confidence': self.confidence
        }


@dataclass
class EditRegion:
    """Region being edited with context."""
    bbox: BoundingBox
    edit_type: EditType
    affected_tokens: List[TokenRecord]
    new_strokes: List[StrokeRecord]
    slot_hint: str = ""  # <FRAC_NUM>, <SUP>, etc.


class EditManager:
    """
    Manages all edit operations mechanically.
    
    The model only does recognition.
    This class handles:
    - Stroke accumulation (snowball)
    - Finding affected tokens by position
    - Triggering re-recognition
    - Updating LaTeX sequence
    """
    
    def __init__(self):
        self.strokes: Dict[int, StrokeRecord] = {}
        self.tokens: Dict[int, TokenRecord] = {}
        self.latex_sequence: List[str] = []
        self.next_stroke_id = 0
        self.next_token_id = 0
    
    # =========================================================================
    # Stroke Management
    # =========================================================================
    
    def add_stroke(self, points: np.ndarray, timestamp: float = 0.0) -> int:
        """Add a new stroke to the canvas."""
        stroke_id = self.next_stroke_id
        self.next_stroke_id += 1
        
        stroke = StrokeRecord.from_points(stroke_id, points, timestamp)
        self.strokes[stroke_id] = stroke
        
        return stroke_id
    
    def get_unconsumed_strokes(self) -> List[StrokeRecord]:
        """Get strokes not yet recognized (the 'snowball')."""
        return [s for s in self.strokes.values() if not s.consumed]
    
    def get_strokes_in_region(self, bbox: BoundingBox, 
                               include_consumed: bool = False) -> List[StrokeRecord]:
        """Get all strokes overlapping with a region."""
        strokes = []
        for s in self.strokes.values():
            if include_consumed or not s.consumed:
                if s.bbox.overlaps(bbox):
                    strokes.append(s)
        return strokes
    
    def consume_strokes(self, stroke_ids: List[int], token_id: int):
        """Mark strokes as consumed by a recognized token."""
        for sid in stroke_ids:
            if sid in self.strokes:
                self.strokes[sid].consumed = True
                self.strokes[sid].token_id = token_id
    
    def unconsume_strokes(self, stroke_ids: List[int]):
        """Mark strokes as unconsumed (for re-recognition)."""
        for sid in stroke_ids:
            if sid in self.strokes:
                self.strokes[sid].consumed = False
                self.strokes[sid].token_id = None
    
    def delete_strokes(self, stroke_ids: List[int]):
        """Delete strokes (e.g., when erased)."""
        for sid in stroke_ids:
            if sid in self.strokes:
                del self.strokes[sid]
    
    # =========================================================================
    # Token Management
    # =========================================================================
    
    def add_token(self, token: str, token_vocab_id: int, 
                  bbox: BoundingBox, stroke_ids: List[int],
                  latex_index: int = -1, confidence: float = 1.0) -> int:
        """Add a recognized token."""
        token_id = self.next_token_id
        self.next_token_id += 1
        
        if latex_index < 0:
            latex_index = len(self.latex_sequence)
        
        record = TokenRecord(
            token=token,
            token_id=token_vocab_id,
            latex_index=latex_index,
            bbox=bbox,
            stroke_ids=stroke_ids,
            confidence=confidence
        )
        
        self.tokens[token_id] = record
        self.consume_strokes(stroke_ids, token_id)
        
        # Insert into LaTeX sequence
        if latex_index >= len(self.latex_sequence):
            self.latex_sequence.append(token)
        else:
            self.latex_sequence.insert(latex_index, token)
            self._reindex_tokens_after(latex_index)
        
        return token_id
    
    def remove_token(self, token_id: int, delete_strokes: bool = True):
        """Remove a token and optionally its strokes."""
        if token_id not in self.tokens:
            return
        
        record = self.tokens[token_id]
        
        # Remove from LaTeX sequence
        if record.latex_index < len(self.latex_sequence):
            self.latex_sequence.pop(record.latex_index)
            self._reindex_tokens_after(record.latex_index - 1)
        
        # Handle strokes
        if delete_strokes:
            self.delete_strokes(record.stroke_ids)
        else:
            self.unconsume_strokes(record.stroke_ids)
        
        del self.tokens[token_id]
    
    def replace_token(self, token_id: int, new_token: str, 
                      new_vocab_id: int, new_stroke_ids: List[int],
                      new_bbox: BoundingBox):
        """Replace a token with new content."""
        if token_id not in self.tokens:
            return
        
        old_record = self.tokens[token_id]
        latex_index = old_record.latex_index
        
        # Unconsume old strokes
        self.unconsume_strokes(old_record.stroke_ids)
        
        # Update record
        old_record.token = new_token
        old_record.token_id = new_vocab_id
        old_record.bbox = new_bbox
        old_record.stroke_ids = new_stroke_ids
        
        # Update LaTeX sequence
        if latex_index < len(self.latex_sequence):
            self.latex_sequence[latex_index] = new_token
        
        # Consume new strokes
        self.consume_strokes(new_stroke_ids, token_id)
    
    def _reindex_tokens_after(self, index: int):
        """Reindex tokens after an insertion/deletion."""
        for token in self.tokens.values():
            if token.latex_index > index:
                token.latex_index += 1
    
    # =========================================================================
    # Edit Detection (Mechanical)
    # =========================================================================
    
    def find_affected_tokens(self, bbox: BoundingBox, 
                             overlap_threshold: float = 0.3) -> List[TokenRecord]:
        """Find tokens whose bbox overlaps with the edit region."""
        affected = []
        for token in self.tokens.values():
            if token.bbox.overlaps(bbox, overlap_threshold):
                affected.append(token)
        return affected
    
    def detect_edit_type(self, new_strokes: List[StrokeRecord]) -> EditType:
        """Detect what kind of edit this is based on stroke pattern."""
        if not new_strokes:
            return EditType.WRITE_NEW
        
        # Compute combined bbox of new strokes
        combined_bbox = new_strokes[0].bbox
        for s in new_strokes[1:]:
            combined_bbox = combined_bbox.union(s.bbox)
        
        # Check for scribble pattern (erase)
        if self._is_scribble(new_strokes):
            return EditType.ERASE
        
        # Check if writing over existing tokens
        affected = self.find_affected_tokens(combined_bbox)
        if affected:
            return EditType.WRITE_OVER
        
        return EditType.WRITE_NEW
    
    def _is_scribble(self, strokes: List[StrokeRecord]) -> bool:
        """Detect if strokes form a scribble/cross-out pattern."""
        if not strokes:
            return False
        
        # Simple heuristic: high point density + many direction changes
        for stroke in strokes:
            if len(stroke.points) < 5:
                continue
            
            # Compute direction changes
            diffs = np.diff(stroke.points, axis=0)
            if len(diffs) < 2:
                continue
            
            # Angle changes
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_changes = np.abs(np.diff(angles))
            
            # Many sharp turns = scribble
            sharp_turns = np.sum(angle_changes > np.pi / 4)
            if sharp_turns > len(angles) * 0.3:
                return True
        
        return False
    
    def get_slot_hint(self, bbox: BoundingBox) -> str:
        """
        Determine what kind of slot user is writing in.
        Returns a hint token for the model.
        """
        # Find nearest structure
        for token in self.tokens.values():
            if '\\frac{' in token.token:
                # Check if above or below fraction bar
                frac_center_y = token.bbox.center[1]
                if bbox.center[1] < frac_center_y:
                    return '<FRAC_NUM>'
                else:
                    return '<FRAC_DEN>'
            
            if '^{' in token.token or token.token == '^':
                if bbox.overlaps(token.bbox):
                    return '<SUP>'
            
            if '_{' in token.token or token.token == '_':
                if bbox.overlaps(token.bbox):
                    return '<SUB>'
        
        return ''  # Normal position
    
    # =========================================================================
    # Edit Operations
    # =========================================================================
    
    def prepare_edit_region(self, new_stroke_ids: List[int]) -> EditRegion:
        """
        Prepare an edit region for recognition.
        
        Returns all info needed for the model to recognize content.
        """
        new_strokes = [self.strokes[sid] for sid in new_stroke_ids 
                       if sid in self.strokes]
        
        if not new_strokes:
            return None
        
        # Compute combined bbox
        combined_bbox = new_strokes[0].bbox
        for s in new_strokes[1:]:
            combined_bbox = combined_bbox.union(s.bbox)
        
        # Detect edit type
        edit_type = self.detect_edit_type(new_strokes)
        
        # Find affected tokens
        affected = self.find_affected_tokens(combined_bbox)
        
        # Get slot hint
        slot_hint = self.get_slot_hint(combined_bbox)
        
        return EditRegion(
            bbox=combined_bbox,
            edit_type=edit_type,
            affected_tokens=affected,
            new_strokes=new_strokes,
            slot_hint=slot_hint
        )
    
    def apply_recognition_result(self, edit_region: EditRegion, 
                                  recognized_token: str,
                                  vocab_id: int,
                                  confidence: float = 1.0):
        """
        Apply recognition result to update LaTeX.
        
        This is called after model recognizes the content.
        """
        if edit_region.edit_type == EditType.ERASE:
            # Delete affected tokens
            for token in edit_region.affected_tokens:
                for tid, t in list(self.tokens.items()):
                    if t.latex_index == token.latex_index:
                        self.remove_token(tid, delete_strokes=True)
                        break
        
        elif edit_region.edit_type == EditType.WRITE_OVER:
            # Replace affected token(s)
            if edit_region.affected_tokens:
                # Replace first affected token
                target = edit_region.affected_tokens[0]
                for tid, t in self.tokens.items():
                    if t.latex_index == target.latex_index:
                        new_stroke_ids = [s.stroke_id for s in edit_region.new_strokes]
                        self.replace_token(tid, recognized_token, vocab_id,
                                          new_stroke_ids, edit_region.bbox)
                        break
                
                # Remove other affected tokens
                for token in edit_region.affected_tokens[1:]:
                    for tid, t in list(self.tokens.items()):
                        if t.latex_index == token.latex_index:
                            self.remove_token(tid)
                            break
        
        else:  # WRITE_NEW
            # Add new token
            stroke_ids = [s.stroke_id for s in edit_region.new_strokes]
            self.add_token(recognized_token, vocab_id, edit_region.bbox,
                          stroke_ids, confidence=confidence)
    
    def get_latex(self) -> str:
        """Get current LaTeX string."""
        return ' '.join(self.latex_sequence)
    
    def get_context_for_model(self) -> Dict:
        """Get context info for model input."""
        return {
            'latex': self.get_latex(),
            'tokens': self.latex_sequence.copy(),
            'token_positions': [
                {'token': t.token, 'bbox': t.bbox, 'index': t.latex_index}
                for t in sorted(self.tokens.values(), key=lambda x: x.latex_index)
            ]
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EditManager Demo")
    print("=" * 60)
    
    manager = EditManager()
    
    # Simulate user writing "x + y"
    print("\n1. User writes 'x'")
    s1 = manager.add_stroke(np.array([[10, 20], [20, 30], [30, 20]]))
    
    # Simulate model recognition
    manager.add_token('x', 23, BoundingBox(10, 20, 30, 30), [s1])
    print(f"   LaTeX: {manager.get_latex()}")
    
    print("\n2. User writes '+'")
    s2 = manager.add_stroke(np.array([[40, 25], [50, 25]]))
    s3 = manager.add_stroke(np.array([[45, 20], [45, 30]]))
    manager.add_token('+', 50, BoundingBox(40, 20, 50, 30), [s2, s3])
    print(f"   LaTeX: {manager.get_latex()}")
    
    print("\n3. User writes 'y'")
    s4 = manager.add_stroke(np.array([[60, 20], [70, 30], [70, 40]]))
    manager.add_token('y', 24, BoundingBox(60, 20, 70, 40), [s4])
    print(f"   LaTeX: {manager.get_latex()}")
    
    print("\n4. User erases 'y' (scribble over it)")
    scribble = np.array([[58, 18], [72, 42], [58, 42], [72, 18], [58, 18]])
    s5 = manager.add_stroke(scribble)
    
    # Prepare edit region
    region = manager.prepare_edit_region([s5])
    print(f"   Edit type: {region.edit_type}")
    print(f"   Affected: {[t.token for t in region.affected_tokens]}")
    
    # Apply as erase
    if region.edit_type == EditType.ERASE:
        manager.apply_recognition_result(region, '', -1)
    
    print(f"   LaTeX: {manager.get_latex()}")
    
    print("\n5. User writes 'z' in place of erased 'y'")
    s6 = manager.add_stroke(np.array([[60, 20], [70, 25], [60, 35], [70, 40]]))
    region = manager.prepare_edit_region([s6])
    manager.apply_recognition_result(region, 'z', 25)
    print(f"   LaTeX: {manager.get_latex()}")

