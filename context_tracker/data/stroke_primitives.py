"""
Stroke-Level Training Data Types and Model Interface

This module provides data structures for stroke-level recognition:
- Stroke, StrokeSequence: Stroke data representation
- RollingSnowball: Patch accumulation for real-time recognition
- ModelInput, ModelOutput: Model I/O structures
- PatchToken: Token representation for conv encoding

For rendering and dataset generation, see:
    from scripts.stroke_renderer import StrokeRenderer, GPUStrokeRenderer
    from scripts.stroke_dataset import StrokeDatasetGenerator

Usage:
    from stroke_primitives import RollingSnowball, ModelInput, ModelOutput
    
    # Real-time accumulation
    snowball = RollingSnowball()
    result = snowball.add_stroke(stroke)
    
    # Model I/O
    model_input = ModelInput(latex_context="x + ")
    model_output = ModelOutput.recognize("y", stroke_ids=[0, 1])
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Stroke Data Structures
# =============================================================================

@dataclass
class Stroke:
    """A single stroke (pen-down to pen-up)."""
    id: int
    # Bounding box (normalized 0-1)
    x0: float
    y0: float
    x1: float
    y1: float
    # Timing
    start_time: float = 0.0
    duration: float = 0.0
    # Points count
    num_points: int = 20
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    @property
    def width(self) -> float:
        return abs(self.x1 - self.x0)
    
    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)
    
    def overlaps_with(self, other: "Stroke", threshold: float = 0.3) -> bool:
        """Check if this stroke is spatially close to another."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        avg_size = (self.width + self.height + other.width + other.height) / 4
        return dist < avg_size * (1 + threshold)


@dataclass
class StrokeSequence:
    """A sequence of strokes forming a symbol or partial symbol."""
    strokes: List[Stroke] = field(default_factory=list)
    target_symbol: Optional[str] = None
    expected_strokes: int = 1
    
    @property
    def is_complete(self) -> bool:
        return len(self.strokes) >= self.expected_strokes
    
    @property
    def completion_ratio(self) -> float:
        if self.expected_strokes == 0:
            return 1.0
        return min(1.0, len(self.strokes) / self.expected_strokes)
    
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        if not self.strokes:
            return (0.5, 0.5, 0.5, 0.5)
        x0 = min(s.x0 for s in self.strokes)
        y0 = min(s.y0 for s in self.strokes)
        x1 = max(s.x1 for s in self.strokes)
        y1 = max(s.y1 for s in self.strokes)
        return (x0, y0, x1, y1)
    
    def add_stroke(self, stroke: Stroke):
        self.strokes.append(stroke)


# =============================================================================
# Rolling Snowball Patch Accumulation
# =============================================================================

@dataclass
class StrokeGroup:
    """A group of spatially-related strokes."""
    strokes: List[Stroke] = field(default_factory=list)
    center_x: float = 0.5
    center_y: float = 0.5
    
    def add_stroke(self, stroke: Stroke):
        self.strokes.append(stroke)
        self._update_center()
    
    def _update_center(self):
        if self.strokes:
            self.center_x = sum((s.x0 + s.x1) / 2 for s in self.strokes) / len(self.strokes)
            self.center_y = sum((s.y0 + s.y1) / 2 for s in self.strokes) / len(self.strokes)
    
    def distance_to(self, stroke: Stroke) -> float:
        sx, sy = stroke.center
        return math.sqrt((self.center_x - sx)**2 + (self.center_y - sy)**2)
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.strokes:
            return (0.5, 0.5, 0.5, 0.5)
        return (
            min(s.x0 for s in self.strokes),
            min(s.y0 for s in self.strokes),
            max(s.x1 for s in self.strokes),
            max(s.y1 for s in self.strokes),
        )


@dataclass
class RollingSnowball:
    """
    Rolling Snowball Patch Accumulator for real-time stroke recognition.
    
    Accumulates strokes, groups them spatially, and tracks what's been consumed
    (recognized) vs what's still rolling forward.
    """
    buffer: List[Stroke] = field(default_factory=list)
    groups: List[StrokeGroup] = field(default_factory=list)
    spatial_threshold: float = 0.25
    min_strokes_to_try: int = 1
    consumed_stroke_ids: set = field(default_factory=set)
    recognition_history: List[Dict] = field(default_factory=list)
    
    def add_stroke(self, stroke: Stroke) -> Dict[str, Any]:
        """Add stroke to buffer, return recognition status."""
        self.buffer.append(stroke)
        
        # Find or create group
        assigned_group = None
        for group in self.groups:
            if group.distance_to(stroke) < self.spatial_threshold:
                group.add_stroke(stroke)
                assigned_group = group
                break
        
        if assigned_group is None:
            new_group = StrokeGroup(strokes=[stroke])
            new_group._update_center()
            self.groups.append(new_group)
            assigned_group = new_group
        
        # Check for recognition candidates
        candidate = None
        for i, group in enumerate(self.groups):
            if len(group.strokes) >= self.min_strokes_to_try:
                candidate = i
        
        return {
            "action": "try_recognize" if candidate is not None else "wait",
            "buffer_size": len(self.buffer),
            "num_groups": len(self.groups),
            "candidate_group": candidate,
            "candidate_bbox": self.groups[candidate].bbox if candidate is not None else None,
        }
    
    def consume_strokes(self, 
                        recognized_symbol: str,
                        stroke_ids: Optional[List[int]] = None,
                        group_index: Optional[int] = None) -> Dict[str, Any]:
        """Remove recognized strokes from buffer."""
        consumed = []
        
        if group_index is not None and group_index < len(self.groups):
            group = self.groups.pop(group_index)
            consumed = [s.id for s in group.strokes]
        elif stroke_ids:
            consumed = stroke_ids
        
        self.consumed_stroke_ids.update(consumed)
        self.buffer = [s for s in self.buffer if s.id not in consumed]
        
        for group in self.groups:
            group.strokes = [s for s in group.strokes if s.id not in consumed]
        self.groups = [g for g in self.groups if len(g.strokes) > 0]
        
        self.recognition_history.append({
            "symbol": recognized_symbol,
            "consumed_strokes": len(consumed),
            "remaining_strokes": len(self.buffer),
        })
        
        return {
            "consumed_count": len(consumed),
            "remaining_count": len(self.buffer),
            "remaining_groups": len(self.groups),
            "consumed_ids": consumed,
        }
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "num_groups": len(self.groups),
            "groups": [
                {"strokes": len(g.strokes), "center": (g.center_x, g.center_y), "bbox": g.bbox}
                for g in self.groups
            ],
            "total_consumed": len(self.consumed_stroke_ids),
            "history": self.recognition_history[-5:] if self.recognition_history else [],
        }


# Legacy alias
PatchAccumulator = RollingSnowball


# =============================================================================
# Model Input/Output Structures
# =============================================================================

@dataclass
class PatchToken:
    """A token derived from a stroke patch region."""
    stroke_id: int
    patch_idx: int
    embedding: Optional[Any] = None
    bbox: Tuple[float, float, float, float] = (0, 0, 1, 1)
    
    @property
    def token_id(self) -> str:
        return f"P_{self.stroke_id}_{self.patch_idx}"


@dataclass
class ModelInput:
    """
    Input structure for the stroke-level model.
    
    Structure: [LaTeX context] | [Patch buffer] | [CLS] | [INT]
    """
    latex_context: str
    latex_token_ids: List[int] = field(default_factory=list)
    patch_buffer: Dict[int, List[PatchToken]] = field(default_factory=dict)
    total_patches: int = 0
    total_strokes: int = 0
    
    def add_stroke_patches(self, stroke_id: int, patches: List[PatchToken]):
        self.patch_buffer[stroke_id] = patches
        self.total_patches += len(patches)
        self.total_strokes = len(self.patch_buffer)
    
    def remove_strokes(self, stroke_ids: List[int]) -> int:
        removed = 0
        for sid in stroke_ids:
            if sid in self.patch_buffer:
                removed += len(self.patch_buffer[sid])
                del self.patch_buffer[sid]
        self.total_patches -= removed
        self.total_strokes = len(self.patch_buffer)
        return removed
    
    def get_patch_sequence(self) -> List[PatchToken]:
        result = []
        for stroke_id in sorted(self.patch_buffer.keys()):
            result.extend(self.patch_buffer[stroke_id])
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "latex_context": self.latex_context,
            "latex_token_ids": self.latex_token_ids,
            "patch_buffer": {sid: [p.token_id for p in patches] 
                            for sid, patches in self.patch_buffer.items()},
            "total_patches": self.total_patches,
            "total_strokes": self.total_strokes,
        }


@dataclass
class ModelOutput:
    """
    Output from the stroke-level model.
    
    Either <WAIT> (keep accumulating) or recognized symbol(s).
    """
    output_tokens: List[str]
    consumed_stroke_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0
    is_wait: bool = False
    
    @property
    def latex_output(self) -> str:
        if self.is_wait:
            return ""
        return "".join(self.output_tokens)
    
    @classmethod
    def wait(cls) -> "ModelOutput":
        return cls(output_tokens=["<WAIT>"], is_wait=True)
    
    @classmethod
    def recognize(cls, symbol: str, stroke_ids: List[int], 
                  confidence: float = 1.0) -> "ModelOutput":
        return cls(
            output_tokens=[symbol],
            consumed_stroke_ids=stroke_ids,
            confidence=confidence,
            is_wait=False,
        )


# =============================================================================
# Attention-Based Stroke Analysis
# =============================================================================

@dataclass
class StrokeAttentionAnalyzer:
    """Analyze attention to determine which strokes to consume."""
    attention_threshold: float = 0.1
    spatial_threshold: float = 0.3
    
    def analyze(self, 
                patch_attention: Dict[str, float],
                patch_tokens: Dict[int, List[PatchToken]],
                strategy: str = "hybrid") -> List[int]:
        """Determine which strokes to consume based on attention."""
        stroke_attention = {}
        for stroke_id, patches in patch_tokens.items():
            total_attn = sum(patch_attention.get(p.token_id, 0.0) for p in patches)
            stroke_attention[stroke_id] = total_attn
        
        if strategy == "attention":
            return [sid for sid, attn in stroke_attention.items() 
                   if attn >= self.attention_threshold]
        
        elif strategy == "spatial":
            if not stroke_attention:
                return []
            max_stroke = max(stroke_attention, key=stroke_attention.get)
            max_bbox = self._get_stroke_bbox(patch_tokens[max_stroke])
            
            consumed = [max_stroke]
            for sid, patches in patch_tokens.items():
                if sid == max_stroke:
                    continue
                if self._is_nearby(max_bbox, self._get_stroke_bbox(patches)):
                    consumed.append(sid)
            return consumed
        
        else:  # hybrid
            significant = [sid for sid, attn in stroke_attention.items() 
                          if attn >= self.attention_threshold]
            if not significant:
                return []
            
            primary = max(significant, key=lambda s: stroke_attention[s])
            primary_bbox = self._get_stroke_bbox(patch_tokens[primary])
            
            consumed = [primary]
            for sid in significant:
                if sid == primary:
                    continue
                if self._is_nearby(primary_bbox, self._get_stroke_bbox(patch_tokens[sid])):
                    consumed.append(sid)
            return consumed
    
    def _get_stroke_bbox(self, patches: List[PatchToken]) -> Tuple[float, float, float, float]:
        if not patches:
            return (0.5, 0.5, 0.5, 0.5)
        return (
            min(p.bbox[0] for p in patches),
            min(p.bbox[1] for p in patches),
            max(p.bbox[2] for p in patches),
            max(p.bbox[3] for p in patches),
        )
    
    def _is_nearby(self, bbox1, bbox2) -> bool:
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return dist < self.spatial_threshold


def infer_consumed_strokes(attention_weights: Dict[str, float],
                           model_input: ModelInput,
                           strategy: str = "hybrid") -> List[int]:
    """Convenience function to infer which strokes to consume."""
    analyzer = StrokeAttentionAnalyzer()
    return analyzer.analyze(attention_weights, model_input.patch_buffer, strategy)


def process_model_output(model_input: ModelInput, 
                         model_output: ModelOutput) -> ModelInput:
    """Process model output and update input for next step."""
    if model_output.is_wait:
        return model_input
    
    new_context = model_input.latex_context + model_output.latex_output
    
    new_input = ModelInput(
        latex_context=new_context,
        latex_token_ids=model_input.latex_token_ids.copy(),
        patch_buffer={
            sid: patches 
            for sid, patches in model_input.patch_buffer.items()
            if sid not in model_output.consumed_stroke_ids
        },
    )
    new_input.total_patches = sum(len(p) for p in new_input.patch_buffer.values())
    new_input.total_strokes = len(new_input.patch_buffer)
    
    return new_input


# =============================================================================
# Special Tokens
# =============================================================================

STROKE_SPECIAL_TOKENS = {
    "<WAIT>": "Model should wait for more strokes",
    "<INCOMPLETE>": "Current patch is incomplete",
    "<MERGE>": "Merge nearby patches",
    "<RECOGNIZE>": "Trigger recognition",
}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("Stroke-Level Data Types")
    print("=" * 60)
    print("\nAvailable classes:")
    print("  - Stroke, StrokeSequence")
    print("  - RollingSnowball (PatchAccumulator)")
    print("  - PatchToken, ModelInput, ModelOutput")
    print("  - StrokeAttentionAnalyzer")
    print("\nFor rendering and dataset generation, use:")
    print("  from scripts.stroke_renderer import StrokeRenderer")
    print("  from scripts.stroke_dataset import StrokeDatasetGenerator")
