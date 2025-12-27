"""
Tree to LaTeX Conversion

Hardcoded conversion from (content tokens + relations) to LaTeX string.

The model outputs:
- Content tokens: "a", "+", "b", "2", "x", "y", etc.
- Relations: RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE

This module converts the tree structure to valid LaTeX:
- ABOVE/BELOW → \\frac{above}{below}
- SUP → ^{...}
- SUB → _{...}
- RIGHT → concatenation
- INSIDE → wrap with parent (\\sqrt{...}, etc.)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Set
import re


class RelationType(IntEnum):
    """Relation types between tokens in the expression tree."""
    RIGHT = 0   # Horizontal sequence: a → b
    SUP = 1     # Superscript: a^b
    SUB = 2     # Subscript: a_b
    ABOVE = 3   # Fraction numerator: above the bar
    BELOW = 4   # Fraction denominator: below the bar
    INSIDE = 5  # Inside a structure: sqrt, matrix cell, etc.


@dataclass
class TreeNode:
    """A node in the expression tree."""
    token: str
    index: int
    children: Dict[RelationType, List['TreeNode']] = field(default_factory=dict)
    parent: Optional['TreeNode'] = None
    parent_relation: Optional[RelationType] = None
    
    def add_child(self, child: 'TreeNode', relation: RelationType):
        if relation not in self.children:
            self.children[relation] = []
        self.children[relation].append(child)
        child.parent = self
        child.parent_relation = relation
    
    def get_children(self, relation: RelationType) -> List['TreeNode']:
        return self.children.get(relation, [])
    
    def has_children(self, relation: RelationType) -> bool:
        return relation in self.children and len(self.children[relation]) > 0


class ExpressionTree:
    """
    Expression tree built from content tokens and actions.
    
    Each action is (parent_idx, relation_type) indicating where
    the new token attaches in the tree.
    """
    
    def __init__(self):
        self.nodes: List[TreeNode] = []
        self.root: Optional[TreeNode] = None
    
    def add_token(self, token: str, action: Tuple[int, RelationType]) -> TreeNode:
        """
        Add a token to the tree.
        
        Args:
            token: Content token (e.g., "a", "+", "2")
            action: (parent_idx, relation) where to attach
            
        Returns:
            The created TreeNode
        """
        node = TreeNode(token=token, index=len(self.nodes))
        self.nodes.append(node)
        
        parent_idx, relation = action
        
        if parent_idx < 0 or self.root is None:
            # First token becomes root
            self.root = node
        else:
            # Attach to parent
            parent = self.nodes[parent_idx]
            parent.add_child(node, relation)
        
        return node
    
    def build_from_sequence(
        self, 
        tokens: List[str], 
        actions: List[Tuple[int, RelationType]]
    ) -> 'ExpressionTree':
        """Build tree from token and action sequences."""
        assert len(tokens) == len(actions), "Tokens and actions must match"
        
        for token, action in zip(tokens, actions):
            self.add_token(token, action)
        
        return self


class TreeToLatex:
    """
    Convert expression tree to LaTeX string.
    
    Hardcoded rules:
    - RIGHT children: concatenate
    - SUP children: ^{...}
    - SUB children: _{...}
    - ABOVE + BELOW: \\frac{above}{below}
    - INSIDE: depends on parent token
    """
    
    # Tokens that need special handling
    FRAC_TRIGGERS = {'/', '-', '—', 'frac', '\\frac'}
    SQRT_TRIGGERS = {'sqrt', '\\sqrt', '√'}
    SUM_TRIGGERS = {'sum', '\\sum', '∑'}
    INT_TRIGGERS = {'int', '\\int', '∫'}
    
    def __init__(self):
        pass
    
    def convert(self, tree: ExpressionTree) -> str:
        """Convert tree to LaTeX string."""
        if tree.root is None:
            return ""
        return self._node_to_latex(tree.root)
    
    def _node_to_latex(self, node: TreeNode) -> str:
        """Recursively convert a node and its children to LaTeX."""
        result = self._token_to_latex(node.token)
        
        # Check for fraction structure (ABOVE + BELOW)
        has_above = node.has_children(RelationType.ABOVE)
        has_below = node.has_children(RelationType.BELOW)
        
        if has_above or has_below:
            # This node is a fraction bar
            above_latex = self._children_to_latex(node, RelationType.ABOVE)
            below_latex = self._children_to_latex(node, RelationType.BELOW)
            
            # If node is a fraction bar, wrap in \frac
            if node.token in self.FRAC_TRIGGERS or (has_above and has_below):
                result = f"\\frac{{{above_latex}}}{{{below_latex}}}"
            else:
                # Partial fraction (only above or below)
                if has_above:
                    result = f"{result}^{{{above_latex}}}"  # Treat as superscript
                if has_below:
                    result = f"{result}_{{{below_latex}}}"  # Treat as subscript
        else:
            # Normal token processing
            result = self._token_to_latex(node.token)
        
        # Handle superscript
        if node.has_children(RelationType.SUP):
            sup_latex = self._children_to_latex(node, RelationType.SUP)
            result = f"{result}^{{{sup_latex}}}"
        
        # Handle subscript
        if node.has_children(RelationType.SUB):
            sub_latex = self._children_to_latex(node, RelationType.SUB)
            result = f"{result}_{{{sub_latex}}}"
        
        # Handle INSIDE (for sqrt, etc.)
        if node.has_children(RelationType.INSIDE):
            inside_latex = self._children_to_latex(node, RelationType.INSIDE)
            if node.token in self.SQRT_TRIGGERS:
                result = f"\\sqrt{{{inside_latex}}}"
            elif node.token in self.SUM_TRIGGERS:
                result = f"\\sum{{{inside_latex}}}"
            elif node.token in self.INT_TRIGGERS:
                result = f"\\int{{{inside_latex}}}"
            else:
                result = f"{result}{{{inside_latex}}}"
        
        # Handle RIGHT children (horizontal sequence)
        if node.has_children(RelationType.RIGHT):
            right_latex = self._children_to_latex(node, RelationType.RIGHT)
            result = f"{result}{right_latex}"
        
        return result
    
    def _children_to_latex(self, node: TreeNode, relation: RelationType) -> str:
        """Convert all children of a specific relation to LaTeX."""
        children = node.get_children(relation)
        if not children:
            return ""
        return "".join(self._node_to_latex(child) for child in children)
    
    def _token_to_latex(self, token: str) -> str:
        """Convert a single token to its LaTeX representation."""
        # Already LaTeX command
        if token.startswith('\\'):
            return token
        
        # Greek letters
        greek_map = {
            'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma',
            'delta': '\\delta', 'epsilon': '\\epsilon', 'theta': '\\theta',
            'lambda': '\\lambda', 'mu': '\\mu', 'pi': '\\pi',
            'sigma': '\\sigma', 'omega': '\\omega', 'phi': '\\phi',
            'psi': '\\psi', 'rho': '\\rho', 'tau': '\\tau',
        }
        if token.lower() in greek_map:
            return greek_map[token.lower()]
        
        # Special symbols
        symbol_map = {
            '*': '\\cdot', '...': '\\ldots', 'inf': '\\infty',
            '>=': '\\geq', '<=': '\\leq', '!=': '\\neq',
            '->': '\\rightarrow', '<-': '\\leftarrow',
            'sqrt': '\\sqrt', 'sum': '\\sum', 'int': '\\int',
            'prod': '\\prod', 'lim': '\\lim',
        }
        if token in symbol_map:
            return symbol_map[token]
        
        # Fraction bar - handled specially in parent
        if token in self.FRAC_TRIGGERS:
            return ""
        
        # Default: return as-is
        return token


def tokens_and_actions_to_latex(
    tokens: List[str],
    actions: List[Tuple[int, int]]  # (parent_idx, relation_int)
) -> str:
    """
    Convenience function to convert tokens + actions to LaTeX.
    
    Args:
        tokens: List of content tokens
        actions: List of (parent_idx, relation_type) tuples
        
    Returns:
        LaTeX string
    """
    # Convert action ints to RelationType
    typed_actions = [(p, RelationType(r)) for p, r in actions]
    
    # Build tree
    tree = ExpressionTree()
    tree.build_from_sequence(tokens, typed_actions)
    
    # Convert to LaTeX
    converter = TreeToLatex()
    return converter.convert(tree)


# =============================================================================
# Spatial Relation Inference
# =============================================================================

class SpatialRelationInferrer:
    """
    Infer relations between symbols based on their bounding boxes.
    
    This is used during training data generation to create
    ground-truth action labels from MathWriting bboxes.
    """
    
    def __init__(
        self,
        vertical_threshold: float = 0.3,   # Fraction of height for SUP/SUB
        fraction_overlap: float = 0.5,     # X-overlap for fraction detection
        size_ratio_threshold: float = 0.6  # Size ratio for SUP/SUB vs ABOVE/BELOW
    ):
        self.vertical_threshold = vertical_threshold
        self.fraction_overlap = fraction_overlap
        self.size_ratio_threshold = size_ratio_threshold
    
    def infer_relation(
        self,
        child_bbox: Tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max)
        parent_bbox: Tuple[float, float, float, float],
        parent_token: str = ""
    ) -> RelationType:
        """
        Infer the relation between child and parent based on positions.
        
        Args:
            child_bbox: Child symbol bounding box
            parent_bbox: Parent symbol bounding box
            parent_token: Parent token (for context)
            
        Returns:
            Inferred RelationType
        """
        cx_min, cy_min, cx_max, cy_max = child_bbox
        px_min, py_min, px_max, py_max = parent_bbox
        
        # Child centers
        cx_center = (cx_min + cx_max) / 2
        cy_center = (cy_min + cy_max) / 2
        
        # Parent centers and dimensions
        px_center = (px_min + px_max) / 2
        py_center = (py_min + py_max) / 2
        p_height = py_max - py_min
        p_width = px_max - px_min
        
        # Child dimensions
        c_height = cy_max - cy_min
        c_width = cx_max - cx_min
        
        # Check horizontal overlap
        x_overlap = self._compute_overlap(cx_min, cx_max, px_min, px_max)
        x_overlap_ratio = x_overlap / max(p_width, 1e-6)
        
        # Vertical relationship
        if cy_max < py_min:
            # Child is above parent
            if x_overlap_ratio > self.fraction_overlap:
                # Significant overlap - likely ABOVE (numerator)
                return RelationType.ABOVE
            elif cx_center > px_max:
                # Child is to the right and above - likely SUP
                return RelationType.SUP
            else:
                return RelationType.ABOVE
        
        elif cy_min > py_max:
            # Child is below parent
            if x_overlap_ratio > self.fraction_overlap:
                # Significant overlap - likely BELOW (denominator)
                return RelationType.BELOW
            elif cx_center > px_max:
                # Child is to the right and below - likely SUB
                return RelationType.SUB
            else:
                return RelationType.BELOW
        
        else:
            # Vertical overlap - check horizontal position
            if cx_min > px_max:
                # Child is to the right
                # Check if it's a superscript (smaller and higher)
                size_ratio = (c_height * c_width) / max(p_height * p_width, 1e-6)
                
                if cy_center < py_center - p_height * self.vertical_threshold:
                    # Higher than center - superscript
                    if size_ratio < self.size_ratio_threshold:
                        return RelationType.SUP
                
                if cy_center > py_center + p_height * self.vertical_threshold:
                    # Lower than center - subscript
                    if size_ratio < self.size_ratio_threshold:
                        return RelationType.SUB
                
                # Default: horizontal sequence
                return RelationType.RIGHT
            
            elif cx_max < px_min:
                # Child is to the left - unusual, treat as error or special case
                return RelationType.RIGHT  # Fallback
            
            else:
                # Overlapping - might be INSIDE
                return RelationType.INSIDE
    
    def _compute_overlap(
        self, 
        a_min: float, a_max: float, 
        b_min: float, b_max: float
    ) -> float:
        """Compute overlap length between two intervals."""
        return max(0, min(a_max, b_max) - max(a_min, b_min))
    
    def infer_sequence_relations(
        self,
        tokens: List[str],
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[int, RelationType]]:
        """
        Infer relations for a sequence of tokens based on bboxes.
        
        Processes tokens in order and assigns each to its most likely parent.
        
        Args:
            tokens: List of content tokens
            bboxes: List of bounding boxes (x_min, y_min, x_max, y_max)
            
        Returns:
            List of (parent_idx, relation) actions
        """
        if len(tokens) == 0:
            return []
        
        if len(tokens) != len(bboxes):
            raise ValueError(f"Tokens ({len(tokens)}) and bboxes ({len(bboxes)}) must match")
        
        actions = []
        
        for i, (token, bbox) in enumerate(zip(tokens, bboxes)):
            if i == 0:
                # First token: attach to root
                actions.append((-1, RelationType.RIGHT))
            else:
                # Find best parent
                parent_idx, relation = self._find_best_parent(
                    i, bbox, tokens[:i], bboxes[:i]
                )
                actions.append((parent_idx, relation))
        
        return actions
    
    def _find_best_parent(
        self,
        child_idx: int,
        child_bbox: Tuple[float, float, float, float],
        prev_tokens: List[str],
        prev_bboxes: List[Tuple[float, float, float, float]]
    ) -> Tuple[int, RelationType]:
        """Find the best parent for a new token."""
        
        # Simple heuristic: find nearest token that makes sense
        cx_min, cy_min, cx_max, cy_max = child_bbox
        cx_center = (cx_min + cx_max) / 2
        cy_center = (cy_min + cy_max) / 2
        
        best_parent = len(prev_tokens) - 1  # Default: attach to previous
        best_relation = RelationType.RIGHT
        best_score = float('inf')
        
        for i, (token, bbox) in enumerate(zip(prev_tokens, prev_bboxes)):
            px_min, py_min, px_max, py_max = bbox
            px_center = (px_min + px_max) / 2
            py_center = (py_min + py_max) / 2
            
            # Compute distance
            dist = ((cx_center - px_center) ** 2 + (cy_center - py_center) ** 2) ** 0.5
            
            # Infer relation
            relation = self.infer_relation(child_bbox, bbox, token)
            
            # Score: prefer RIGHT relations, penalize ABOVE/BELOW unless close
            score = dist
            if relation == RelationType.RIGHT:
                score *= 0.8  # Prefer horizontal
            elif relation in [RelationType.SUP, RelationType.SUB]:
                score *= 0.9  # Slightly prefer over ABOVE/BELOW
            
            if score < best_score:
                best_score = score
                best_parent = i
                best_relation = relation
        
        return best_parent, best_relation


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tree to LaTeX Conversion Tests")
    print("=" * 60)
    
    converter = TreeToLatex()
    
    # Test 1: Simple sequence a+b
    print("\n1. Simple sequence: a + b")
    tokens = ["a", "+", "b"]
    actions = [(-1, 0), (0, 0), (1, 0)]  # All RIGHT
    latex = tokens_and_actions_to_latex(tokens, actions)
    print(f"   Tokens: {tokens}")
    print(f"   Actions: {actions}")
    print(f"   LaTeX: {latex}")
    assert latex == "a+b", f"Expected 'a+b', got '{latex}'"
    
    # Test 2: Superscript x^2
    print("\n2. Superscript: x^2")
    tokens = ["x", "2"]
    actions = [(-1, 0), (0, 1)]  # 2 is SUP of x
    latex = tokens_and_actions_to_latex(tokens, actions)
    print(f"   Tokens: {tokens}")
    print(f"   Actions: {actions}")
    print(f"   LaTeX: {latex}")
    assert latex == "x^{2}", f"Expected 'x^{{2}}', got '{latex}'"
    
    # Test 3: Subscript a_i
    print("\n3. Subscript: a_i")
    tokens = ["a", "i"]
    actions = [(-1, 0), (0, 2)]  # i is SUB of a
    latex = tokens_and_actions_to_latex(tokens, actions)
    print(f"   Tokens: {tokens}")
    print(f"   Actions: {actions}")
    print(f"   LaTeX: {latex}")
    assert latex == "a_{i}", f"Expected 'a_{{i}}', got '{latex}'"
    
    # Test 4: Fraction a/b (using ABOVE/BELOW)
    print("\n4. Fraction: a/b")
    tokens = ["/", "a", "b"]  # / is the fraction bar
    actions = [(-1, 0), (0, 3), (0, 4)]  # a ABOVE /, b BELOW /
    latex = tokens_and_actions_to_latex(tokens, actions)
    print(f"   Tokens: {tokens}")
    print(f"   Actions: {actions}")
    print(f"   LaTeX: {latex}")
    assert "\\frac{a}{b}" in latex, f"Expected '\\frac{{a}}{{b}}', got '{latex}'"
    
    # Test 5: Complex: x^2 + y_i
    print("\n5. Complex: x^2 + y_i")
    tokens = ["x", "2", "+", "y", "i"]
    actions = [(-1, 0), (0, 1), (0, 0), (2, 0), (3, 2)]
    # x, 2^x, + right of x, y right of +, i sub of y
    latex = tokens_and_actions_to_latex(tokens, actions)
    print(f"   Tokens: {tokens}")
    print(f"   Actions: {actions}")
    print(f"   LaTeX: {latex}")
    
    # Test 6: Spatial relation inference
    print("\n6. Spatial Relation Inference")
    inferrer = SpatialRelationInferrer()
    
    # Superscript case
    parent_bbox = (0, 10, 20, 30)  # x at y=10-30
    child_bbox = (22, 5, 30, 15)   # 2 at y=5-15 (higher, smaller, to the right)
    relation = inferrer.infer_relation(child_bbox, parent_bbox)
    print(f"   Parent bbox: {parent_bbox}")
    print(f"   Child bbox: {child_bbox}")
    print(f"   Inferred: {relation.name}")
    assert relation == RelationType.SUP, f"Expected SUP, got {relation.name}"
    
    # Fraction case
    parent_bbox = (0, 20, 40, 25)  # fraction bar
    child_bbox = (5, 5, 35, 18)    # numerator (above)
    relation = inferrer.infer_relation(child_bbox, parent_bbox)
    print(f"   Fraction bar: {parent_bbox}")
    print(f"   Numerator: {child_bbox}")
    print(f"   Inferred: {relation.name}")
    assert relation == RelationType.ABOVE, f"Expected ABOVE, got {relation.name}"
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)



