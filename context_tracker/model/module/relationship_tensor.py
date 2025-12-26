"""
Relationship Tensor Module

Converts LaTeX strings into structured representations:
- Content tokens (no syntax like _, ^, {, })
- 3D Relationship tensor [num_tokens × num_tokens × num_relations]

Relations encode tree structure:
- RIGHT: horizontal sequence (a + b → + is RIGHT of a)
- SUP: superscript (a^x → x is SUP of a)
- SUB: subscript (a_t → t is SUB of a)
- ABOVE: numerator (frac{a}{b} → a is ABOVE frac)
- BELOW: denominator (frac{a}{b} → b is BELOW frac)
- INSIDE: content of sqrt, etc.
"""

import re
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import IntEnum


class RelationType(IntEnum):
    """Relation types for tree structure"""
    RIGHT = 0    # Horizontal sequence
    SUP = 1      # Superscript
    SUB = 2      # Subscript
    ABOVE = 3    # Numerator
    BELOW = 4    # Denominator
    INSIDE = 5   # Content of sqrt, overline, etc.
    
    @classmethod
    def num_types(cls) -> int:
        return len(cls)
    
    @classmethod
    def names(cls) -> List[str]:
        return [r.name for r in cls]


@dataclass
class ParsedNode:
    """A node in the parsed LaTeX tree"""
    content: str
    index: int = -1
    children: Dict[str, List['ParsedNode']] = field(default_factory=dict)
    
    def add_child(self, relation: str, child: 'ParsedNode'):
        if relation not in self.children:
            self.children[relation] = []
        self.children[relation].append(child)


@dataclass
class ParsedLatex:
    """Result of parsing a LaTeX string"""
    tokens: List[str]
    relations: List[Tuple[int, int, RelationType]]  # (child_idx, parent_idx, relation)
    tensor: Optional[torch.Tensor] = None  # [num_tokens, num_tokens, num_relations]
    
    def build_tensor(self) -> torch.Tensor:
        """Build 3D relationship tensor"""
        n = len(self.tokens)
        r = RelationType.num_types()
        tensor = torch.zeros(n, n, r, dtype=torch.float32)
        
        for child_idx, parent_idx, rel_type in self.relations:
            if 0 <= child_idx < n and 0 <= parent_idx < n:
                tensor[child_idx, parent_idx, rel_type] = 1.0
        
        self.tensor = tensor
        return tensor


class LatexRelationshipParser:
    """
    Parse LaTeX strings into content tokens and relationship tensors.
    
    Handles:
    - Basic symbols: a, b, x, 1, 2, +, -, =, etc.
    - Subscripts: a_t, a_{xy}
    - Superscripts: a^2, a^{xy}
    - Fractions: \\frac{a}{b}
    - Square roots: \\sqrt{x}, \\sqrt[n]{x}
    - Greek letters: \\alpha, \\beta, etc.
    - Operators: \\sum, \\int, \\prod
    - Grouping: {a+b}
    """
    
    # Tokens to skip (LaTeX syntax, not content)
    SKIP_TOKENS = {'{', '}', '_', '^', '\\left', '\\right', '\\,', '\\;', '\\:', '\\ '}
    
    # Commands that take arguments
    COMMANDS_WITH_ARGS = {
        'frac': 2,      # \frac{num}{denom}
        'sqrt': 1,      # \sqrt{x} or \sqrt[n]{x}
        'overline': 1,
        'underline': 1,
        'hat': 1,
        'bar': 1,
        'vec': 1,
        'dot': 1,
        'ddot': 1,
        'tilde': 1,
        'text': 1,
        'mathrm': 1,
        'mathbf': 1,
        'mathit': 1,
    }
    
    # Greek letters and symbols (map to clean names)
    SYMBOLS = {
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
        'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Pi', 'Rho', 'Sigma',
        'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega',
        'infty', 'partial', 'nabla', 'forall', 'exists', 'emptyset',
        'sum', 'prod', 'int', 'oint', 'lim', 'log', 'ln', 'sin', 'cos', 'tan',
        'cdot', 'times', 'div', 'pm', 'mp', 'leq', 'geq', 'neq', 'approx',
        'equiv', 'subset', 'supset', 'in', 'notin', 'cup', 'cap',
        'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow', 'leftrightarrow',
        'to', 'mapsto', 'implies', 'iff',
    }
    
    # Operators that are content
    OPERATORS = {'+', '-', '=', '<', '>', '/', '*', '!', '|', ',', '.', ';', ':', '(', ')', '[', ']'}
    
    def __init__(self):
        self.tokens: List[str] = []
        self.relations: List[Tuple[int, int, RelationType]] = []
        self.pos = 0
        self.latex = ""
    
    def parse(self, latex: str) -> ParsedLatex:
        """Parse a LaTeX string into tokens and relations"""
        self.tokens = []
        self.relations = []
        self.pos = 0
        self.latex = latex.strip()
        
        # Parse the expression
        root_nodes = self._parse_sequence()
        
        # Connect sequential nodes with RIGHT relation
        self._connect_sequence(root_nodes)
        
        result = ParsedLatex(
            tokens=self.tokens,
            relations=self.relations
        )
        result.build_tensor()
        return result
    
    def _parse_sequence(self, until_chars: str = "") -> List[int]:
        """Parse a sequence of tokens, return their indices"""
        node_indices = []
        
        while self.pos < len(self.latex):
            # Check for termination
            if self.latex[self.pos] in until_chars:
                break
            
            # Skip whitespace
            if self.latex[self.pos].isspace():
                self.pos += 1
                continue
            
            # Parse next element
            idx = self._parse_element()
            if idx >= 0:
                node_indices.append(idx)
        
        return node_indices
    
    def _parse_element(self) -> int:
        """Parse a single element, return its token index (or -1 if skipped)"""
        if self.pos >= len(self.latex):
            return -1
        
        char = self.latex[self.pos]
        
        # Skip certain chars
        if char in ' \t\n':
            self.pos += 1
            return -1
        
        # Grouping: {content}
        if char == '{':
            self.pos += 1
            inner_indices = self._parse_sequence('}')
            if self.pos < len(self.latex) and self.latex[self.pos] == '}':
                self.pos += 1
            # Return first element of group (simplified)
            return inner_indices[0] if inner_indices else -1
        
        # End of group
        if char == '}':
            return -1
        
        # LaTeX command
        if char == '\\':
            return self._parse_command()
        
        # Subscript
        if char == '_':
            self.pos += 1
            return -1  # Handled by caller
        
        # Superscript
        if char == '^':
            self.pos += 1
            return -1  # Handled by caller
        
        # Operators
        if char in self.OPERATORS:
            self.pos += 1
            return self._add_token(char)
        
        # Alphanumeric
        if char.isalnum():
            self.pos += 1
            idx = self._add_token(char)
            # Check for subscript/superscript
            self._check_sub_sup(idx)
            return idx
        
        # Unknown char - skip
        self.pos += 1
        return -1
    
    def _parse_command(self) -> int:
        """Parse a LaTeX command like \\frac, \\alpha, etc."""
        self.pos += 1  # Skip backslash
        
        # Read command name
        cmd_start = self.pos
        while self.pos < len(self.latex) and self.latex[self.pos].isalpha():
            self.pos += 1
        cmd_name = self.latex[cmd_start:self.pos]
        
        if not cmd_name:
            return -1
        
        # Handle special commands
        if cmd_name == 'frac':
            return self._parse_frac()
        elif cmd_name == 'sqrt':
            return self._parse_sqrt()
        elif cmd_name in self.COMMANDS_WITH_ARGS:
            return self._parse_command_with_arg(cmd_name)
        elif cmd_name in self.SYMBOLS:
            idx = self._add_token(cmd_name)
            self._check_sub_sup(idx)
            return idx
        else:
            # Unknown command - treat as symbol
            idx = self._add_token(cmd_name)
            self._check_sub_sup(idx)
            return idx
    
    def _parse_frac(self) -> int:
        """Parse \\frac{numerator}{denominator}"""
        # Add frac token
        frac_idx = self._add_token('frac')
        
        # Skip whitespace
        while self.pos < len(self.latex) and self.latex[self.pos].isspace():
            self.pos += 1
        
        # Parse numerator
        if self.pos < len(self.latex) and self.latex[self.pos] == '{':
            self.pos += 1
            num_indices = self._parse_sequence('}')
            if self.pos < len(self.latex) and self.latex[self.pos] == '}':
                self.pos += 1
            # Connect numerator to frac with ABOVE
            self._connect_sequence(num_indices)
            if num_indices:
                self.relations.append((num_indices[0], frac_idx, RelationType.ABOVE))
        
        # Skip whitespace
        while self.pos < len(self.latex) and self.latex[self.pos].isspace():
            self.pos += 1
        
        # Parse denominator
        if self.pos < len(self.latex) and self.latex[self.pos] == '{':
            self.pos += 1
            denom_indices = self._parse_sequence('}')
            if self.pos < len(self.latex) and self.latex[self.pos] == '}':
                self.pos += 1
            # Connect denominator to frac with BELOW
            self._connect_sequence(denom_indices)
            if denom_indices:
                self.relations.append((denom_indices[0], frac_idx, RelationType.BELOW))
        
        self._check_sub_sup(frac_idx)
        return frac_idx
    
    def _parse_sqrt(self) -> int:
        """Parse \\sqrt{content} or \\sqrt[n]{content}"""
        sqrt_idx = self._add_token('sqrt')
        
        # Skip whitespace
        while self.pos < len(self.latex) and self.latex[self.pos].isspace():
            self.pos += 1
        
        # Optional index [n]
        if self.pos < len(self.latex) and self.latex[self.pos] == '[':
            self.pos += 1
            # Parse index
            idx_start = self.pos
            while self.pos < len(self.latex) and self.latex[self.pos] != ']':
                self.pos += 1
            index_content = self.latex[idx_start:self.pos]
            if self.pos < len(self.latex):
                self.pos += 1  # Skip ]
            if index_content:
                idx_token = self._add_token(index_content)
                self.relations.append((idx_token, sqrt_idx, RelationType.SUP))
        
        # Skip whitespace
        while self.pos < len(self.latex) and self.latex[self.pos].isspace():
            self.pos += 1
        
        # Parse content
        if self.pos < len(self.latex) and self.latex[self.pos] == '{':
            self.pos += 1
            content_indices = self._parse_sequence('}')
            if self.pos < len(self.latex) and self.latex[self.pos] == '}':
                self.pos += 1
            self._connect_sequence(content_indices)
            if content_indices:
                self.relations.append((content_indices[0], sqrt_idx, RelationType.INSIDE))
        
        return sqrt_idx
    
    def _parse_command_with_arg(self, cmd_name: str) -> int:
        """Parse a command that takes one argument"""
        cmd_idx = self._add_token(cmd_name)
        
        # Skip whitespace
        while self.pos < len(self.latex) and self.latex[self.pos].isspace():
            self.pos += 1
        
        # Parse argument
        if self.pos < len(self.latex) and self.latex[self.pos] == '{':
            self.pos += 1
            arg_indices = self._parse_sequence('}')
            if self.pos < len(self.latex) and self.latex[self.pos] == '}':
                self.pos += 1
            self._connect_sequence(arg_indices)
            if arg_indices:
                self.relations.append((arg_indices[0], cmd_idx, RelationType.INSIDE))
        
        self._check_sub_sup(cmd_idx)
        return cmd_idx
    
    def _check_sub_sup(self, parent_idx: int):
        """Check for subscript/superscript after current element"""
        while self.pos < len(self.latex):
            # Skip whitespace
            while self.pos < len(self.latex) and self.latex[self.pos].isspace():
                self.pos += 1
            
            if self.pos >= len(self.latex):
                break
            
            char = self.latex[self.pos]
            
            if char == '_':
                self.pos += 1
                # Parse subscript
                sub_indices = self._parse_subscript_content()
                self._connect_sequence(sub_indices)
                if sub_indices:
                    self.relations.append((sub_indices[0], parent_idx, RelationType.SUB))
            elif char == '^':
                self.pos += 1
                # Parse superscript
                sup_indices = self._parse_subscript_content()
                self._connect_sequence(sup_indices)
                if sup_indices:
                    self.relations.append((sup_indices[0], parent_idx, RelationType.SUP))
            else:
                break
    
    def _parse_subscript_content(self) -> List[int]:
        """Parse content after _ or ^"""
        # Skip whitespace
        while self.pos < len(self.latex) and self.latex[self.pos].isspace():
            self.pos += 1
        
        if self.pos >= len(self.latex):
            return []
        
        if self.latex[self.pos] == '{':
            self.pos += 1
            indices = self._parse_sequence('}')
            if self.pos < len(self.latex) and self.latex[self.pos] == '}':
                self.pos += 1
            return indices
        else:
            # Single character or command
            idx = self._parse_element()
            return [idx] if idx >= 0 else []
    
    def _connect_sequence(self, indices: List[int]):
        """Connect a sequence of nodes with RIGHT relations"""
        for i in range(1, len(indices)):
            if indices[i] >= 0 and indices[i-1] >= 0:
                self.relations.append((indices[i], indices[i-1], RelationType.RIGHT))
    
    def _add_token(self, content: str) -> int:
        """Add a token and return its index"""
        idx = len(self.tokens)
        self.tokens.append(content)
        return idx


def visualize_relationship_tensor(parsed: ParsedLatex, title: str = "Relationship Tensor") -> str:
    """
    Create ASCII visualization of the relationship tensor.
    
    Returns a string that can be printed.
    """
    tokens = parsed.tokens
    relations = parsed.relations
    n = len(tokens)
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"  {title}")
    lines.append("=" * 80)
    lines.append("")
    
    # Token list
    lines.append("TOKENS (content only, no LaTeX syntax):")
    lines.append("-" * 40)
    for i, tok in enumerate(tokens):
        lines.append(f"  [{i}] {tok}")
    lines.append("")
    
    # Relations list
    lines.append("RELATIONS (child -> parent, type):")
    lines.append("-" * 40)
    for child_idx, parent_idx, rel_type in relations:
        child_tok = tokens[child_idx] if child_idx < len(tokens) else "?"
        parent_tok = tokens[parent_idx] if parent_idx < len(tokens) else "?"
        lines.append(f"  {child_tok}[{child_idx}] -> {parent_tok}[{parent_idx}], {RelationType(rel_type).name}")
    lines.append("")
    
    # Tensor visualization (one slice per relation type)
    lines.append("TENSOR SLICES [child × parent]:")
    lines.append("-" * 40)
    
    if parsed.tensor is not None:
        tensor = parsed.tensor
        
        for rel_idx, rel_name in enumerate(RelationType.names()):
            # Check if this relation type has any entries
            slice_sum = tensor[:, :, rel_idx].sum().item()
            if slice_sum == 0:
                continue
            
            lines.append(f"\n  {rel_name} relations:")
            
            # Header row
            max_tok_len = max(len(t) for t in tokens) if tokens else 3
            header = "  " + " " * (max_tok_len + 4)
            for j, tok in enumerate(tokens):
                header += f" {tok[:3]:>3}"
            lines.append(header)
            
            # Data rows
            for i, child_tok in enumerate(tokens):
                row = f"  {child_tok:>{max_tok_len}} [{i}]"
                for j in range(n):
                    val = tensor[i, j, rel_idx].item()
                    if val > 0:
                        row += "   X"
                    else:
                        row += "   ."
                lines.append(row)
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def visualize_tree(parsed: ParsedLatex) -> str:
    """
    Create ASCII tree visualization of the parsed structure.
    """
    tokens = parsed.tokens
    relations = parsed.relations
    
    if not tokens:
        return "Empty expression"
    
    # Build adjacency for tree drawing
    # Find root nodes (tokens with no parent)
    has_parent = set()
    children_map = {}  # parent_idx -> [(child_idx, rel_type), ...]
    
    for child_idx, parent_idx, rel_type in relations:
        has_parent.add(child_idx)
        if parent_idx not in children_map:
            children_map[parent_idx] = []
        children_map[parent_idx].append((child_idx, rel_type))
    
    roots = [i for i in range(len(tokens)) if i not in has_parent]
    
    lines = []
    lines.append("TREE STRUCTURE:")
    lines.append("-" * 40)
    
    def draw_node(idx: int, prefix: str = "", is_last: bool = True, relation: str = ""):
        connector = "+-- " if is_last else "|-- "
        rel_str = f"[{relation}] " if relation else ""
        lines.append(f"{prefix}{connector}{rel_str}{tokens[idx]}")
        
        children = children_map.get(idx, [])
        # Sort children by relation type for consistent output
        children.sort(key=lambda x: (x[1], x[0]))
        
        child_prefix = prefix + ("    " if is_last else "|   ")
        for i, (child_idx, rel_type) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            draw_node(child_idx, child_prefix, is_last_child, RelationType(rel_type).name)
    
    # Draw from roots (connect roots with RIGHT if multiple)
    for i, root_idx in enumerate(roots):
        if i > 0:
            lines.append("    |")
            lines.append("    +-- [NEXT]")
        draw_node(root_idx, "    ", True, "ROOT" if i == 0 else "RIGHT")
    
    return "\n".join(lines)


class RelationshipTensorModule(nn.Module):
    """
    Neural network module for working with relationship tensors.
    
    Can be used to:
    1. Embed relationship information into token representations
    2. Provide action embeddings for position prediction
    """
    
    def __init__(self, d_model: int, num_relations: int = None):
        super().__init__()
        if num_relations is None:
            num_relations = RelationType.num_types()
        
        self.d_model = d_model
        self.num_relations = num_relations
        
        # Learnable embeddings for each relation type
        self.relation_embeddings = nn.Embedding(num_relations, d_model)
        
        # Initialize with small values
        nn.init.normal_(self.relation_embeddings.weight, std=0.02)
    
    def get_relation_embedding(self, rel_type: int) -> torch.Tensor:
        """Get embedding for a relation type"""
        return self.relation_embeddings(torch.tensor(rel_type))
    
    def build_action_embeddings(
        self, 
        token_embeddings: torch.Tensor,  # [B, N, d_model]
    ) -> torch.Tensor:
        """
        Build action embeddings for position prediction.
        
        Each action = (parent_token, relation_type)
        Action embedding = parent_embedding + relation_embedding
        
        Returns: [B, N * num_relations, d_model]
        """
        B, N, d = token_embeddings.shape
        R = self.num_relations
        
        # Get all relation embeddings: [R, d]
        rel_emb = self.relation_embeddings.weight
        
        # Outer sum: token_emb[i] + rel_emb[j] for all i,j
        # [B, N, 1, d] + [1, 1, R, d] → [B, N, R, d]
        action_emb = (
            token_embeddings.unsqueeze(2) + 
            rel_emb.unsqueeze(0).unsqueeze(0)
        )
        
        # Reshape to [B, N*R, d]
        action_emb = action_emb.view(B, N * R, d)
        
        return action_emb
    
    def decode_action_index(self, action_idx: int, num_tokens: int) -> Tuple[int, str]:
        """
        Convert action index to (parent_idx, relation_name).
        
        Actions are ordered: [(tok0, rel0), (tok0, rel1), ..., (tok1, rel0), ...]
        """
        parent_idx = action_idx // self.num_relations
        rel_idx = action_idx % self.num_relations
        rel_name = RelationType(rel_idx).name
        return parent_idx, rel_name
    
    def encode_action(self, parent_idx: int, rel_type: RelationType) -> int:
        """Convert (parent_idx, relation) to action index"""
        return parent_idx * self.num_relations + int(rel_type)


# Convenience function
def parse_latex(latex: str) -> ParsedLatex:
    """Parse a LaTeX string into tokens and relationship tensor"""
    parser = LatexRelationshipParser()
    return parser.parse(latex)


if __name__ == "__main__":
    # Demo/test
    test_cases = [
        r"a_t",
        r"a^2",
        r"a_t^2",
        r"a+b",
        r"\frac{a}{b}",
        r"\frac{a_t+b}{c}",
        r"\sqrt{x}",
        r"\sqrt[3]{x}",
        r"x^2 + y^2 = z^2",
        r"\frac{\alpha + \beta}{\gamma}",
    ]
    
    for latex in test_cases:
        print(f"\n{'='*80}")
        print(f"INPUT: {latex}")
        print("="*80)
        
        parsed = parse_latex(latex)
        print(visualize_relationship_tensor(parsed, f"Parsed: {latex}"))
        print(visualize_tree(parsed))
        print()

