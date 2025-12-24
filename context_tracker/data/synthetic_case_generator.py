"""
Comprehensive Data Generation for LaTeX Editing Model

Generates diverse training data covering:
- ADD: Adding new symbols to expressions
- REPLACE: Correcting/changing existing symbols
- INSERT: Inserting symbols at specific positions

Categories:
1. Commutative Diagrams (arrows, nodes, labels)
2. Mathematical Expressions (equations, series)
3. Greek Letters (α, β, γ, θ, etc.)
4. Matrix Operations (elements, rows, columns)
5. Calculus (integrals, derivatives, limits)
6. Subscripts/Superscripts
7. Fractions and Complex Structures

Training Data Format:
    Input:  (latex_context_tokens, edit_crop_region)
    Output: (updated_latex_tokens)

Usage:
    python cases.py --output ./generated_cases --count 1000
"""

import random
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from enum import Enum
import string
from datetime import datetime


# =============================================================================
# Expression Tree Depth & Complexity Analysis
# =============================================================================

@dataclass
class ExpressionComplexity:
    """
    Tree-depth based complexity metrics for LaTeX expressions.
    
    Mathematical expressions form trees where:
    - Depth = nesting level (fractions in fractions, nested scripts)
    - Breadth = number of terms at each level
    - Special constructs add complexity multipliers
    
    Difficulty Classification:
    - easy: depth ≤ 2, score < 0.3
    - medium: depth 3-4, score 0.3-0.6  
    - hard: depth ≥ 5, score > 0.6
    """
    depth: int                    # Maximum brace nesting depth
    num_braces: int               # Total brace pairs {}
    num_operators: int            # Binary operators (+, -, *, /)
    num_functions: int            # Functions (sin, cos, log, etc.)
    num_fractions: int            # \frac constructs
    num_scripts: int              # Subscripts/superscripts (^, _)
    num_special: int              # Special: int, sum, prod, partial, matrix, tikzcd
    num_greek: int                # Greek letters
    num_environments: int         # LaTeX environments (begin/end)
    total_tokens: int             # Approximate token count
    difficulty_score: float       # Normalized difficulty (0-1)
    difficulty: str               # Classification: easy, medium, hard
    
    def __repr__(self):
        return (f"Complexity(depth={self.depth}, score={self.difficulty_score:.2f}, "
                f"difficulty='{self.difficulty}')")


def _compute_semantic_depth(latex: str) -> int:
    """
    Compute semantic tree depth for math expressions.
    
    Unlike simple brace counting, this tracks "semantic depth" where:
    - \frac{a}{b} puts both a and b at depth+1
    - ^{x} and _{x} put x at depth+1
    - \sqrt{x} puts x at depth+1
    - Nested structures stack: \frac{\frac{1}{2}}{3} has depth 2
    
    For TikZ-CD diagrams, depth = max(num_rows, num_cols) as a proxy
    for diagram complexity.
    """
    if not latex:
        return 0
    
    # Special handling for TikZ-CD diagrams
    if r'\begin{tikzcd}' in latex:
        # Count rows (\\) and columns (&) to estimate grid size
        num_rows = latex.count(r'\\') + 1
        num_cols = latex.count('&') // max(1, num_rows) + 1
        num_arrows = latex.count(r'\arrow')
        # Diagram depth = grid size + arrow complexity
        return max(num_rows, num_cols) + (1 if num_arrows > 4 else 0)
    
    # For regular math: compute semantic depth by tracking constructs
    # that create tree structure
    
    # Method: Parse the expression tracking depth-increasing constructs
    max_depth = 1  # Base depth is 1 for any non-empty expression
    current_depth = 1
    
    i = 0
    while i < len(latex):
        # Check for depth-increasing constructs
        remaining = latex[i:]
        
        # Fractions: \frac{}{} - numerator and denominator each at depth+1
        if remaining.startswith(r'\frac'):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
            i += 5
            continue
        
        # Square roots: \sqrt{} or \sqrt[n]{}
        if remaining.startswith(r'\sqrt'):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
            i += 5
            continue
        
        # Superscript/subscript with braces: ^{} or _{}
        if i > 0 and remaining.startswith('{'):
            prev_char = latex[i-1]
            if prev_char in ['^', '_']:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
        
        # Integrals, sums with limits add depth
        if any(remaining.startswith(cmd) for cmd in [r'\int', r'\sum', r'\prod', r'\oint']):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
            i += 4
            continue
        
        # Environments add depth
        if remaining.startswith(r'\begin{'):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif remaining.startswith(r'\end{'):
            current_depth = max(1, current_depth - 1)
        
        # Track brace nesting for depth within constructs
        if latex[i] == '{':
            pass  # Handled by construct detection
        elif latex[i] == '}':
            current_depth = max(1, current_depth - 1)
        
        i += 1
    
    return max_depth


def analyze_latex_complexity(latex: str) -> ExpressionComplexity:
    """
    Analyze LaTeX expression and compute tree-depth based complexity.
    
    Uses semantic depth (not just brace counting) where:
    - \frac{}{} adds depth to its contents
    - ^{} and _{} add depth to their contents  
    - TikZ-CD depth = max(rows, cols) + arrow_complexity
    
    Args:
        latex: LaTeX expression string
        
    Returns:
        ExpressionComplexity with all metrics computed
    """
    if not latex:
        return ExpressionComplexity(
            depth=0, num_braces=0, num_operators=0, num_functions=0,
            num_fractions=0, num_scripts=0, num_special=0, num_greek=0,
            num_environments=0, total_tokens=0, difficulty_score=0.0,
            difficulty="easy"
        )
    
    # Compute semantic tree depth
    semantic_depth = _compute_semantic_depth(latex)
    
    # Also count raw brace nesting for reference
    brace_depth = 0
    max_brace_depth = 0
    num_braces = 0
    for char in latex:
        if char == '{':
            brace_depth += 1
            num_braces += 1
            max_brace_depth = max(max_brace_depth, brace_depth)
        elif char == '}':
            brace_depth = max(0, brace_depth - 1)
    
    # Use maximum of semantic and brace depth
    max_depth = max(semantic_depth, max_brace_depth)
    
    # Count operators
    operators = ['+', '-', r'\times', r'\cdot', r'\div', r'\pm', r'\mp', '=', r'\neq', r'\leq', r'\geq']
    num_operators = sum(latex.count(op) for op in operators)
    
    # Count functions
    functions = [r'\sin', r'\cos', r'\tan', r'\log', r'\ln', r'\exp', r'\lim', r'\max', r'\min']
    num_functions = sum(latex.count(fn) for fn in functions)
    
    # Count fractions
    num_fractions = latex.count(r'\frac')
    
    # Count scripts (subscripts and superscripts)
    num_scripts = latex.count('_') + latex.count('^')
    
    # Count special constructs (high complexity)
    special = [
        r'\int', r'\iint', r'\iiint', r'\oint',  # Integrals
        r'\sum', r'\prod',                         # Summations
        r'\partial', r'\nabla',                    # Derivatives
        r'\begin{pmatrix}', r'\begin{bmatrix}', r'\begin{vmatrix}',  # Matrices
        r'\begin{tikzcd}',                         # Diagrams
        r'\begin{aligned}', r'\begin{cases}',      # Alignments
    ]
    num_special = sum(latex.count(s) for s in special)
    
    # Count Greek letters
    greek = [r'\alpha', r'\beta', r'\gamma', r'\delta', r'\epsilon', r'\theta',
             r'\lambda', r'\mu', r'\sigma', r'\phi', r'\psi', r'\omega',
             r'\Gamma', r'\Delta', r'\Theta', r'\Lambda', r'\Sigma', r'\Phi', r'\Psi', r'\Omega']
    num_greek = sum(latex.count(g) for g in greek)
    
    # Count environments
    num_environments = latex.count(r'\begin{')
    
    # Approximate token count
    total_tokens = len(re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]|[+\-*/=<>^_{}]', latex))
    
    # Compute difficulty score (0-1 scale)
    # Weighted combination with semantic depth having high weight
    score = (
        0.30 * min(max_depth / 5, 1.0) +           # Semantic depth (normalized to 5)
        0.15 * min(num_fractions / 3, 1.0) +       # Fractions
        0.15 * min(num_scripts / 6, 1.0) +         # Scripts
        0.15 * min(num_special / 2, 1.0) +         # Special constructs
        0.10 * min(num_environments / 2, 1.0) +    # Environments
        0.10 * min(num_operators / 5, 1.0) +       # Operators
        0.05 * min(total_tokens / 50, 1.0)         # Token count
    )
    
    # Classify difficulty
    if max_depth <= 2 and score < 0.3:
        difficulty = "easy"
    elif max_depth >= 5 or score > 0.6:
        difficulty = "hard"
    else:
        difficulty = "medium"
    
    return ExpressionComplexity(
        depth=max_depth,
        num_braces=num_braces,
        num_operators=num_operators,
        num_functions=num_functions,
        num_fractions=num_fractions,
        num_scripts=num_scripts,
        num_special=num_special,
        num_greek=num_greek,
        num_environments=num_environments,
        total_tokens=total_tokens,
        difficulty_score=round(score, 3),
        difficulty=difficulty
    )


def compute_case_difficulty(before_latex: str, after_latex: str) -> str:
    """
    Compute difficulty for an edit case based on before/after expressions.
    
    Uses the more complex expression (usually after) to determine difficulty.
    """
    before_complexity = analyze_latex_complexity(before_latex)
    after_complexity = analyze_latex_complexity(after_latex)
    
    # Use the maximum complexity
    if after_complexity.difficulty_score >= before_complexity.difficulty_score:
        return after_complexity.difficulty
    return before_complexity.difficulty


def generate_by_difficulty(target_difficulty: str, base_generator, count: int = 50) -> List:
    """
    Generate cases filtered by target difficulty.
    
    Args:
        target_difficulty: "easy", "medium", or "hard"
        base_generator: Generator function that creates EditCase list
        count: Target number of cases
        
    Returns:
        List of EditCase matching target difficulty
    """
    # Generate more than needed to filter
    candidates = base_generator(count * 3)
    
    # Filter by difficulty
    filtered = []
    for case in candidates:
        actual_difficulty = compute_case_difficulty(case.before_latex, case.after_latex)
        if actual_difficulty == target_difficulty:
            case.difficulty = actual_difficulty  # Update with computed value
            filtered.append(case)
        if len(filtered) >= count:
            break
    
    return filtered[:count]


# =============================================================================
# Bottom-Up Expression Builder (Dynamic Programming)
# =============================================================================

class ExpressionBuilder:
    """
    Build LaTeX expressions of arbitrary depth using bottom-up dynamic programming.
    
    This class constructs expressions by:
    1. Starting with atomic symbols (depth 0) from the full vocabulary
    2. Applying production rules to build depth N+1 from depth N
    3. Caching intermediate results for efficiency
    
    Production Rules (each increases depth by 1):
        - FRACTION: expr -> frac{expr}{expr}
        - SCRIPT: expr -> expr^{expr} or expr_{expr}
        - SQRT: expr -> sqrt{expr} or sqrt[expr]{expr}
        - INTEGRAL: expr -> int_{expr}^{expr} expr
        - SUM: expr -> sum_{expr}^{expr} expr
        - PARENTHESIS: expr -> left( expr right)
        - FUNCTION: expr -> func(expr) where func in {sin, cos, log, ...}
    
    The builder can use symbols from:
        - Tokenizer vocabulary (782 tokens)
        - Stroke data (236 handwriting-ready symbols)
        - Custom symbol lists
    
    Usage:
        builder = ExpressionBuilder()
        expr = builder.build(target_depth=5)
        expr = builder.build(target_depth=10, min_breadth=3)
        
        # Use only stroke-available symbols
        builder = ExpressionBuilder(use_stroke_vocab=True)
    """
    
    # Default atomic symbols (depth 0) - comprehensive list
    DEFAULT_VARIABLES = list('xyzabcdefghijklmnopqrstuvw')
    DEFAULT_DIGITS = list('0123456789')
    DEFAULT_GREEK_LOWER = [
        '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta',
        '\\eta', '\\theta', '\\iota', '\\kappa', '\\lambda', '\\mu',
        '\\nu', '\\xi', '\\pi', '\\rho', '\\sigma', '\\tau',
        '\\upsilon', '\\phi', '\\chi', '\\psi', '\\omega'
    ]
    DEFAULT_GREEK_UPPER = [
        '\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi', 
        '\\Pi', '\\Sigma', '\\Upsilon', '\\Phi', '\\Psi', '\\Omega'
    ]
    DEFAULT_CONSTANTS = ['0', '1', '2', '3', 'e', '\\pi', '\\infty', 'n', 'N']
    
    # Binary operators
    BINARY_OPS = ['+', '-', '\\cdot', '\\times', '\\pm', '\\mp', '\\div']
    RELATIONS = ['=', '\\neq', '\\leq', '\\geq', '<', '>', '\\approx', '\\equiv', '\\sim']
    SET_OPS = ['\\cup', '\\cap', '\\subset', '\\supset', '\\in', '\\setminus']
    
    # Functions that take arguments
    FUNCTIONS = [
        '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc',
        '\\log', '\\ln', '\\exp', '\\arcsin', '\\arccos', '\\arctan',
        '\\sinh', '\\cosh', '\\tanh', '\\det', '\\dim', '\\ker'
    ]
    
    # Stroke-available symbol mapping (LaTeX -> Unicode for validation)
    LATEX_TO_UNICODE = {
        '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ',
        '\\epsilon': 'ε', '\\zeta': 'ζ', '\\eta': 'η', '\\theta': 'θ',
        '\\iota': 'ι', '\\kappa': 'κ', '\\lambda': 'λ', '\\mu': 'μ',
        '\\nu': 'ν', '\\xi': 'ξ', '\\pi': 'π', '\\rho': 'ρ',
        '\\sigma': 'σ', '\\tau': 'τ', '\\upsilon': 'υ', '\\phi': 'φ',
        '\\chi': 'χ', '\\psi': 'ψ', '\\omega': 'ω',
        '\\Gamma': 'Γ', '\\Delta': 'Δ', '\\Theta': 'Θ', '\\Lambda': 'Λ',
        '\\Xi': 'Ξ', '\\Pi': 'Π', '\\Sigma': 'Σ', '\\Phi': 'Φ',
        '\\Psi': 'Ψ', '\\Omega': 'Ω',
        '\\sum': '∑', '\\prod': '∏', '\\int': '∫', '\\infty': '∞',
        '\\partial': '∂', '\\nabla': '∇', '\\forall': '∀', '\\exists': '∃',
        '\\emptyset': '∅', '\\neg': '¬', '\\wedge': '∧', '\\vee': '∨',
    }
    
    def __init__(self, seed: int = None, use_stroke_vocab: bool = False, 
                 custom_symbols: List[str] = None):
        """
        Initialize builder with optional random seed and vocabulary source.
        
        Args:
            seed: Random seed for reproducibility
            use_stroke_vocab: If True, only use symbols with stroke data available
            custom_symbols: Custom list of atomic symbols to use
        """
        if seed is not None:
            random.seed(seed)
        
        self.use_stroke_vocab = use_stroke_vocab
        self._stroke_symbols = None
        
        # Initialize symbol lists
        if custom_symbols:
            self.VARIABLES = [s for s in custom_symbols if len(s) == 1 and s.isalpha()]
            self.DIGITS = [s for s in custom_symbols if len(s) == 1 and s.isdigit()]
            self.GREEK_LOWER = [s for s in custom_symbols if s.startswith('\\') and s[1:].islower()]
            self.GREEK_UPPER = [s for s in custom_symbols if s.startswith('\\') and s[1:].isupper()]
            self.CONSTANTS = self.DIGITS[:3] + ['e', '\\pi', '\\infty']
        else:
            self.VARIABLES = self.DEFAULT_VARIABLES.copy()
            self.DIGITS = self.DEFAULT_DIGITS.copy()
            self.GREEK_LOWER = self.DEFAULT_GREEK_LOWER.copy()
            self.GREEK_UPPER = self.DEFAULT_GREEK_UPPER.copy()
            self.CONSTANTS = self.DEFAULT_CONSTANTS.copy()
        
        # Filter to stroke-available if requested
        if use_stroke_vocab:
            self._filter_to_stroke_vocab()
        
        # Cache: depth -> list of expressions at that depth
        self._cache: Dict[int, List[str]] = {}
        self._cache[0] = self._generate_atomics()
    
    def _get_stroke_symbols(self) -> set:
        """Lazy load stroke symbols."""
        if self._stroke_symbols is None:
            try:
                from scripts.stroke_renderer import StrokeDataLoader
                loader = StrokeDataLoader()
                self._stroke_symbols = set(loader.get_all_symbols())
            except ImportError:
                self._stroke_symbols = set()
        return self._stroke_symbols
    
    def _filter_to_stroke_vocab(self):
        """Filter symbol lists to only include stroke-available symbols."""
        stroke_syms = self._get_stroke_symbols()
        
        # Filter variables and digits
        self.VARIABLES = [v for v in self.VARIABLES if v in stroke_syms]
        self.DIGITS = [d for d in self.DIGITS if d in stroke_syms]
        
        # Filter Greek by checking Unicode mapping
        self.GREEK_LOWER = [g for g in self.GREEK_LOWER 
                           if self.LATEX_TO_UNICODE.get(g, '') in stroke_syms]
        self.GREEK_UPPER = [g for g in self.GREEK_UPPER 
                           if self.LATEX_TO_UNICODE.get(g, '') in stroke_syms]
        
        # Update constants
        self.CONSTANTS = [c for c in self.CONSTANTS 
                         if c in stroke_syms or c in self.DIGITS or 
                         self.LATEX_TO_UNICODE.get(c, '') in stroke_syms]
    
    def _generate_atomics(self, count: int = 50) -> List[str]:
        """Generate depth-0 atomic expressions."""
        atomics = []
        
        # Simple variables and digits
        atomics.extend(self.VARIABLES)
        atomics.extend(self.DIGITS)
        atomics.extend(self.GREEK_LOWER)
        
        # Simple expressions (still depth 0)
        for _ in range(count):
            expr_type = random.choice(['var', 'digit', 'greek', 'const'])
            if expr_type == 'var':
                atomics.append(random.choice(self.VARIABLES))
            elif expr_type == 'digit':
                atomics.append(random.choice(self.DIGITS))
            elif expr_type == 'greek':
                atomics.append(random.choice(self.GREEK_LOWER + self.GREEK_UPPER))
            else:
                atomics.append(random.choice(self.CONSTANTS))
        
        return list(set(atomics))  # Remove duplicates
    
    def _get_expr_at_depth(self, depth: int) -> str:
        """Get a random expression at exactly the given depth."""
        if depth < 0:
            depth = 0
        
        # Build up to required depth if not cached
        while depth not in self._cache or not self._cache[depth]:
            self._build_next_depth()
        
        return random.choice(self._cache[depth])
    
    def _build_next_depth(self):
        """Build expressions at the next depth level using production rules."""
        current_max = max(self._cache.keys())
        next_depth = current_max + 1
        
        new_exprs = []
        
        # Apply each production rule multiple times
        for _ in range(20):  # Generate multiple variants
            # Get expressions from previous depth(s)
            base = self._get_expr_at_depth(current_max)
            base2 = self._get_expr_at_depth(random.randint(0, current_max))
            base3 = self._get_expr_at_depth(random.randint(0, current_max))
            
            rule = random.choice([
                'fraction', 'superscript', 'subscript', 'both_scripts',
                'sqrt', 'integral', 'sum', 'product', 'function', 
                'parenthesis', 'nested_fraction'
            ])
            
            if rule == 'fraction':
                new_exprs.append(f'\\frac{{{base}}}{{{base2}}}')
            
            elif rule == 'superscript':
                new_exprs.append(f'{{{base}}}^{{{base2}}}')
            
            elif rule == 'subscript':
                new_exprs.append(f'{{{base}}}_{{{base2}}}')
            
            elif rule == 'both_scripts':
                new_exprs.append(f'{{{base}}}_{{{base2}}}^{{{base3}}}')
            
            elif rule == 'sqrt':
                if random.random() < 0.3:
                    # nth root
                    n = random.choice(['3', 'n', '4'])
                    new_exprs.append(f'\\sqrt[{n}]{{{base}}}')
                else:
                    new_exprs.append(f'\\sqrt{{{base}}}')
            
            elif rule == 'integral':
                lower = self._get_expr_at_depth(random.randint(0, min(2, current_max)))
                upper = self._get_expr_at_depth(random.randint(0, min(2, current_max)))
                integrand = base
                new_exprs.append(f'\\int_{{{lower}}}^{{{upper}}} {integrand} \\, dx')
            
            elif rule == 'sum':
                idx = random.choice(['i', 'j', 'k', 'n'])
                lower = f'{idx}={random.choice(["0", "1"])}'
                upper = random.choice(['n', 'N', '\\infty', 'm'])
                new_exprs.append(f'\\sum_{{{lower}}}^{{{upper}}} {base}')
            
            elif rule == 'product':
                idx = random.choice(['i', 'j', 'k'])
                lower = f'{idx}=1'
                upper = random.choice(['n', 'N', 'm'])
                new_exprs.append(f'\\prod_{{{lower}}}^{{{upper}}} {base}')
            
            elif rule == 'function':
                func = random.choice(self.FUNCTIONS)
                new_exprs.append(f'{func}\\left({base}\\right)')
            
            elif rule == 'parenthesis':
                # Combine with operator
                op = random.choice(self.BINARY_OPS)
                new_exprs.append(f'\\left({base} {op} {base2}\\right)')
            
            elif rule == 'nested_fraction':
                # Fraction with fraction inside
                inner = f'\\frac{{{base2}}}{{{base3}}}'
                if random.random() < 0.5:
                    new_exprs.append(f'\\frac{{{inner}}}{{{base}}}')
                else:
                    new_exprs.append(f'\\frac{{{base}}}{{{inner}}}')
        
        self._cache[next_depth] = new_exprs
    
    def build(self, 
              target_depth: int, 
              min_breadth: int = 1,
              include_operators: bool = True) -> str:
        """
        Build an expression with the target depth.
        
        Args:
            target_depth: Target tree depth (1-15 typically)
            min_breadth: Minimum number of terms combined
            include_operators: Whether to combine with binary operators
            
        Returns:
            LaTeX expression string
        """
        if target_depth <= 0:
            return self._get_expr_at_depth(0)
        
        # Build expression at target depth
        main_expr = self._get_expr_at_depth(target_depth)
        
        # Optionally add breadth (multiple terms)
        if min_breadth > 1 and include_operators:
            terms = [main_expr]
            for _ in range(min_breadth - 1):
                # Get expression at varying depths for variety
                term_depth = random.randint(max(0, target_depth - 2), target_depth)
                terms.append(self._get_expr_at_depth(term_depth))
            
            op = random.choice(self.BINARY_OPS + self.RELATIONS)
            main_expr = f' {op} '.join(terms)
        
        return main_expr
    
    def build_equation(self, 
                       lhs_depth: int,
                       rhs_depth: int,
                       relation: str = '=') -> str:
        """Build an equation: LHS = RHS with specified depths."""
        lhs = self.build(lhs_depth)
        rhs = self.build(rhs_depth)
        return f'{lhs} {relation} {rhs}'
    
    def build_with_exact_depth(self, target_depth: int, attempts: int = 20) -> str:
        """
        Build an expression and verify it has exactly the target depth.
        
        Args:
            target_depth: Required depth
            attempts: Max attempts to find exact match
            
        Returns:
            Expression with verified depth
        """
        for _ in range(attempts):
            expr = self.build(target_depth)
            complexity = analyze_latex_complexity(expr)
            if complexity.depth >= target_depth:
                return expr
        
        # If we can't hit exact depth, return best effort
        return self.build(target_depth)
    
    @classmethod
    def generate_depth_progression(cls, 
                                   min_depth: int = 1,
                                   max_depth: int = 10,
                                   count_per_depth: int = 10,
                                   seed: int = None) -> List[Tuple[str, int]]:
        """
        Generate expressions across a range of depths.
        
        Useful for curriculum learning.
        
        Returns:
            List of (expression, depth) tuples
        """
        builder = cls(seed=seed)
        results = []
        
        for depth in range(min_depth, max_depth + 1):
            for _ in range(count_per_depth):
                expr = builder.build(depth)
                results.append((expr, depth))
        
        return results
    
    def build_with_empty_slot(self, 
                              target_depth: int,
                              slot_type: str = 'random') -> Tuple[str, str, str]:
        """
        Build expression with empty braces {} for FILL operation.
        
        Adjacent braces {} indicate empty content. The brace positions
        preserve WHERE the content should go:
            - { position = left edge of empty region
            - } position = right edge of empty region
        
        Args:
            target_depth: Target tree depth
            slot_type: Type of empty slot - 'fraction_num', 'fraction_denom', 
                      'superscript', 'subscript', 'sqrt', 'integral', 'random'
        
        Returns:
            Tuple of (with_empty, filled, fill_content)
            
        Example:
            (r'\\frac{}{z}', r'\\frac{x+y}{z}', 'x+y')
        """
        if slot_type == 'random':
            slot_type = random.choice([
                'fraction_num', 'fraction_denom', 'superscript', 
                'subscript', 'sqrt', 'integral_lower', 'integral_upper'
            ])
        
        base_depth = max(0, target_depth - 1)
        
        if slot_type == 'fraction_num':
            content = self._get_expr_at_depth(random.randint(0, base_depth))
            denom = self._get_expr_at_depth(random.randint(0, base_depth))
            with_empty = f'\\frac{{}}{{{denom}}}'  # Empty numerator
            filled = f'\\frac{{{content}}}{{{denom}}}'
            fill_content = content
            
        elif slot_type == 'fraction_denom':
            num = self._get_expr_at_depth(random.randint(0, base_depth))
            content = self._get_expr_at_depth(random.randint(0, base_depth))
            with_empty = f'\\frac{{{num}}}{{}}'  # Empty denominator
            filled = f'\\frac{{{num}}}{{{content}}}'
            fill_content = content
            
        elif slot_type == 'superscript':
            base = self._get_expr_at_depth(random.randint(0, base_depth))
            content = self._get_expr_at_depth(random.randint(0, min(2, base_depth)))
            with_empty = f'{{{base}}}^{{}}'  # Empty superscript
            filled = f'{{{base}}}^{{{content}}}'
            fill_content = content
            
        elif slot_type == 'subscript':
            base = self._get_expr_at_depth(random.randint(0, base_depth))
            content = self._get_expr_at_depth(random.randint(0, min(2, base_depth)))
            with_empty = f'{{{base}}}_{{}}'  # Empty subscript
            filled = f'{{{base}}}_{{{content}}}'
            fill_content = content
            
        elif slot_type == 'sqrt':
            content = self._get_expr_at_depth(random.randint(0, base_depth))
            with_empty = '\\sqrt{}'  # Empty sqrt
            filled = f'\\sqrt{{{content}}}'
            fill_content = content
            
        elif slot_type == 'integral_lower':
            lower = self._get_expr_at_depth(random.randint(0, min(2, base_depth)))
            upper = self._get_expr_at_depth(random.randint(0, min(2, base_depth)))
            integrand = self._get_expr_at_depth(random.randint(0, base_depth))
            with_empty = f'\\int_{{}}^{{{upper}}} {integrand} \\, dx'  # Empty lower
            filled = f'\\int_{{{lower}}}^{{{upper}}} {integrand} \\, dx'
            fill_content = lower
            
        else:  # integral_upper
            lower = self._get_expr_at_depth(random.randint(0, min(2, base_depth)))
            upper = self._get_expr_at_depth(random.randint(0, min(2, base_depth)))
            integrand = self._get_expr_at_depth(random.randint(0, base_depth))
            with_empty = f'\\int_{{{lower}}}^{{}} {integrand} \\, dx'  # Empty upper
            filled = f'\\int_{{{lower}}}^{{{upper}}} {integrand} \\, dx'
            fill_content = upper
        
        return with_empty, filled, fill_content
    
    def erase_subexpression(self, expr: str) -> Tuple[str, str, str]:
        """
        Simulate erasing a subexpression from a complete expression.
        
        This models user erasing content, leaving empty {} braces.
        The {} braces preserve the position of the erased region.
        
        Returns:
            Tuple of (with_empty, original, erased_content)
            
        Example:
            Original: \\frac{x+y}{z}
            Erased:   \\frac{}{z}   (empty braces)
            Content:  x+y
        """
        import re
        
        # Find erasable subexpressions (content inside braces)
        # Pattern: {content} where content is not empty
        brace_pattern = r'\{([^{}]+)\}'
        matches = list(re.finditer(brace_pattern, expr))
        
        if matches:
            # Pick a random brace group to erase
            match = random.choice(matches)
            erased_content = match.group(1)
            
            # Replace content with empty (just {})
            with_empty = expr[:match.start()] + '{}' + expr[match.end():]
            
            return with_empty, expr, erased_content
        
        # Fallback: wrap in fraction and erase denominator
        denom = self._get_expr_at_depth(random.randint(0, 2))
        original = f'\\frac{{{expr}}}{{{denom}}}'
        with_empty = f'\\frac{{{expr}}}{{}}'
        
        return with_empty, original, denom


def build_expression_at_depth(target_depth: int, seed: int = None) -> str:
    """Convenience function to build a single expression at target depth."""
    builder = ExpressionBuilder(seed=seed)
    return builder.build(target_depth)


def build_expression_pair(before_depth: int, 
                         after_depth: int,
                         operation: str = 'ADD',
                         seed: int = None) -> Tuple[str, str, str]:
    """
    Build a before/after expression pair for edit training.
    
    Args:
        before_depth: Depth of before expression
        after_depth: Depth of after expression
        operation: Type of edit (ADD, REPLACE, etc.)
        
    Returns:
        (before_latex, after_latex, edit_symbol)
    """
    builder = ExpressionBuilder(seed=seed)
    
    before_expr = builder.build(before_depth)
    
    if operation == 'ADD':
        # Add something to make it deeper
        addition = builder.build(max(0, after_depth - before_depth))
        after_expr = f'{before_expr} + {addition}'
        edit_symbol = addition.strip()
        
    elif operation == 'WRAP':
        # Wrap the expression in a structure
        after_expr = f'\\frac{{{before_expr}}}{{1}}'
        edit_symbol = '\\frac'
        
    elif operation == 'REPLACE':
        # Replace a variable with something more complex
        replacement = builder.build(after_depth)
        # Simple replacement of first variable found
        for var in builder.VARIABLES:
            if var in before_expr:
                after_expr = before_expr.replace(var, replacement, 1)
                edit_symbol = replacement
                break
        else:
            after_expr = before_expr
            edit_symbol = ''
    else:
        after_expr = before_expr
        edit_symbol = ''
    
    return before_expr, after_expr, edit_symbol


class ComplexExpressionCases:
    """
    Generate edit cases with arbitrary-depth expressions using ExpressionBuilder.
    
    This replaces hardcoded template-based generation with dynamic construction
    that can produce expressions of any complexity level.
    """
    
    @classmethod
    def generate_add_cases(cls, count: int = 50, 
                          min_depth: int = 1,
                          max_depth: int = 10) -> List['EditCase']:
        """Generate ADD cases with expressions at varying depths."""
        cases = []
        builder = ExpressionBuilder()
        
        for i in range(count):
            # Random depths (ensure valid range)
            before_depth = random.randint(min_depth, max(min_depth, max_depth - 1))
            after_depth = random.randint(before_depth, max(before_depth + 1, max_depth))
            
            before_expr = builder.build(before_depth)
            
            # Choose addition type
            add_type = random.choice(['term', 'subscript', 'superscript', 'fraction', 'function'])
            
            if add_type == 'term':
                addition = builder.build(random.randint(1, 3))
                op = random.choice(['+', '-', '\\cdot'])
                after_expr = f'{before_expr} {op} {addition}'
                edit_symbols = [addition.split()[0] if ' ' in addition else addition]
                
            elif add_type == 'subscript':
                sub = builder._get_expr_at_depth(random.randint(0, 2))
                after_expr = f'{{{before_expr}}}_{{{sub}}}'
                edit_symbols = [sub]
                
            elif add_type == 'superscript':
                sup = builder._get_expr_at_depth(random.randint(0, 2))
                after_expr = f'{{{before_expr}}}^{{{sup}}}'
                edit_symbols = [sup]
                
            elif add_type == 'fraction':
                denom = builder.build(random.randint(1, 3))
                after_expr = f'\\frac{{{before_expr}}}{{{denom}}}'
                edit_symbols = ['\\frac', denom]
                
            else:  # function
                func = random.choice(builder.FUNCTIONS)
                after_expr = f'{func}\\left({before_expr}\\right)'
                edit_symbols = [func]
            
            # Compute actual complexity
            complexity = analyze_latex_complexity(after_expr)
            
            cases.append(EditCase(
                id=f"complex_add_{i:04d}",
                category="complex_expression",
                subcategory=add_type,
                operation="ADD",
                before_latex=before_expr,
                after_latex=after_expr,
                edit_description=f"Add {add_type} to depth-{before_depth} expression",
                difficulty=complexity.difficulty,
                metadata={'edit_symbols': edit_symbols, 'depth': complexity.depth}
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 50,
                              min_depth: int = 2,
                              max_depth: int = 8) -> List['EditCase']:
        """Generate REPLACE cases with expressions at varying depths."""
        cases = []
        builder = ExpressionBuilder()
        
        for i in range(count):
            depth = random.randint(min_depth, max_depth)
            expr = builder.build(depth)
            
            # Replace a variable with something else
            replace_type = random.choice(['var_to_var', 'var_to_expr', 'num_to_num', 'greek_swap'])
            
            if replace_type == 'var_to_var':
                old_var = random.choice(builder.VARIABLES[:10])
                new_var = random.choice([v for v in builder.VARIABLES[:10] if v != old_var])
                if old_var in expr:
                    after_expr = expr.replace(old_var, new_var, 1)
                    edit_symbols = [new_var]
                else:
                    after_expr = expr
                    edit_symbols = []
                    
            elif replace_type == 'var_to_expr':
                old_var = random.choice(builder.VARIABLES[:5])
                new_expr = builder.build(random.randint(1, 2))
                if old_var in expr:
                    after_expr = expr.replace(old_var, f'({new_expr})', 1)
                    edit_symbols = [new_expr]
                else:
                    after_expr = expr
                    edit_symbols = []
                    
            elif replace_type == 'num_to_num':
                old_num = random.choice(builder.DIGITS)
                new_num = random.choice([d for d in builder.DIGITS if d != old_num])
                if old_num in expr:
                    after_expr = expr.replace(old_num, new_num, 1)
                    edit_symbols = [new_num]
                else:
                    after_expr = expr
                    edit_symbols = []
                    
            else:  # greek_swap
                if builder.GREEK_LOWER:
                    old_greek = random.choice(builder.GREEK_LOWER)
                    new_greek = random.choice([g for g in builder.GREEK_LOWER if g != old_greek])
                    if old_greek in expr:
                        after_expr = expr.replace(old_greek, new_greek, 1)
                        edit_symbols = [new_greek]
                    else:
                        after_expr = expr
                        edit_symbols = []
                else:
                    after_expr = expr
                    edit_symbols = []
            
            if after_expr != expr:
                complexity = analyze_latex_complexity(after_expr)
                cases.append(EditCase(
                    id=f"complex_replace_{i:04d}",
                    category="complex_expression",
                    subcategory=replace_type,
                    operation="REPLACE",
                    before_latex=expr,
                    after_latex=after_expr,
                    edit_description=f"Replace in depth-{depth} expression",
                    difficulty=complexity.difficulty,
                    metadata={'edit_symbols': edit_symbols, 'depth': complexity.depth}
                ))
        
        return cases
    
    @classmethod
    def generate_fill_cases(cls, count: int = 50,
                           min_depth: int = 1,
                           max_depth: int = 8) -> List['EditCase']:
        """
        Generate FILL cases using empty braces {} for empty slots.
        
        Adjacent braces {} indicate empty content:
            - \\frac{}{z} -> user erased numerator
            - x^{} -> user erased exponent
        
        The brace positions preserve the empty region:
            - { position = left edge of erased content
            - } position = right edge of erased content
        """
        cases = []
        builder = ExpressionBuilder()
        
        slot_types = [
            'fraction_num', 'fraction_denom', 'superscript',
            'subscript', 'sqrt', 'integral_lower', 'integral_upper'
        ]
        
        for i in range(count):
            depth = random.randint(min_depth, max_depth)
            
            # 50% build with empty slot, 50% erase from complete expression
            if random.random() < 0.5:
                slot_type = random.choice(slot_types)
                with_empty, filled, fill_content = builder.build_with_empty_slot(depth, slot_type)
                subcat = slot_type
            else:
                # Build complete expression then erase something
                complete_expr = builder.build(depth)
                with_empty, filled, fill_content = builder.erase_subexpression(complete_expr)
                subcat = 'erased'
            
            complexity = analyze_latex_complexity(filled)
            
            cases.append(EditCase(
                id=f"complex_fill_{i:04d}",
                category="complex_expression",
                subcategory=subcat,
                operation="FILL",
                before_latex=with_empty,
                after_latex=filled,
                edit_description=f"Fill empty slot with {fill_content}",
                difficulty=complexity.difficulty,
                metadata={'edit_symbols': [fill_content], 'depth': complexity.depth}
            ))
        
        return cases
    
    @classmethod
    def generate_curriculum_cases(cls, 
                                  depths: List[int] = None,
                                  count_per_depth: int = 20) -> List['EditCase']:
        """
        Generate cases for curriculum learning with progressive depths.
        
        Args:
            depths: List of target depths (default: 1-10)
            count_per_depth: Cases per depth level
            
        Returns:
            List of EditCase sorted by increasing depth
        """
        if depths is None:
            depths = list(range(1, 11))
        
        cases = []
        
        for depth in depths:
            # Mix of ADD, REPLACE and FILL at each depth
            n_each = count_per_depth // 3
            add_cases = cls.generate_add_cases(
                count=n_each,
                min_depth=max(1, depth - 1),
                max_depth=depth
            )
            replace_cases = cls.generate_replace_cases(
                count=n_each,
                min_depth=depth,
                max_depth=depth
            )
            fill_cases = cls.generate_fill_cases(
                count=n_each,
                min_depth=max(1, depth - 1),
                max_depth=depth
            )
            
            for case in add_cases + replace_cases + fill_cases:
                case.metadata['curriculum_depth'] = depth
            
            cases.extend(add_cases + replace_cases + fill_cases)
        
        return cases


# =============================================================================
# Enums and Data Classes
# =============================================================================

class EditOperation(Enum):
    ADD = "ADD"           # Add to end/edge: x → x^{2}
    REPLACE = "REPLACE"   # Replace existing: α → β  
    INSERT = "INSERT"     # Insert in middle: x+y → x+z+y
    FILL = "FILL"         # Fill empty slot: \frac{2}{} → \frac{2}{3}
    DELETE = "DELETE"     # Erase/remove: x+y → x (cross out y)
    WRAP = "WRAP"         # Wrap in structure: x+1 → \frac{x+1}{y} (draw fraction bar)
    UNWRAP = "UNWRAP"     # Remove structure: \frac{x}{1} → x


@dataclass
class CropRegionSpec:
    """Specification for the cropped edit region."""
    # Relative position in canvas (0-1 normalized)
    x: float = 0.5
    y: float = 0.5
    width: float = 0.3
    height: float = 0.3
    
    # Noise/distraction elements in crop region (detailed specs)
    noise_specs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Simple noise type list for quick reference
    noise_types: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y, 
            "width": self.width,
            "height": self.height,
            "noise_specs": self.noise_specs,
            "noise_types": self.noise_types,
            "has_noise": len(self.noise_specs) > 0,
        }


@dataclass 
class TrainingExample:
    """
    Complete training example with input-output pairs.
    
    Input:
        - latex_context: Previous LaTeX tokens (can be empty for initial state)
        - crop_region: Cropped image region around edit area
        - noise_spec: What noise/distractions are in the crop
        
    Output:
        - updated_latex: The new LaTeX after edit
    """
    id: str
    
    # INPUT
    latex_context: str           # Previous LaTeX (empty string = initial state)
    crop_spec: CropRegionSpec    # Where/what the edit region contains
    
    # OUTPUT  
    updated_latex: str           # New LaTeX after edit
    
    # METADATA
    operation: str               # ADD, REPLACE, INSERT
    edit_symbol: str             # The symbol being added/edited
    category: str
    subcategory: str
    is_initial_state: bool       # True if starting from empty
    difficulty: str
    
    def to_dict(self):
        d = asdict(self)
        d['crop_spec'] = self.crop_spec.to_dict()
        return d


@dataclass
class EditCase:
    """A single editing case for training."""
    id: str
    category: str
    subcategory: str
    operation: str
    before_latex: str
    after_latex: str
    edit_description: str
    position_hint: Optional[str] = None  # e.g., "superscript", "after:x", "row:2"
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    def to_training_example(self, add_noise: bool = True, 
                            noise_probability: float = 0.4) -> TrainingExample:
        """Convert to TrainingExample format with optional noise."""
        
        # Determine crop position based on operation
        if self.position_hint == "superscript":
            crop_y = 0.3  # Upper region
        elif self.position_hint == "subscript":
            crop_y = 0.7  # Lower region
        else:
            crop_y = 0.5  # Center
            
        # Generate noise specifications
        noise_specs = []
        noise_types = []
        if add_noise:
            noise_specs = NoiseGenerator.generate_noise_spec(
                noise_probability=noise_probability
            )
            noise_types = [spec["type"] for spec in noise_specs]
        
        crop_spec = CropRegionSpec(
            x=random.uniform(0.3, 0.7),
            y=crop_y,
            width=random.uniform(0.2, 0.4),
            height=random.uniform(0.2, 0.4),
            noise_specs=noise_specs,
            noise_types=noise_types,
        )
        
        # Extract the edit symbol from description or diff
        edit_symbol = self._extract_edit_symbol()
        
        return TrainingExample(
            id=self.id,
            latex_context=self.before_latex,
            crop_spec=crop_spec,
            updated_latex=self.after_latex,
            operation=self.operation,
            edit_symbol=edit_symbol,
            category=self.category,
            subcategory=self.subcategory,
            is_initial_state=(self.before_latex == "" or self.before_latex.strip() == ""),
            difficulty=self.difficulty,
        )
    
    def _extract_edit_symbol(self) -> str:
        """Extract the symbol being edited from before/after diff."""
        # Simple heuristic: find what's new in after_latex
        if not self.before_latex:
            return self.after_latex
        # For more complex cases, return the description hint
        return self.metadata.get("edit_symbol", "unknown")
    
    def compute_difficulty(self) -> str:
        """
        Compute difficulty based on tree-depth analysis of expressions.
        Updates self.difficulty and returns the computed value.
        """
        self.difficulty = compute_case_difficulty(self.before_latex, self.after_latex)
        return self.difficulty
    
    def get_complexity(self) -> Tuple[ExpressionComplexity, ExpressionComplexity]:
        """
        Get full complexity analysis for before and after expressions.
        
        Returns:
            Tuple of (before_complexity, after_complexity)
        """
        return (
            analyze_latex_complexity(self.before_latex),
            analyze_latex_complexity(self.after_latex)
        )


# =============================================================================
# Noise/Distraction Generator for Crop Regions
# =============================================================================

class NoiseGenerator:
    """
    Generate noise/distraction specifications for crop regions.
    These represent visual elements that may appear near the edit symbol
    but should be ignored by the model.
    """
    
    NOISE_TYPES = {
        # ===========================================
        # Line-based noise (slashes through crop)
        # ===========================================
        "random_line": {
            "description": "Random straight line segment",
            "params": ["angle", "length", "thickness"]
        },
        "dashed_line": {
            "description": "Dashed or dotted line",
            "params": ["angle", "dash_pattern", "length"]
        },
        "curved_line": {
            "description": "Random curved stroke (like cursive)",
            "params": ["curvature", "length"]
        },
        "slash_mark": {
            "description": "Diagonal slash across crop (like crossing out)",
            "params": ["direction", "thickness"]  # direction: / or \
        },
        "scribble": {
            "description": "Quick scribble/zigzag line",
            "params": ["intensity", "direction"]
        },
        
        # ===========================================
        # Edge intrusion (artifacts at crop borders)
        # ===========================================
        "edge_stroke": {
            "description": "Partial stroke entering from edge (neighbor symbol)",
            "params": ["edge", "penetration", "angle"]  # edge: top/bottom/left/right
        },
        "border_artifact": {
            "description": "Partial content at crop border",
            "params": ["edge", "content_type", "size"]
        },
        "corner_mark": {
            "description": "Mark in corner (adjacent writing)",
            "params": ["corner", "shape"]  # corner: tl/tr/bl/br
        },
        
        # ===========================================
        # Symbol-based noise (near edit region)
        # ===========================================
        "nearby_symbol": {
            "description": "Part of adjacent symbol visible in crop",
            "params": ["symbol", "visibility_pct", "position"]
        },
        "partial_symbol": {
            "description": "Incomplete/partial symbol stroke",
            "params": ["symbol", "completeness"]
        },
        "stray_mark": {
            "description": "Small accidental mark or dot",
            "params": ["size", "shape"]
        },
        "floating_dot": {
            "description": "Stray dot (from pen tap or i-dot)",
            "params": ["position", "size"]
        },
        
        # ===========================================
        # Correction-based noise (user errors)
        # ===========================================
        "crossed_out": {
            "description": "Crossed out previous attempt",
            "params": ["underlying_symbol", "cross_style"]
        },
        "smudge": {
            "description": "Erased/smudged area",
            "params": ["intensity", "size"]
        },
        "overwrite": {
            "description": "Symbol written over another",
            "params": ["underlying_symbol", "opacity"]
        },
        "scratch_out": {
            "description": "Heavy scratch-out lines",
            "params": ["num_lines", "intensity"]
        },
        
        # ===========================================
        # Structural noise (background)
        # ===========================================
        "grid_line": {
            "description": "Background grid line (from paper)",
            "params": ["orientation", "spacing"]
        },
        "margin_line": {
            "description": "Page margin or border",
            "params": ["position"]
        },
        "texture_noise": {
            "description": "Paper texture/grain",
            "params": ["intensity"]
        },
    }
    
    @classmethod
    def generate_noise_spec(cls, 
                           num_elements: int = None,
                           noise_probability: float = 0.4) -> List[Dict[str, Any]]:
        """Generate random noise specification for a crop region."""
        
        if random.random() > noise_probability:
            return []  # No noise
        
        if num_elements is None:
            num_elements = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        
        noise_specs = []
        selected_types = random.sample(list(cls.NOISE_TYPES.keys()), 
                                       min(num_elements, len(cls.NOISE_TYPES)))
        
        for noise_type in selected_types:
            spec = {
                "type": noise_type,
                "description": cls.NOISE_TYPES[noise_type]["description"],
            }
            
            # =============================================
            # Line-based noise parameters
            # =============================================
            if noise_type == "random_line":
                spec["angle"] = random.uniform(0, 180)
                spec["length"] = random.uniform(0.1, 0.5)  # Relative to crop size
                spec["thickness"] = random.choice(["thin", "medium", "thick"])
                
            elif noise_type == "dashed_line":
                spec["angle"] = random.uniform(0, 180)
                spec["dash_pattern"] = random.choice(["dashed", "dotted", "dash-dot"])
                spec["length"] = random.uniform(0.1, 0.4)
                
            elif noise_type == "curved_line":
                spec["curvature"] = random.uniform(0.2, 0.8)
                spec["length"] = random.uniform(0.2, 0.5)
                spec["start_edge"] = random.choice(["left", "right", "top", "bottom"])
                
            elif noise_type == "slash_mark":
                spec["direction"] = random.choice(["/", "\\"])  # Forward or back slash
                spec["thickness"] = random.choice([1, 2, 3])
                spec["spans_full"] = random.random() < 0.3  # 30% chance to span entire crop
                
            elif noise_type == "scribble":
                spec["intensity"] = random.choice(["light", "medium", "heavy"])
                spec["direction"] = random.choice(["horizontal", "vertical", "diagonal"])
                spec["num_zigzags"] = random.randint(3, 8)
                
            # =============================================
            # Edge intrusion parameters
            # =============================================
            elif noise_type == "edge_stroke":
                spec["edge"] = random.choice(["left", "right", "top", "bottom"])
                spec["penetration"] = random.uniform(0.1, 0.4)  # How far into crop
                spec["angle"] = random.uniform(-30, 30)  # Angle from perpendicular
                spec["thickness"] = random.choice([1, 2, 3])
                
            elif noise_type == "border_artifact":
                spec["edge"] = random.choice(["left", "right", "top", "bottom"])
                spec["content_type"] = random.choice(["curve", "line", "partial_symbol"])
                spec["size"] = random.uniform(0.1, 0.3)
                
            elif noise_type == "corner_mark":
                spec["corner"] = random.choice(["tl", "tr", "bl", "br"])
                spec["shape"] = random.choice(["dot", "curve", "angle", "partial"])
                spec["size"] = random.uniform(0.05, 0.15)
                
            # =============================================
            # Symbol-based noise parameters
            # =============================================
            elif noise_type == "nearby_symbol":
                spec["symbol"] = random.choice(list("xyzabc") + [r"\alpha", r"\beta"])
                spec["visibility_pct"] = random.uniform(0.1, 0.4)
                spec["position"] = random.choice(["left", "right", "top", "bottom"])
                
            elif noise_type == "partial_symbol":
                spec["symbol"] = random.choice(list("xyzabc123"))
                spec["completeness"] = random.uniform(0.2, 0.6)
                
            elif noise_type == "stray_mark":
                spec["size"] = random.choice(["small", "tiny"])
                spec["shape"] = random.choice(["dot", "short_stroke", "curve"])
                
            elif noise_type == "floating_dot":
                spec["position"] = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
                spec["size"] = random.randint(1, 4)
                
            # =============================================
            # Correction-based noise parameters
            # =============================================
            elif noise_type == "crossed_out":
                spec["underlying_symbol"] = random.choice(list("xyzabc123"))
                spec["cross_style"] = random.choice(["X", "single_line", "scribble"])
                
            elif noise_type == "scratch_out":
                spec["num_lines"] = random.randint(2, 5)
                spec["intensity"] = random.choice(["light", "medium", "heavy"])
                spec["region"] = (random.uniform(0.2, 0.5), random.uniform(0.2, 0.5))  # x, y center
                
            elif noise_type == "smudge":
                spec["intensity"] = random.uniform(0.1, 0.4)
                spec["size"] = random.uniform(0.1, 0.3)
                
            noise_specs.append(spec)
        
        return noise_specs
    
    @classmethod
    def get_noise_description(cls, noise_specs: List[Dict]) -> str:
        """Get human-readable description of noise elements."""
        if not noise_specs:
            return "clean (no noise)"
        return ", ".join(spec["type"] for spec in noise_specs)


# =============================================================================
# Symbol Libraries
# =============================================================================

GREEK_LOWER = [
    ("alpha", r"\alpha"), ("beta", r"\beta"), ("gamma", r"\gamma"),
    ("delta", r"\delta"), ("epsilon", r"\epsilon"), ("zeta", r"\zeta"),
    ("eta", r"\eta"), ("theta", r"\theta"), ("iota", r"\iota"),
    ("kappa", r"\kappa"), ("lambda", r"\lambda"), ("mu", r"\mu"),
    ("nu", r"\nu"), ("xi", r"\xi"), ("pi", r"\pi"),
    ("rho", r"\rho"), ("sigma", r"\sigma"), ("tau", r"\tau"),
    ("upsilon", r"\upsilon"), ("phi", r"\phi"), ("chi", r"\chi"),
    ("psi", r"\psi"), ("omega", r"\omega"),
]

GREEK_UPPER = [
    ("Gamma", r"\Gamma"), ("Delta", r"\Delta"), ("Theta", r"\Theta"),
    ("Lambda", r"\Lambda"), ("Xi", r"\Xi"), ("Pi", r"\Pi"),
    ("Sigma", r"\Sigma"), ("Phi", r"\Phi"), ("Psi", r"\Psi"),
    ("Omega", r"\Omega"),
]

OPERATORS = [
    ("+", "+"), ("-", "-"), ("times", r"\times"), ("div", r"\div"),
    ("cdot", r"\cdot"), ("pm", r"\pm"), ("mp", r"\mp"),
]

RELATIONS = [
    ("=", "="), ("neq", r"\neq"), ("leq", r"\leq"), ("geq", r"\geq"),
    ("<", "<"), (">", ">"), ("approx", r"\approx"), ("equiv", r"\equiv"),
    ("sim", r"\sim"), ("propto", r"\propto"), ("subset", r"\subset"),
    ("supset", r"\supset"), ("in", r"\in"), ("ni", r"\ni"),
]

ARROWS = [
    ("to", r"\to"), ("rightarrow", r"\rightarrow"), ("leftarrow", r"\leftarrow"),
    ("Rightarrow", r"\Rightarrow"), ("Leftarrow", r"\Leftarrow"),
    ("leftrightarrow", r"\leftrightarrow"), ("mapsto", r"\mapsto"),
    ("hookrightarrow", r"\hookrightarrow"), ("twoheadrightarrow", r"\twoheadrightarrow"),
]

FUNCTIONS = [
    ("sin", r"\sin"), ("cos", r"\cos"), ("tan", r"\tan"),
    ("log", r"\log"), ("ln", r"\ln"), ("exp", r"\exp"),
    ("lim", r"\lim"), ("max", r"\max"), ("min", r"\min"),
    ("sup", r"\sup"), ("inf", r"\inf"),
]

VARIABLES = list("xyzuvwabcdefghijklmnpqrst")
DIGITS = list("0123456789")


# =============================================================================
# Commutative Diagram Cases
# =============================================================================

class DiagramCases:
    """Generate commutative diagram editing cases."""
    
    ARROW_DIRECTIONS = ["r", "d", "l", "u", "dr", "dl", "ur", "ul"]
    ARROW_STYLES = ["", "dashed", "hook", "two heads", "tail"]
    NODE_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
        r"X'", r"Y'", r"Z'", r"A_1", r"B_2", r"X_n", r"Y_m",
        r"\mathcal{F}", r"\mathcal{G}", r"\mathcal{C}", r"\mathcal{D}",
    ]
    MORPHISM_LABELS = list("fghijklmnpqrstu") + [
        r"\varphi", r"\psi", r"\eta", r"\mu", r"\nu",
        r"f'", r"g'", r"f^*", r"g_*", r"f \circ g",
    ]
    
    @classmethod
    def simple_arrow(cls) -> str:
        """A -> B"""
        a, b = random.sample(cls.NODE_LABELS, 2)
        f = random.choice(cls.MORPHISM_LABELS)
        return rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] & {b} \end{{tikzcd}}"
    
    @classmethod
    def triangle(cls) -> str:
        """Commutative triangle."""
        a, b, c = random.sample(cls.NODE_LABELS, 3)
        f, g, h = random.sample(cls.MORPHISM_LABELS, 3)
        return rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] \arrow[dr, \"{h}\"'] & {b} \arrow[d, \"{g}\"] \\ & {c} \end{{tikzcd}}"
    
    @classmethod
    def square(cls) -> str:
        """Commutative square."""
        a, b, c, d = random.sample(cls.NODE_LABELS, 4)
        f, g, h, k = random.sample(cls.MORPHISM_LABELS, 4)
        return rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] \arrow[d, \"{g}\"'] & {b} \arrow[d, \"{h}\"] \\ {c} \arrow[r, \"{k}\"'] & {d} \end{{tikzcd}}"
    
    @classmethod
    def generate_add_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate ADD operation cases for diagrams."""
        cases = []
        
        for i in range(count):
            case_type = random.choice(["add_arrow", "add_node", "add_label", "extend_sequence"])
            
            if case_type == "add_arrow":
                # Add diagonal to L-shape
                a, b, c = random.sample(cls.NODE_LABELS, 3)
                f, g, h = random.sample(cls.MORPHISM_LABELS, 3)
                before = rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] & {b} \arrow[d, \"{g}\"] \\ & {c} \end{{tikzcd}}"
                after = rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] \arrow[dr, \"{h}\"'] & {b} \arrow[d, \"{g}\"] \\ & {c} \end{{tikzcd}}"
                desc = f"Add diagonal arrow {h}: {a} -> {c}"
                
            elif case_type == "add_node":
                # Extend A -> B to A -> B -> C
                a, b, c = random.sample(cls.NODE_LABELS, 3)
                f, g = random.sample(cls.MORPHISM_LABELS, 2)
                before = rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] & {b} \end{{tikzcd}}"
                after = rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] & {b} \arrow[r, \"{g}\"] & {c} \end{{tikzcd}}"
                desc = f"Add node {c} with arrow {g}"
                
            elif case_type == "add_label":
                # Add label to unlabeled arrow
                a, b = random.sample(cls.NODE_LABELS, 2)
                f = random.choice(cls.MORPHISM_LABELS)
                before = rf"\begin{{tikzcd}} {a} \arrow[r] & {b} \end{{tikzcd}}"
                after = rf"\begin{{tikzcd}} {a} \arrow[r, \"{f}\"] & {b} \end{{tikzcd}}"
                desc = f"Add label {f} to arrow"
                
            else:  # extend_sequence
                # Add to exact sequence
                nodes = random.sample(cls.NODE_LABELS, 4)
                morph = random.sample(cls.MORPHISM_LABELS, 3)
                before = rf"\begin{{tikzcd}} {nodes[0]} \arrow[r, \"{morph[0]}\"] & {nodes[1]} \arrow[r, \"{morph[1]}\"] & {nodes[2]} \end{{tikzcd}}"
                after = rf"\begin{{tikzcd}} {nodes[0]} \arrow[r, \"{morph[0]}\"] & {nodes[1]} \arrow[r, \"{morph[1]}\"] & {nodes[2]} \arrow[r, \"{morph[2]}\"] & {nodes[3]} \end{{tikzcd}}"
                desc = f"Extend sequence with {nodes[3]}"
            
            cases.append(EditCase(
                id=f"diagram_add_{i:04d}",
                category="diagram",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty=random.choice(["easy", "medium"]),
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate REPLACE operation cases for diagrams."""
        cases = []
        
        for i in range(count):
            case_type = random.choice(["replace_label", "replace_node", "replace_arrow_style"])
            
            if case_type == "replace_label":
                a, b = random.sample(cls.NODE_LABELS, 2)
                f_old, f_new = random.sample(cls.MORPHISM_LABELS, 2)
                before = rf"\begin{{tikzcd}} {a} \arrow[r, \"{f_old}\"] & {b} \end{{tikzcd}}"
                after = rf"\begin{{tikzcd}} {a} \arrow[r, \"{f_new}\"] & {b} \end{{tikzcd}}"
                desc = f"Replace morphism label {f_old} -> {f_new}"
                
            elif case_type == "replace_node":
                a_old, a_new, b = random.sample(cls.NODE_LABELS, 3)
                f = random.choice(cls.MORPHISM_LABELS)
                before = rf"\begin{{tikzcd}} {a_old} \arrow[r, \"{f}\"] & {b} \end{{tikzcd}}"
                after = rf"\begin{{tikzcd}} {a_new} \arrow[r, \"{f}\"] & {b} \end{{tikzcd}}"
                desc = f"Replace node {a_old} -> {a_new}"
                
            else:  # replace_arrow_style
                a, b = random.sample(cls.NODE_LABELS, 2)
                f = random.choice(cls.MORPHISM_LABELS)
                style_old, style_new = random.sample(["", "dashed", "hook", "two heads"], 2)
                style_str_old = f", {style_old}" if style_old else ""
                style_str_new = f", {style_new}" if style_new else ""
                before = rf"\begin{{tikzcd}} {a} \arrow[r{style_str_old}, \"{f}\"] & {b} \end{{tikzcd}}"
                after = rf"\begin{{tikzcd}} {a} \arrow[r{style_str_new}, \"{f}\"] & {b} \end{{tikzcd}}"
                desc = f"Change arrow style: {style_old or 'solid'} -> {style_new or 'solid'}"
            
            cases.append(EditCase(
                id=f"diagram_replace_{i:04d}",
                category="diagram",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty=random.choice(["easy", "medium"]),
            ))
        
        return cases
    
    @classmethod
    def generate_insert_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate INSERT operation cases for diagrams."""
        cases = []
        
        for i in range(count):
            # Insert node in sequence
            nodes = random.sample(cls.NODE_LABELS, 4)
            morph = random.sample(cls.MORPHISM_LABELS, 3)
            
            # Before: A -> C, After: A -> B -> C
            before = rf"\begin{{tikzcd}} {nodes[0]} \arrow[r, \"{morph[0]}\"] & {nodes[2]} \end{{tikzcd}}"
            after = rf"\begin{{tikzcd}} {nodes[0]} \arrow[r, \"{morph[1]}\"] & {nodes[1]} \arrow[r, \"{morph[2]}\"] & {nodes[2]} \end{{tikzcd}}"
            desc = f"Insert {nodes[1]} between {nodes[0]} and {nodes[2]}"
            
            cases.append(EditCase(
                id=f"diagram_insert_{i:04d}",
                category="diagram",
                subcategory="insert_node",
                operation="INSERT",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                position_hint=f"between:{nodes[0]}:{nodes[2]}",
                difficulty="medium",
            ))
        
        return cases


# =============================================================================
# Calculus and Analysis Cases
# =============================================================================

class CalculusCases:
    """Generate calculus expression editing cases."""
    
    @staticmethod
    def random_var() -> str:
        return random.choice(VARIABLES[:5])  # x, y, z, u, v
    
    @staticmethod
    def random_const() -> str:
        return random.choice(["a", "b", "c", "k", "n", "m"])
    
    @staticmethod
    def random_func() -> str:
        func = random.choice(FUNCTIONS[:6])  # sin, cos, tan, log, ln, exp
        return func[1]
    
    @classmethod
    def integral_definite(cls) -> str:
        var = cls.random_var()
        lower = random.choice(["0", "a", "-1", r"-\infty"])
        upper = random.choice(["1", "b", r"\infty", r"\pi"])
        integrand = random.choice([
            f"{var}^2",
            rf"\sin({var})",
            rf"e^{{{var}}}",
            rf"\frac{{1}}{{{var}}}",
            f"{var}^3 + {var}",
        ])
        return rf"\int_{{{lower}}}^{{{upper}}} {integrand} \, d{var}"
    
    @classmethod
    def integral_indefinite(cls) -> str:
        var = cls.random_var()
        integrand = random.choice([
            f"{var}^2",
            rf"\cos({var})",
            rf"e^{{{var}}}",
            rf"\ln({var})",
        ])
        return rf"\int {integrand} \, d{var}"
    
    @classmethod
    def derivative(cls) -> str:
        var = cls.random_var()
        func = random.choice([
            f"{var}^2",
            rf"\sin({var})",
            rf"e^{{{var}}}",
            rf"\ln({var})",
            f"{var}^3 - {var}",
        ])
        notation = random.choice(["frac", "prime"])
        if notation == "frac":
            return rf"\frac{{d}}{{d{var}}} \left( {func} \right)"
        else:
            return rf"({func})'"
    
    @classmethod
    def partial_derivative(cls) -> str:
        vars_ = random.sample(VARIABLES[:3], 2)
        func = f"f({vars_[0]}, {vars_[1]})"
        return rf"\frac{{\partial {func}}}{{\partial {vars_[0]}}}"
    
    @classmethod
    def limit(cls) -> str:
        var = cls.random_var()
        approach = random.choice(["0", r"\infty", r"-\infty", "a", "1"])
        func = random.choice([
            rf"\frac{{\sin({var})}}{{{var}}}",
            rf"\frac{{{var}^2 - 1}}{{{var} - 1}}",
            rf"\left(1 + \frac{{1}}{{{var}}}\right)^{{{var}}}",
            rf"e^{{-{var}}}",
        ])
        return rf"\lim_{{{var} \to {approach}}} {func}"
    
    @classmethod
    def series(cls) -> str:
        var = random.choice(["n", "k", "i"])
        term = random.choice([
            rf"\frac{{1}}{{{var}^2}}",
            rf"\frac{{(-1)^{var}}}{{{var}}}",
            rf"\frac{{{var}}}{{{var}!}}",
            rf"x^{var}",
        ])
        return rf"\sum_{{{var}=1}}^{{\infty}} {term}"
    
    @classmethod
    def generate_add_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate ADD cases for calculus expressions."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_bounds", "add_term", "add_limits", "add_constant"
            ])
            
            if case_type == "add_bounds":
                # Indefinite -> Definite integral
                var = cls.random_var()
                integrand = f"{var}^2"
                before = rf"\int {integrand} \, d{var}"
                after = rf"\int_{{0}}^{{1}} {integrand} \, d{var}"
                desc = "Add integration bounds [0, 1]"
                
            elif case_type == "add_term":
                # Add term to expression
                var = cls.random_var()
                before = rf"\int {var}^2 \, d{var}"
                after = rf"\int ({var}^2 + {var}) \, d{var}"
                desc = f"Add term +{var} to integrand"
                
            elif case_type == "add_limits":
                # Add limit notation
                var = cls.random_var()
                before = rf"\frac{{\sin({var})}}{{{var}}}"
                after = rf"\lim_{{{var} \to 0}} \frac{{\sin({var})}}{{{var}}}"
                desc = f"Add limit as {var} -> 0"
                
            else:  # add_constant
                var = cls.random_var()
                before = rf"\int {var}^2 \, d{var} = \frac{{{var}^3}}{{3}}"
                after = rf"\int {var}^2 \, d{var} = \frac{{{var}^3}}{{3}} + C"
                desc = "Add constant of integration C"
            
            cases.append(EditCase(
                id=f"calculus_add_{i:04d}",
                category="calculus",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty=random.choice(["easy", "medium"]),
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate REPLACE cases for calculus expressions."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "replace_bound", "replace_variable", "replace_function"
            ])
            
            if case_type == "replace_bound":
                var = cls.random_var()
                old_bound, new_bound = random.sample(["0", "1", "a", r"\pi"], 2)
                before = rf"\int_{{{old_bound}}}^{{1}} {var}^2 \, d{var}"
                after = rf"\int_{{{new_bound}}}^{{1}} {var}^2 \, d{var}"
                desc = f"Replace lower bound {old_bound} -> {new_bound}"
                
            elif case_type == "replace_variable":
                old_var, new_var = random.sample(VARIABLES[:5], 2)
                before = rf"\int {old_var}^2 \, d{old_var}"
                after = rf"\int {new_var}^2 \, d{new_var}"
                desc = f"Replace variable {old_var} -> {new_var}"
                
            else:  # replace_function
                var = cls.random_var()
                old_func = rf"\sin({var})"
                new_func = rf"\cos({var})"
                before = rf"\int {old_func} \, d{var}"
                after = rf"\int {new_func} \, d{var}"
                desc = f"Replace sin -> cos"
            
            cases.append(EditCase(
                id=f"calculus_replace_{i:04d}",
                category="calculus",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty=random.choice(["easy", "medium"]),
            ))
        
        return cases


# =============================================================================
# Differential Equations Cases
# =============================================================================

class DiffEqCases:
    """Generate differential equation editing cases."""
    
    @classmethod
    def ode_first_order(cls) -> str:
        var = random.choice(["y", "u", "f"])
        indep = random.choice(["x", "t"])
        rhs = random.choice([
            f"{var}",
            f"{indep}",
            f"{var} + {indep}",
            rf"{var}^2",
            rf"\sin({indep})",
        ])
        return rf"\frac{{d{var}}}{{d{indep}}} = {rhs}"
    
    @classmethod
    def ode_second_order(cls) -> str:
        var = random.choice(["y", "u"])
        indep = random.choice(["x", "t"])
        coef = random.choice(["", "2", "k"])
        rhs = random.choice(["0", f"{var}", rf"\sin({indep})"])
        return rf"\frac{{d^2 {var}}}{{d{indep}^2}} + {coef}{var} = {rhs}"
    
    @classmethod
    def pde_heat(cls) -> str:
        return rf"\frac{{\partial u}}{{\partial t}} = k \frac{{\partial^2 u}}{{\partial x^2}}"
    
    @classmethod
    def pde_wave(cls) -> str:
        return rf"\frac{{\partial^2 u}}{{\partial t^2}} = c^2 \frac{{\partial^2 u}}{{\partial x^2}}"
    
    @classmethod
    def generate_add_cases(cls, count: int = 30) -> List[EditCase]:
        """Generate ADD cases for differential equations."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_initial_condition", "add_term", "add_boundary"
            ])
            
            if case_type == "add_initial_condition":
                before = rf"\frac{{dy}}{{dx}} = y"
                after = rf"\frac{{dy}}{{dx}} = y, \quad y(0) = 1"
                desc = "Add initial condition y(0) = 1"
                
            elif case_type == "add_term":
                var = random.choice(["y", "u"])
                before = rf"\frac{{d{var}}}{{dx}} = {var}"
                after = rf"\frac{{d{var}}}{{dx}} = {var} + x"
                desc = "Add forcing term +x"
                
            else:  # add_boundary
                before = rf"\frac{{\partial^2 u}}{{\partial x^2}} = 0"
                after = rf"\frac{{\partial^2 u}}{{\partial x^2}} = 0, \quad u(0) = 0, \, u(1) = 1"
                desc = "Add boundary conditions"
            
            cases.append(EditCase(
                id=f"diffeq_add_{i:04d}",
                category="diffeq",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="medium",
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 30) -> List[EditCase]:
        """Generate REPLACE cases for differential equations."""
        cases = []
        
        for i in range(count):
            case_type = random.choice(["replace_order", "replace_coefficient", "replace_rhs"])
            
            if case_type == "replace_order":
                before = rf"\frac{{dy}}{{dx}} = y"
                after = rf"\frac{{d^2 y}}{{dx^2}} = y"
                desc = "Change from first to second order"
                
            elif case_type == "replace_coefficient":
                old_coef = random.choice(["k", "2", "a"])
                new_coef = random.choice(["c", "3", "b"])
                before = rf"\frac{{dy}}{{dx}} + {old_coef}y = 0"
                after = rf"\frac{{dy}}{{dx}} + {new_coef}y = 0"
                desc = f"Replace coefficient {old_coef} -> {new_coef}"
                
            else:  # replace_rhs
                before = rf"\frac{{dy}}{{dx}} = 0"
                rhs = random.choice(["y", "x", r"\sin(x)"])
                after = rf"\frac{{dy}}{{dx}} = {rhs}"
                desc = f"Replace RHS 0 -> {rhs}"
            
            cases.append(EditCase(
                id=f"diffeq_replace_{i:04d}",
                category="diffeq",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="medium",
            ))
        
        return cases


# =============================================================================
# Matrix Cases
# =============================================================================

class MatrixCases:
    """Generate matrix editing cases."""
    
    @staticmethod
    def random_element() -> str:
        return random.choice([
            *list("abcdefghijk"),
            "0", "1", "-1", "2",
            r"\lambda", r"\mu",
            "x", "y",
        ])
    
    @classmethod
    def matrix_2x2(cls) -> str:
        elems = [cls.random_element() for _ in range(4)]
        return rf"\begin{{pmatrix}} {elems[0]} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{pmatrix}}"
    
    @classmethod
    def matrix_3x3(cls) -> str:
        elems = [cls.random_element() for _ in range(9)]
        return rf"\begin{{pmatrix}} {elems[0]} & {elems[1]} & {elems[2]} \\ {elems[3]} & {elems[4]} & {elems[5]} \\ {elems[6]} & {elems[7]} & {elems[8]} \end{{pmatrix}}"
    
    @classmethod
    def determinant(cls, size: int = 2) -> str:
        if size == 2:
            elems = [cls.random_element() for _ in range(4)]
            return rf"\begin{{vmatrix}} {elems[0]} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{vmatrix}}"
        else:
            elems = [cls.random_element() for _ in range(9)]
            return rf"\begin{{vmatrix}} {elems[0]} & {elems[1]} & {elems[2]} \\ {elems[3]} & {elems[4]} & {elems[5]} \\ {elems[6]} & {elems[7]} & {elems[8]} \end{{vmatrix}}"
    
    @classmethod
    def generate_add_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate ADD cases for matrices."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_row", "add_column", "add_element", "expand_matrix"
            ])
            
            if case_type == "add_row":
                # 1x2 -> 2x2
                a, b = cls.random_element(), cls.random_element()
                c, d = cls.random_element(), cls.random_element()
                before = rf"\begin{{pmatrix}} {a} & {b} \end{{pmatrix}}"
                after = rf"\begin{{pmatrix}} {a} & {b} \\ {c} & {d} \end{{pmatrix}}"
                desc = f"Add row [{c}, {d}]"
                pos_hint = "row:2"
                
            elif case_type == "add_column":
                # 2x1 -> 2x2
                a, c = cls.random_element(), cls.random_element()
                b, d = cls.random_element(), cls.random_element()
                before = rf"\begin{{pmatrix}} {a} \\ {c} \end{{pmatrix}}"
                after = rf"\begin{{pmatrix}} {a} & {b} \\ {c} & {d} \end{{pmatrix}}"
                desc = f"Add column [{b}, {d}]"
                pos_hint = "col:2"
                
            elif case_type == "add_element":
                # Add subscript to element
                elem = cls.random_element()
                sub = random.choice(["1", "2", "ij", "11"])
                before = rf"\begin{{pmatrix}} {elem} & 0 \\ 0 & {elem} \end{{pmatrix}}"
                after = rf"\begin{{pmatrix}} {elem}_{{{sub}}} & 0 \\ 0 & {elem}_{{{sub}}} \end{{pmatrix}}"
                desc = f"Add subscript {sub} to {elem}"
                pos_hint = "subscript"
                
            else:  # expand_matrix
                # 2x2 -> 3x3
                elems = [cls.random_element() for _ in range(4)]
                new_elems = [cls.random_element() for _ in range(5)]
                before = rf"\begin{{pmatrix}} {elems[0]} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{pmatrix}}"
                after = rf"\begin{{pmatrix}} {elems[0]} & {elems[1]} & {new_elems[0]} \\ {elems[2]} & {elems[3]} & {new_elems[1]} \\ {new_elems[2]} & {new_elems[3]} & {new_elems[4]} \end{{pmatrix}}"
                desc = "Expand 2x2 to 3x3"
                pos_hint = "expand"
            
            cases.append(EditCase(
                id=f"matrix_add_{i:04d}",
                category="matrix",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                position_hint=pos_hint if 'pos_hint' in dir() else None,
                difficulty=random.choice(["easy", "medium", "hard"]),
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate REPLACE cases for matrices."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "replace_element", "replace_bracket", "swap_rows"
            ])
            
            if case_type == "replace_element":
                elems = [cls.random_element() for _ in range(4)]
                old_elem = elems[0]
                new_elem = cls.random_element()
                while new_elem == old_elem:
                    new_elem = cls.random_element()
                before = rf"\begin{{pmatrix}} {elems[0]} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{pmatrix}}"
                after = rf"\begin{{pmatrix}} {new_elem} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{pmatrix}}"
                desc = f"Replace element {old_elem} -> {new_elem}"
                
            elif case_type == "replace_bracket":
                elems = [cls.random_element() for _ in range(4)]
                before = rf"\begin{{pmatrix}} {elems[0]} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{pmatrix}}"
                after = rf"\begin{{bmatrix}} {elems[0]} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{bmatrix}}"
                desc = "Change brackets: () -> []"
                
            else:  # swap_rows
                elems = [cls.random_element() for _ in range(4)]
                before = rf"\begin{{pmatrix}} {elems[0]} & {elems[1]} \\ {elems[2]} & {elems[3]} \end{{pmatrix}}"
                after = rf"\begin{{pmatrix}} {elems[2]} & {elems[3]} \\ {elems[0]} & {elems[1]} \end{{pmatrix}}"
                desc = "Swap rows 1 and 2"
            
            cases.append(EditCase(
                id=f"matrix_replace_{i:04d}",
                category="matrix",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty=random.choice(["easy", "medium"]),
            ))
        
        return cases


# =============================================================================
# Single Symbol Cases (Atomic Edits)
# =============================================================================

class SingleSymbolCases:
    """
    Generate ATOMIC single-symbol editing cases.
    Each case involves exactly ONE symbol change.
    """
    
    # All possible single symbols
    ENGLISH_LOWER = list("abcdefghijklmnopqrstuvwxyz")
    ENGLISH_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    DIGITS = list("0123456789")
    
    GREEK_SYMBOLS = [
        r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
        r"\zeta", r"\eta", r"\theta", r"\iota", r"\kappa",
        r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi",
        r"\rho", r"\sigma", r"\tau", r"\upsilon", r"\phi",
        r"\chi", r"\psi", r"\omega",
        r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
        r"\Pi", r"\Sigma", r"\Phi", r"\Psi", r"\Omega",
    ]
    
    OPERATORS = ["+", "-", r"\times", r"\cdot", r"\pm", r"\mp", r"\div"]
    
    RELATIONS = ["=", r"\neq", r"\leq", r"\geq", "<", ">", r"\approx", r"\equiv", r"\sim"]
    
    ARROWS = [r"\to", r"\rightarrow", r"\leftarrow", r"\Rightarrow", r"\mapsto"]
    
    CALCULUS = [r"\partial", r"\nabla", r"\infty", r"\int", r"\sum", r"\prod"]
    
    SUBSCRIPTS = ["0", "1", "2", "i", "j", "k", "n", "m", "x", "y", "max", "min"]
    
    SUPERSCRIPTS = ["2", "3", "n", "-1", r"\prime", "*", r"\dagger", "T"]
    
    @classmethod
    def random_single_symbol(cls) -> str:
        """Get a random single symbol from any category."""
        pool = (
            cls.ENGLISH_LOWER + 
            cls.ENGLISH_UPPER + 
            cls.DIGITS +
            cls.GREEK_SYMBOLS
        )
        return random.choice(pool)
    
    @classmethod
    def random_base_symbol(cls) -> str:
        """Get a random base symbol (for adding sub/superscripts)."""
        pool = cls.ENGLISH_LOWER[:10] + cls.GREEK_SYMBOLS[:10]  # x, y, z, etc + α, β, etc
        return random.choice(pool)
    
    @classmethod
    def generate_initial_state_cases(cls, count: int = 100) -> List[EditCase]:
        """
        Generate cases starting from EMPTY initial state.
        These represent the first symbol drawn on a blank canvas.
        """
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "first_letter",
                "first_greek",
                "first_digit",
                "first_operator_symbol",
                "first_bracket",
                "first_calculus",
            ])
            
            if case_type == "first_letter":
                symbol = random.choice(cls.ENGLISH_LOWER + cls.ENGLISH_UPPER)
                desc = f"Draw first letter: {symbol}"
                
            elif case_type == "first_greek":
                symbol = random.choice(cls.GREEK_SYMBOLS)
                desc = f"Draw first Greek letter: {symbol}"
                
            elif case_type == "first_digit":
                symbol = random.choice(cls.DIGITS)
                desc = f"Draw first digit: {symbol}"
                
            elif case_type == "first_operator_symbol":
                symbol = random.choice(["+", "-", "=", r"\pm"])
                desc = f"Draw first operator: {symbol}"
                
            elif case_type == "first_bracket":
                symbol = random.choice(["(", ")", "[", "]", r"\{", r"\}"])
                desc = f"Draw first bracket: {symbol}"
                
            else:  # first_calculus
                symbol = random.choice([r"\int", r"\sum", r"\partial", r"\nabla", r"\infty"])
                desc = f"Draw first calculus symbol: {symbol}"
            
            cases.append(EditCase(
                id=f"initial_{i:04d}",
                category="single_symbol",
                subcategory=case_type,
                operation="ADD",
                before_latex="",  # EMPTY initial state
                after_latex=symbol,
                edit_description=desc,
                position_hint="start",
                difficulty="easy",
                metadata={"atomic": True, "initial_state": True, "edit_symbol": symbol}
            ))
        
        return cases
    
    @classmethod
    def generate_add_single_symbol(cls, count: int = 100) -> List[EditCase]:
        """
        ADD one single symbol to an existing expression.
        (Initial state cases are generated separately via generate_initial_state_cases)
        Examples: 
            - "x" -> "x + y"
            - "a" -> "a_1"
            - "x" -> "x^2"
        """
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_letter_after",       # x -> xy
                "add_operator",           # x -> x+
                "add_subscript",          # x -> x_1
                "add_superscript",        # x -> x^2
                "add_greek",              # x -> x\alpha
                "add_digit",              # x -> x1 or x_1
            ])
            
            if case_type == "add_letter_after":
                # Add one letter after existing
                base = cls.random_base_symbol()
                new_sym = random.choice(cls.ENGLISH_LOWER[:10])
                before = base
                after = f"{base} {new_sym}"
                desc = f"Add letter {new_sym} after {base}"
                pos = "after"
                
            elif case_type == "add_operator":
                # Add operator (implies next symbol coming)
                base = cls.random_base_symbol()
                op = random.choice(cls.OPERATORS)
                before = base
                after = f"{base} {op}"
                desc = f"Add operator {op}"
                pos = "after"
                
            elif case_type == "add_subscript":
                # Add subscript to base
                base = cls.random_base_symbol()
                sub = random.choice(cls.SUBSCRIPTS)
                before = base
                after = f"{base}_{{{sub}}}"
                desc = f"Add subscript {sub} to {base}"
                pos = "subscript"
                
            elif case_type == "add_superscript":
                # Add superscript to base
                base = cls.random_base_symbol()
                sup = random.choice(cls.SUPERSCRIPTS)
                before = base
                after = f"{base}^{{{sup}}}"
                desc = f"Add superscript {sup} to {base}"
                pos = "superscript"
                
            elif case_type == "add_greek":
                # Add a Greek letter
                context = random.choice(["", "x + ", "= "])
                greek = random.choice(cls.GREEK_SYMBOLS)
                before = context.strip() if context else ""
                after = f"{context}{greek}".strip()
                desc = f"Add Greek letter {greek}"
                pos = "after" if context else "start"
                
            else:  # add_digit
                # Add digit as subscript or coefficient
                base = cls.random_base_symbol()
                digit = random.choice(cls.DIGITS)
                style = random.choice(["subscript", "coefficient"])
                if style == "subscript":
                    before = base
                    after = f"{base}_{{{digit}}}"
                    desc = f"Add digit {digit} as subscript"
                    pos = "subscript"
                else:
                    before = base
                    after = f"{digit}{base}"
                    desc = f"Add coefficient {digit}"
                    pos = "before"
            
            cases.append(EditCase(
                id=f"single_add_{i:04d}",
                category="single_symbol",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                position_hint=pos,
                difficulty="easy",
                metadata={"atomic": True, "symbol_count": 1}
            ))
        
        return cases
    
    @classmethod
    def generate_replace_single_symbol(cls, count: int = 100) -> List[EditCase]:
        """
        REPLACE exactly one symbol with another.
        Examples:
            - "x" -> "y"
            - "α" -> "β"  
            - "x^2" -> "x^3"
            - "x_1" -> "x_2"
        """
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "replace_letter",         # x -> y
                "replace_greek",          # α -> β
                "replace_digit",          # 2 -> 3
                "replace_subscript",      # x_1 -> x_2
                "replace_superscript",    # x^2 -> x^3
                "replace_operator",       # + -> -
                "letter_to_greek",        # x -> α
                "greek_to_letter",        # α -> x
            ])
            
            if case_type == "replace_letter":
                old_letter = random.choice(cls.ENGLISH_LOWER[:10])
                new_letter = random.choice([l for l in cls.ENGLISH_LOWER[:10] if l != old_letter])
                before = old_letter
                after = new_letter
                desc = f"Replace letter {old_letter} -> {new_letter}"
                
            elif case_type == "replace_greek":
                old_greek = random.choice(cls.GREEK_SYMBOLS[:15])
                new_greek = random.choice([g for g in cls.GREEK_SYMBOLS[:15] if g != old_greek])
                before = old_greek
                after = new_greek
                desc = f"Replace Greek {old_greek} -> {new_greek}"
                
            elif case_type == "replace_digit":
                old_digit = random.choice(cls.DIGITS)
                new_digit = random.choice([d for d in cls.DIGITS if d != old_digit])
                before = old_digit
                after = new_digit
                desc = f"Replace digit {old_digit} -> {new_digit}"
                
            elif case_type == "replace_subscript":
                base = cls.random_base_symbol()
                old_sub = random.choice(cls.SUBSCRIPTS[:6])
                new_sub = random.choice([s for s in cls.SUBSCRIPTS[:6] if s != old_sub])
                before = f"{base}_{{{old_sub}}}"
                after = f"{base}_{{{new_sub}}}"
                desc = f"Replace subscript {old_sub} -> {new_sub}"
                
            elif case_type == "replace_superscript":
                base = cls.random_base_symbol()
                old_sup = random.choice(cls.SUPERSCRIPTS[:4])
                new_sup = random.choice([s for s in cls.SUPERSCRIPTS[:4] if s != old_sup])
                before = f"{base}^{{{old_sup}}}"
                after = f"{base}^{{{new_sup}}}"
                desc = f"Replace superscript {old_sup} -> {new_sup}"
                
            elif case_type == "replace_operator":
                base = cls.random_base_symbol()
                var2 = random.choice(cls.ENGLISH_LOWER[:5])
                old_op = random.choice(["+", "-"])
                new_op = "+" if old_op == "-" else "-"
                before = f"{base} {old_op} {var2}"
                after = f"{base} {new_op} {var2}"
                desc = f"Replace operator {old_op} -> {new_op}"
                
            elif case_type == "letter_to_greek":
                letter = random.choice(cls.ENGLISH_LOWER[:10])
                greek = random.choice(cls.GREEK_SYMBOLS[:10])
                before = letter
                after = greek
                desc = f"Replace {letter} -> {greek}"
                
            else:  # greek_to_letter
                greek = random.choice(cls.GREEK_SYMBOLS[:10])
                letter = random.choice(cls.ENGLISH_LOWER[:10])
                before = greek
                after = letter
                desc = f"Replace {greek} -> {letter}"
            
            cases.append(EditCase(
                id=f"single_replace_{i:04d}",
                category="single_symbol",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
                metadata={"atomic": True, "symbol_count": 1}
            ))
        
        return cases
    
    @classmethod
    def generate_calculus_symbol_cases(cls, count: int = 50) -> List[EditCase]:
        """
        ADD/REPLACE calculus-specific symbols: ∂, ∇, d, ∫, etc.
        """
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_partial",            # f -> ∂f
                "add_derivative_d",       # y -> dy
                "add_nabla",              # f -> ∇f
                "add_integral",           # f -> ∫f
                "replace_d_to_partial",   # df -> ∂f
            ])
            
            var = random.choice(["f", "u", "v", "y", "g"])
            
            if case_type == "add_partial":
                before = var
                after = rf"\partial {var}"
                desc = f"Add partial symbol to {var}"
                op = "ADD"
                
            elif case_type == "add_derivative_d":
                before = var
                after = f"d{var}"
                desc = f"Add derivative d to {var}"
                op = "ADD"
                
            elif case_type == "add_nabla":
                before = var
                after = rf"\nabla {var}"
                desc = f"Add nabla to {var}"
                op = "ADD"
                
            elif case_type == "add_integral":
                before = var
                after = rf"\int {var}"
                desc = f"Add integral to {var}"
                op = "ADD"
                
            else:  # replace_d_to_partial
                before = f"d{var}"
                after = rf"\partial {var}"
                desc = f"Replace d -> ∂"
                op = "REPLACE"
            
            cases.append(EditCase(
                id=f"calculus_symbol_{i:04d}",
                category="single_symbol",
                subcategory=case_type,
                operation=op,
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
                metadata={"atomic": True, "calculus": True}
            ))
        
        return cases
    
    @classmethod
    def generate_matrix_element_cases(cls, count: int = 50) -> List[EditCase]:
        """
        ADD/REPLACE single matrix elements.
        """
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "replace_element",        # a_11 -> b_11
                "add_subscript_to_elem",  # a -> a_{11}
                "replace_element_digit",  # 1 -> 2 in matrix
            ])
            
            if case_type == "replace_element":
                old_elem = random.choice(cls.ENGLISH_LOWER[:5])
                new_elem = random.choice([e for e in cls.ENGLISH_LOWER[:5] if e != old_elem])
                row = random.choice(["1", "2", "i"])
                col = random.choice(["1", "2", "j"])
                before = f"{old_elem}_{{{row}{col}}}"
                after = f"{new_elem}_{{{row}{col}}}"
                desc = f"Replace matrix element {old_elem} -> {new_elem}"
                op = "REPLACE"
                
            elif case_type == "add_subscript_to_elem":
                elem = random.choice(cls.ENGLISH_LOWER[:5])
                row = random.choice(["1", "2", "i"])
                col = random.choice(["1", "2", "j"])
                before = elem
                after = f"{elem}_{{{row}{col}}}"
                desc = f"Add matrix subscript {row}{col} to {elem}"
                op = "ADD"
                
            else:  # replace_element_digit
                elem = random.choice(cls.ENGLISH_LOWER[:3])
                old_idx = random.choice(["1", "2"])
                new_idx = "2" if old_idx == "1" else "1"
                before = f"{elem}_{{{old_idx}1}}"
                after = f"{elem}_{{{new_idx}1}}"
                desc = f"Replace row index {old_idx} -> {new_idx}"
                op = "REPLACE"
            
            cases.append(EditCase(
                id=f"matrix_elem_{i:04d}",
                category="single_symbol",
                subcategory=case_type,
                operation=op,
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
                metadata={"atomic": True, "matrix_element": True}
            ))
        
        return cases
    
    @classmethod
    def generate_arrow_cases(cls, count: int = 50) -> List[EditCase]:
        """
        ADD/REPLACE single arrows (for diagrams).
        """
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_arrow",              # A B -> A -> B
                "replace_arrow_type",     # -> to =>
            ])
            
            if case_type == "add_arrow":
                a = random.choice(cls.ENGLISH_UPPER[:10])
                b = random.choice([l for l in cls.ENGLISH_UPPER[:10] if l != a])
                arrow = random.choice(cls.ARROWS)
                before = f"{a} \\quad {b}"
                after = f"{a} {arrow} {b}"
                desc = f"Add arrow {arrow} between {a} and {b}"
                op = "ADD"
                
            else:  # replace_arrow_type
                a = random.choice(cls.ENGLISH_UPPER[:10])
                b = random.choice([l for l in cls.ENGLISH_UPPER[:10] if l != a])
                old_arrow = random.choice(cls.ARROWS[:3])
                new_arrow = random.choice([arr for arr in cls.ARROWS[:3] if arr != old_arrow])
                before = f"{a} {old_arrow} {b}"
                after = f"{a} {new_arrow} {b}"
                desc = f"Replace arrow {old_arrow} -> {new_arrow}"
                op = "REPLACE"
            
            cases.append(EditCase(
                id=f"arrow_{i:04d}",
                category="single_symbol",
                subcategory=case_type,
                operation=op,
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
                metadata={"atomic": True, "arrow": True}
            ))
        
        return cases


# =============================================================================
# Subscript/Superscript Cases
# =============================================================================

class SubSupCases:
    """Generate subscript/superscript editing cases."""
    
    @classmethod
    def generate_add_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate ADD cases for sub/superscripts."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_superscript", "add_subscript", "add_both", "add_power"
            ])
            
            base = random.choice(VARIABLES[:10])
            
            if case_type == "add_superscript":
                exp = random.choice(["2", "3", "n", "-1", "*", r"\prime"])
                before = base
                after = f"{base}^{{{exp}}}"
                desc = f"Add superscript {exp} to {base}"
                pos = "superscript"
                
            elif case_type == "add_subscript":
                sub = random.choice(["1", "2", "i", "j", "n", "0", "max"])
                before = base
                after = f"{base}_{{{sub}}}"
                desc = f"Add subscript {sub} to {base}"
                pos = "subscript"
                
            elif case_type == "add_both":
                sub = random.choice(["i", "j", "1"])
                sup = random.choice(["2", "n", "*"])
                before = base
                after = f"{base}_{{{sub}}}^{{{sup}}}"
                desc = f"Add both subscript {sub} and superscript {sup}"
                pos = "both"
                
            else:  # add_power
                # Expression power: (x+1)^2
                expr = random.choice([
                    f"{base}+1", f"{base}-1", f"a+b", f"1-{base}"
                ])
                exp = random.choice(["2", "3", "n", "-1"])
                before = rf"({expr})"
                after = rf"({expr})^{{{exp}}}"
                desc = f"Add power {exp} to parentheses"
                pos = "superscript"
            
            cases.append(EditCase(
                id=f"subsup_add_{i:04d}",
                category="subscript_superscript",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                position_hint=pos,
                difficulty="easy",
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate REPLACE cases for sub/superscripts."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "replace_exponent", "replace_index", "sub_to_sup"
            ])
            
            base = random.choice(VARIABLES[:10])
            
            if case_type == "replace_exponent":
                old_exp = random.choice(["2", "3", "n"])
                new_exp = random.choice(["4", "m", "-1"])
                while new_exp == old_exp:
                    new_exp = random.choice(["4", "m", "-1", "k"])
                before = f"{base}^{{{old_exp}}}"
                after = f"{base}^{{{new_exp}}}"
                desc = f"Replace exponent {old_exp} -> {new_exp}"
                
            elif case_type == "replace_index":
                old_sub = random.choice(["1", "i", "n"])
                new_sub = random.choice(["2", "j", "m"])
                while new_sub == old_sub:
                    new_sub = random.choice(["2", "j", "m", "k"])
                before = f"{base}_{{{old_sub}}}"
                after = f"{base}_{{{new_sub}}}"
                desc = f"Replace subscript {old_sub} -> {new_sub}"
                
            else:  # sub_to_sup
                # Change subscript to superscript position
                idx = random.choice(["2", "n"])
                before = f"{base}_{{{idx}}}"
                after = f"{base}^{{{idx}}}"
                desc = f"Change subscript to superscript"
            
            cases.append(EditCase(
                id=f"subsup_replace_{i:04d}",
                category="subscript_superscript",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
            ))
        
        return cases


# =============================================================================
# Greek Letter Cases
# =============================================================================

class GreekCases:
    """Generate Greek letter editing cases."""
    
    @classmethod
    def generate_add_cases(cls, count: int = 30) -> List[EditCase]:
        """Generate ADD cases with Greek letters."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_greek_term", "add_greek_subscript", "add_greek_expression"
            ])
            
            greek = random.choice(GREEK_LOWER)
            
            if case_type == "add_greek_term":
                var = random.choice(VARIABLES[:5])
                before = f"{var}^2"
                after = f"{var}^2 + {greek[1]}"
                desc = f"Add Greek term +{greek[0]}"
                
            elif case_type == "add_greek_subscript":
                var = random.choice(VARIABLES[:5])
                before = var
                after = f"{var}_{{{greek[1]}}}"
                desc = f"Add Greek subscript {greek[0]}"
                
            else:  # add_greek_expression
                greek2 = random.choice(GREEK_LOWER)
                before = f"{greek[1]}"
                after = f"{greek[1]} + {greek2[1]}"
                desc = f"Add +{greek2[0]} to expression"
            
            cases.append(EditCase(
                id=f"greek_add_{i:04d}",
                category="greek",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 30) -> List[EditCase]:
        """Generate REPLACE cases with Greek letters."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "replace_greek", "latin_to_greek", "lower_to_upper"
            ])
            
            if case_type == "replace_greek":
                old_greek, new_greek = random.sample(GREEK_LOWER, 2)
                before = old_greek[1]
                after = new_greek[1]
                desc = f"Replace {old_greek[0]} -> {new_greek[0]}"
                
            elif case_type == "latin_to_greek":
                latin = random.choice(VARIABLES[:5])
                greek = random.choice(GREEK_LOWER)
                before = latin
                after = greek[1]
                desc = f"Replace {latin} -> {greek[0]}"
                
            else:  # lower_to_upper
                # Find matching upper/lower pairs
                lower = random.choice([g for g in GREEK_LOWER if any(
                    u[0].lower() == g[0].lower() for u in GREEK_UPPER
                )])
                upper = [u for u in GREEK_UPPER if u[0].lower() == lower[0].lower()][0]
                before = lower[1]
                after = upper[1]
                desc = f"Capitalize {lower[0]} -> {upper[0]}"
            
            cases.append(EditCase(
                id=f"greek_replace_{i:04d}",
                category="greek",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
            ))
        
        return cases


# =============================================================================
# Fraction Cases
# =============================================================================

class FractionCases:
    """Generate fraction editing cases."""
    
    @classmethod
    def generate_add_cases(cls, count: int = 30) -> List[EditCase]:
        """Generate ADD cases for fractions."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_numerator_term", "add_denominator_term", "wrap_in_fraction"
            ])
            
            if case_type == "add_numerator_term":
                var = random.choice(VARIABLES[:5])
                before = rf"\frac{{{var}}}{{2}}"
                after = rf"\frac{{{var} + 1}}{{2}}"
                desc = "Add +1 to numerator"
                
            elif case_type == "add_denominator_term":
                var = random.choice(VARIABLES[:5])
                before = rf"\frac{{1}}{{{var}}}"
                after = rf"\frac{{1}}{{{var} + 1}}"
                desc = "Add +1 to denominator"
                
            else:  # wrap_in_fraction
                var = random.choice(VARIABLES[:5])
                before = f"{var} + 1"
                after = rf"\frac{{{var} + 1}}{{2}}"
                desc = "Wrap expression in fraction"
            
            cases.append(EditCase(
                id=f"fraction_add_{i:04d}",
                category="fraction",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="medium",
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 30) -> List[EditCase]:
        """Generate REPLACE cases for fractions."""
        cases = []
        
        for i in range(count):
            old_num = random.choice(["1", "a", "x"])
            new_num = random.choice(["2", "b", "y"])
            while new_num == old_num:
                new_num = random.choice(["2", "b", "y", "n"])
            
            denom = random.choice(["2", "n", "x+1"])
            before = rf"\frac{{{old_num}}}{{{denom}}}"
            after = rf"\frac{{{new_num}}}{{{denom}}}"
            desc = f"Replace numerator {old_num} -> {new_num}"
            
            cases.append(EditCase(
                id=f"fraction_replace_{i:04d}",
                category="fraction",
                subcategory="replace_numerator",
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
            ))
        
        return cases


# =============================================================================
# Incomplete Structure Cases (FILL operation)
# =============================================================================

class IncompleteCases:
    """
    Generate FILL operation cases for completing incomplete LaTeX structures.
    
    Uses empty braces {} to mark empty slots (NO special token needed):
    - \\frac{2}{} -> \\frac{2}{3}  (empty denominator)
    - x^{} -> x^{2}              (empty superscript)
    - x_{} -> x_{i}              (empty subscript)
    - \\sqrt{} -> \\sqrt{x}        (empty sqrt)
    
    The {} braces preserve position info:
    - { position = left edge of empty region
    - } position = right edge of empty region
    - Model learns: adjacent {} = empty slot to fill
    """
    
    @classmethod
    def generate_fill_cases(cls, count: int = 100) -> List[EditCase]:
        """Generate FILL cases for incomplete structures using empty braces."""
        cases = []
        case_id = 0
        
        def add_case(before, after, symbol, subcat, diff="easy"):
            nonlocal case_id
            cases.append(EditCase(
                id=f"incomplete_fill_{case_id:04d}",
                category="incomplete",
                subcategory=subcat,
                operation="FILL",
                before_latex=before,
                after_latex=after,
                edit_description=f"Fill empty slot with {symbol}",
                difficulty=diff,
                metadata={"edit_symbol": symbol},
            ))
            case_id += 1
        
        # Extract just LaTeX commands from Greek tuples
        greek_symbols = [g[1] for g in GREEK_LOWER[:5]]  # ['\alpha', '\beta', ...]
        
        # Empty fraction denominators: \frac{X}{} → \frac{X}{Y}
        for _ in range(count // 5):
            num = random.choice(VARIABLES + list("123") + greek_symbols)
            denom = random.choice(VARIABLES + list("234") + greek_symbols)
            before = f"\\frac{{{num}}}{{}}"
            after = f"\\frac{{{num}}}{{{denom}}}"
            add_case(before, after, denom, "fraction_denom")
        
        # Empty fraction numerators: \frac{}{Y} → \frac{X}{Y}
        for _ in range(count // 5):
            num = random.choice(VARIABLES + list("123") + greek_symbols)
            denom = random.choice(VARIABLES + list("234") + greek_symbols)
            before = f"\\frac{{}}{{{denom}}}"
            after = f"\\frac{{{num}}}{{{denom}}}"
            add_case(before, after, num, "fraction_num")
        
        # Empty superscripts: x^{} → x^{2}
        for _ in range(count // 5):
            base = random.choice(VARIABLES[:5])
            exp = random.choice(list("234") + greek_symbols[:3] + ["n", "m", "k"])
            before = f"{base}^{{}}"
            after = f"{base}^{{{exp}}}"
            add_case(before, after, exp, "superscript")
        
        # Empty subscripts: x_{} → x_{i}
        for _ in range(count // 5):
            base = random.choice(VARIABLES[:5] + ["a", "b", "c"])
            sub = random.choice(list("ijknm012") + greek_symbols[:3])
            before = f"{base}_{{}}"
            after = f"{base}_{{{sub}}}"
            add_case(before, after, sub, "subscript")
        
        # Empty sqrt: \sqrt{} → \sqrt{x}
        for _ in range(count // 10):
            content = random.choice(VARIABLES[:5] + list("234") + ["x+1", "a^2", "n"])
            before = "\\sqrt{}"
            after = f"\\sqrt{{{content}}}"
            add_case(before, after, content, "sqrt")
        
        # Complex: nested incomplete structures
        # \frac{x^{}}{y} → \frac{x^{2}}{y}
        for _ in range(count // 10):
            base = random.choice(VARIABLES[:3])
            exp = random.choice(list("234n"))
            denom = random.choice(VARIABLES[:3] + list("234"))
            before = f"\\frac{{{base}^{{}}}}{{{denom}}}"
            after = f"\\frac{{{base}^{{{exp}}}}}{{{denom}}}"
            add_case(before, after, exp, "nested", "medium")
        
        # Integral bounds: \int_{}^{b} → \int_{a}^{b}
        for _ in range(count // 10):
            lower = random.choice(list("0ab") + ["-1", r"-\infty"])
            upper = random.choice(list("1nb") + [r"\infty", "t"])
            before = f"\\int_{{}}^{{{upper}}} f(x) dx"
            after = f"\\int_{{{lower}}}^{{{upper}}} f(x) dx"
            add_case(before, after, lower, "integral_bound", "medium")
        
        # Sum bounds: \sum_{}^{n} → \sum_{i=0}^{n}
        for _ in range(count // 10):
            lower = random.choice(["i=0", "i=1", "k=0", "n=0", "j=1"])
            upper = random.choice(["n", "N", r"\infty", "m", "k"])
            before = f"\\sum_{{}}^{{{upper}}} a_i"
            after = f"\\sum_{{{lower}}}^{{{upper}}} a_i"
            add_case(before, after, lower, "sum_bound", "medium")
        
        random.shuffle(cases)
        return cases[:count]


# =============================================================================
# DELETE Operation Cases (Erase/Remove)
# =============================================================================

class DeleteCases:
    """
    Generate DELETE operation cases - erasing/removing parts of expressions.
    
    Real-world scenarios:
    - User crosses out a term they don't want
    - User erases a symbol to correct a mistake
    - User removes an entire sub-expression
    """
    
    @classmethod
    def generate_delete_cases(cls, count: int = 100) -> List[EditCase]:
        """Generate DELETE cases for various structures."""
        cases = []
        case_id = 0
        
        def add_case(before, after, deleted, subcat, diff="easy"):
            nonlocal case_id
            cases.append(EditCase(
                id=f"delete_{case_id:04d}",
                category="delete",
                subcategory=subcat,
                operation="DELETE",
                before_latex=before,
                after_latex=after,
                edit_description=f"Delete '{deleted}'",
                difficulty=diff,
                metadata={"deleted_content": deleted},
            ))
            case_id += 1
        
        # Delete term from sum: x + y → x
        for _ in range(count // 6):
            a = random.choice(VARIABLES[:5])
            b = random.choice(VARIABLES[:5])
            if a != b:
                add_case(f"{a} + {b}", a, f"+ {b}", "delete_term")
                add_case(f"{a} + {b}", b, f"{a} +", "delete_term")
        
        # Delete subscript: x_{i} → x
        for _ in range(count // 6):
            base = random.choice(VARIABLES[:5])
            sub = random.choice(list("ijk012"))
            add_case(f"{base}_{{{sub}}}", base, f"_{{{sub}}}", "delete_subscript")
        
        # Delete superscript: x^{2} → x
        for _ in range(count // 6):
            base = random.choice(VARIABLES[:5])
            exp = random.choice(list("234n"))
            add_case(f"{base}^{{{exp}}}", base, f"^{{{exp}}}", "delete_superscript")
        
        # Delete from product: xy → x or y
        for _ in range(count // 6):
            a = random.choice(VARIABLES[:5])
            b = random.choice(VARIABLES[:5])
            if a != b:
                add_case(f"{a}{b}", a, b, "delete_factor")
                add_case(f"{a}{b}", b, a, "delete_factor")
        
        # Delete operator: x + y - z → x + y (remove last term)
        for _ in range(count // 6):
            a, b, c = random.sample(VARIABLES[:6], 3)
            op = random.choice(["+", "-"])
            add_case(f"{a} + {b} {op} {c}", f"{a} + {b}", f"{op} {c}", "delete_last_term")
        
        # Delete entire numerator/denominator content (leave empty for FILL later)
        for _ in range(count // 6):
            num = random.choice(VARIABLES[:3])
            denom = random.choice(list("234"))
            # This represents crossing out the numerator
            add_case(f"\\frac{{{num}}}{{{denom}}}", f"\\frac{{}}{{{denom}}}", num, "delete_numerator", "medium")
        
        random.shuffle(cases)
        return cases[:count]


# =============================================================================
# WRAP Operation Cases (Wrap in Structure)
# =============================================================================

class WrapCases:
    """
    Generate WRAP operation cases - wrapping existing content in a structure.
    
    Real-world scenarios:
    - User draws a horizontal line under x+1 to create a fraction
    - User draws a line over content to add a bar/hat
    - User draws sqrt symbol around content
    - User adds parentheses around expression
    """
    
    @classmethod
    def generate_wrap_cases(cls, count: int = 100) -> List[EditCase]:
        """Generate WRAP cases for various structures."""
        cases = []
        case_id = 0
        
        def add_case(before, after, wrapper, subcat, diff="medium"):
            nonlocal case_id
            cases.append(EditCase(
                id=f"wrap_{case_id:04d}",
                category="wrap",
                subcategory=subcat,
                operation="WRAP",
                before_latex=before,
                after_latex=after,
                edit_description=f"Wrap in {wrapper}",
                difficulty=diff,
                metadata={"wrapper_type": wrapper},
            ))
            case_id += 1
        
        # Wrap in fraction (draw horizontal line): x+1 → \frac{x+1}{y}
        for _ in range(count // 5):
            content = random.choice([
                "x+1", "a+b", "x", "2x", "n+1", "x-1",
                r"\alpha", r"x^{2}", "ab"
            ])
            denom = random.choice(VARIABLES[:5] + list("234") + ["n", "m"])
            add_case(content, f"\\frac{{{content}}}{{{denom}}}", "fraction", "wrap_fraction")
        
        # Wrap in sqrt (draw sqrt symbol): x → \sqrt{x}
        for _ in range(count // 5):
            content = random.choice([
                "x", "2", "a+b", "x^{2}", r"\alpha", "n", "ab"
            ])
            add_case(content, f"\\sqrt{{{content}}}", "sqrt", "wrap_sqrt")
        
        # Wrap in parentheses: x+y → (x+y)
        for _ in range(count // 5):
            content = random.choice([
                "x+y", "a-b", "x+1", "2x+3", r"\alpha+\beta"
            ])
            add_case(content, f"({content})", "parentheses", "wrap_parens", "easy")
        
        # Wrap in absolute value: x → |x|
        for _ in range(count // 10):
            content = random.choice(VARIABLES[:5] + ["x-a", "x+1"])
            add_case(content, f"|{content}|", "absolute", "wrap_abs", "easy")
        
        # Wrap in overline/bar: x → \bar{x}
        for _ in range(count // 10):
            content = random.choice(VARIABLES[:5] + [r"\alpha", r"\beta"])
            add_case(content, f"\\bar{{{content}}}", "bar", "wrap_bar")
        
        # Wrap in hat: x → \hat{x}
        for _ in range(count // 10):
            content = random.choice(VARIABLES[:5] + [r"\theta", r"\phi"])
            add_case(content, f"\\hat{{{content}}}", "hat", "wrap_hat")
        
        # Wrap in vector arrow: v → \vec{v}
        for _ in range(count // 10):
            content = random.choice(["v", "u", "a", "b", "F", "r"])
            add_case(content, f"\\vec{{{content}}}", "vector", "wrap_vec")
        
        # Wrap expression in exponent: x → e^{x}
        for _ in range(count // 10):
            content = random.choice(["x", "-x", "2x", "ix", r"\alpha"])
            add_case(content, f"e^{{{content}}}", "exp", "wrap_exp")
        
        random.shuffle(cases)
        return cases[:count]


# =============================================================================
# UNWRAP Operation Cases (Remove Structure)  
# =============================================================================

class UnwrapCases:
    """
    Generate UNWRAP operation cases - removing wrapper structures.
    
    Real-world scenarios:
    - User crosses out fraction bar to "flatten" a fraction
    - User removes sqrt symbol
    - User removes parentheses
    - Simplification: \frac{x}{1} → x
    """
    
    @classmethod
    def generate_unwrap_cases(cls, count: int = 100) -> List[EditCase]:
        """Generate UNWRAP cases for various structures."""
        cases = []
        case_id = 0
        
        def add_case(before, after, unwrapped, subcat, diff="medium"):
            nonlocal case_id
            cases.append(EditCase(
                id=f"unwrap_{case_id:04d}",
                category="unwrap",
                subcategory=subcat,
                operation="UNWRAP",
                before_latex=before,
                after_latex=after,
                edit_description=f"Unwrap from {unwrapped}",
                difficulty=diff,
                metadata={"removed_structure": unwrapped},
            ))
            case_id += 1
        
        # Unwrap fraction with denominator 1: \frac{x}{1} → x
        for _ in range(count // 5):
            content = random.choice(VARIABLES[:5] + ["x+1", "ab", r"\alpha"])
            add_case(f"\\frac{{{content}}}{{1}}", content, "fraction/1", "unwrap_frac_one")
        
        # Unwrap sqrt of perfect square concept: \sqrt{x^2} → x (simplification)
        for _ in range(count // 5):
            base = random.choice(VARIABLES[:5])
            add_case(f"\\sqrt{{{base}^{{2}}}}", base, "sqrt", "unwrap_sqrt")
        
        # Unwrap unnecessary parentheses: (x) → x
        for _ in range(count // 5):
            content = random.choice(VARIABLES[:5] + [r"\alpha", "2", "n"])
            add_case(f"({content})", content, "parentheses", "unwrap_parens", "easy")
        
        # Unwrap single-term fraction: \frac{x}{y} → x/y (alternate notation)
        for _ in range(count // 5):
            num = random.choice(VARIABLES[:3])
            denom = random.choice(VARIABLES[:3] + list("234"))
            if num != denom:
                add_case(f"\\frac{{{num}}}{{{denom}}}", f"{num}/{denom}", "frac_to_slash", "unwrap_inline_frac")
        
        # Unwrap bar: \bar{x} → x
        for _ in range(count // 10):
            content = random.choice(VARIABLES[:5])
            add_case(f"\\bar{{{content}}}", content, "bar", "unwrap_bar")
        
        # Unwrap double negation: --x → x
        for _ in range(count // 10):
            content = random.choice(VARIABLES[:5])
            add_case(f"--{content}", content, "double_neg", "unwrap_neg", "easy")
        
        random.shuffle(cases)
        return cases[:count]


# =============================================================================
# Algebraic Expression Cases
# =============================================================================

class AlgebraCases:
    """Generate general algebraic expression editing cases."""
    
    @classmethod
    def polynomial(cls, degree: int = 2) -> str:
        var = random.choice(VARIABLES[:3])
        terms = []
        for d in range(degree, -1, -1):
            coef = random.choice(["", "2", "3", "a", "b"])
            if d == 0:
                terms.append(random.choice(["1", "c", "k"]))
            elif d == 1:
                terms.append(f"{coef}{var}")
            else:
                terms.append(f"{coef}{var}^{d}")
        return " + ".join(terms)
    
    @classmethod
    def generate_add_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate ADD cases for algebraic expressions."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "add_term", "add_operator", "add_parentheses", "add_equality"
            ])
            
            var = random.choice(VARIABLES[:5])
            
            if case_type == "add_term":
                term = random.choice([
                    f"+ {var}", f"- {var}", f"+ {var}^2", "+ 1", "- 1"
                ])
                before = f"{var}^2"
                after = f"{var}^2 {term}"
                desc = f"Add term {term}"
                
            elif case_type == "add_operator":
                var2 = random.choice([v for v in VARIABLES[:5] if v != var])
                op = random.choice(["+", "-", r"\cdot"])
                before = f"{var}"
                after = f"{var} {op} {var2}"
                desc = f"Add {op} {var2}"
                
            elif case_type == "add_parentheses":
                before = f"{var} + 1 \\cdot 2"
                after = f"({var} + 1) \\cdot 2"
                desc = "Add parentheses for grouping"
                
            else:  # add_equality
                rhs = random.choice(["0", "1", f"{var}^2", "a"])
                before = f"{var}^2 + {var}"
                after = f"{var}^2 + {var} = {rhs}"
                desc = f"Add = {rhs}"
            
            cases.append(EditCase(
                id=f"algebra_add_{i:04d}",
                category="algebra",
                subcategory=case_type,
                operation="ADD",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty=random.choice(["easy", "medium"]),
            ))
        
        return cases
    
    @classmethod
    def generate_replace_cases(cls, count: int = 50) -> List[EditCase]:
        """Generate REPLACE cases for algebraic expressions."""
        cases = []
        
        for i in range(count):
            case_type = random.choice([
                "replace_variable", "replace_coefficient", "replace_operator"
            ])
            
            if case_type == "replace_variable":
                old_var, new_var = random.sample(VARIABLES[:10], 2)
                before = f"{old_var}^2 + {old_var}"
                after = f"{new_var}^2 + {new_var}"
                desc = f"Replace variable {old_var} -> {new_var}"
                
            elif case_type == "replace_coefficient":
                var = random.choice(VARIABLES[:5])
                old_coef = random.choice(["2", "3", "a"])
                new_coef = random.choice(["4", "5", "b"])
                before = f"{old_coef}{var}^2"
                after = f"{new_coef}{var}^2"
                desc = f"Replace coefficient {old_coef} -> {new_coef}"
                
            else:  # replace_operator
                var = random.choice(VARIABLES[:5])
                old_op, new_op = random.sample(["+", "-"], 2)
                before = f"x^2 {old_op} {var}"
                after = f"x^2 {new_op} {var}"
                desc = f"Replace operator {old_op} -> {new_op}"
            
            cases.append(EditCase(
                id=f"algebra_replace_{i:04d}",
                category="algebra",
                subcategory=case_type,
                operation="REPLACE",
                before_latex=before,
                after_latex=after,
                edit_description=desc,
                difficulty="easy",
            ))
        
        return cases


# =============================================================================
# Main Generator
# =============================================================================

class CaseGenerator:
    """Main case generator that combines all categories."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
    
    def generate_all(self, count_per_category: int = 50, 
                     initial_state_pct: float = 0.2) -> List[EditCase]:
        """Generate cases from all categories."""
        all_cases = []
        
        # Initial state cases (blank canvas -> first symbol) - 20% of total
        # Total cases will be ~1000 for count=50, so initial should be ~200
        initial_count = int(count_per_category * 4)  # ~200 initial state cases
        all_cases.extend(SingleSymbolCases.generate_initial_state_cases(initial_count))
        
        # Single Symbol (ATOMIC) - highest priority
        all_cases.extend(SingleSymbolCases.generate_add_single_symbol(count_per_category * 2))
        all_cases.extend(SingleSymbolCases.generate_replace_single_symbol(count_per_category * 2))
        all_cases.extend(SingleSymbolCases.generate_calculus_symbol_cases(count_per_category))
        all_cases.extend(SingleSymbolCases.generate_matrix_element_cases(count_per_category))
        all_cases.extend(SingleSymbolCases.generate_arrow_cases(count_per_category))
        
        # Diagrams
        all_cases.extend(DiagramCases.generate_add_cases(count_per_category))
        all_cases.extend(DiagramCases.generate_replace_cases(count_per_category))
        all_cases.extend(DiagramCases.generate_insert_cases(count_per_category // 2))
        
        # Calculus
        all_cases.extend(CalculusCases.generate_add_cases(count_per_category))
        all_cases.extend(CalculusCases.generate_replace_cases(count_per_category))
        
        # Differential Equations
        all_cases.extend(DiffEqCases.generate_add_cases(count_per_category // 2))
        all_cases.extend(DiffEqCases.generate_replace_cases(count_per_category // 2))
        
        # Matrices
        all_cases.extend(MatrixCases.generate_add_cases(count_per_category))
        all_cases.extend(MatrixCases.generate_replace_cases(count_per_category))
        
        # Subscripts/Superscripts
        all_cases.extend(SubSupCases.generate_add_cases(count_per_category))
        all_cases.extend(SubSupCases.generate_replace_cases(count_per_category))
        
        # Greek Letters
        all_cases.extend(GreekCases.generate_add_cases(count_per_category // 2))
        all_cases.extend(GreekCases.generate_replace_cases(count_per_category // 2))
        
        # Fractions
        all_cases.extend(FractionCases.generate_add_cases(count_per_category // 2))
        all_cases.extend(FractionCases.generate_replace_cases(count_per_category // 2))
        
        # Algebra
        all_cases.extend(AlgebraCases.generate_add_cases(count_per_category))
        all_cases.extend(AlgebraCases.generate_replace_cases(count_per_category))
        
        # Incomplete structures (FILL operation) - important for real-time editing!
        all_cases.extend(IncompleteCases.generate_fill_cases(count_per_category))
        
        # DELETE operation - erasing/removing content
        all_cases.extend(DeleteCases.generate_delete_cases(count_per_category))
        
        # WRAP operation - wrapping content in structures (draw fraction bar, sqrt, etc.)
        all_cases.extend(WrapCases.generate_wrap_cases(count_per_category))
        
        # UNWRAP operation - removing structures
        all_cases.extend(UnwrapCases.generate_unwrap_cases(count_per_category // 2))
        
        random.shuffle(all_cases)
        return all_cases
    
    def generate_by_operation(self, operation: str, count: int = 100) -> List[EditCase]:
        """Generate cases for a specific operation."""
        cases = []
        
        generators = {
            "ADD": [
                SingleSymbolCases.generate_add_single_symbol,  # Atomic single symbol
                SingleSymbolCases.generate_calculus_symbol_cases,
                SingleSymbolCases.generate_matrix_element_cases,
                SingleSymbolCases.generate_arrow_cases,
                DiagramCases.generate_add_cases,
                CalculusCases.generate_add_cases,
                DiffEqCases.generate_add_cases,
                MatrixCases.generate_add_cases,
                SubSupCases.generate_add_cases,
                GreekCases.generate_add_cases,
                FractionCases.generate_add_cases,
                AlgebraCases.generate_add_cases,
            ],
            "REPLACE": [
                SingleSymbolCases.generate_replace_single_symbol,  # Atomic single symbol
                DiagramCases.generate_replace_cases,
                CalculusCases.generate_replace_cases,
                DiffEqCases.generate_replace_cases,
                MatrixCases.generate_replace_cases,
                SubSupCases.generate_replace_cases,
                GreekCases.generate_replace_cases,
                FractionCases.generate_replace_cases,
                AlgebraCases.generate_replace_cases,
            ],
            "INSERT": [
                DiagramCases.generate_insert_cases,
            ],
            "FILL": [
                IncompleteCases.generate_fill_cases,
            ],
            "DELETE": [
                DeleteCases.generate_delete_cases,
            ],
            "WRAP": [
                WrapCases.generate_wrap_cases,
            ],
            "UNWRAP": [
                UnwrapCases.generate_unwrap_cases,
            ],
        }
        
        per_gen = max(1, count // len(generators.get(operation, [])))
        for gen in generators.get(operation, []):
            cases.extend(gen(per_gen))
        
        random.shuffle(cases)
        return cases[:count]
    
    def generate_by_category(self, category: str, count: int = 100) -> List[EditCase]:
        """Generate cases for a specific category."""
        
        # Special handling for single_symbol category
        if category == "single_symbol":
            cases = []
            per_type = max(1, count // 5)
            cases.extend(SingleSymbolCases.generate_add_single_symbol(per_type))
            cases.extend(SingleSymbolCases.generate_replace_single_symbol(per_type))
            cases.extend(SingleSymbolCases.generate_calculus_symbol_cases(per_type))
            cases.extend(SingleSymbolCases.generate_matrix_element_cases(per_type))
            cases.extend(SingleSymbolCases.generate_arrow_cases(per_type))
            random.shuffle(cases)
            return cases[:count]
        
        generators = {
            "diagram": (DiagramCases, ["add", "replace", "insert"]),
            "calculus": (CalculusCases, ["add", "replace"]),
            "diffeq": (DiffEqCases, ["add", "replace"]),
            "matrix": (MatrixCases, ["add", "replace"]),
            "subscript_superscript": (SubSupCases, ["add", "replace"]),
            "greek": (GreekCases, ["add", "replace"]),
            "fraction": (FractionCases, ["add", "replace"]),
            "algebra": (AlgebraCases, ["add", "replace"]),
        }
        
        if category not in generators:
            return []
        
        cls, ops = generators[category]
        cases = []
        per_op = max(1, count // len(ops))
        
        for op in ops:
            method = getattr(cls, f"generate_{op}_cases", None)
            if method:
                cases.extend(method(per_op))
        
        random.shuffle(cases)
        return cases[:count]
    
    def generate_by_difficulty(self, target_difficulty: str, count: int = 100,
                               recompute: bool = True) -> List[EditCase]:
        """
        Generate cases filtered by computed difficulty.
        
        Args:
            target_difficulty: "easy", "medium", or "hard"
            count: Target number of cases
            recompute: If True, recompute difficulty using tree-depth analysis
            
        Returns:
            List of EditCase matching target difficulty
        """
        # Generate more cases than needed (we'll filter)
        all_cases = self.generate_all(count_per_category=count // 5)
        
        if recompute:
            self.recompute_all_difficulties(all_cases)
        
        # Filter by target difficulty
        filtered = [c for c in all_cases if c.difficulty == target_difficulty]
        
        if len(filtered) < count:
            # Need more - generate additional
            extra = self.generate_all(count_per_category=count)
            if recompute:
                self.recompute_all_difficulties(extra)
            filtered.extend([c for c in extra if c.difficulty == target_difficulty])
        
        random.shuffle(filtered)
        return filtered[:count]
    
    def generate_curriculum(self, total_count: int = 300,
                           easy_pct: float = 0.4,
                           medium_pct: float = 0.4,
                           hard_pct: float = 0.2) -> List[EditCase]:
        """
        Generate a curriculum-ordered dataset with specified difficulty distribution.
        
        Args:
            total_count: Total number of cases
            easy_pct: Percentage of easy cases (default 40%)
            medium_pct: Percentage of medium cases (default 40%)
            hard_pct: Percentage of hard cases (default 20%)
            
        Returns:
            List of EditCase ordered by difficulty (easy -> medium -> hard)
        """
        easy_count = int(total_count * easy_pct)
        medium_count = int(total_count * medium_pct)
        hard_count = total_count - easy_count - medium_count
        
        easy_cases = self.generate_by_difficulty("easy", easy_count)
        medium_cases = self.generate_by_difficulty("medium", medium_count)
        hard_cases = self.generate_by_difficulty("hard", hard_count)
        
        # Curriculum order: easy first, then medium, then hard
        return easy_cases + medium_cases + hard_cases
    
    @staticmethod
    def recompute_all_difficulties(cases: List[EditCase]) -> Dict[str, int]:
        """
        Recompute difficulties for all cases using tree-depth analysis.
        
        Args:
            cases: List of EditCase to update
            
        Returns:
            Dict with counts per difficulty level
        """
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for case in cases:
            case.compute_difficulty()
            counts[case.difficulty] = counts.get(case.difficulty, 0) + 1
        return counts
    
    @staticmethod
    def get_complexity_stats(cases: List[EditCase]) -> Dict[str, Any]:
        """
        Get complexity statistics for a set of cases.
        
        Returns:
            Dict with min/max/avg depth, score distributions, etc.
        """
        if not cases:
            return {}
        
        depths = []
        scores = []
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        
        for case in cases:
            before_c, after_c = case.get_complexity()
            max_c = after_c if after_c.difficulty_score >= before_c.difficulty_score else before_c
            depths.append(max_c.depth)
            scores.append(max_c.difficulty_score)
            difficulties[max_c.difficulty] = difficulties.get(max_c.difficulty, 0) + 1
        
        return {
            "depth": {
                "min": min(depths),
                "max": max(depths),
                "avg": sum(depths) / len(depths),
            },
            "score": {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores),
            },
            "difficulty_distribution": difficulties,
            "total": len(cases),
        }
    
    def convert_to_training_examples(self, cases: List[EditCase], 
                                      add_noise: bool = True,
                                      noise_probability: float = 0.4) -> List[TrainingExample]:
        """Convert EditCases to TrainingExamples with noise augmentation."""
        return [
            case.to_training_example(add_noise=add_noise, 
                                     noise_probability=noise_probability)
            for case in cases
        ]
    
    def save_to_json(self, cases: List[EditCase], filepath: str, 
                     as_training_examples: bool = False,
                     add_noise: bool = True,
                     noise_probability: float = 0.4):
        """Save cases to JSON file."""
        
        if as_training_examples:
            examples = self.convert_to_training_examples(
                cases, add_noise=add_noise, noise_probability=noise_probability
            )
            
            # Count statistics
            initial_count = sum(1 for e in examples if e.is_initial_state)
            noisy_count = sum(1 for e in examples if e.crop_spec.noise_specs)
            
            data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "format": "training_examples",
                    "total_examples": len(examples),
                    "initial_state_count": initial_count,
                    "initial_state_pct": f"{100*initial_count/len(examples):.1f}%",
                    "noisy_examples_count": noisy_count,
                    "noisy_examples_pct": f"{100*noisy_count/len(examples):.1f}%",
                    "categories": list(set(e.category for e in examples)),
                    "operations": list(set(e.operation for e in examples)),
                },
                "schema": {
                    "input": ["latex_context", "crop_spec"],
                    "output": ["updated_latex"],
                    "crop_spec_fields": ["x", "y", "width", "height", "noise_specs", "noise_types"],
                },
                "examples": [e.to_dict() for e in examples]
            }
        else:
            data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "format": "edit_cases",
                    "total_cases": len(cases),
                    "categories": list(set(c.category for c in cases)),
                    "operations": list(set(c.operation for c in cases)),
                },
                "cases": [c.to_dict() for c in cases]
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(cases)} {'training examples' if as_training_examples else 'cases'} to {filepath}")
    
    def print_statistics(self, cases: List[EditCase], show_complexity: bool = True):
        """Print statistics about generated cases."""
        print("\n" + "=" * 60)
        print("CASE GENERATION STATISTICS")
        print("=" * 60)
        
        # By category
        print("\nBy Category:")
        categories = {}
        for c in cases:
            categories[c.category] = categories.get(c.category, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat:25} {count:5} cases")
        
        # By operation
        print("\nBy Operation:")
        operations = {}
        for c in cases:
            operations[c.operation] = operations.get(c.operation, 0) + 1
        for op, count in sorted(operations.items()):
            print(f"  {op:25} {count:5} cases")
        
        # By difficulty
        print("\nBy Difficulty:")
        difficulties = {}
        for c in cases:
            difficulties[c.difficulty] = difficulties.get(c.difficulty, 0) + 1
        for diff, count in sorted(difficulties.items()):
            print(f"  {diff:25} {count:5} cases")
        
        # Complexity statistics (tree-depth based)
        if show_complexity:
            print("\nComplexity Analysis (Tree Depth):")
            stats = self.get_complexity_stats(cases)
            if stats:
                print(f"  Depth:    min={stats['depth']['min']}, max={stats['depth']['max']}, avg={stats['depth']['avg']:.1f}")
                print(f"  Score:    min={stats['score']['min']:.3f}, max={stats['score']['max']:.3f}, avg={stats['score']['avg']:.3f}")
        
        print(f"\nTotal: {len(cases)} cases")
        print("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    import sys
    
    # Fix encoding for Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Generate LaTeX editing training cases")
    parser.add_argument("--output", "-o", default="./generated_cases.json",
                       help="Output JSON file path")
    parser.add_argument("--count", "-n", type=int, default=50,
                       help="Number of cases per category (default: 50)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--category", "-c", default=None,
                       choices=["single_symbol", "diagram", "calculus", "diffeq", "matrix", 
                               "subscript_superscript", "greek", "fraction", "algebra"],
                       help="Generate only specific category (single_symbol for atomic edits)")
    parser.add_argument("--operation", "-op", default=None,
                       choices=["ADD", "REPLACE", "INSERT", "FILL", "DELETE", "WRAP", "UNWRAP"],
                       help="Generate only specific operation")
    parser.add_argument("--difficulty", "-d", default=None,
                       choices=["easy", "medium", "hard"],
                       help="Generate only cases of specific difficulty (uses tree-depth analysis)")
    parser.add_argument("--curriculum", action="store_true",
                       help="Generate curriculum-ordered dataset (easy -> medium -> hard)")
    parser.add_argument("--recompute-difficulty", action="store_true",
                       help="Recompute all difficulties using tree-depth analysis")
    parser.add_argument("--preview", "-p", type=int, default=5,
                       help="Number of sample cases to preview (default: 5)")
    parser.add_argument("--training-format", "-t", action="store_true",
                       help="Output in training example format (with noise specs)")
    parser.add_argument("--noise-prob", type=float, default=0.4,
                       help="Probability of adding noise to crop region (default: 0.4)")
    parser.add_argument("--no-noise", action="store_true",
                       help="Disable noise augmentation")
    
    args = parser.parse_args()
    
    generator = CaseGenerator(seed=args.seed)
    
    # Generate cases based on mode
    if args.curriculum:
        # Curriculum learning: ordered by difficulty
        print("Generating curriculum-ordered dataset...")
        cases = generator.generate_curriculum(total_count=args.count * 10)
    elif args.difficulty:
        # Filter by specific difficulty
        print(f"Generating cases with difficulty: {args.difficulty}")
        cases = generator.generate_by_difficulty(args.difficulty, args.count * 5)
    elif args.category:
        cases = generator.generate_by_category(args.category, args.count * 3)
    elif args.operation:
        cases = generator.generate_by_operation(args.operation, args.count * 8)
    else:
        cases = generator.generate_all(args.count)
    
    # Optionally recompute difficulties using tree-depth analysis
    if args.recompute_difficulty:
        print("Recomputing difficulties using tree-depth analysis...")
        counts = CaseGenerator.recompute_all_difficulties(cases)
        print(f"  Updated: easy={counts['easy']}, medium={counts['medium']}, hard={counts['hard']}")
    
    # Print statistics
    generator.print_statistics(cases)
    
    # Preview samples
    if args.preview > 0:
        print(f"\n{'=' * 60}")
        print(f"SAMPLE CASES (showing {min(args.preview, len(cases))})")
        print("=" * 60)
        for case in random.sample(cases, min(args.preview, len(cases))):
            # Get complexity for display
            _, after_c = case.get_complexity()
            print(f"\n[{case.id}] {case.category}/{case.subcategory}")
            print(f"  Operation: {case.operation}")
            print(f"  Difficulty: {case.difficulty} (depth={after_c.depth}, score={after_c.difficulty_score:.2f})")
            print(f"  Before: {case.before_latex}")
            print(f"  After:  {case.after_latex}")
            print(f"  Desc:   {case.edit_description}")
    
    # Save to file
    generator.save_to_json(
        cases, 
        args.output,
        as_training_examples=args.training_format,
        add_noise=not args.no_noise,
        noise_probability=args.noise_prob
    )


if __name__ == "__main__":
    main()
