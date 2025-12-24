"""
LaTeX Preprocessing Utilities

Handles incomplete LaTeX normalization and validation for real-time editing scenarios.

Functions:
    - normalize_incomplete_latex: Fix unclosed braces, incomplete fractions, etc.
    - can_render_latex: Quick check if LaTeX can likely be rendered
    - generate_incomplete_training_examples: Create training data for partial states
"""

import re
from typing import List, Tuple, Dict, Any


def normalize_incomplete_latex(latex: str) -> Tuple[str, bool, List[str]]:
    r"""
    Normalize incomplete LaTeX for rendering and model input.
    
    Handles edge cases during real-time editing:
    - Unclosed braces: x^{2 -> x^{2}
    - Incomplete fractions: \frac{2}{ -> \frac{2}{\square}
    - Partial commands: \alph -> \alpha (or keep as-is)
    - Incomplete environments: \begin{matrix} -> add \end{matrix}
    
    Args:
        latex: Potentially incomplete LaTeX string
        
    Returns:
        Tuple of (normalized_latex, is_complete, list_of_issues)
    """
    issues = []
    normalized = latex
    
    # Count braces
    open_braces = latex.count('{')
    close_braces = latex.count('}')
    
    # Fix unclosed braces
    if open_braces > close_braces:
        missing = open_braces - close_braces
        normalized += '}' * missing
        issues.append(f"added {missing} closing brace(s)")
    elif close_braces > open_braces:
        # Extra closing braces - prepend opening
        extra = close_braces - open_braces
        normalized = '{' * extra + normalized
        issues.append(f"added {extra} opening brace(s)")
    
    # Fix incomplete fractions: \frac{...}{ or \frac{
    # \frac{X}{ without content → \frac{X}{\square}
    normalized = re.sub(r'\\frac\{([^}]*)\}\{(\s*)$', r'\\frac{\1}{\\square}', normalized)
    if r'\square' in normalized and r'\square' not in latex:
        issues.append("added placeholder for incomplete fraction")
    
    # \frac{ without numerator
    normalized = re.sub(r'\\frac\{(\s*)$', r'\\frac{\\square}{\\square}', normalized)
    
    # Fix incomplete environments
    env_begins = re.findall(r'\\begin\{(\w+)\}', normalized)
    env_ends = re.findall(r'\\end\{(\w+)\}', normalized)
    
    for env in env_begins:
        if env_begins.count(env) > env_ends.count(env):
            normalized += f'\\end{{{env}}}'
            issues.append(f"added \\end{{{env}}}")
    
    # Check for partial commands (common ones)
    partial_commands = {
        r'\alph': r'\alpha',
        r'\bet': r'\beta', 
        r'\gam': r'\gamma',
        r'\fra': r'\frac{}{}',
        r'\sqr': r'\sqrt{}',
        r'\sum': r'\sum',  # complete
        r'\int': r'\int',  # complete
    }
    
    for partial, complete in partial_commands.items():
        if normalized.endswith(partial) and partial != complete:
            # Don't auto-complete, just flag it
            issues.append(f"partial command: {partial}")
    
    is_complete = len(issues) == 0
    
    return normalized, is_complete, issues


def can_render_latex(latex: str) -> bool:
    """
    Quick check if LaTeX can likely be rendered.
    
    Doesn't actually render - just checks for obvious issues.
    """
    # Basic brace balance
    if latex.count('{') != latex.count('}'):
        return False
    
    # Check environment balance
    begins = re.findall(r'\\begin\{(\w+)\}', latex)
    ends = re.findall(r'\\end\{(\w+)\}', latex)
    if sorted(begins) != sorted(ends):
        return False
    
    return True


def generate_incomplete_training_examples(
    complete_latex: str,
    num_steps: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate training examples for incomplete LaTeX states.
    
    Simulates a user typing the expression step by step,
    creating intermediate incomplete states.
    
    Args:
        complete_latex: The final complete LaTeX
        num_steps: Number of intermediate states to generate
        
    Returns:
        List of training examples with incomplete states
    """
    import random
    
    examples = []
    
    # Identify "breakpoints" where we can split the LaTeX
    # Prefer splitting at: spaces, operators, braces, after commands
    
    # Find good split points
    split_points = [0]
    
    # Split at spaces and operators
    for m in re.finditer(r'[\s+\-={}]', complete_latex):
        split_points.append(m.end())
    
    # Split after commands (fixed-width lookbehind not needed)
    for m in re.finditer(r'\\[a-z]+', complete_latex):
        split_points.append(m.end())
    
    split_points.append(len(complete_latex))
    split_points = sorted(set(split_points))
    
    # Sample intermediate states
    if len(split_points) > num_steps:
        indices = sorted(random.sample(range(1, len(split_points)-1), num_steps-1))
        indices = [0] + indices + [len(split_points)-1]
    else:
        indices = range(len(split_points))
    
    for i, idx in enumerate(indices[:-1]):
        partial = complete_latex[:split_points[indices[i+1]]]
        normalized, is_complete, issues = normalize_incomplete_latex(partial)
        
        examples.append({
            "step": i,
            "partial_latex": partial,
            "normalized_latex": normalized,
            "is_complete": is_complete,
            "issues": issues,
            "final_latex": complete_latex,
            "completion_pct": len(partial) / len(complete_latex),
        })
    
    return examples

