"""
Symbol-Level LaTeX Tokenizer for Real-Time Editing Model

Design choices:
- ~800 tokens vocabulary (small, edge-friendly)
- LaTeX commands as single tokens (e.g., \alpha, \frac{)
- Structure-preserving (^{, _{, } as tokens)
- Edit operation tokens (<ADD>, <REPLACE>, <INSERT>)

Usage:
    tokenizer = LaTeXTokenizer()
    ids = tokenizer.encode(r"x^{2} + \alpha")
    latex = tokenizer.decode(ids)
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


# =============================================================================
# Vocabulary Definition
# =============================================================================

def build_vocabulary() -> Dict[str, int]:
    """Build the complete LaTeX symbol vocabulary."""
    
    vocab = {}
    idx = 0
    
    # =========================================================================
    # Special Tokens (0-19)
    # =========================================================================
    special_tokens = [
        "<PAD>",      # Padding
        "<BOS>",      # Beginning of sequence  
        "<EOS>",      # End of sequence
        "<UNK>",      # Unknown token
        "<SEP>",      # Separator (between context and output)
        # Model output tokens
        "<DEL>",      # Delete operation (cross-out detected)
        "<WAIT>",     # Incomplete symbol, wait for more strokes
        # Position context tokens (optional, for auxiliary info)
        "<SUP>",      # Superscript position hint
        "<SUB>",      # Subscript position hint
        "<FRAC_NUM>", # Fraction numerator hint
        "<FRAC_DEN>", # Fraction denominator hint
        "<MATRIX>",   # Matrix context
        "<DIAGRAM>",  # Diagram context
        # Reserved for future use
        "<RESERVED_14>", "<RESERVED_15>", 
        "<RESERVED_16>", "<RESERVED_17>", "<RESERVED_18>", "<RESERVED_19>",
        # NOTE: No <ADD>, <REPLACE>, <INSERT>, <FILL> needed!
        # Position query determines operation type implicitly:
        #   - Empty {} slot → INSERT/FILL
        #   - Existing token → REPLACE  
        #   - End of sequence → ADD
        #   - <DEL> output → DELETE
    ]
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # English Letters (20-71)
    # =========================================================================
    for c in "abcdefghijklmnopqrstuvwxyz":
        vocab[c] = idx
        idx += 1
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        vocab[c] = idx
        idx += 1
    
    # =========================================================================
    # Digits (72-81)
    # =========================================================================
    for c in "0123456789":
        vocab[c] = idx
        idx += 1
    
    # =========================================================================
    # Greek Letters - Lowercase (82-105)
    # =========================================================================
    greek_lower = [
        r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
        r"\zeta", r"\eta", r"\theta", r"\iota", r"\kappa",
        r"\lambda", r"\mu", r"\nu", r"\xi", r"\omicron",
        r"\pi", r"\rho", r"\sigma", r"\tau", r"\upsilon",
        r"\phi", r"\chi", r"\psi", r"\omega",
    ]
    for token in greek_lower:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Greek Letters - Uppercase (106-119)
    # =========================================================================
    greek_upper = [
        r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
        r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi",
        r"\Omega",
        # Variants
        r"\varepsilon", r"\varphi", r"\vartheta",
    ]
    for token in greek_upper:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Basic Operators (120-149)
    # =========================================================================
    operators = [
        "+", "-", "=", 
        r"\times", r"\cdot", r"\div",
        r"\pm", r"\mp",
        r"\ast", r"\star", r"\circ", r"\bullet",
        r"\oplus", r"\otimes", r"\odot",
        r"\wedge", r"\vee", r"\cap", r"\cup",
        r"\setminus", r"\sqcup",
        # Negation
        r"\neg", r"\lnot",
        # Reserved
        "<OP_RESERVED_1>", "<OP_RESERVED_2>", "<OP_RESERVED_3>",
        "<OP_RESERVED_4>", "<OP_RESERVED_5>", "<OP_RESERVED_6>",
    ]
    for token in operators:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Relations (150-189)
    # =========================================================================
    relations = [
        "<", ">", 
        r"\leq", r"\geq", r"\neq",
        r"\le", r"\ge", r"\ne",
        r"\ll", r"\gg",
        r"\approx", r"\equiv", r"\sim", r"\simeq", r"\cong",
        r"\propto", r"\parallel", r"\perp",
        r"\subset", r"\supset", r"\subseteq", r"\supseteq",
        r"\in", r"\ni", r"\notin",
        r"\vdash", r"\dashv", r"\models",
        r"\prec", r"\succ", r"\preceq", r"\succeq",
        # Reserved
        "<REL_RESERVED_1>", "<REL_RESERVED_2>", "<REL_RESERVED_3>",
        "<REL_RESERVED_4>", "<REL_RESERVED_5>", "<REL_RESERVED_6>",
    ]
    for token in relations:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Brackets and Grouping (190-219)
    # =========================================================================
    brackets = [
        "(", ")", "[", "]",
        "{", "}",  # Plain braces
        r"\{", r"\}",  # Escaped braces
        r"\langle", r"\rangle",
        r"\lfloor", r"\rfloor", r"\lceil", r"\rceil",
        r"\left(", r"\right)", r"\left[", r"\right]",
        r"\left\{", r"\right\}",
        r"\left.", r"\right.",
        # Grouping (single tokens for efficiency)
        "^{", "_{",
        "^", "_",  # Without braces (single char sup/sub)
        # Reserved
        "<BRACKET_RESERVED_1>", "<BRACKET_RESERVED_2>", 
        "<BRACKET_RESERVED_3>", "<BRACKET_RESERVED_4>",
    ]
    for token in brackets:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Fractions, Roots, Binomials (220-239)
    # =========================================================================
    fractions = [
        r"\frac{", r"\dfrac{", r"\tfrac{",
        r"\sqrt{", r"\sqrt[",
        r"\binom{", r"\dbinom{",
        r"\over",
        # Reserved
        "<FRAC_RESERVED_1>", "<FRAC_RESERVED_2>", "<FRAC_RESERVED_3>",
        "<FRAC_RESERVED_4>", "<FRAC_RESERVED_5>", "<FRAC_RESERVED_6>",
        "<FRAC_RESERVED_7>", "<FRAC_RESERVED_8>", "<FRAC_RESERVED_9>",
        "<FRAC_RESERVED_10>", "<FRAC_RESERVED_11>", "<FRAC_RESERVED_12>",
    ]
    for token in fractions:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Calculus Symbols (240-279)
    # =========================================================================
    calculus = [
        r"\int", r"\iint", r"\iiint", r"\oint",
        r"\int_{", r"\int_",  # With subscript start
        r"\sum", r"\sum_{", r"\sum_",
        r"\prod", r"\prod_{", r"\prod_",
        r"\coprod",
        r"\partial",
        r"\nabla",
        r"\infty",
        r"\lim", r"\lim_{",
        r"\limsup", r"\liminf",
        "d",  # Differential d (context-dependent)
        r"\mathrm{d}",  # Upright d
        r"\,",  # Thin space (often used in dx)
        # Reserved
        "<CALC_RESERVED_1>", "<CALC_RESERVED_2>", "<CALC_RESERVED_3>",
        "<CALC_RESERVED_4>", "<CALC_RESERVED_5>", "<CALC_RESERVED_6>",
        "<CALC_RESERVED_7>", "<CALC_RESERVED_8>", "<CALC_RESERVED_9>",
        "<CALC_RESERVED_10>", "<CALC_RESERVED_11>", "<CALC_RESERVED_12>",
        "<CALC_RESERVED_13>", "<CALC_RESERVED_14>", "<CALC_RESERVED_15>",
    ]
    for token in calculus:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Arrows (280-319)
    # =========================================================================
    arrows = [
        r"\to", r"\rightarrow", r"\leftarrow",
        r"\Rightarrow", r"\Leftarrow", r"\Leftrightarrow",
        r"\leftrightarrow",
        r"\mapsto", r"\longmapsto",
        r"\hookrightarrow", r"\hookleftarrow",
        r"\twoheadrightarrow", r"\twoheadleftarrow",
        r"\uparrow", r"\downarrow", r"\updownarrow",
        r"\Uparrow", r"\Downarrow",
        r"\nearrow", r"\searrow", r"\nwarrow", r"\swarrow",
        r"\leadsto",
        r"\rightharpoonup", r"\rightharpoondown",
        r"\leftharpoonup", r"\leftharpoondown",
        # Reserved
        "<ARROW_RESERVED_1>", "<ARROW_RESERVED_2>", "<ARROW_RESERVED_3>",
        "<ARROW_RESERVED_4>", "<ARROW_RESERVED_5>", "<ARROW_RESERVED_6>",
        "<ARROW_RESERVED_7>", "<ARROW_RESERVED_8>", "<ARROW_RESERVED_9>",
        "<ARROW_RESERVED_10>", "<ARROW_RESERVED_11>", "<ARROW_RESERVED_12>",
    ]
    for token in arrows:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Functions (320-349)
    # =========================================================================
    functions = [
        r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
        r"\arcsin", r"\arccos", r"\arctan",
        r"\sinh", r"\cosh", r"\tanh",
        r"\log", r"\ln", r"\lg", r"\exp",
        r"\max", r"\min", r"\sup", r"\inf",
        r"\arg", r"\det", r"\dim", r"\ker", r"\deg",
        r"\gcd", r"\lcm",
        r"\hom", r"\Hom",
        # Reserved
        "<FUNC_RESERVED_1>", "<FUNC_RESERVED_2>", "<FUNC_RESERVED_3>",
    ]
    for token in functions:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Accents and Modifiers (350-379)
    # =========================================================================
    accents = [
        r"\hat{", r"\bar{", r"\vec{", r"\tilde{",
        r"\dot{", r"\ddot{", r"\acute{", r"\grave{",
        r"\check{", r"\breve{",
        r"\widehat{", r"\widetilde{", r"\overline{", r"\underline{",
        r"\overbrace{", r"\underbrace{",
        r"\overrightarrow{", r"\overleftarrow{",
        r"\mathbf{", r"\mathit{", r"\mathrm{", r"\mathcal{",
        r"\mathbb{", r"\mathfrak{", r"\mathsf{", r"\mathtt{",
        r"\boldsymbol{", r"\text{",
    ]
    for token in accents:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Matrix Environments (380-419)
    # =========================================================================
    matrices = [
        r"\begin{matrix}", r"\end{matrix}",
        r"\begin{pmatrix}", r"\end{pmatrix}",
        r"\begin{bmatrix}", r"\end{bmatrix}",
        r"\begin{vmatrix}", r"\end{vmatrix}",
        r"\begin{Vmatrix}", r"\end{Vmatrix}",
        r"\begin{cases}", r"\end{cases}",
        r"\begin{array}", r"\end{array}",
        r"\begin{aligned}", r"\end{aligned}",
        r"\\",  # Row separator
        "&",    # Column separator
        r"\hline", r"\vline",
        r"\cdots", r"\ldots", r"\vdots", r"\ddots",
        # Reserved
        "<MATRIX_RESERVED_1>", "<MATRIX_RESERVED_2>", "<MATRIX_RESERVED_3>",
        "<MATRIX_RESERVED_4>", "<MATRIX_RESERVED_5>", "<MATRIX_RESERVED_6>",
        "<MATRIX_RESERVED_7>", "<MATRIX_RESERVED_8>", "<MATRIX_RESERVED_9>",
        "<MATRIX_RESERVED_10>", "<MATRIX_RESERVED_11>", "<MATRIX_RESERVED_12>",
        "<MATRIX_RESERVED_13>", "<MATRIX_RESERVED_14>", "<MATRIX_RESERVED_15>",
        "<MATRIX_RESERVED_16>",
    ]
    for token in matrices:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # TikZ-CD Diagram Tokens (420-499)
    # =========================================================================
    tikzcd = [
        r"\begin{tikzcd}", r"\end{tikzcd}",
        r"\arrow[",  # Arrow start
        "]",         # Arrow end (also used elsewhere)
        # Directions
        "r", "l", "u",  # Note: 'd' already in calculus
        "rr", "ll", "uu", "dd",
        "dr", "dl", "ur", "ul",
        "drr", "dll", "urr", "ull",
        "ddr", "ddl", "uur", "uul",
        # Arrow styles
        "dashed", "dotted",
        "hook", "hook'",
        "two heads",
        "tail",
        "Rightarrow", "Leftarrow",
        "phantom",
        "crossing over",
        "bend left", "bend right",
        "bend left=", "bend right=",
        "loop left", "loop right", "loop above", "loop below",
        # Labels
        '\"', "'",  # Label delimiters
        "description", "marking", "pos=",
        "very near start", "near start", "near end", "very near end",
        # Special
        r"\lrcorner", r"\ulcorner",  # Pullback/pushout corners
        # Reserved for more TikZ-CD
        "<TIKZ_RESERVED_1>", "<TIKZ_RESERVED_2>", "<TIKZ_RESERVED_3>",
        "<TIKZ_RESERVED_4>", "<TIKZ_RESERVED_5>", "<TIKZ_RESERVED_6>",
        "<TIKZ_RESERVED_7>", "<TIKZ_RESERVED_8>", "<TIKZ_RESERVED_9>",
        "<TIKZ_RESERVED_10>", "<TIKZ_RESERVED_11>", "<TIKZ_RESERVED_12>",
        "<TIKZ_RESERVED_13>", "<TIKZ_RESERVED_14>", "<TIKZ_RESERVED_15>",
        "<TIKZ_RESERVED_16>", "<TIKZ_RESERVED_17>", "<TIKZ_RESERVED_18>",
        "<TIKZ_RESERVED_19>", "<TIKZ_RESERVED_20>",
    ]
    for token in tikzcd:
        if token not in vocab:  # Avoid duplicates
            vocab[token] = idx
            idx += 1
    
    # =========================================================================
    # Special Symbols (500-549)
    # =========================================================================
    special_symbols = [
        r"\emptyset", r"\varnothing",
        r"\forall", r"\exists", r"\nexists",
        r"\therefore", r"\because",
        r"\angle", r"\triangle", r"\square",
        r"\diamond", r"\clubsuit", r"\heartsuit", r"\spadesuit",
        r"\prime", r"\backprime",
        r"\dagger", r"\ddagger",
        r"\ell", r"\hbar", r"\imath", r"\jmath",
        r"\Re", r"\Im",
        r"\wp", r"\aleph", r"\beth",
        r"\N", r"\Z", r"\Q", r"\R", r"\C",  # Number sets (may need \mathbb)
        r"\top", r"\bot",
        r"\Box", r"\Diamond",
        r"\S", r"\P",
        r"\copyright", r"\pounds",
        # Reserved
        "<SPECIAL_RESERVED_1>", "<SPECIAL_RESERVED_2>", "<SPECIAL_RESERVED_3>",
        "<SPECIAL_RESERVED_4>", "<SPECIAL_RESERVED_5>", "<SPECIAL_RESERVED_6>",
        "<SPECIAL_RESERVED_7>", "<SPECIAL_RESERVED_8>", "<SPECIAL_RESERVED_9>",
        "<SPECIAL_RESERVED_10>",
    ]
    for token in special_symbols:
        vocab[token] = idx
        idx += 1
    
    # =========================================================================
    # Spacing and Misc (550-579)
    # =========================================================================
    spacing = [
        r"\quad", r"\qquad",
        r"\,", r"\:", r"\;", r"\ ",
        r"\!", r"\>",
        r"\hspace{", r"\vspace{",
        r"\phantom{", r"\hphantom{", r"\vphantom{",
        ",", ".", ";", ":", "!", "?",
        "/",  # Division slash
        "'",  # Prime (for labels)
        r"\ldotp", r"\cdotp",
        # Reserved
        "<SPACE_RESERVED_1>", "<SPACE_RESERVED_2>", "<SPACE_RESERVED_3>",
        "<SPACE_RESERVED_4>", "<SPACE_RESERVED_5>", "<SPACE_RESERVED_6>",
        "<SPACE_RESERVED_7>", "<SPACE_RESERVED_8>", "<SPACE_RESERVED_9>",
        "<SPACE_RESERVED_10>",
    ]
    for token in spacing:
        if token not in vocab:
            vocab[token] = idx
            idx += 1
    
    # =========================================================================
    # Reserved for Future (580-799)
    # =========================================================================
    for i in range(580, 800):
        vocab[f"<RESERVED_{i}>"] = i
    
    return vocab


# =============================================================================
# Tokenizer Class
# =============================================================================

class LaTeXTokenizer:
    """
    Symbol-level tokenizer for LaTeX expressions.
    
    Features:
    - ~800 token vocabulary
    - Greedy longest-match tokenization
    - Special handling for ^{ and _{ groupings
    - Edit operation tokens for our model
    """
    
    def __init__(self, vocab: Dict[str, int] = None):
        self.vocab = vocab or build_vocabulary()
        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special tokens that can appear in input/output
        # NOTE: Edit operation type is determined by position, not explicit tokens
        self.inline_special_tokens = ['<SEP>', '<DEL>', '<WAIT>']
        
        # Sort patterns by length (longest first) for greedy matching
        # Include inline special tokens in the pattern list
        self.patterns = sorted(
            [k for k in self.vocab.keys() if not k.startswith("<")] + 
            [t for t in self.inline_special_tokens if t in self.vocab],
            key=len, 
            reverse=True
        )
        
        # Special token IDs
        self.pad_id = self.vocab["<PAD>"]
        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]
        self.unk_id = self.vocab["<UNK>"]
        self.sep_id = self.vocab["<SEP>"]
        
        # Model output IDs
        self.del_id = self.vocab["<DEL>"]    # Delete operation
        self.wait_id = self.vocab["<WAIT>"]  # Wait for more strokes
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def encode(self, latex: str, 
               add_bos: bool = False, 
               add_eos: bool = False) -> List[int]:
        """
        Encode LaTeX string to token IDs.
        
        Args:
            latex: LaTeX string to encode
            add_bos: Add <BOS> at start
            add_eos: Add <EOS> at end
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_id)
        
        i = 0
        while i < len(latex):
            # Skip whitespace (except significant ones like \, \quad)
            if latex[i] == ' ':
                i += 1
                continue
            
            # Try to match longest pattern first
            matched = False
            for pattern in self.patterns:
                if latex[i:].startswith(pattern):
                    tokens.append(self.token_to_id[pattern])
                    i += len(pattern)
                    matched = True
                    break
            
            if not matched:
                # Unknown character - use <UNK>
                tokens.append(self.unk_id)
                i += 1
        
        if add_eos:
            tokens.append(self.eos_id)
        
        return tokens
    
    def decode(self, ids: List[int], 
               skip_special: bool = True) -> str:
        """
        Decode token IDs back to LaTeX string.
        
        Args:
            ids: List of token IDs
            skip_special: Skip special tokens like <PAD>, <BOS>, <EOS>
            
        Returns:
            LaTeX string
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
        
        tokens = []
        for id in ids:
            if skip_special and id in special_ids:
                continue
            token = self.id_to_token.get(id, "")
            if not token.startswith("<"):  # Skip reserved tokens
                tokens.append(token)
        
        # Join with smart spacing
        result = self._smart_join(tokens)
        return result
    
    def _smart_join(self, tokens: List[str]) -> str:
        """Join tokens with appropriate spacing."""
        if not tokens:
            return ""
        
        result = []
        for i, token in enumerate(tokens):
            # Add space before if needed
            if i > 0:
                prev = tokens[i-1]
                # Don't add space after { or before }
                if prev.endswith("{") or token == "}":
                    pass
                # Don't add space between letter and ^{ or _{
                elif token in ["^{", "_{", "^", "_"]:
                    pass
                # Add space between letters/numbers
                elif (prev[-1].isalnum() and token[0].isalnum() and 
                      not prev.startswith("\\")):
                    result.append(" ")
            
            result.append(token)
        
        return "".join(result)
    
    def encode_edit(self, 
                    context_latex: str,
                    target_content: str,
                    target_position: int = -1) -> Dict[str, any]:
        """
        Encode an editing example for training.
        
        NEW DESIGN: No explicit edit operation tokens needed!
        - Position determines operation type implicitly
        - Empty {} slot → FILL
        - Existing token position → REPLACE
        - End position → ADD
        - <DEL> content → DELETE
        
        Args:
            context_latex: LaTeX context (may contain empty {} slots)
            target_content: What was written (or "<DEL>" for delete)
            target_position: Token index where edit applies (-1 = end)
            
        Returns:
            Dict with 'context_ids', 'target_content_ids', 'target_position'
        """
        # Encode context
        context_ids = self.encode(context_latex, add_bos=True)
        
        # Encode target content (what was written)
        if target_content == "<DEL>":
            target_content_ids = [self.del_id]
        else:
            target_content_ids = self.encode(target_content, add_eos=True)
        
        # Position: -1 means end (ADD), otherwise specific index
        if target_position < 0:
            target_position = len(context_ids)
        
        return {
            "context_ids": context_ids,
            "target_content_ids": target_content_ids,
            "target_position": target_position,
        }
    
    def save(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "LaTeXTokenizer":
        """Load vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab)


# =============================================================================
# Utility Functions
# =============================================================================

def get_token_statistics(tokenizer: LaTeXTokenizer) -> Dict:
    """Get vocabulary statistics."""
    stats = {
        "total_vocab_size": tokenizer.vocab_size,
        "categories": {},
    }
    
    # Count by category (based on ID ranges)
    ranges = [
        ("special", 0, 20),
        ("english_lower", 20, 46),
        ("english_upper", 46, 72),
        ("digits", 72, 82),
        ("greek_lower", 82, 106),
        ("greek_upper", 106, 120),
        ("operators", 120, 150),
        ("relations", 150, 190),
        ("brackets", 190, 220),
        ("fractions", 220, 240),
        ("calculus", 240, 280),
        ("arrows", 280, 320),
        ("functions", 320, 350),
        ("accents", 350, 380),
        ("matrices", 380, 420),
        ("tikzcd", 420, 500),
        ("special_symbols", 500, 550),
        ("spacing", 550, 580),
        ("reserved", 580, 800),
    ]
    
    for name, start, end in ranges:
        count = sum(1 for v in tokenizer.vocab.values() if start <= v < end)
        stats["categories"][name] = count
    
    return stats


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    tokenizer = LaTeXTokenizer()
    stats = get_token_statistics(tokenizer)
    
    print("=" * 60)
    print("LaTeX Symbol-Level Tokenizer")
    print("=" * 60)
    print(f"\nVocabulary Size: {stats['total_vocab_size']}")
    print("\nTokens by Category:")
    for cat, count in stats['categories'].items():
        if count > 0 and not cat == "reserved":
            print(f"  {cat:20} {count:4} tokens")
    
    # Test examples
    print("\n" + "=" * 60)
    print("Test Examples")
    print("=" * 60)
    
    test_cases = [
        r"x^{2} + y^{2} = r^{2}",
        r"\frac{1}{2}",
        r"\int_{0}^{\infty} e^{-x} \, dx",
        r"\sum_{n=1}^{\infty} \frac{1}{n^2}",
        r"\alpha + \beta = \gamma",
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        r"\partial f / \partial x",
        r"A \to B",
    ]
    
    for latex in test_cases:
        ids = tokenizer.encode(latex)
        decoded = tokenizer.decode(ids)
        print(f"\nOriginal: {latex}")
        print(f"Token IDs: {ids}")
        print(f"Decoded:  {decoded}")
        print(f"Tokens ({len(ids)}): {[tokenizer.id_to_token[i] for i in ids]}")
    
    # Test edit encoding
    print("\n" + "=" * 60)
    print("Edit Encoding Test")
    print("=" * 60)
    
    edit = tokenizer.encode_edit(
        context_latex=r"x",
        target_latex=r"x^{2}",
        operation="ADD"
    )
    print(f"\nContext: 'x' -> Target: 'x^{{2}}'")
    print(f"Input IDs:  {edit['input_ids']}")
    print(f"Target IDs: {edit['target_ids']}")
    print(f"Full IDs:   {edit['full_ids']}")
    print(f"Input tokens:  {[tokenizer.id_to_token[i] for i in edit['input_ids']]}")
    print(f"Target tokens: {[tokenizer.id_to_token[i] for i in edit['target_ids']]}")
    
    # Save vocabulary
    tokenizer.save("vocabulary.json")
    print(f"\nVocabulary saved to vocabulary.json")


if __name__ == "__main__":
    main()
