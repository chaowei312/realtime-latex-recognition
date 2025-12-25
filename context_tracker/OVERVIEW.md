# Context Tracker: Architecture Overview

> **Real-time stroke-level LaTeX editing with reasoning trajectory capture**

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CONTEXT TRACKER PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   USER INPUT                    PROCESSING                      OUTPUT          │
│   ══════════                    ══════════                      ══════          │
│                                                                                  │
│   ┌─────────┐                                                                   │
│   │ Stroke 1│────┐                                                              │
│   └─────────┘    │         ┌──────────────────────┐                            │
│   ┌─────────┐    │         │                      │         ┌──────────────┐   │
│   │ Stroke 2│────┼────────►│  Dual-Token Model    │────────►│ LaTeX: x^{2} │   │
│   └─────────┘    │         │  [CLS] + [INT]       │         └──────────────┘   │
│   ┌─────────┐    │         │                      │                ▼           │
│   │ Stroke 3│────┘         └──────────────────────┘         ┌──────────────┐   │
│   └─────────┘                       ▲                       │ Edit: ADD    │   │
│        │                            │                       │ Position: 3  │   │
│        ▼                     ┌──────┴──────┐                └──────────────┘   │
│   ┌─────────────┐            │             │                       ▼           │
│   │ Text Context│────────────┘     KV-Cache                 ┌──────────────┐   │
│   │ "x + y"     │              (incremental)                │ Reasoning    │   │
│   └─────────────┘                                           │ Trajectory   │   │
│                                                             └──────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Input Sequence Format

### Token Sequence Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  INPUT SEQUENCE (Single Forward Pass)                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────┬─────┬──────────────────────┬─────┬─────┬────────────┐ │
│  │   LaTeX Context     │ SEP │   Stroke Patches     │ CLS │ INT │ AR Decode  │ │
│  ├─────────────────────┼─────┼──────────────────────┼─────┼─────┼────────────┤ │
│  │ "x" "+" "y" "^{" "2"│ [S] │ [P₀][P₁][P₂]...[Pₖ] │ [C] │ [I] │ [e₁]..[EOS]│ │
│  │       "}" "="       │     │                      │     │     │            │ │
│  └─────────────────────┴─────┴──────────────────────┴─────┴─────┴────────────┘ │
│           │                           │                │     │        │        │
│           │                           │                │     │        │        │
│      Committed                   Current            Local  Global   Edit      │
│      symbols                     stroke             recog  reason   output    │
│      (text)                      (visual)                                      │
│                                                                                  │
│  EXAMPLE:                                                                        │
│  ════════                                                                        │
│  User has written "x + y" and now draws "²" above y                             │
│                                                                                  │
│  Input:  <x> <+> <y> [SEP] [P₀][P₁][P₂][P₃] [CLS] [INT]                        │
│  Output: [ADD] <^{> <2> <}> [EOS]                                               │
│                                                                                  │
│  Result: "x + y^{2}"                                                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Concrete Examples

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 1: Adding Superscript                                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Before: x + y          User draws: ²          After: x + y^{2}                 │
│                                                                                  │
│  Input Tokens:                                                                   │
│  ┌───┬───┬───┬─────┬────┬────┬────┬────┬─────┬─────┐                           │
│  │ x │ + │ y │[SEP]│ P₀ │ P₁ │ P₂ │ P₃ │[CLS]│[INT]│                           │
│  └───┴───┴───┴─────┴────┴────┴────┴────┴─────┴─────┘                           │
│    ↑   ↑   ↑          ↑    ↑    ↑    ↑     ↑     ↑                              │
│   text context       stroke patches      dual tokens                            │
│                                                                                  │
│  Output: [ADD] [^{] [2] [}] [EOS]                                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 2: Replacing Symbol                                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Before: x^{2}          User draws: 3          After: x^{3}                     │
│          (over the 2)                                                           │
│                                                                                  │
│  Input Tokens:                                                                   │
│  ┌───┬────┬───┬───┬─────┬────┬────┬────┬─────┬─────┐                           │
│  │ x │ ^{ │ 2 │ } │[SEP]│ P₀ │ P₁ │ P₂ │[CLS]│[INT]│                           │
│  └───┴────┴───┴───┴─────┴────┴────┴────┴─────┴─────┘                           │
│                                                                                  │
│  Output: [REPLACE] [pos=2] [3] [EOS]                                            │
│          (replace token at position 2)                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  EXAMPLE 3: Incomplete Stroke (WAIT)                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User drawing "α" but only 2 of 3 strokes complete                              │
│                                                                                  │
│  Input Tokens:                                                                   │
│  ┌───┬───┬─────┬────┬────┬─────┬─────┐                                         │
│  │ x │ + │[SEP]│ P₀ │ P₁ │[CLS]│[INT]│    (partial strokes)                    │
│  └───┴───┴─────┴────┴────┴─────┴─────┘                                         │
│                                                                                  │
│  Output: [WAIT]                                                                  │
│          (model recognizes incomplete symbol, waits for more strokes)           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Attention Pattern Designs

We propose **two experimental attention patterns** to compare:

### Option A: Isolated Patches (Dual-Token Design)

**Philosophy**: Keep visual processing pure. [CLS] focuses only on stroke recognition, [INT] handles context integration.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OPTION A: ISOLATED PATCHES (Recommended)                                        │
│  ════════════════════════════════════════                                        │
│                                                                                  │
│  Attention Matrix:                                                               │
│                                                                                  │
│              L₀  L₁  L₂  SEP  P₀  P₁  P₂  P₃  CLS INT  e₁  e₂ EOS             │
│         ┌────────────────────────────────────────────────────────┐              │
│     L₀  │  ■   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   │  LaTeX:      │
│     L₁  │  ■   ■   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   │  CAUSAL      │
│     L₂  │  ■   ■   ■   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   │  (triangle)  │
│    SEP  │  ■   ■   ■   ■   ·   ·   ·   ·   ·   ·   ·   ·   ·   │              │
│     ────┼────────────────────────────────────────────────────────              │
│     P₀  │  ·   ·   ·   ·   ■   ·   ·   ·   ·   ·   ·   ·   ·   │  Patches:    │
│     P₁  │  ·   ·   ·   ·   ■   ■   ·   ·   ·   ·   ·   ·   ·   │  ISOLATED    │
│     P₂  │  ·   ·   ·   ·   ■   ■   ■   ·   ·   ·   ·   ·   ·   │  CAUSAL      │
│     P₃  │  ·   ·   ·   ·   ■   ■   ■   ■   ·   ·   ·   ·   ·   │  (no LaTeX)  │
│     ────┼────────────────────────────────────────────────────────              │
│    CLS  │  ·   ·   ·   ·   ■   ■   ■   ■   ■   ·   ·   ·   ·   │  [CLS]:      │
│         │                                                        │  patches     │
│         │                                                        │  ONLY        │
│     ────┼────────────────────────────────────────────────────────              │
│    INT  │  ■   ■   ■   ■   ·   ·   ·   ·   ■   ■   ·   ·   ·   │  [INT]:      │
│         │                                                        │  LaTeX+CLS   │
│     ────┼────────────────────────────────────────────────────────              │
│     e₁  │  ■   ■   ■   ■   ·   ·   ·   ·   ■   ■   ■   ·   ·   │  AR decode:  │
│     e₂  │  ■   ■   ■   ■   ·   ·   ·   ·   ■   ■   ■   ■   ·   │  CAUSAL      │
│    EOS  │  ■   ■   ■   ■   ·   ·   ·   ·   ■   ■   ■   ■   ■   │              │
│         └────────────────────────────────────────────────────────┘              │
│                                                                                  │
│  ■ = attends    · = masked                                                      │
│                                                                                  │
│  ADVANTAGES:                                                                     │
│  ✓ [CLS] has PURE visual features (no context dilution)                         │
│  ✓ Clear separation: visual recognition vs semantic reasoning                   │
│  ✓ O(1) REPLACE: only recompute patches + [CLS] + [INT]                        │
│                                                                                  │
│  DISADVANTAGES:                                                                  │
│  ✗ Cannot use context to disambiguate similar symbols (1 vs l)                  │
│  ✗ Two special tokens (slightly more complex)                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Option B: Context-Aware Patches

**Philosophy**: Let patches see context for better disambiguation of visually similar symbols.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OPTION B: CONTEXT-AWARE PATCHES                                                 │
│  ═══════════════════════════════                                                 │
│                                                                                  │
│  Attention Matrix:                                                               │
│                                                                                  │
│              L₀  L₁  L₂  SEP  P₀  P₁  P₂  P₃  CLS  e₁  e₂ EOS                  │
│         ┌─────────────────────────────────────────────────────┐                 │
│     L₀  │  ■   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   │  LaTeX:          │
│     L₁  │  ■   ■   ·   ·   ·   ·   ·   ·   ·   ·   ·   ·   │  CAUSAL          │
│     L₂  │  ■   ■   ■   ·   ·   ·   ·   ·   ·   ·   ·   ·   │                  │
│    SEP  │  ■   ■   ■   ■   ·   ·   ·   ·   ·   ·   ·   ·   │                  │
│     ────┼─────────────────────────────────────────────────────                  │
│     P₀  │  ■   ■   ■   ■   ■   ·   ·   ·   ·   ·   ·   ·   │  Patches:        │
│     P₁  │  ■   ■   ■   ■   ■   ■   ·   ·   ·   ·   ·   ·   │  SEE LATEX       │
│     P₂  │  ■   ■   ■   ■   ■   ■   ■   ·   ·   ·   ·   ·   │  + causal        │
│     P₃  │  ■   ■   ■   ■   ■   ■   ■   ■   ·   ·   ·   ·   │  among patches   │
│     ────┼─────────────────────────────────────────────────────                  │
│    CLS  │  ■   ■   ■   ■   ■   ■   ■   ■   ■   ·   ·   ·   │  [CLS]:          │
│         │                                                     │  sees ALL       │
│         │                                                     │  (no [INT])     │
│     ────┼─────────────────────────────────────────────────────                  │
│     e₁  │  ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ·   ·   │  AR decode       │
│     e₂  │  ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ·   │                  │
│    EOS  │  ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   │                  │
│         └─────────────────────────────────────────────────────┘                 │
│                                                                                  │
│  ■ = attends    · = masked                                                      │
│                                                                                  │
│  ADVANTAGES:                                                                     │
│  ✓ Context helps disambiguate: "1" vs "l" based on "x + ?" vs "hello"          │
│  ✓ Simpler (single [CLS], no [INT])                                             │
│  ✓ End-to-end learning of context-visual interaction                            │
│                                                                                  │
│  DISADVANTAGES:                                                                  │
│  ✗ CONTEXT DILUTION: patch attention split between visual + text               │
│  ✗ O(n) REPLACE: patches depend on context → must recompute all                │
│  ✗ Harder to interpret what [CLS] learned                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Option C: Hybrid (Best of Both?)

**Philosophy**: Patches stay isolated for visual purity, but low-confidence candidates get context access.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OPTION C: HYBRID (Confidence-Gated Context)                                     │
│  ═══════════════════════════════════════════                                     │
│                                                                                  │
│  STEP 1: Initial recognition (isolated patches)                                 │
│  ┌────────────────────────────────────────────────────────────┐                 │
│  │  [P₀][P₁][P₂][P₃] → [CLS] → softmax → {                   │                 │
│  │                                         "1": 0.45          │                 │
│  │                                         "l": 0.40          │                 │
│  │                                         "I": 0.10          │                 │
│  │                                         ...                │                 │
│  │                                        }                   │                 │
│  │                                                            │                 │
│  │  Confidence = 0.45 < threshold (0.7)                       │                 │
│  │  → LOW CONFIDENCE → need context!                          │                 │
│  └────────────────────────────────────────────────────────────┘                 │
│                                                                                  │
│  STEP 2: Context-assisted disambiguation                                        │
│  ┌────────────────────────────────────────────────────────────┐                 │
│  │                                                            │                 │
│  │  Context: "x + y ="                                        │                 │
│  │           (math expression → likely "1" not "l")           │                 │
│  │                                                            │                 │
│  │  [INT] attends to:                                         │                 │
│  │    - LaTeX context ["x", "+", "y", "="]                    │                 │
│  │    - [CLS] embedding (visual: "1" or "l")                  │                 │
│  │    - Top-k candidates ["1", "l", "I"]                      │                 │
│  │                                                            │                 │
│  │  Output: "1" (math context favors digit)                   │                 │
│  │                                                            │                 │
│  └────────────────────────────────────────────────────────────┘                 │
│                                                                                  │
│  ADVANTAGES:                                                                     │
│  ✓ Visual features stay pure for easy symbols                                   │
│  ✓ Context only used when needed (efficiency)                                   │
│  ✓ Interpretable: can see when/why context was consulted                        │
│                                                                                  │
│  DISADVANTAGES:                                                                  │
│  ✗ Two-stage inference (slightly more complex)                                  │
│  ✗ Threshold tuning required                                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Visual Comparison of Attention Patterns

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION PATTERN COMPARISON                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  OPTION A (Isolated)          OPTION B (Context-Aware)      OPTION C (Hybrid)   │
│                                                                                  │
│   LaTeX │ Patches │ CLS       LaTeX │ Patches │ CLS        LaTeX │ Patches│CLS │
│  ───────┼─────────┼────      ───────┼─────────┼────       ───────┼────────┼────│
│  ████   │         │          ████   │         │           ████   │        │    │
│  ████   │         │          ████   │ ████    │           ████   │        │    │
│  ████   │         │          ████   │ ████    │           ████   │        │    │
│         │ ████    │ █               │ ████    │ █                │ ████   │ █  │
│         │ ████    │ █               │ ████    │ █                │ ████   │ █  │
│         │ ████    │ █               │ ████    │ █                │ ████   │ █  │
│         │    ▲    │ ▲               │    ▲    │ ▲                │    ▲   │ ▲  │
│         │    │    │ │               │    │    │ │                │    │   │ │  │
│         │ no cross│ patches         │  cross  │ ALL              │isolated│CLS │
│         │ attend  │ only            │ attend  │                  │ first  │+ctx│
│                                                                                  │
│  [CLS] feature:              [CLS] feature:               [CLS] feature:        │
│  PURE visual                 visual + context             visual (+ context     │
│  "looks like 1/l/I"          "probably 1 given math"      if low confidence)    │
│                                                                                  │
│  Cost per stroke:            Cost per stroke:             Cost per stroke:      │
│  O(p² + n)                   O((n+p)²)                    O(p² + n) or O((n+p)²)│
│                                                                                  │
│  REPLACE cost:               REPLACE cost:                REPLACE cost:         │
│  O(p² + 1)                   O((n+p)²)                    O(p² + 1) usually     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Pipeline Visualization

### Training Data Composition

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATA COMPOSITION                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ATOMIC SAMPLES (depth-based)                     COMPOSITIONAL CONTEXT        │
│   ════════════════════════════                     ══════════════════════        │
│                                                                                  │
│   depth=1: x, y, α, +, =, ...                     ┌─────────────────────────┐   │
│   depth=2: x², \frac{1}{2}, x_i                   │ \int_0^1 f(x)dx,        │   │
│   depth=3: \frac{x^2}{y}, \sum_{i=1}^n            │ \begin{pmatrix}...      │   │
│   depth=4+: complex expressions                   │ H_7 \to D               │   │
│             + commutative diagrams                │                         │   │
│                                                   │ (multiple atomic +      │   │
│                                                   │  diagrams composed)     │   │
│                                                   └─────────────────────────┘   │
│                                                                                  │
│   STROKE VARIATIONS                                                             │
│   ═════════════════                                                             │
│                                                                                  │
│   Symbol "x" has multiple handwritten versions:                                 │
│                                                                                  │
│   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                  │
│   │  X  │  │  X  │  │  X  │  │  X  │  │  X  │   × 5 style variations          │
│   │ ╲ ╱ │  │ ╳   │  │╲  ╱ │  │ X   │  │ ⤫   │   × N augmentations             │
│   └─────┘  └─────┘  └─────┘  └─────┘  └─────┘   = Millions of examples         │
│   style 1  style 2  style 3  style 4  style 5                                   │
│                                                                                  │
│   STROKE PROGRESSION (complete vs WAIT)                                         │
│   ═════════════════════════════════════                                         │
│                                                                                  │
│   Symbol "y" composed of 3 strokes:                                             │
│                                                                                  │
│   ┌─────┐  ┌─────┐  ┌─────┐                                                    │
│   │  \  │  │  \ /│  │  Y  │                                                    │
│   │     │  │   | │  │  |  │                                                    │
│   └─────┘  └─────┘  └─────┘                                                    │
│   1/3 WAIT  2/3 WAIT  3/3 ✓y                                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Demo: Context Visualization

![Context Visualization](data/demos/context_visualization.png)

*Compositional context with math expressions, matrices, and commutative diagram. Red dots show position jittering for training.*

---

## 6. Reasoning Trajectory Capture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      REASONING TRAJECTORY EXAMPLE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  USER SESSION: Solving quadratic equation                                       │
│  ═══════════════════════════════════════                                        │
│                                                                                  │
│  t=0   │ stroke → "x"        │ ADD      │ LaTeX: "x"                           │
│  t=1   │ stroke → "²"        │ ADD      │ LaTeX: "x^{2}"                        │
│  t=2   │ stroke → "+"        │ ADD      │ LaTeX: "x^{2} +"                      │
│  t=3   │ stroke → "2"        │ ADD      │ LaTeX: "x^{2} + 2"                    │
│  t=4   │ stroke → "x"        │ ADD      │ LaTeX: "x^{2} + 2x"                   │
│  t=5   │ stroke → "+"        │ ADD      │ LaTeX: "x^{2} + 2x +"                 │
│  t=6   │ stroke → "1"        │ ADD      │ LaTeX: "x^{2} + 2x + 1"              │
│  t=7   │ PAUSE (3 sec)       │ ---      │ (user thinking...)                   │
│  t=8   │ stroke → "="        │ ADD      │ LaTeX: "x^{2} + 2x + 1 ="            │
│  t=9   │ stroke → "0"        │ ADD      │ LaTeX: "x^{2} + 2x + 1 = 0"          │
│  t=10  │ (scratch gesture)   │ DELETE   │ LaTeX: "x^{2} + 2x + 1"  (undo)      │
│  t=11  │ stroke → "("        │ WRAP     │ LaTeX: "(x^{2} + 2x + 1)"            │
│  t=12  │ ...                 │          │                                       │
│                                                                                  │
│  TRAJECTORY FEATURES EXTRACTED:                                                 │
│  ══════════════════════════════                                                 │
│                                                                                  │
│  • Operation sequence: [ADD, ADD, ADD, ADD, ADD, ADD, ADD, ADD, ADD, DEL, WRAP] │
│  • Pause patterns: long pause before "=" (decision point)                       │
│  • Self-correction: deleted "= 0", suggests rethinking approach                 │
│  • Structure: recognized factoring pattern (x+1)²                               │
│                                                                                  │
│  ALIGNMENT TO REASONING:                                                        │
│  ══════════════════════                                                         │
│                                                                                  │
│  trajectory embedding ──┬── align to ──► knowledge graph node                   │
│                         │               "quadratic_equation.factoring"          │
│                         │                                                       │
│                         └── predict ──► hint: "This looks factorable..."        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Experiments to Run

| Experiment | Question | Metric |
|------------|----------|--------|
| **A vs B** | Does context dilute visual features? | Symbol accuracy on isolated symbols |
| **A vs B** | Does context help disambiguation? | Accuracy on "1/l/I" confusion set |
| **A vs B** | REPLACE latency difference? | ms per edit operation |
| **A vs C** | Is hybrid worth the complexity? | F1 on ambiguous symbols |
| **All** | Reasoning trajectory utility? | Prediction of student errors |

---

## 8. Quick Reference: Token Types

| Token | Symbol | Attends To | Purpose |
|-------|--------|------------|---------|
| LaTeX context | `<x>`, `<+>`, `<\frac>` | Previous LaTeX (causal) | Committed recognition |
| Separator | `[SEP]` | All LaTeX | Boundary marker |
| Patches | `[P₀]`...`[Pₖ]` | Option A: patches only<br>Option B: LaTeX + patches | Visual features |
| CLS | `[CLS]` | Patches (+ LaTeX in B) | Aggregate visual → symbol |
| INT | `[INT]` | LaTeX + [CLS] | Global reasoning |
| Edit tokens | `[ADD]`, `<^{>`, ... | All context (causal) | Output sequence |

---

*Last updated: December 2024*

