# Context Tracker: Architecture Overview

> **Real-time stroke-level LaTeX editing with reasoning trajectory capture**

---

## 0. Motivation & Design Intent

### Why This Architecture?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         THE PROBLEM WITH EXISTING APPROACHES                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TRADITIONAL VLM (Frame-wise Full Self-Attention)                               │
│  ════════════════════════════════════════════════                               │
│                                                                                  │
│   Frame 1        Frame 2        Frame 3        Frame N                          │
│  ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐                        │
│  │ image │      │ image │      │ image │      │ image │                        │
│  │ patch │      │ patch │      │ patch │      │ patch │                        │
│  │ tokens│      │ tokens│      │ tokens│      │ tokens│                        │
│  └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘                        │
│      │              │              │              │                             │
│      └──────────────┴──────────────┴──────────────┘                             │
│                          │                                                       │
│                          ▼                                                       │
│              ┌─────────────────────────┐                                        │
│              │  FULL SELF-ATTENTION    │   O(N² × P²) per inference            │
│              │  ALL patches × ALL time │   Memory: O(N × P) patches            │
│              └─────────────────────────┘   Latency: grows with history         │
│                          │                                                       │
│                          ▼                                                       │
│              ┌─────────────────────────┐                                        │
│              │     Single output       │   No edit operations                   │
│              │  (full re-recognition)  │   No reasoning trajectory             │
│              └─────────────────────────┘   No incremental updates              │
│                                                                                  │
│  PROBLEMS:                                                                       │
│  ✗ Quadratic cost: every new stroke reprocesses ALL visual history             │
│  ✗ No knowledge accumulation: treats each frame independently                  │
│  ✗ No trajectory: only captures WHAT was written, not HOW                      │
│  ✗ Edit = full re-inference: no efficient REPLACE/INSERT                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OUR SOLUTION: CONTEXT TRACKER                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  KEY INSIGHT: "Snowball" recognized symbols into TEXT, discard visual history  │
│  ════════════════════════════════════════════════════════════════════════════   │
│                                                                                  │
│   Stroke 1       Stroke 2       Stroke 3       Stroke N                         │
│  ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐                        │
│  │patches│      │patches│      │patches│      │patches│   Current stroke       │
│  └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘   (visual)             │
│      │              │              │              │                             │
│      ▼              ▼              ▼              ▼                             │
│  ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐                        │
│  │ [CLS] │      │ [CLS] │      │ [CLS] │      │ [CLS] │   Local recognition    │
│  └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘                        │
│      │              │              │              │                             │
│      ▼              ▼              ▼              ▼                             │
│   "x" ─────────► "x +" ───────► "x + y" ────► "x + y^{2}"                      │
│                                                                                  │
│   ▲               ▲               ▲               ▲                             │
│   │               │               │               │                             │
│   └───────────────┴───────────────┴───────────────┘                             │
│            TEXT CONTEXT (KV-cached, O(n) tokens)                                │
│            Committed symbols = COMPRESSED knowledge                             │
│                                                                                  │
│  ADVANTAGES:                                                                     │
│  ✓ Linear growth: text context O(n) vs visual history O(n × p²)                │
│  ✓ Knowledge accumulation: each symbol COMMITTED to symbolic form              │
│  ✓ Trajectory captured: sequence of [ADD, REPLACE, DELETE] operations          │
│  ✓ O(1) edits: REPLACE only recomputes current stroke + [INT]                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

| Principle | Implementation | Benefit |
|-----------|---------------|---------|
| **Symbolic Commitment** | Recognized strokes → text tokens | Memory: O(n) vs O(n×p²) |
| **Sparse Attention** | Causal on text, isolated patches | Compute: 60-70% savings |
| **Dual-Token Separation** | [CLS]=visual, [INT]=semantic | Clean feature spaces |
| **Incremental KV-Cache** | Text cached, patches ephemeral | O(1) edit operations |
| **Trajectory Logging** | Edit operations as first-class | Reasoning path capture |

### Reasoning Trajectory & Knowledge Graph Alignment

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    BEYOND OCR: REASONING TRAJECTORY CAPTURE                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Traditional OCR captures:     Context Tracker captures:                        │
│  ═════════════════════════     ═════════════════════════                        │
│                                                                                  │
│  • Final expression            • Final expression                               │
│    "x² + 2x + 1 = 0"            "x² + 2x + 1 = 0"                              │
│                                                                                  │
│                                • HOW it was constructed:                        │
│                                  t=0: ADD "x"                                   │
│                                  t=1: ADD "²"     ──┐                           │
│                                  t=2: ADD "+"       │ Trajectory                │
│                                  t=3: ADD "2x"      │ embedding                 │
│                                  t=4: PAUSE 3s    ──┤                           │
│                                  t=5: ADD "+1"      │ → Knowledge               │
│                                  t=6: ADD "=0"      │    graph node             │
│                                  t=7: DELETE "=0" ──┘    alignment              │
│                                  t=8: WRAP "(...)²"                             │
│                                                                                  │
│  KNOWLEDGE GRAPH MODELING:                                                      │
│  ═════════════════════════                                                      │
│                                                                                  │
│  Edit trajectory ──────► Trajectory encoder ──────► KG node prediction         │
│                                │                                                │
│                                ▼                                                │
│                    ┌─────────────────────────┐                                  │
│                    │  "quadratic_equation"   │                                  │
│                    │         │               │                                  │
│                    │    ┌────┴────┐          │                                  │
│                    │    ▼         ▼          │                                  │
│                    │ "factoring" "formula"   │                                  │
│                    │    │                    │                                  │
│                    │    ▼                    │                                  │
│                    │ "(x+1)²=0" ◄── user's trajectory suggests this path       │
│                    └─────────────────────────┘                                  │
│                                                                                  │
│  APPLICATIONS:                                                                  │
│  • Tutoring hints (without giving answers)                                      │
│  • Misconception detection (from hesitation/correction patterns)                │
│  • Learning analytics (problem-solving strategy identification)                 │
│  • Collaborative math (real-time shared derivations)                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Attention Pattern Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION DESIGN: WHY SPARSE + DUAL-TOKEN?                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FULL SELF-ATTENTION (BERT-style VLM):                                          │
│  ═════════════════════════════════════                                          │
│                                                                                  │
│       Every token attends to every other token                                  │
│       ┌─────────────────────────────────────┐                                   │
│       │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │   Cost: O(N²)                       │
│       │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │   N = text + patches                 │
│       │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │                                      │
│       │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │   Problem:                           │
│       │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │   • Visual features DILUTED         │
│       │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ │     by attending to text            │
│       └─────────────────────────────────────┘   • No KV-cache on edits         │
│                                                                                  │
│  OUR SPARSE + DUAL-TOKEN DESIGN:                                                │
│  ═══════════════════════════════                                                │
│                                                                                  │
│       Structured sparsity with role separation                                  │
│       ┌─────────────────────────────────────┐                                   │
│       │ ■ ■ ■ · · · · · · · · · │ text     │   Text: causal (O(n²/2))          │
│       │ ■ ■ ■ · · · · · · · · · │ context  │                                    │
│       │ ■ ■ ■ · · · · · · · · · │          │   Patches: isolated group          │
│       │ · · · ■ ■ ■ · · · · · · │ patches  │   (O(p²), no text dilution)        │
│       │ · · · ■ ■ ■ · · · · · · │          │                                    │
│       │ · · · ■ ■ ■ · · · · · · │          │   [CLS]: patches only              │
│       │ · · · ■ ■ ■ ■ · · · · · │ [CLS]    │   (pure visual aggregation)        │
│       │ ■ ■ ■ · · · ■ ■ · · · · │ [INT]    │                                    │
│       │ ■ ■ ■ · · · ■ ■ ■ · · · │ decode   │   [INT]: text + [CLS]              │
│       └─────────────────────────────────────┘   (semantic reasoning)            │
│                                                                                  │
│       Cost: O(n²/2 + p² + n) ≈ 35% of full attention                           │
│       Benefit: Visual purity + O(1) edit capability                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

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

