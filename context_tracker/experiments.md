# Experiments: Real-Time Stroke-Level LaTeX Editing

## Overview

This document outlines experimental designs for a **real-time LaTeX editing system** that processes handwritten input at the **stroke level** (pen-down to pen-up). The architecture uses a **multimodal sequence** combining LaTeX text context with image patches, processed through a **dual-token ([CLS] + [INT])** design with **autoregressive decoding**.

---

## 0. Related Work & Novelty Analysis

### 0.1 What's Well-Studied (Existing)

| Component | Status | References |
|-----------|--------|------------|
| **Patch → [CLS] → Classification** | ✅ Standard | ViT (Dosovitskiy 2020) |
| **Image → Encoder → Decoder → Text** | ✅ Standard | Show-Attend-Tell, TrOCR |
| **Full image → LaTeX (offline HMER)** | ✅ Well-studied | TAMER, CoMER, BTTR, WAP, PosFormer |
| **Online ink → LaTeX (coordinate-based)** | ✅ Studied | GRU encoder-decoder (2017), InkSight |
| **Commercial real-time math OCR** | ✅ Exists | MyScript Math, Mathpix, Equatio, MathBrush |

### 0.2 Key Existing Approaches

**Offline HMER** (Full image processing):
- TAMER (AAAI 2025): Tree-aware transformer, processes complete expression image
- CoMER (ECCV 2022): Coverage modeling for attention
- PosFormer (2024): Position forest for structure parsing

**Online HMER** (Stroke sequence processing):
- GRU-based: Encodes (x, y, t) stroke coordinates → decoder → LaTeX
- InkSight (2024): Ink tokenizer for converting strokes to discrete tokens

**Commercial Products**:
- MyScript Math: Real-time recognition with gesture editing (scratch-to-erase)
- Mathpix Digital Ink: Live rendering, scribble-to-erase
- MathBrush: Interactive recognition and manipulation

### 0.3 What's Novel in Our Proposal

| Novel Aspect | Why It's Different | Existing Gap |
|--------------|-------------------|--------------|
| **Parallel 2D Conv along trajectory** | Extract conv tokens along stroke path with (x,y) positions; not bounding box crops | Efficient, preserves spatial info, no wasted background |
| **Multimodal (LaTeX + Conv tokens)** | Existing HMER is image-only; we combine text context + stroke tokens | Clean symbolic context + visual recognition |
| **Dual [CLS] + [INT] architecture** | Novel separation of local vs global reasoning | Solves cascade problem elegantly |
| **Autoregressive edit decoding** | Generate edit tokens from [INT] with KV-cache | Efficient for short LaTeX edits (1-5 tokens) |
| **Explicit ADD/REPLACE/INSERT** | Commercial products have editing but architecture is proprietary | Open, documented mechanism |
| **2D RoPE for edit localization** | RoPE used for 1D sequences; 2D RoPE for spatial math editing is new | Position-aware editing |
| **Per-stroke combinatorial augmentation** | Mix stroke variations for exponential diversity from small manual effort | Efficient data generation |

### 0.4 Comparison with Closest Work

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    APPROACH COMPARISON                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Offline HMER (TAMER, CoMER):                                               │
│  ┌────────────────┐    ┌──────────┐    ┌────────────┐                       │
│  │ Full Image     │ →  │ Encoder  │ →  │ Decoder    │ → LaTeX               │
│  │ (complete)     │    │ (CNN/ViT)│    │ (one-shot) │                       │
│  └────────────────┘    └──────────┘    └────────────┘                       │
│  ❌ No incremental editing                                                   │
│                                                                              │
│  Online HMER (GRU-based):                                                   │
│  ┌────────────────┐    ┌──────────┐    ┌────────────┐                       │
│  │ Stroke coords  │ →  │ GRU      │ →  │ Decoder    │ → LaTeX               │
│  │ (x, y, t)      │    │ Encoder  │    │ (one-shot) │                       │
│  └────────────────┘    └──────────┘    └────────────┘                       │
│  ❌ No visual patches, ❌ No editing                                         │
│                                                                              │
│  Commercial (MyScript, Mathpix):                                            │
│  ┌────────────────┐    ┌──────────┐    ┌────────────┐                       │
│  │ Strokes        │ →  │ ???      │ →  │ ???        │ → LaTeX + Edit        │
│  │ (proprietary)  │    │ (black   │    │ (black     │                       │
│  └────────────────┘    │  box)    │    │  box)      │                       │
│                        └──────────┘    └────────────┘                       │
│  ✅ Has editing, ❌ Proprietary architecture                                 │
│                                                                              │
│  OURS (Multimodal + Dual-Token + AR):                                       │
│  ┌────────────────┐    ┌──────────────────────┐    ┌────────────┐           │
│  │ LaTeX Context  │ +  │ Stroke PATCHES       │ →  │ [CLS]+[INT]│ → AR     │
│  │ (text tokens)  │    │ (current stroke)     │    │ (dual-tok) │   decode │
│  └────────────────┘    └──────────────────────┘    └────────────┘           │
│  ✅ Clean context, ✅ Visual recognition, ✅ Efficient AR decoding           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.5 Research Questions

Given the existing work, our key research questions are:

1. **Does patch embedding outperform coordinate encoding for stroke-level recognition?**
   - Hypothesis: Patches capture visual shape (loops, curves) better than (x,y) trajectories

2. **Does multimodal (LaTeX text + patches) outperform image-only approaches?**
   - Hypothesis: Text tokens provide clean symbolic context, avoiding visual contamination

3. **Can the dual [CLS]+[INT] architecture match single-model accuracy while enabling O(1) edits?**
   - Hypothesis: [CLS] handles local recognition, [INT] handles global reasoning — cleanly separated

4. **Is autoregressive decoding efficient enough for real-time LaTeX editing?**
   - Hypothesis: Most edits are 1-5 tokens, so AR overhead is minimal compared to NAR/diffusion

5. **Does 2D RoPE improve edit localization over learned embeddings?**
   - Hypothesis: Explicit spatial encoding helps with superscript vs. adjacent ambiguity

---

## 0.6 Input/Output Token Sequence Format

Before diving into architecture details, here's the concrete token format:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    INPUT/OUTPUT TOKEN SEQUENCE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT SEQUENCE:                                                                 │
│  ═══════════════                                                                 │
│                                                                                  │
│  ┌───────────────────┬─────┬─────────────────────┬─────┬─────┬──────────────┐  │
│  │   LaTeX Context   │[SEP]│   Stroke Patches    │[CLS]│[INT]│  AR Decode   │  │
│  ├───────────────────┼─────┼─────────────────────┼─────┼─────┼──────────────┤  │
│  │ <x> <+> <y> <^{>  │     │ [P₀] [P₁] [P₂] [P₃]│     │     │ [e₁]...[EOS] │  │
│  │ <2> <}>           │     │                     │     │     │              │  │
│  └───────────────────┴─────┴─────────────────────┴─────┴─────┴──────────────┘  │
│         │                           │               │     │          │         │
│    Committed text              Visual patches    Local  Global   Output        │
│    (causal attn)               (current stroke)  recog  reason   tokens        │
│                                                                                  │
│  CONCRETE EXAMPLES:                                                             │
│  ══════════════════                                                             │
│                                                                                  │
│  Example 1: ADD (append superscript)                                            │
│  ────────────────────────────────────                                           │
│  Context: "x + y"     Stroke: draws "²"     Result: "x + y^{2}"                │
│                                                                                  │
│  Input:  <x> <+> <y> [SEP] [P₀][P₁][P₂][P₃] [CLS] [INT]                        │
│  Output: [ADD] <^{> <2> <}> [EOS]                                               │
│                                                                                  │
│                                                                                  │
│  Example 2: REPLACE (correct symbol)                                            │
│  ───────────────────────────────────                                            │
│  Context: "x^{2}"     Stroke: draws "3" over "2"     Result: "x^{3}"           │
│                                                                                  │
│  Input:  <x> <^{> <2> <}> [SEP] [P₀][P₁][P₂] [CLS] [INT]                       │
│  Output: [REPLACE] [pos=2] <3> [EOS]                                            │
│                                                                                  │
│                                                                                  │
│  Example 3: WAIT (incomplete stroke)                                            │
│  ───────────────────────────────────                                            │
│  Context: "x + "      Stroke: 2/3 strokes of "α"     Result: wait              │
│                                                                                  │
│  Input:  <x> <+> [SEP] [P₀][P₁] [CLS] [INT]    (partial α)                     │
│  Output: [WAIT]                                                                  │
│                                                                                  │
│                                                                                  │
│  Example 4: INSERT (add in middle)                                              │
│  ─────────────────────────────────                                              │
│  Context: "xy"        Stroke: draws "+" between     Result: "x + y"            │
│                                                                                  │
│  Input:  <x> <y> [SEP] [P₀][P₁][P₂] [CLS] [INT]                                │
│  Output: [INSERT] [pos=1] <+> [EOS]                                             │
│                                                                                  │
│                                                                                  │
│  Example 5: WRAP (structural change)                                            │
│  ───────────────────────────────────                                            │
│  Context: "x + 1"     Stroke: draws "─" below (fraction line)                  │
│                                                                                  │
│  Input:  <x> <+> <1> [SEP] [P₀][P₁]...[Pₙ] [CLS] [INT]                         │
│  Output: [WRAP] <\frac{> [COPY_PREV] <}{}> [EOS]                                │
│          → Result: "\frac{x + 1}{}"                                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Token Vocabulary Summary

| Category | Examples | Count |
|----------|----------|-------|
| **Special** | `[SEP]`, `[CLS]`, `[INT]`, `[EOS]`, `[PAD]` | ~10 |
| **Operations** | `[ADD]`, `[REPLACE]`, `[INSERT]`, `[WRAP]`, `[WAIT]` | ~8 |
| **Position** | `[pos=0]`...`[pos=255]` | 256 |
| **LaTeX symbols** | `<x>`, `<\frac>`, `<\alpha>`, `<^{>`, `<}>` | ~800 |
| **Total** | | ~1074 |

---

## 1. Core Architecture Design

### 1.1 Multimodal Sequence Architecture (Primary Design)

The architecture combines **LaTeX text context** (what's already written) with **image patches** (current stroke), processed through **dual tokens ([CLS] + [INT])** with **autoregressive decoding**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL AR ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Sequence:                                                             │
│  ┌─────────────────┬───────────────────┬───────┬───────┬───────────────────┐│
│  │ LaTeX Context   │ Image Patches     │ [CLS] │ [INT] │ AR Edit Tokens    ││
│  │ "x" "+" "y"     │ [P'₁][P'₂]..[P'ₖ] │       │       │ [edit₁]...[EOS]   ││
│  │ (text tokens)   │ (current stroke)  │(local)│(global│ (output)          ││
│  └─────────────────┴───────────────────┴───────┴───────┴───────────────────┘│
│                                                                              │
│  Token Roles:                                                                │
│  ═══════════                                                                 │
│  • LaTeX tokens: Clean symbolic context (no visual contamination)           │
│  • Patches: Visual features of the NEW stroke being drawn                   │
│  • [CLS]: Aggregates patch features → local symbol recognition              │
│  • [INT]: Aggregates LaTeX + [CLS] → global reasoning + edit decision       │
│  • Edit tokens: Generated autoregressively from [INT]                       │
│                                                                              │
│  Decoding:                                                                   │
│  ═════════                                                                   │
│  [INT] → [edit₁] → [edit₂] → ... → [EOS]   (autoregressive)                │
│     │       │         │                                                      │
│     └───────┴─────────┴── Each token sees all previous via causal mask     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Autoregressive Decoding?

**Comparison with alternatives:**

| Method | Forward Passes | For 5-token edit | Issue for LaTeX |
|--------|---------------|------------------|-----------------|
| **Autoregressive (AR)** | T tokens → T passes | ~5 (with KV-cache) | ✅ Fast, syntactically correct |
| **Masked LM (CMLM)** | K iterations | 5-10 passes | ⚠️ Must predict mask positions first |
| **Diffusion LM** | N denoising steps | 50-100 passes | ❌ Too slow, weak syntax guarantees |

**AR advantages for LaTeX editing:**
1. **Short outputs**: Most edits are 1-5 tokens → AR overhead is minimal
2. **Syntax guarantees**: Left-to-right ensures matching braces, proper nesting
3. **KV-cache efficiency**: Previous tokens cached, only new token computed
4. **Streaming**: Can show partial results immediately

### 1.3 Attention Mask Design

```
Custom attention mask for dual-token + AR:

           L₀  L₁  L₂  P'₁ P'₂ P'₃ CLS INT ed₁ ed₂ EOS
    L₀   [  1   0   0   0   0   0   0   0   0   0   0  ]  (LaTeX causal)
    L₁   [  1   1   0   0   0   0   0   0   0   0   0  ]
    L₂   [  1   1   1   0   0   0   0   0   0   0   0  ]
    P'₁  [  0   0   0   1   0   0   0   0   0   0   0  ]  (patches isolated)
    P'₂  [  0   0   0   1   1   0   0   0   0   0   0  ]
    P'₃  [  0   0   0   1   1   1   0   0   0   0   0  ]
    CLS  [  0   0   0   1   1   1   1   0   0   0   0  ]  ← patches only
    INT  [  1   1   1   0   0   0   1   1   0   0   0  ]  ← LaTeX + CLS
    ed₁  [  1   1   1   0   0   0   1   1   1   0   0  ]  ← all context + AR
    ed₂  [  1   1   1   0   0   0   1   1   1   1   0  ]
    EOS  [  1   1   1   0   0   0   1   1   1   1   1  ]

Key design choices:
• [CLS] only sees patches → pure local symbol recognition
• [INT] sees LaTeX + [CLS] → global reasoning
• Edit tokens see everything + causal among themselves → coherent output
```

### 1.4 Stroke Input Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STROKE INPUT STAGE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Pen Press ────► Stroke Drawing ────► Pen Lift                             │
│       │                                    │                                 │
│       │         ┌──────────────────┐       │                                 │
│       └────────►│ Stroke Points    │◄──────┘                                 │
│                 │ (x,y) trajectory │                                         │
│                 └────────┬─────────┘                                         │
│                          ▼                                                   │
│                 ┌──────────────────┐                                         │
│                 │ Sample Along     │  Every N pixels along trajectory       │
│                 │ Trajectory       │                                         │
│                 └────────┬─────────┘                                         │
│                          ▼                                                   │
│                 ┌──────────────────┐                                         │
│                 │ Parallel 2D Conv │  Extract local patch at each point     │
│                 │ [t₁,pos₁]...[tₘ,posₘ]│  Conv tokens WITH positions       │
│                 └────────┬─────────┘                                         │
│                          ▼                                                   │
│                 ┌──────────────────┐                                         │
│                 │ Concat with      │                                         │
│                 │ LaTeX context    │                                         │
│                 └────────┬─────────┘                                         │
│                          ▼                                                   │
│                 ┌──────────────────┐                                         │
│                 │ [CLS] + [INT]    │  Dual-token processing                  │
│                 │ + 2D RoPE        │  Position-aware attention               │
│                 │ + AR decode      │                                         │
│                 └────────┬─────────┘                                         │
│                          ▼                                                   │
│                 ┌──────────────────┐                                         │
│                 │ Edit tokens      │  e.g., "^{" "2" "}"                     │
│                 │ + operation type │  e.g., ADD / REPLACE / INSERT           │
│                 └──────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design**: Instead of bounding box crops divided into ViT patches, we extract **conv tokens along the stroke trajectory**. Each token has (x,y) position for 2D RoPE attention. See `data/data_augmentation.md` for details.

### 1.5 Complete Example: Adding "²" to "x + y"

```
Step-by-step through the architecture:

1. Current LaTeX context: "x + y"
   Tokenized: ["x", "+", "y"]

2. User draws "²" above and to the right of "y"
   → Stroke cropped → Patch embedding: [P'₁, P'₂, P'₃, P'₄]

3. Full input sequence:
   ["x", "+", "y"] | [P'₁, P'₂, P'₃, P'₄] | [CLS] | [INT]
       text              patches           local  global

4. Attention computation:
   • [CLS] attends to patches → recognizes "²" symbol
   • [INT] attends to ["x", "+", "y", CLS] → understands:
     - Current expression structure
     - New symbol is "²" 
     - Position suggests superscript on "y"

5. Autoregressive decoding from [INT]:
   [INT] → [ADD] → "^{" → "2" → "}" → [EOS]
   
6. Result: "x + y^{2}"
```

---

## 2. Autoregressive Output Format

### 2.1 Edit Token Sequence

The model outputs edit tokens autoregressively from [INT]:

```
Output sequence: [OP_TYPE] [position?] [tex₁] [tex₂] ... [EOS]

Examples:
  [ADD] "y" [EOS]                    → append "y" to expression
  [ADD] "^{" "2" "}" [EOS]           → append superscript
  [REPLACE] [pos=3] "3" [EOS]        → replace token at position 3
  [INSERT] [pos=1] "+" [EOS]         → insert "+" after position 1
```

### 2.2 Operation Types

| Operation | When Used | Position Token | Example |
|-----------|-----------|----------------|---------|
| **ADD** | New symbol at end | Not needed | `x` → `x + 1` |
| **REPLACE** | Overwrite existing | Required | `x^2` → `x^3` |
| **INSERT** | Insert in middle | Required | `xy` → `x+y` |

### 2.3 Position Encoding with 2D RoPE

For REPLACE/INSERT, the model must identify which token to modify:

```python
# Position prediction via attention
# [INT] token's attention weights over LaTeX context reveal edit position

attention_weights = softmax(INT_query @ LaTeX_keys.T)
# High attention to token k → REPLACE/INSERT at position k

# 2D RoPE helps disambiguate spatial positions
# e.g., "2" drawn above vs. beside "x" → superscript vs. coefficient
```

### 2.4 KV-Cache for Efficient Decoding

```
Decoding step-by-step:

Step 0: [LaTeX...] [Patches...] [CLS] [INT]
        └────────── KV-cache ──────────┘
                                        ↓
Step 1: [INT] → [ADD]                  (1 new token computed)
        └── cached ──┘
                      ↓
Step 2: [ADD] → "^{"                   (1 new token computed)
        └─ cached ─┘
                    ↓
Step 3: "^{" → "2"                     (1 new token computed)
        ...

Total: 5 passes for 5 output tokens (with KV-cache reuse)
```

---

## 3. Dual-Token Design: [CLS] + [INT]

### 3.1 Why Two Tokens?

| Token | Attends To | Purpose |
|-------|------------|---------|
| **[CLS]** | Only current patches | Local symbol recognition |
| **[INT]** | LaTeX context + [CLS] | Global reasoning + edit decision |

**Key Benefit**: Clean separation enables O(1) updates on REPLACE:
- [CLS] is independent of other strokes → no cascade
- Only [INT] needs recomputation (single token)

### 3.2 Attention Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DUAL-TOKEN ATTENTION ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LaTeX Context:        Patches:           [CLS]     [INT]                   │
│  "x" "+" "y"          [P'₁][P'₂][P'₃]       │         │                     │
│     │   │   │            │    │    │        │         │                     │
│     └───┴───┴────────────┼────┼────┼────────┤         │                     │
│         (causal)         │    │    │        │         │                     │
│                          └────┴────┘        │         │                     │
│                          (within patches)   │         │                     │
│                                   ┌─────────┘         │                     │
│                                   │                   │                     │
│                               [CLS] sees              │                     │
│                               patches only            │                     │
│                                   │                   │                     │
│           ┌───────────────────────┼───────────────────┘                     │
│           │                       │                                         │
│        [INT] sees:             [CLS]                                        │
│        • LaTeX context          output                                      │
│        • [CLS] feature           "²"                                        │
│                                                                              │
│        [INT] output → [ADD] "^{" "2" "}" [EOS]                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Why This Solves the Cascade Problem

In a **single-token** architecture where each [CLS] attends to previous [CLS] tokens:
```
REPLACE CLS₃ → CLS₄, CLS₅ have stale KV-cache → O(n) re-encoding needed
```

In our **dual-token** architecture:
```
REPLACE: Just update LaTeX context tokens + recompute [INT]
         [CLS] tokens are never chained → no cascade!
         Cost: O(1) for [INT] recomputation
```

### 3.4 Edit Operations

#### 3.4.1 ADD Operation

```
Before: LaTeX context = "x + y"
Stroke: User draws "²" as superscript

Output: [ADD] "^{" "2" "}" [EOS]

After: LaTeX context = "x + y^{2}"
```

#### 3.4.2 REPLACE Operation

```
Before: LaTeX context = "x^{2}"
Stroke: User draws "3" over the "2"

Output: [REPLACE] [pos=3] "3" [EOS]

After: LaTeX context = "x^{3}"
       (token at position 3 changed from "2" to "3")
```

#### 3.4.3 INSERT Operation

```
Before: LaTeX context = "xy"
Stroke: User draws "+" between x and y

Output: [INSERT] [pos=1] "+" [EOS]

After: LaTeX context = "x + y"
       (token inserted after position 1)
```

### 3.5 Extended Use Cases for [INT] Token

#### 1. Edit Operation Decision
```
[INT] analyzes LaTeX context + [CLS] →
  - What symbol is being drawn? (from [CLS])
  - Where does it fit in the expression? (from context + 2D position)
  - Is it adding new content or correcting existing?
  
Output: [ADD] / [REPLACE pos=N] / [INSERT pos=N]
```

#### 2. Integration with Reasoning Models (Optional)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REASONING-ENHANCED ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User's handwritten strokes → [CLS₁][CLS₂][CLS₃]... (symbol recognition)    │
│                                        │                                     │
│                                        ▼                                     │
│                                     [INT]                                    │
│                                        │                                     │
│                    ┌───────────────────┼───────────────────┐                │
│                    │                   │                   │                │
│                    ▼                   ▼                   ▼                │
│            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│            │ Edit Head   │    │ Hint Head   │    │ Trajectory  │            │
│            │             │    │             │    │ Analyzer    │            │
│            │ ADD/REPLACE │    │ Next step?  │    │             │            │
│            │ INSERT      │    │ Common      │    │ Learning    │            │
│            │             │    │ mistakes?   │    │ patterns    │            │
│            └─────────────┘    └──────┬──────┘    └─────────────┘            │
│                                      │                                       │
│                                      ▼                                       │
│                    ┌─────────────────────────────────────┐                  │
│                    │ DeepSeek IMO / Math Reasoning Model │                  │
│                    │                                     │                  │
│                    │ Input: [INT] embedding + context    │                  │
│                    │                                     │                  │
│                    │ Output:                             │                  │
│                    │ - "Consider: what if x = 0?"        │                  │
│                    │ - "You might want to factor..."     │                  │
│                    │ - "Check the sign of the exponent"  │                  │
│                    │                                     │                  │
│                    │ (Hints WITHOUT giving solution!)    │                  │
│                    └─────────────────────────────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 2D RoPE for Spatial Position Encoding

Use 2D Rotary Position Embedding to help disambiguate spatial positions (e.g., superscript vs. adjacent):

```python
class RoPE2D:
    """
    2D Rotary Position Embedding for spatial positioning.
    Helps distinguish: x² (superscript) vs. x2 (adjacent)
    """
    
    def __init__(self, dim, max_freq=10000):
        self.dim = dim
        self.max_freq = max_freq
        
    def forward(self, x_pos, y_pos, features):
        # Split dim for x and y
        half_dim = self.dim // 2
        
        # Apply rotary encoding separately for x and y
        x_encoded = self.apply_rope(x_pos, features[..., :half_dim])
        y_encoded = self.apply_rope(y_pos, features[..., half_dim:])
        
        return torch.cat([x_encoded, y_encoded], dim=-1)
```

**Use Cases**:
- Distinguish superscript (y offset) from coefficient (x offset)
- Detect REPLACE when stroke overlaps existing symbol position
- Understand nested structures (fraction numerator vs. denominator)

---

## 4. Difficult Cases: Multi-Token Output Scenarios

### 4.1 Case 1: Adding a Horizontal Line (Fraction Creation)

**Scenario**: User has written "x+1" and draws a horizontal line underneath.

```
Before LaTeX: x + 1
Visual:       x + 1
              
After drawing line:
Visual:       x + 1
              ─────
              
Required LaTeX: \frac{x+1}{}
```

**Challenge**: Single stroke must output 4+ tokens: `\frac`, `{`, `x+1`, `}`, `{`, `}`

**Solution**: Multi-token decoder with structure-aware attention
```
Input stroke (horizontal line) →
  Cross-attend to existing [CLS] tokens →
  Detect "fraction creation" pattern →
  Output: [\frac, {, <COPY_PREV>, }, {, }]
```

### 4.2 Case 2: Adding Exponent to Complex Expression

**Scenario**: Expression "x+1" → user adds "2" as superscript.

```
Before: x + 1
After:  (x + 1)²

Before LaTeX: x + 1
After LaTeX:  (x + 1)^{2}
              ↑     ↑ ↑↑↑
              New tokens added
```

**Challenge**: Adding "²" requires wrapping existing expression with parentheses.

**Token Changes**:
```
Before chunks: [x] [+] [1]
After chunks:  [(] [x] [+] [1] [)] [^] [{] [2] [}]
               ↑                 ↑   New chunks
```

### 4.3 Case 3: Converting to Subscript/Superscript Stack

**Scenario**: Change "x₁" to "x₁²" (subscript + superscript).

```
Before: x_1        → LaTeX: x_{1}
After:  x₁² (both) → LaTeX: x_{1}^{2}
```

**Challenge**: Structural reorganization required.

---

### 4.3.1 Architecture for Subscript/Superscript Editing (Cropped Patch Approach)

**Key Insight**: For spatial edits like subscripts/superscripts, we crop around the **editing area** rather than the entire canvas.

#### Cropping Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROPPED PATCH FOR SUB/SUPERSCRIPT                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Original Canvas:                                                    │
│  ┌──────────────────────────────────────┐                           │
│  │                                      │                           │
│  │           x + y                      │                           │
│  │              ↑                       │                           │
│  │     User adds "²" here               │                           │
│  │                                      │                           │
│  └──────────────────────────────────────┘                           │
│                                                                      │
│  Cropped Edit Region (localized patch):                             │
│  ┌─────────┐                                                        │
│  │  y  ²   │  ← Small patch around edit area                        │
│  │    ↑    │    Contains: base symbol + new stroke                  │
│  └─────────┘                                                        │
│                                                                      │
│  Why crop?                                                          │
│  • Reduces computation (small patch vs full image)                  │
│  • Provides spatial context (see "y" to know it's superscript)      │
│  • CNN focuses on relevant region                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Architecture: Multimodal Sequence with Cropped Patches

For subscript/superscript edits, we use the **multimodal** approach where:
- LaTeX context is provided as text tokens (clean, no contamination)
- Cropped patches around the edit region provide visual features
- [CLS] aggregates patches, [INT] reasons globally, then AR decodes

```
┌─────────────────────────────────────────────────────────────────────┐
│              MULTIMODAL SEQUENCE FOR INCREMENTAL EDITING             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Current LaTeX: "x + y"                                              │
│  User draws: "²" above "y"                                           │
│                                                                      │
│  Input Sequence:                                                     │
│  ┌─────────────────┬───────────────────┬───────┬───────┬───────────┐│
│  │ LaTeX tokens    │ Image patches     │ [CLS] │ [INT] │ AR output ││
│  │ "x" "+" "y"     │ [P'₁][P'₂]..[P'ₖ] │       │       │→ edit tok ││
│  │ (KV-cached)     │ (current stroke)  │(local)│(global)│          ││
│  └─────────────────┴───────────────────┴───────┴───────┴───────────┘│
│                                                                      │
│  Processing:                                                         │
│  1. Retrieve KV-cache for ["x", "+", "y"]                           │
│  2. CNN(cropped_patch) → [P'₁...P'ₖ] (patches around "y²")          │
│  3. [CLS] attends to patches → recognizes "²"                       │
│  4. [INT] attends to text + [CLS] → understands "add superscript"   │
│  5. AR decode from [INT] → [ADD] "^{" "2" "}" [EOS]                 │
│                                                                      │
│  Output:                                                             │
│  - Operation: ADD                                                    │
│  - New LaTeX: "x + y^{2}"                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Cropping Strategy for Spatial Edits

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROPPED PATCH FOR SUB/SUPERSCRIPT                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Original Canvas:                                                    │
│  ┌──────────────────────────────────────┐                           │
│  │                                      │                           │
│  │           x + y                      │                           │
│  │              ↑                       │                           │
│  │     User adds "²" here               │                           │
│  │                                      │                           │
│  └──────────────────────────────────────┘                           │
│                                                                      │
│  Cropped Edit Region:                                               │
│  ┌─────────┐                                                        │
│  │  y  ²   │  ← Small patch around edit area                        │
│  └─────────┘    (may include neighboring symbols)                   │
│                                                                      │
│  But we use MULTIMODAL: text gives clean context,                   │
│  patches just for visual stroke recognition                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Why Multimodal Is Better Here

| Challenge | Image-Only | Multimodal (Ours) |
|-----------|------------|-------------------|
| **Visual contamination** | Cropped patch includes "y" → confusing | Text has clean "x + y" |
| **Context precision** | [CLS] chain may accumulate errors | Text tokens are exact |
| **Spatial reasoning** | Must infer from patch positions | 2D RoPE on patches + text position |

### 4.4 Case 4: Matrix Row Addition

**Scenario**: Adding a row to existing matrix.

```
Before: \begin{pmatrix} a & b \end{pmatrix}

After:  \begin{pmatrix} a & b \\ c & d \end{pmatrix}
        User draws new row underneath
```

**Challenge**: Must insert `\\`, then new elements, then close properly.

### 4.5 Case 5: Equation System Creation

**Scenario**: User draws second line in an equation.

```
Before: x + y = 5
After:  \begin{cases} x + y = 5 \\ x - y = 1 \end{cases}
```

**Challenge**: Single stroke triggers complete structural transformation.

---

## 5. Experiment Design

### 5.1 Experiment 1: [CLS] Aggregation Quality

**Goal**: Evaluate how well [CLS] tokens capture symbol semantics.

**Architecture**: `[CLS₁][CLS₂]...[P'₁]...[P'ₘ][CLS_new]` (CLS at END + history)

**Setup**:
1. Train single-stroke symbol recognizer with [CLS] at END
2. Compare [CLS] representations vs. mean-pooled patches
3. Metric: Symbol classification accuracy

**Ablations**:
- Number of self-attention layers (1, 2, 4, 6)
- Patch size (8×8, 16×16, 32×32)
- [CLS] embedding dimension (256, 512, 768)

### 5.2 Experiment 2: Attention Pattern Analysis

**Goal**: Visualize [INT] attention patterns to understand edit decisions.

**Expected patterns:**
```
Case: User draws "3" to replace existing "2" in "x^{2}"

Attention from [INT] over LaTeX tokens:
┌─────────────────────────────────────────────┐
│ Token        │ Attention Weight │ Meaning   │
├─────────────────────────────────────────────┤
│ "x"          │      0.08       │ low       │
│ "^{"         │      0.12       │ low       │
│ "2"          │      0.45       │ HIGH ← target for REPLACE │
│ "}"          │      0.10       │ low       │
│ [CLS]        │      0.25       │ medium (new symbol info) │
└─────────────────────────────────────────────┘

→ High attention to "2" token suggests REPLACE at that position
→ [CLS] attention captures that new symbol is "3"
```

**Metrics**:
- Attention entropy (should be low for clear decisions)
- Position accuracy from attention weights
- Correlation between 2D proximity and attention

### 5.3 Experiment 3: Autoregressive vs NAR Decoding

**Goal**: Validate that AR decoding is efficient enough for real-time LaTeX editing.

**Comparison:**

| Method | Implementation | Metric |
|--------|---------------|--------|
| **AR (ours)** | [INT] → decode tokens one by one with KV-cache | Latency, accuracy |
| **NAR** | Predict all edit tokens in parallel | Latency, accuracy |
| **Hybrid** | Predict length first, then parallel | Latency, accuracy |

**Expected Results:**
```
Typical edit: 3-5 tokens (e.g., "^{" "2" "}")

AR with KV-cache:
  - 3-5 forward passes (each incremental)
  - Total: ~10-15ms on GPU
  
NAR:
  - 1 forward pass
  - But: may miss inter-token dependencies (mismatched braces)
```

**Hypothesis**: For short edits (< 10 tokens), AR overhead is negligible and provides better syntax guarantees.

### 5.4 Experiment 4: Multimodal vs Image-Only

**Goal**: Validate that multimodal (LaTeX text + patches) outperforms image-only approaches.

**Architectures:**

| Architecture | Context | New Symbol |
|--------------|---------|------------|
| **Image-only** | [CLS] chain from previous strokes | Current patches + [CLS] |
| **Multimodal (ours)** | LaTeX text tokens | Current patches + [CLS] |

**Expected Benefits of Multimodal:**

| Aspect | Image-Only | Multimodal |
|--------|------------|------------|
| Context quality | Fuzzy [CLS] embeddings | Exact text tokens |
| Visual contamination | Cropped patches include neighbors | Text is clean |
| KV-cache updates | Must recompute on REPLACE | Just update text tokens |
| Interpretability | Hard to debug | Can inspect text context |

**Key Questions**:
1. Does text context provide cleaner signal than image-based context?
2. Is the overhead of text encoding justified by accuracy gains?

### 5.5 Experiment 5: [CLS] Attention Scope — Isolated vs Full Context

**Goal**: Compare two attention designs for [CLS] token:
- **Option A (Dual-Token)**: [CLS] attends to patches only, [INT] handles context
- **Option B (Single-Token)**: [CLS] attends to everything (LaTeX + patches), no [INT]

**Architectures:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  OPTION A: Dual-Token ([CLS] isolated + [INT])                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Attention mask:                                                     │
│         L₀  L₁  L₂  P'₁ P'₂ P'₃ CLS INT ed₁                         │
│  L₀   [  1   0   0   0   0   0   0   0   0  ]                       │
│  L₁   [  1   1   0   0   0   0   0   0   0  ]                       │
│  L₂   [  1   1   1   0   0   0   0   0   0  ]                       │
│  P'₁  [  0   0   0   1   0   0   0   0   0  ]  ← patches isolated   │
│  P'₂  [  0   0   0   1   1   0   0   0   0  ]                       │
│  P'₃  [  0   0   0   1   1   1   0   0   0  ]                       │
│  CLS  [  0   0   0   1   1   1   1   0   0  ]  ← patches ONLY       │
│  INT  [  1   1   1   0   0   0   1   1   0  ]  ← LaTeX + CLS        │
│  ed₁  [  1   1   1   0   0   0   1   1   1  ]  ← AR decode          │
│                                                                      │
│  [CLS] = pure visual feature (richer image representation?)         │
│  [INT] = context reasoning (combines text + visual)                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  OPTION B: Single-Token ([CLS] sees all, NO [INT])                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Attention mask:                                                     │
│         L₀  L₁  L₂  P'₁ P'₂ P'₃ CLS ed₁                             │
│  L₀   [  1   0   0   0   0   0   0   0  ]                           │
│  L₁   [  1   1   0   0   0   0   0   0  ]                           │
│  L₂   [  1   1   1   0   0   0   0   0  ]                           │
│  P'₁  [  1   1   1   1   0   0   0   0  ]  ← patches see LaTeX      │
│  P'₂  [  1   1   1   1   1   0   0   0  ]                           │
│  P'₃  [  1   1   1   1   1   1   0   0  ]                           │
│  CLS  [  1   1   1   1   1   1   1   0  ]  ← [CLS] sees ALL         │
│  ed₁  [  1   1   1   1   1   1   1   1  ]  ← AR decode from CLS     │
│                                                                      │
│  [CLS] = context-aware symbol recognition                           │
│  No [INT] needed (but image features may be diluted by text)        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Trade-off Analysis:**

| Aspect | Option A (Dual) | Option B (Single) |
|--------|-----------------|-------------------|
| **Image feature richness** | ✅ [CLS] focuses purely on patches | ⚠️ Attention diluted across text+patches |
| **Context awareness** | Via [INT] (separate step) | ✅ Built into [CLS] directly |
| **Model complexity** | 2 special tokens | 1 special token |
| **Interpretability** | Clear: [CLS]=visual, [INT]=semantic | Mixed: [CLS] does both |
| **REPLACE cost** | O(1): just recompute [INT] | O(n): [CLS] depends on context |

**Hypothesis**: 
- Option A has **richer visual features** because [CLS] dedicates all attention to patches
- Option B may have **better ambiguity resolution** (e.g., "1" vs "l") if context helps
- For symbols with clear visual identity, Option A should match or beat Option B

### 5.5.1 Option C: Hybrid (Confidence-Gated Context)

A third option combines the benefits of both:

```
┌─────────────────────────────────────────────────────────────────────┐
│  OPTION C: HYBRID (Confidence-Gated Context Access)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STAGE 1: Isolated Recognition (like Option A)                      │
│  ─────────────────────────────────────────────                      │
│                                                                      │
│    [P₀][P₁][P₂][P₃] → [CLS] → softmax predictions                  │
│                                                                      │
│    Example: { "1": 0.45, "l": 0.40, "I": 0.10, ... }                │
│                                                                      │
│    Confidence = max_prob = 0.45                                     │
│                                                                      │
│    IF confidence > threshold (e.g., 0.7):                           │
│        → Output directly (skip context lookup)                      │
│    ELSE:                                                            │
│        → Proceed to Stage 2                                         │
│                                                                      │
│  STAGE 2: Context-Assisted Disambiguation (when needed)             │
│  ─────────────────────────────────────────────────────              │
│                                                                      │
│    [INT] attends to:                                                │
│      • LaTeX context: "x + y ="                                     │
│      • [CLS] visual embedding                                       │
│      • Top-k candidate tokens: ["1", "l", "I"]                      │
│                                                                      │
│    Context reasoning: "math expression → likely digit"              │
│                                                                      │
│    Output: "1" with context-boosted confidence                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

| Aspect | Option A | Option B | Option C (Hybrid) |
|--------|----------|----------|-------------------|
| Visual purity | ✅ Pure | ❌ Diluted | ✅ Pure (stage 1) |
| Context disambiguation | ❌ None | ✅ Always | ✅ When needed |
| Computational cost | Low | High | Adaptive |
| REPLACE latency | O(1) | O(n) | O(1) usually |
| Complexity | Medium | Low | High |

### 5.5.2 Visual Comparison of All Options

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION PATTERN VISUAL SUMMARY                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Legend: ■ = attends, · = masked, ▒ = conditional (hybrid only)                 │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  OPTION A: Isolated Dual-Token          OPTION B: Context-Aware Single          │
│  ───────────────────────────            ────────────────────────────            │
│                                                                                  │
│       LaTeX  Patch  CLS INT                  LaTeX  Patch  CLS                  │
│      ┌─────┬─────┬────┬────┐               ┌─────┬─────┬────┐                   │
│ LaTeX│ ■■■ │     │    │    │          LaTeX│ ■■■ │     │    │                   │
│      │ ■■■ │     │    │    │               │ ■■■ │     │    │                   │
│      ├─────┼─────┼────┼────┤               ├─────┼─────┼────┤                   │
│ Patch│     │ ■■■ │    │    │          Patch│ ■■■ │ ■■■ │    │  ← sees LaTeX!   │
│      │     │ ■■■ │    │    │               │ ■■■ │ ■■■ │    │                   │
│      ├─────┼─────┼────┼────┤               ├─────┼─────┼────┤                   │
│  CLS │     │ ■■■ │ ■  │    │           CLS │ ■■■ │ ■■■ │ ■  │  ← sees ALL      │
│      ├─────┼─────┼────┼────┤               └─────┴─────┴────┘                   │
│  INT │ ■■■ │     │ ■  │ ■  │               (no INT needed)                      │
│      └─────┴─────┴────┴────┘                                                    │
│                                                                                  │
│                                                                                  │
│  OPTION C: Hybrid (Two-Stage)                                                   │
│  ────────────────────────────                                                   │
│                                                                                  │
│  Stage 1 (always):           Stage 2 (if low confidence):                       │
│       Patch  CLS                   LaTeX  CLS  INT                              │
│      ┌─────┬────┐                 ┌─────┬────┬────┐                             │
│ Patch│ ■■■ │    │            LaTeX│ ■■■ │    │    │                             │
│      │ ■■■ │    │                 ├─────┼────┼────┤                             │
│      ├─────┼────┤              CLS│ ▒▒▒ │ ■  │    │  ← conditional access       │
│  CLS │ ■■■ │ ■  │                 ├─────┼────┼────┤                             │
│      └─────┴────┘              INT│ ■■■ │ ■  │ ■  │                             │
│   (pure visual)                   └─────┴────┴────┘                             │
│                                   (context reasoning)                           │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│  COMPUTATIONAL COST COMPARISON                                                  │
│  ─────────────────────────────                                                  │
│                                                                                  │
│                    Per Stroke              REPLACE Operation                    │
│  Option A:         O(p² + n)               O(p²) + O(1)  ← fastest edit        │
│  Option B:         O((n+p)²)               O((n+p)²)     ← full recompute      │
│  Option C:         O(p²) to O(p² + n²)     O(p²) usually ← adaptive            │
│                                                                                  │
│  where: n = context length, p = patch count                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Experimental Setup:**

```python
# Option A: Dual-token
class DualTokenModel:
    def forward(self, latex_tokens, patches):
        # LaTeX: self-attention
        latex_kv = self.encode_latex(latex_tokens)
        
        # Patches: self-attention only (isolated)
        patch_features = self.encode_patches(patches)  # no cross-attn to latex
        
        # [CLS]: attends to patches only
        cls_feature = self.cls_aggregate(patch_features)
        
        # [INT]: attends to latex + cls
        int_feature = self.int_reason(latex_kv, cls_feature)
        
        # AR decode from [INT]
        return self.decode(int_feature)

# Option B: Single-token
class SingleTokenModel:
    def forward(self, latex_tokens, patches):
        # All in one sequence with full causal attention
        sequence = concat(latex_tokens, patches, [CLS])
        
        # [CLS] sees everything
        features = self.transformer(sequence)
        cls_feature = features[-1]  # [CLS] at end
        
        # AR decode directly from [CLS]
        return self.decode(cls_feature)
```

**Metrics:**

| Metric | What it measures |
|--------|------------------|
| **Symbol accuracy (isolated)** | Test [CLS] on symbols with NO context → measures visual richness |
| **Symbol accuracy (ambiguous)** | Test on "1" vs "l" cases → measures context benefit |
| **Edit operation accuracy** | ADD/REPLACE/INSERT classification |
| **REPLACE latency** | Time to process edit at position k |
| **Attention entropy** | Lower = more focused (expected lower for Option A on patches) |

**Key Questions:**
1. Does [CLS] attending only to patches give richer image features?
2. Does Option B's context-awareness help with ambiguous symbols?
3. Is the REPLACE cost difference significant in practice?

### 5.6 Experiment 6: KV-Cache Consistency on REPLACE

**Note**: This experiment applies to **Option B (Single-Token)** from Experiment 5. Option A (Dual-Token) solves cascade by design.

**Goal**: Evaluate accuracy vs. latency trade-offs for cache update strategies.

**Setup**:
1. Create test set with REPLACE operations at various positions
2. Compare strategies A (re-encode), C (approximate), D (constrained)

**Test Scenarios**:
```
Scenario 1: Replace last chunk (pos = n-1)
  [CLS₁][CLS₂][CLS₃][CLS₄] → Replace CLS₄
  All strategies equivalent (no subsequent chunks)

Scenario 2: Replace middle chunk (pos = n/2)
  [CLS₁][CLS₂][CLS₃][CLS₄][CLS₅] → Replace CLS₃
  Strategy A: re-encode CLS₄, CLS₅
  Strategy C: just swap CLS₃
  
Scenario 3: Replace early chunk (pos = 1)
  [CLS₁][CLS₂][CLS₃][CLS₄][CLS₅] → Replace CLS₁
  Strategy A: re-encode CLS₂, CLS₃, CLS₄, CLS₅ (expensive!)
  Strategy C: just swap CLS₁ (risky?)
```

**Metrics**:
| Strategy | Latency (expected) | Accuracy (expected) |
|----------|-------------------|---------------------|
| A: Re-encode | O(n - pos) | 100% (baseline) |
| C: Approximate | O(1) | ? (measure degradation) |
| D: Constrained (N=3) | O(3) max | 100% within window |

**Key Question**: How much does accuracy degrade with Strategy C?
- If < 2% drop → use Strategy C for all edits
- If > 5% drop → use Dual-Token architecture instead!

### 5.7 Experiment 7: Context Memory Strategies

**Goal**: Compare memory vs. accuracy trade-offs for what to keep in context.

| Strategy | Memory | Accuracy (expected) |
|----------|--------|---------------------|
| Full patches | High | Baseline |
| [CLS] only | Low | -2-5% |
| Hybrid (N=3) | Medium | -1-2% |
| Hybrid (N=5) | Medium | -0.5-1% |

**Metrics**:
- Edit accuracy (correct operation type)
- LaTeX exact match after edit
- Memory usage (tokens/expression)
- Latency per stroke (ms)

### 5.8 Experiment 8: Multi-Token Decoding

**Goal**: Handle cases requiring multiple output tokens per stroke.

**Setup**:
1. Create synthetic dataset with structure-changing edits
2. Train with variable-length output sequences
3. Use special tokens: `<WRAP>`, `<COPY>`, `<STRUCT_CHANGE>`

**Test Cases**:
- Fraction creation (1 stroke → 5+ tokens)
- Exponent addition to expression (1 stroke → 6+ tokens)
- Matrix row addition (1 stroke → 4+ tokens)

### 5.9 Experiment 9: 2D RoPE Effectiveness

**Goal**: Validate spatial position encoding for edit localization.

**Setup**:
1. Compare: 2D RoPE vs. learned 2D embedding vs. concatenated (x, y) coordinates
2. Test on spatially ambiguous edits (superscript vs. adjacent symbol)

**Metrics**:
- Position localization accuracy
- Edit type classification accuracy
- Robustness to handwriting variation

### 5.10 Experiment 10: Reasoning Model Integration (DeepSeek IMO)

**Goal**: Integrate reasoning model for educational hints.

**Setup**:
1. Fine-tune DeepSeek IMO model to accept [INT] embeddings as input
2. Train hint generation: provide guidance WITHOUT solutions
3. Evaluate on math problem-solving sessions

**Tasks**:
```
Task A: Hint Quality
  Input: User writing "x² + 2x + 1 = 0"
  [INT] embedding captures: "completing the square" vs "quadratic formula" trajectory
  
  Good hint: "This expression looks factorable..."
  Bad hint: "The answer is x = -1"

Task B: Misconception Detection
  Input: User writes "√(a² + b²) = a + b"
  [INT] should flag: common misconception about square roots
  
  Hint: "Try plugging in a=3, b=4. Does it work?"

Task C: Trajectory-Aware Feedback
  Input: User's stroke sequence reveals hesitation/backtracking
  [INT] captures: "user unsure about next step"
  
  Hint: "What property of exponents might help here?"
```

**Metrics**:
- Hint helpfulness (user study)
- Solution leakage rate (should be ~0%)
- Learning outcome improvement

### 5.11 Experiment 11: Edge Device Deployment

**Goal**: Profile real-time performance on mobile/tablet.

**Target Devices**:
- iPad Pro (M2)
- Android tablet (Snapdragon 8 Gen 2)
- Raspberry Pi 4 (edge case)

**Metrics**:
- Latency: pen-lift to LaTeX render (target: < 100ms)
- Memory footprint (target: < 500MB)
- Battery consumption per hour of use

---

## 6. Implementation Plan

### Phase 1: Single-Stroke Recognition (Week 1-2)
- [ ] Implement patch embedding for stroke regions
- [ ] Train [CLS] token aggregation
- [ ] Single symbol → LaTeX token mapping
- [ ] Benchmark against CROHME symbols

### Phase 2: Incremental Editing (Week 3-4)
- [ ] Implement chunk-based storage
- [ ] Cross-attention between [CLS] and new patches
- [ ] ADD/REPLACE/INSERT classification head
- [ ] 2D RoPE position encoding

### Phase 3: Multi-Token Scenarios (Week 5-6)
- [ ] Variable-length decoder for structure changes
- [ ] Synthetic data generation for edge cases
- [ ] Handle fraction, matrix, equation system creation

### Phase 4: Optimization & Deployment (Week 7-8)
- [ ] Quantization (INT8, FP16)
- [ ] ONNX export for cross-platform deployment
- [ ] Mobile SDK integration (Core ML, TensorFlow Lite)
- [ ] User study with handwriting input

---

## 7. Evaluation Datasets

### 7.1 Existing Datasets with Stroke Sequences ✅

**Great news**: Multiple datasets have **InkML format** with full stroke-level data!

| Dataset | Size | Has Strokes | Symbol Grouping | Timestamps |
|---------|------|-------------|-----------------|------------|
| **CROHME 2014/16/19** | 12K+ | ✅ InkML | ✅ traceGroup | ✅ (x,y,t) |
| **MathWriting** | 230K human + 400K synthetic | ✅ InkML | ✅ | ✅ (x,y,t) |
| **HME100K** | 100K | ❌ Images only | ❌ | ❌ |

#### InkML Format Structure (CROHME/MathWriting)

```xml
<ink xmlns="http://www.w3.org/2003/InkML">
  <!-- Individual strokes with coordinates and timestamps -->
  <trace id="t1">10 20 0, 15 25 10, 20 30 20, ...</trace>  <!-- stroke 1: x -->
  <trace id="t2">30 15 100, 35 20 110, ...</trace>         <!-- stroke 2: ^ -->
  <trace id="t3">35 10 200, 40 15 210, ...</trace>         <!-- stroke 3: 2 -->
  
  <!-- Symbol grouping: which strokes form each symbol -->
  <traceGroup>
    <traceGroup xml:id="g1">
      <annotation type="truth">x</annotation>
      <traceView traceDataRef="t1"/>     <!-- stroke t1 = symbol "x" -->
    </traceGroup>
    <traceGroup xml:id="g2">
      <annotation type="truth">^</annotation>
      <traceView traceDataRef="t2"/>     <!-- stroke t2 = symbol "^" -->
    </traceGroup>
    <traceGroup xml:id="g3">
      <annotation type="truth">2</annotation>
      <traceView traceDataRef="t3"/>     <!-- stroke t3 = symbol "2" -->
    </traceGroup>
  </traceGroup>
  
  <!-- Full LaTeX ground truth -->
  <annotation type="truth">x^{2}</annotation>
</ink>
```

**What InkML gives us for FREE:**
1. **Stroke order**: Temporal sequence of how user wrote the expression
2. **Symbol-stroke mapping**: Which strokes form each symbol
3. **Timestamps**: Exact timing (useful for detecting hesitation, corrections)
4. **Coordinates**: (x, y) for rendering stroke images
5. **Ground truth**: Symbol labels and full LaTeX

#### MathWriting Statistics (Stroke-Level)

| Metric | Median | Mean | Max |
|--------|--------|------|-----|
| Strokes per expression | 14 | 16.8 | 100+ |
| Points per expression | 350 | 420 | 2000+ |
| Writing duration | 6.03s | 7.2s | 60s+ |

This means **MathWriting alone** gives us **230K expressions × ~14 strokes = 3.2M stroke-level training examples!**

### 7.2 Dataset Construction for Editing Operations

**Challenge**: No existing dataset has stroke-level editing annotations. We need to **synthesize** editing data from existing HMER datasets.

#### 7.2.1 Synthetic Data Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDIT DATASET GENERATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Existing HMER dataset (CROHME, HME100K)                             │
│         - Full expression images                                             │
│         - LaTeX labels                                                       │
│         - (Optional) Stroke-level InkML data                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: Parse LaTeX into Symbol Sequence                            │    │
│  │                                                                      │    │
│  │   "x^{2} + y = 5"  →  [x, ^, {, 2, }, +, y, =, 5]                   │    │
│  │                        │  │     │     │  │  │  │                    │    │
│  │                       s1 s2    s3    s4 s5 s6 s7                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: Generate Edit Operations by Subtraction/Substitution        │    │
│  │                                                                      │    │
│  │   ADD:     Remove last symbol(s), train to add back                 │    │
│  │   REPLACE: Substitute symbol, train to correct                      │    │
│  │   INSERT:  Remove middle symbol, train to insert                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: Create Image Pairs                                          │    │
│  │                                                                      │    │
│  │   For each edit operation:                                          │    │
│  │   - "Before" image (partial expression)                             │    │
│  │   - "Edit stroke" image (the symbol being added/edited)             │    │
│  │   - "After" LaTeX (target output)                                   │    │
│  │   - Operation label (ADD/REPLACE/INSERT + position)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.2.2 ADD Operation Data Generation

```python
def generate_add_examples(expression_data):
    """
    Generate ADD training examples by progressively building expressions.
    
    From: "x^{2} + y = 5"
    Generate sequence of ADD operations:
    """
    examples = []
    
    # Parse into symbols with bounding boxes
    symbols = parse_expression(expression_data)
    # symbols = [(sym="x", bbox=(10,20,30,40)), (sym="^", bbox=(...)), ...]
    
    for i in range(1, len(symbols)):
        # Context: symbols 0 to i-1
        context_symbols = symbols[:i]
        context_image = render_symbols(context_symbols)
        context_latex = to_latex(context_symbols)
        
        # New stroke: symbol i
        new_symbol = symbols[i]
        new_stroke_image = crop_symbol_region(expression_data.image, new_symbol.bbox)
        
        # Target
        target_latex = to_latex(symbols[:i+1])
        
        examples.append({
            "context_image": context_image,      # What's already written
            "context_latex": context_latex,       
            "new_stroke_image": new_stroke_image, # New stroke region
            "new_stroke_bbox": new_symbol.bbox,   # 2D position
            "operation": "ADD",
            "target_latex": target_latex,
        })
    
    return examples

# Example output:
# Step 1: context=""      + stroke="x"  → target="x"         (ADD)
# Step 2: context="x"     + stroke="^"  → target="x^"        (ADD)
# Step 3: context="x^"    + stroke="2"  → target="x^{2}"     (ADD)
# Step 4: context="x^{2}" + stroke="+"  → target="x^{2}+"    (ADD)
# ...
```

#### 7.2.3 REPLACE Operation Data Generation

```python
def generate_replace_examples(expression_data, symbol_substitution_map):
    """
    Generate REPLACE training examples by substituting symbols.
    
    symbol_substitution_map = {
        "2": ["3", "4", "5", "n"],      # Numbers
        "+": ["-", "×", "÷"],            # Operators
        "x": ["y", "z", "a", "b"],       # Variables
        "sin": ["cos", "tan", "log"],    # Functions
    }
    """
    examples = []
    symbols = parse_expression(expression_data)
    
    for i, symbol in enumerate(symbols):
        if symbol.text in symbol_substitution_map:
            for replacement in symbol_substitution_map[symbol.text]:
                # Create "wrong" expression with substitution
                wrong_symbols = symbols.copy()
                wrong_symbols[i] = Symbol(text=replacement, bbox=symbol.bbox)
                wrong_image = render_symbols(wrong_symbols)
                wrong_latex = to_latex(wrong_symbols)
                
                # The "correction" stroke is the original symbol
                correction_stroke = crop_symbol_region(expression_data.image, symbol.bbox)
                
                # Target is the correct expression
                target_latex = to_latex(symbols)
                
                examples.append({
                    "context_image": wrong_image,
                    "context_latex": wrong_latex,
                    "new_stroke_image": correction_stroke,
                    "new_stroke_bbox": symbol.bbox,
                    "operation": "REPLACE",
                    "replace_position": i,
                    "target_latex": target_latex,
                })
    
    return examples

# Example output:
# context="x^{3}+y=5" + stroke="2" at pos 2 → target="x^{2}+y=5" (REPLACE 3→2)
# context="x^{2}-y=5" + stroke="+" at pos 3 → target="x^{2}+y=5" (REPLACE -→+)
```

#### 7.2.4 INSERT Operation Data Generation

```python
def generate_insert_examples(expression_data):
    """
    Generate INSERT training examples by removing middle symbols.
    
    From: "x^{2} + y = 5"
    Remove "^{2}" → "x + y = 5"
    Train to INSERT "^{2}" at correct position
    """
    examples = []
    symbols = parse_expression(expression_data)
    
    # Find removable sub-expressions (exponents, subscripts, terms)
    removable_spans = find_removable_spans(symbols)
    
    for span_start, span_end in removable_spans:
        # Create incomplete expression (with span removed)
        incomplete_symbols = symbols[:span_start] + symbols[span_end:]
        incomplete_image = render_symbols(incomplete_symbols)
        incomplete_latex = to_latex(incomplete_symbols)
        
        # The missing part becomes the "insert stroke"
        missing_symbols = symbols[span_start:span_end]
        insert_stroke_image = crop_symbol_region(
            expression_data.image, 
            get_combined_bbox(missing_symbols)
        )
        
        examples.append({
            "context_image": incomplete_image,
            "context_latex": incomplete_latex,
            "new_stroke_image": insert_stroke_image,
            "new_stroke_bbox": get_combined_bbox(missing_symbols),
            "operation": "INSERT",
            "insert_position": span_start,
            "target_latex": to_latex(symbols),
        })
    
    return examples

# Example output:
# context="x+y=5" + stroke="^{2}" at pos 1 → target="x^{2}+y=5" (INSERT exponent)
# context="x^{2}+y" + stroke="=5" at pos 4 → target="x^{2}+y=5" (INSERT ending)
```

#### 7.2.5 Direct InkML-Based Data Generation (Recommended!)

Since CROHME and MathWriting have InkML with **actual stroke sequences**, we can generate **realistic** editing data directly!

```python
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

@dataclass
class Stroke:
    id: str
    points: List[Tuple[float, float, float]]  # (x, y, t)
    
@dataclass  
class Symbol:
    label: str
    strokes: List[Stroke]
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

def parse_inkml(inkml_path: str) -> Tuple[List[Symbol], str]:
    """
    Parse InkML file to extract symbols with their strokes.
    
    Returns:
        symbols: List of Symbol objects in writing order
        latex: Full LaTeX ground truth
    """
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    ns = {'ink': 'http://www.w3.org/2003/InkML'}
    
    # Parse all traces (strokes)
    traces = {}
    for trace in root.findall('.//ink:trace', ns):
        trace_id = trace.get('id')
        points = []
        for point_str in trace.text.strip().split(','):
            coords = point_str.strip().split()
            x, y = float(coords[0]), float(coords[1])
            t = float(coords[2]) if len(coords) > 2 else 0
            points.append((x, y, t))
        traces[trace_id] = Stroke(id=trace_id, points=points)
    
    # Parse symbol groupings
    symbols = []
    for trace_group in root.findall('.//ink:traceGroup/ink:traceGroup', ns):
        label = trace_group.find('ink:annotation[@type="truth"]', ns)
        if label is not None:
            symbol_strokes = []
            for trace_view in trace_group.findall('.//ink:traceView', ns):
                trace_ref = trace_view.get('traceDataRef')
                if trace_ref in traces:
                    symbol_strokes.append(traces[trace_ref])
            
            if symbol_strokes:
                bbox = compute_bbox(symbol_strokes)
                symbols.append(Symbol(
                    label=label.text,
                    strokes=symbol_strokes,
                    bbox=bbox
                ))
    
    # Sort by temporal order (first stroke timestamp)
    symbols.sort(key=lambda s: min(p[2] for stroke in s.strokes for p in stroke.points))
    
    # Get full LaTeX
    latex_annotation = root.find('.//ink:annotation[@type="truth"]', ns)
    latex = latex_annotation.text if latex_annotation else ""
    
    return symbols, latex

def render_strokes_to_image(strokes: List[Stroke], canvas_size=(256, 256), 
                            line_width=2, padding=10) -> np.ndarray:
    """Render strokes to image."""
    # Get bounding box
    all_points = [(p[0], p[1]) for s in strokes for p in s.points]
    min_x = min(p[0] for p in all_points) - padding
    min_y = min(p[1] for p in all_points) - padding
    max_x = max(p[0] for p in all_points) + padding
    max_y = max(p[1] for p in all_points) + padding
    
    # Scale to canvas
    scale = min(canvas_size[0] / (max_x - min_x), canvas_size[1] / (max_y - min_y))
    
    img = Image.new('L', canvas_size, color=255)
    draw = ImageDraw.Draw(img)
    
    for stroke in strokes:
        scaled_points = [
            ((p[0] - min_x) * scale, (p[1] - min_y) * scale)
            for p in stroke.points
        ]
        if len(scaled_points) > 1:
            draw.line(scaled_points, fill=0, width=line_width)
    
    return np.array(img)


def generate_add_examples_from_inkml(inkml_path: str) -> List[dict]:
    """
    Generate ADD training examples from InkML.
    
    Each symbol in temporal order becomes one ADD example.
    """
    symbols, full_latex = parse_inkml(inkml_path)
    examples = []
    
    for i in range(len(symbols)):
        # Context: all symbols written before this one
        context_strokes = [s for sym in symbols[:i] for s in sym.strokes]
        context_image = render_strokes_to_image(context_strokes) if context_strokes else np.ones((256,256))*255
        
        # New stroke(s): the current symbol
        new_strokes = symbols[i].strokes
        new_stroke_image = render_strokes_to_image(new_strokes)
        
        # Partial LaTeX up to this symbol
        partial_latex = build_partial_latex(symbols[:i+1])
        
        examples.append({
            "context_image": context_image,
            "context_symbols": [s.label for s in symbols[:i]],
            "new_stroke_image": new_stroke_image,
            "new_stroke_bbox": symbols[i].bbox,
            "new_symbol_label": symbols[i].label,
            "operation": "ADD",
            "target_latex": partial_latex,
            "timestamp": min(p[2] for s in new_strokes for p in s.points),
        })
    
    return examples


def generate_replace_examples_from_inkml(inkml_path: str, 
                                          confusion_pairs: dict) -> List[dict]:
    """
    Generate REPLACE training examples.
    
    For each symbol, create a "corrupted" version and train to correct it.
    """
    symbols, full_latex = parse_inkml(inkml_path)
    examples = []
    
    for i, symbol in enumerate(symbols):
        if symbol.label in confusion_pairs:
            for wrong_label in confusion_pairs[symbol.label]:
                # Render expression with WRONG symbol at position i
                wrong_context_image = render_with_substitution(
                    symbols, i, wrong_label
                )
                
                # The correction stroke is the CORRECT symbol
                correct_stroke_image = render_strokes_to_image(symbol.strokes)
                
                examples.append({
                    "context_image": wrong_context_image,
                    "context_latex": build_latex_with_sub(symbols, i, wrong_label),
                    "new_stroke_image": correct_stroke_image,
                    "new_stroke_bbox": symbol.bbox,
                    "operation": "REPLACE",
                    "replace_position": i,
                    "replace_from": wrong_label,
                    "replace_to": symbol.label,
                    "target_latex": full_latex,
                })
    
    return examples

# Confusion pairs for REPLACE generation
CONFUSION_PAIRS = {
    # Numbers
    "0": ["O", "o", "6", "9"],
    "1": ["l", "I", "7", "|"],
    "2": ["z", "Z"],
    "3": ["8"],
    "5": ["S", "s"],
    "6": ["0", "b", "9"],
    "9": ["6", "q"],
    
    # Variables
    "x": ["X", "×", "+"],
    "y": ["Y", "v"],
    "z": ["2", "Z"],
    
    # Operators
    "+": ["-", "×", "t"],
    "-": ["+", "−", "—"],
    "=": ["−", "≡"],
    
    # Greek
    "α": ["a", "∝"],
    "β": ["B", "ß"],
    "θ": ["0", "O"],
    "π": ["n", "∏"],
}
```

#### 7.2.6 Scale: What InkML Datasets Give Us

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATASET SCALE FOR EDITING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MathWriting (230K expressions):                                            │
│  ├─ Average 14 strokes/expression                                           │
│  ├─ = 3.2M stroke-level ADD examples                                        │
│  ├─ + ~500K REPLACE examples (with confusion pairs)                         │
│  └─ + ~200K INSERT examples (sub-expression removal)                        │
│                                                                              │
│  CROHME (12K expressions):                                                  │
│  ├─ Average 10 strokes/expression                                           │
│  ├─ = 120K stroke-level ADD examples                                        │
│  └─ + proportional REPLACE/INSERT                                           │
│                                                                              │
│  TOTAL: ~4M+ training examples for editing!                                 │
│                                                                              │
│  This is MUCH more than we'd get from synthetic generation alone!           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.2.7 Dataset Statistics (Actual from InkML!)

| Operation | Source | Examples | Method |
|-----------|--------|----------|--------|
| **ADD** | MathWriting | 3,200,000+ | Direct from stroke sequence |
| **ADD** | CROHME | 120,000+ | Direct from stroke sequence |
| **REPLACE** | Both | 500,000+ | Confusion pair substitution |
| **INSERT** | Both | 200,000+ | Sub-expression removal |
| **Total** | | **~4,000,000** | Combined |

**Key Advantage**: Using InkML gives us **REAL** user writing patterns, not synthetic!

**Splits**:
- Train: 80% (3.2M examples)
- Validation: 10% (400K examples)
- Test: 10% (400K examples)

**Cross-dataset evaluation**:
- Train on MathWriting → Test on CROHME (domain transfer)
- Train on CROHME → Test on MathWriting (scale transfer)

#### 7.2.7 Data Augmentation

```python
augmentations = [
    # Spatial augmentations
    RandomRotation(degrees=5),
    RandomScale(range=(0.9, 1.1)),
    RandomTranslate(pixels=10),
    
    # Stroke augmentations
    StrokeThicknessVariation(range=(1, 3)),
    StrokeNoiseInjection(sigma=2),
    
    # Handwriting style augmentation
    ElasticDeformation(alpha=10, sigma=3),
    
    # Background variations
    PaperTextureAugmentation(),
    GridLineAugmentation(),
]
```

### 7.3 Evaluation Protocol

#### 7.3.1 Metrics

| Metric | Description |
|--------|-------------|
| **Operation Accuracy** | Correct prediction of ADD/REPLACE/INSERT |
| **Position Accuracy** | Correct position for REPLACE/INSERT |
| **LaTeX Exact Match** | Full LaTeX string matches target |
| **Symbol Error Rate** | Edit distance at symbol level |
| **Latency** | Time from pen-lift to LaTeX output |

#### 7.3.2 Test Scenarios

```
Scenario A: Sequential Building (Easy)
  Build "x^2+1" stroke by stroke
  Expected: All ADD operations, high accuracy

Scenario B: Simple Replacement (Medium)
  Change "x^2" to "x^3"
  Expected: Detect REPLACE, correct position

Scenario C: Structural Edit (Hard)
  Add fraction line under "x+1" to make "\frac{x+1}{}"
  Expected: Detect structural change, multi-token output

Scenario D: Ambiguous Position (Hard)
  Add "2" near "x" - is it "x2" or "x^2" or "x_2"?
  Expected: Use 2D position to disambiguate
```

### 7.4 Data Preparation Implementation

We have implemented the data generation pipeline in the `data/` folder:

#### 7.4.1 Files Overview

| File | Purpose |
|------|---------|
| `cases.py` | Training case generator (ADD/REPLACE/INSERT) |
| `tokenizer.py` | Symbol-level LaTeX tokenizer (~800 vocab) |
| `data_prep.ipynb` | Visualization demo notebook |
| `generate_tikzcd.py` | TikZ-CD diagram rendering |

#### 7.4.2 Training Data Format

```
Input:
  - latex_context: Previous LaTeX string (empty for initial state)
  - crop_spec: {x, y, width, height, noise_specs}
  
Output:
  - updated_latex: New LaTeX string after edit

Token Sequence:
  [BOS] context_tokens [OP] [SEP] target_tokens [EOS]
```

Example JSON:
```json
{
  "id": "single_add_0042",
  "latex_context": "x",
  "crop_spec": {
    "x": 0.6, "y": 0.3,
    "width": 0.25, "height": 0.3,
    "noise_specs": [
      {"type": "stray_mark", "size": "small"},
      {"type": "nearby_symbol", "symbol": "y", "visibility_pct": 0.3}
    ]
  },
  "updated_latex": "x^{2}",
  "operation": "ADD",
  "is_initial_state": false
}
```

#### 7.4.3 Symbol-Level Tokenization

**Design choice**: Symbol-level (~800 tokens) instead of character-level or BPE.

| Category | Count | Examples |
|----------|-------|----------|
| Special tokens | 20 | `<PAD>`, `<BOS>`, `<ADD>`, `<REPLACE>` |
| English letters | 51 | `a-z`, `A-Z` |
| Digits | 10 | `0-9` |
| Greek | 38 | `\alpha`, `\beta`, `\Gamma` |
| Operators | 30 | `+`, `-`, `\times`, `\cdot` |
| Relations | 40 | `=`, `\leq`, `\subset`, `\in` |
| Brackets | 30 | `(`, `)`, `^{`, `_{`, `}` |
| Calculus | 40 | `\int`, `\sum`, `\partial`, `\nabla` |
| Arrows | 40 | `\to`, `\rightarrow`, `\mapsto` |
| Functions | 30 | `\sin`, `\cos`, `\log`, `\lim` |
| TikZ-CD | 80 | `\begin{tikzcd}`, `\arrow[`, `dashed` |
| **Total** | **~782** | |

Tokenization example:
```
"x^{2} + \alpha"
  → ['x', '^{', '2', '}', '+', '\alpha']
  → [43, 209, 74, 192, 120, 82]
  → 6 tokens (vs ~14 for character-level)
```

#### 7.4.4 Case Categories

| Category | Subcases | Example |
|----------|----------|---------|
| **Single Symbol** | add_letter, add_greek, add_subscript, replace_letter | `x` → `x^{2}` |
| **Initial State** | first_letter, first_greek, first_calculus | `` → `\alpha` |
| **Diagram** | add_arrow, add_node, replace_label | `A \to B` → `A \to B \to C` |
| **Calculus** | add_integral, add_limit, replace_bound | `\int f` → `\int_0^1 f` |
| **Matrix** | add_row, add_element, replace_bracket | 2×2 → 3×3 |
| **Diff. Eq.** | add_term, add_boundary | `dy/dx = y` → `dy/dx = y + x` |

#### 7.4.5 Noise Augmentation

40-50% of training examples include noise in crop region:

| Noise Type | Description |
|------------|-------------|
| `random_line` | Straight line segment |
| `dashed_line` | Dashed/dotted line |
| `nearby_symbol` | Part of adjacent symbol visible |
| `partial_symbol` | Incomplete symbol stroke |
| `stray_mark` | Accidental dot/mark |
| `crossed_out` | Previous attempt crossed out |
| `smudge` | Erased area |
| `grid_line` | Background paper grid |

#### 7.4.6 Data Generation Commands

```bash
# Generate all categories
python cases.py -t --count 100 -o training_data.json

# Single symbol cases only (atomic edits)
python cases.py -c single_symbol -t --count 200 -o atomic.json

# Without noise (clean)
python cases.py -t --no-noise -o clean_training.json

# Test tokenizer
python tokenizer.py
```

#### 7.4.7 Dataset Statistics (default generation)

```
Total Examples:        ~1000
Initial State:         ~16% (empty → first symbol)
With Noise:            ~40-50%
ADD Operations:        ~55%
REPLACE Operations:    ~40%
INSERT Operations:     ~5%

Categories:
  single_symbol:       ~350 (highest priority)
  diagram:             ~125
  matrix:              ~100
  calculus:            ~100
  algebra:             ~100
  subscript/super:     ~100
  diffeq:              ~50
  greek:               ~50
  fraction:            ~50
```

#### 7.4.8 Compositional Data Augmentation (`augmentation.py`)

**Problem**: Real LaTeX documents contain multiple expressions. Training on isolated expressions doesn't teach the model to focus on the edit region amid complex surrounding context.

**Solution**: Chunk Pool + Compositional Context

```
┌─────────────────────────────────────────────────────────────────┐
│  CHUNK POOL (generated via cases.py, organized by depth)        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │depth=1 │ │depth=2 │ │depth=3 │ │depth=4 │ │depth=5+│        │
│  │x + y   │ │\frac{} │ │nested  │ │complex │ │tikzcd  │        │
│  │\alpha  │ │x_{i}^j │ │fracs   │ │integrals│ │diagrams│       │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Random Composition
┌─────────────────────────────────────────────────────────────────┐
│  COMPOSED TRAINING EXAMPLE                                       │
│                                                                  │
│  Context: [chunk₃ \quad chunk₁ \quad TARGET \quad chunk₂]       │
│                                    ↑                            │
│                               Edit this chunk                   │
│                               (change α → β)                    │
│                                                                  │
│  Full Before: "x^{2} \quad \frac{1}{n} \quad \alpha + y \quad z"│
│  Full After:  "x^{2} \quad \frac{1}{n} \quad \beta + y \quad z" │
│                                         ↑                       │
│                                    Single edit                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Benefits**:

| Benefit | Why It Matters |
|---------|----------------|
| **Combinatorial explosion** | 100 chunks → millions of unique contexts |
| **Realistic scenarios** | Real documents have multiple expressions |
| **Focus learning** | Model learns to ignore irrelevant context |
| **Variable context length** | Handles 1-10 surrounding chunks |
| **Difficulty mixing** | Easy targets with hard context, and vice versa |

**Implementation** (`compositional_augmentation.py`):

```python
from augmentation import ChunkPool, ContextComposer
from synthetic_case_generator import CaseGenerator

# Step 1: Build chunk pool (one-time)
generator = CaseGenerator(seed=42)
pool = ChunkPool.from_case_generator(generator, chunks_per_depth=100)
pool.save("chunk_pool.json")

# Step 2: Compose training examples (on-the-fly or batch)
composer = ContextComposer(pool)
examples = composer.compose_curriculum(
    total_examples=10000,
    min_context=1,     # Start with 1 context chunk
    max_context=5,     # Build up to 5 context chunks
)
```

**Composition Strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `random` | Randomly sample context chunks | General training |
| `depth_matched` | Context depth ≈ target depth | Curriculum learning |
| `diverse` | Mix categories and depths | Robustness |

**Curriculum Learning** (default):
1. **Phase 1 (40%)**: Easy targets, 1 context chunk
2. **Phase 2 (40%)**: Medium targets, 2-3 context chunks
3. **Phase 3 (20%)**: Hard targets, 4-5 context chunks

**Tree Depth Calculation**:

For semantic depth (not just brace counting):
- `\frac{a}{b}` → depth +1 for contents
- `^{x}` and `_{x}` → depth +1 for subscript/superscript
- TikZ-CD diagrams → depth = max(rows, cols) + arrow_bonus

```python
from synthetic_case_generator import analyze_latex_complexity

# Example depths:
analyze_latex_complexity(r"x + y")              # depth=1
analyze_latex_complexity(r"\frac{a}{b}")        # depth=2
analyze_latex_complexity(r"\frac{\frac{1}{2}}{3}")  # depth=3
analyze_latex_complexity(r"\sum_{n=0}^{\infty} \frac{(-1)^n}{n!}")  # depth=4
```

**CLI Usage**:

```bash
# Build dataset with compositional augmentation
python augmentation.py \
    --chunks-per-depth 100 \
    --num-examples 5000 \
    --min-context 1 \
    --max-context 5 \
    --output-pool chunk_pool.json \
    --output-examples training_composed.json
```

#### 7.4.9 Stroke-Level Data Augmentation

> **📖 See `data/data_augmentation.md` for complete documentation** on the stroke-level data generation pipeline.

**Key Design Decisions**:

1. **Parallel 2D Conv Along Stroke Trajectory** (not bounding box patches)
   - Extract conv tokens along stroke path, each with (x,y) position
   - Preserves spatial relationships for 2D RoPE attention
   - Efficient: processes only stroke pixels, not empty background

2. **Per-Stroke Variations with Combinatorial Mixing**
   - Manual stroke capture via web tool (`stroke_corpus/capture_tool.html`)
   - 3-5 variations per stroke → 27-125 unique combinations per symbol
   - Global normalization preserves relative stroke positions

3. **Stroke-Level Noise** (not image artifacts)
   - Noise strokes generate conv tokens like real strokes
   - Model learns to ignore via low attention weights
   - Types: stray dots, partial neighbors, edge intrusions

**Data Scale**:
- 238 symbols × 27+ combinations × 450 augmentations ≈ **22M training examples**

```
┌─────────────────────────────────────────────────────────────────┐
│  STROKE ENCODING PIPELINE                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stroke trajectory: (x₁,y₁) → (x₂,y₂) → ... → (xₙ,yₙ)          │
│                          │                                       │
│                          ▼ Sample every N pixels                │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                         │
│  │patch│   │patch│   │patch│   │patch│   Local 2D conv         │
│  └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘                         │
│     │         │         │         │                             │
│     ▼         ▼         ▼         ▼      (parallel GPU batch)  │
│  [t₁,pos₁] [t₂,pos₂] [t₃,pos₃] [t₄,pos₄]                       │
│                          │                                       │
│                          ▼                                       │
│         Transformer with 2D RoPE attention                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single-symbol accuracy | > 95% | CROHME symbol test set |
| Edit operation accuracy | > 90% | Synthetic edit dataset |
| Multi-token F1 | > 85% | Structure change test set |
| End-to-end latency | < 100ms | iPad Pro benchmark |
| Memory per stroke | < 5KB | [CLS]-only mode |
| User satisfaction | > 4/5 | User study (n=20) |

---

## 9. Open Questions

1. **Stroke Segmentation**: How to handle multi-symbol strokes (e.g., "∞" as two loops)?
2. **Undo/Redo**: How to efficiently reverse operations without full re-computation?
3. **Ambiguity Resolution**: When 2D position is ambiguous, should model ask for user confirmation?
4. **Error Accumulation**: How to prevent errors from cascading through [CLS] tokens?
5. **Training Curriculum**: Should we train stroke-by-stroke or full expression first?

---

*Last updated: December 18, 2025 (simplified to Multimodal + Dual-Token + AR architecture)*

