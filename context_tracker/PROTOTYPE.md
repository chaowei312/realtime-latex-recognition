# Context Tracker: Minimal Prototype Plan

> **Goal**: Prove trajectory-based reasoning capture works in 2-3 weeks  
> **NOT**: Production-ready SOTA system

---

## 0. Key Innovation Demonstration

### Innovation vs Existing Methods

| Aspect | Existing Methods | **Ours** | Gain |
|--------|------------------|----------|------|
| **What's captured** | Final expression only | **Reasoning trajectory** (edit sequence) | New capability |
| **Visual processing** | All patches every step | **Current chunk only** → commit → discard | O(n³) → O(n²) |
| **Memory scaling** | O(n × p²) visual history | **O(n) text tokens** | ~100× reduction |
| **Edit operation** | Full re-inference | **O(1) local chunk re-embed** | ~10× faster |
| **Noise handling** | Clean input assumed | **Stroke artifacts** (model filters noise) | Robustness |

### Complexity: Why O(n³) → O(n²)

```
EXISTING: Keep all patches → Σ(ip)² = O(n³)
OURS:     Commit & discard → Σ(p² + i) = O(n²)

n=20, p=16:  Existing ~683K ops  vs  Ours ~5K ops  →  ~130× speedup
```

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  DUAL-HEAD MODEL: [VC] = Visual Class, [INT] = Integration                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  [LaTeX] ──causal──► [Patches] ──causal──► [VC] ─────────► [INT]               │
│                          │                   │                │                 │
│                          │                   └────────┬───────┘                 │
│                          │                      concat([INT],[VC])              │
│                          │                            │                         │
│                          │                     ┌──────┴──────┐                  │
│                          │                     │ QUERY PROJ  │                  │
│                          │                     │ W_q_h → q_h │                  │
│                          │                     └──────┬──────┘                  │
│                          │                            │                         │
│                   ┌──────┴──────┐                     │                         │
│                   │  KEY PROJ   │                     │                         │
│                   │ W_k_h → k_ij│                     │                         │
│                   └──────┬──────┘                     │                         │
│                          │                            │                         │
│                          └────────────┬───────────────┘                         │
│                                       ▼                                         │
│                          ┌─────────────────────────────┐                        │
│                          │ SCORE FIRST, THEN POOL      │                        │
│                          │                             │                        │
│                          │ Per-patch:  score_ij = q·k_ij                       │
│                          │ Per-stroke: score_i = mean(score_ij)                │
│                          │ Per-head:   repeat for h=1..4                       │
│                          │ Final:      mean across heads                       │
│                          │ mask = σ(final_score)                               │
│                          │                             │                        │
│                          │ HEAD B: Stroke Selection    │                        │
│                          └──────────────┬──────────────┘                        │
│                                         │                                       │
│  ┌────────────────────────┐             │                                       │
│  │ HEAD A: Recognition    │◄────────────┘                                       │
│  │ h_INT → FFN → P(sym)   │                                                     │
│  └────────────────────────┘                                                     │
│                                                                                  │
│  Attention: [VC] sees patches only, [INT] sees LaTeX + [VC]                    │
│  Head A: h_INT → FFN → symbol (context-aware recognition)                      │
│  Head B: Stroke Modification Module (SMM) - multi-head selection gate          │
│  Head C: Position Head - per-token edit position during AR                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Head C: Position Prediction (During AR Decoding)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  POSITION HEAD: Per-Token Edit Position (Each AR Step)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  At each AR step t, model outputs BOTH:                                         │
│    • Symbol: what token to insert (via Head A)                                 │
│    • Action: where to insert it (via Head C)                                   │
│                                                                                  │
│  Example: User writes "t" small, below "a" (subscript)                         │
│                                                                                  │
│    Step t:  h_t → (token="t", action=(a, SUB))                                 │
│             → Result: a_t  (t becomes subscript of a)                          │
│                                                                                  │
│  Key: User does NOT write "_" - it's LaTeX syntax!                             │
│       Model infers SUB relation from position of handwritten "t"               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Relationship Tensor (Parsing LaTeX to Structure)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LaTeX SYNTAX → PARSED INTO RELATIONS (not stored as tokens)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LaTeX String          Tokens (content only)      Relationship Tensor          │
│  ════════════════════  ════════════════════════   ══════════════════════════   │
│                                                                                  │
│  "a_t"                 ["a", "t"]                 (t -> a, SUB)                 │
│  "a^x"                 ["a", "x"]                 (x -> a, SUP)                 │
│  "\frac{a_t+b}{c}"     ["frac","a","t","+","b","c"]                             │
│                                                    (a -> frac, ABOVE)            │
│                                                    (c -> frac, BELOW)            │
│                                                    (t -> a, SUB)                 │
│                                                    (+ -> a, RIGHT)               │
│                                                    (b -> +, RIGHT)               │
│                                                                                  │
│  3D Tensor: [num_tokens x num_tokens x num_relations]                          │
│  Relations: RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE                              │
│                                                                                  │
│  Benefits:                                                                      │
│    • Smaller vocab (no _, ^, {, } syntax tokens)                               │
│    • Matches user input (user writes "t", not "_")                             │
│    • Structure explicit in relations                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Position Head Architecture: Tree-aware Position Module (TPM)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TPM: ACTION EMBEDDINGS = PARENT EMBEDDING + RELATION EMBEDDING                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Query: concat(h_t, h_VC)  ← h_t for context, h_VC for spatial stroke location │
│  Keys:  Action embeddings (parent_token + relation_type)                       │
│                                                                                  │
│  Why concat(h_t, h_VC)?                                                         │
│  ═══════════════════════════════════════════════════════════                   │
│  • h_t: Already attended to LaTeX context + [INT], knows WHAT symbol           │
│  • h_VC: Contains WHERE strokes are placed (spatial information)               │
│  • Together: shares positional load from h_t, avoids overloading               │
│                                                                                  │
│  Compare to SMM query concat(h_INT, h_VC):                                     │
│  • SMM needs global context (h_INT) to disambiguate (I vs 1)                   │
│  • TPM needs token-specific position (h_t) + stroke location (h_VC)            │
│                                                                                  │
│  Action Space (grows as we generate):                                          │
│  ┌────────────────────────────────────────┐                                    │
│  │ (frac, RIGHT)  (frac, SUP)  (frac, SUB)│                                    │
│  │ (a, RIGHT)     (a, SUP)     (a, SUB)   │ ← can attach to "a"                │
│  │ (c, RIGHT)     (c, SUP)     (c, SUB)   │                                    │
│  │ ... + newly generated tokens           │                                    │
│  └────────────────────────────────────────┘                                    │
│                                                                                  │
│  Multi-head attention (same pattern as SMM):                                   │
│    q_h = W_q_h(concat(h_t, h_VC))  → [d_head]                                  │
│    k_h = W_k_h(action_emb)         → [N*R, d_head]                             │
│    scores = q_h · k_h^T            → [N*R]                                     │
│    action_probs = softmax(avg_across_heads(scores))                            │
│                                                                                  │
│  Output: selected_action = argmax(action_probs)                                │
│          → (parent_idx, relation_type)                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### TPM vs SMM Query Design

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  WHY DIFFERENT QUERIES FOR SMM AND TPM?                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MODULE     │ QUERY                │ REASON                                    │
│  ═══════════╪══════════════════════╪════════════════════════════════════════   │
│  SMM        │ concat(h_INT, h_VC)  │ Global context (I vs 1) + visual patches │
│  TPM        │ concat(h_t, h_VC)    │ Token-specific + where stroke placed     │
│                                                                                  │
│  SMM Decision: "Which strokes made this symbol?"                               │
│    • Needs INT: global context disambiguates similar strokes (I/1)             │
│    • Needs VC: knows which patches are candidates                              │
│                                                                                  │
│  TPM Decision: "Where does this token attach in the tree?"                     │
│    • Needs h_t: current token's identity and context                           │
│    • Needs VC: WHERE user wrote it (spatial position → SUB/SUP/RIGHT)          │
│    • Does NOT need INT: position is local, not global disambiguation           │
│                                                                                  │
│  Key Insight: VC carries SPATIAL info that h_t doesn't have                    │
│               This shares the positional burden from h_t                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### AR Decoding with Dual Output

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EACH AR STEP = (SYMBOL, POSITION)                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Example: Adding "_t+b" to \frac{a}{c}                                         │
│                                                                                  │
│  Step │ Symbol │ Action        │ Meaning                │ LaTeX After          │
│  ═════╪════════╪═══════════════╪════════════════════════╪══════════════════════│
│   0   │   t    │ (a, SUB)      │ t subscript of a       │ \frac{a_t}{c}        │
│   1   │   +    │ (t, RIGHT)    │ + right of t           │ \frac{a_t+}{c}       │
│   2   │   b    │ (+, RIGHT)    │ b right of +           │ \frac{a_t+b}{c}      │
│   3   │ <EOS>  │   —           │ done                   │ \frac{a_t+b}{c}      │
│                                                                                  │
│  Note: No "_" token! SUB relation encodes subscript.                           │
│        Tree-to-LaTeX conversion adds syntax in post-processing.                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Head B Detail: Score First, Then Pool

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MULTI-HEAD STROKE SELECTION (Score First, Then Pool)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DIMENSION DESIGN: Query has LESS compression (preserves context+visual)       │
│  ════════════════════════════════════════════════════════════════════          │
│                                                                                  │
│  d_concat = d_int + d_vc = 512 + 512 = 1024                                    │
│  d_patch = 512                                                                  │
│  num_heads = 4                                                                  │
│  d_head = d_concat / num_heads = 1024 / 4 = 256                                │
│                                                                                  │
│  Query: W_q_h(1024 → 256)  = 4× compression                                    │
│  Key:   W_k_h(512 → 256)   = 2× compression  ← LESS compressed                 │
│                                                                                  │
│  Why? Query carries richer signal (context + visual decision)                  │
│       Keys just need to be matchable, can afford more capacity                 │
│                                                                                  │
│  ────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│  Patch tokens (grouped by stroke):                                              │
│  ════════════════════════════════                                               │
│  Stroke 1: [P₁₀, P₁₁, P₁₂]        (3 patches)                                  │
│  Stroke 2: [P₂₀, P₂₁]             (2 patches, incomplete)                      │
│                                                                                  │
│  Step 1: Project query and keys (asymmetric compression)                       │
│  ═══════════════════════════════════════════════════════                       │
│                                                                                  │
│  q_h = W_q_h(concat([INT],[VC]))   [1024] → [256]  (4× compression)           │
│  k_ij = W_k_h(P_ij)                [512]  → [256]  (2× compression)           │
│                                                                                  │
│  Step 2: Score EACH patch (per head)                                           │
│  ═══════════════════════════════════                                            │
│                                                                                  │
│  Head 1:  q₁·k₁₀=0.8  q₁·k₁₁=0.3  q₁·k₁₂=0.9  │  q₁·k₂₀=0.2  q₁·k₂₁=0.1     │
│  Head 2:  q₂·k₁₀=0.7  q₂·k₁₁=0.4  q₂·k₁₂=0.8  │  q₂·k₂₀=0.3  q₂·k₂₁=0.2     │
│  Head 3:  q₃·k₁₀=0.9  q₃·k₁₁=0.2  q₃·k₁₂=0.7  │  q₃·k₂₀=0.1  q₃·k₂₁=0.1     │
│  Head 4:  q₄·k₁₀=0.6  q₄·k₁₁=0.5  q₄·k₁₂=0.8  │  q₄·k₂₀=0.2  q₄·k₂₁=0.3     │
│                                                                                  │
│  Step 3: Average within each stroke (per head)                                  │
│  ═════════════════════════════════════════════                                  │
│                                                                                  │
│  Head 1:  stroke₁ = mean(0.8,0.3,0.9) = 0.67   stroke₂ = mean(0.2,0.1) = 0.15 │
│  Head 2:  stroke₁ = mean(0.7,0.4,0.8) = 0.63   stroke₂ = mean(0.3,0.2) = 0.25 │
│  Head 3:  stroke₁ = mean(0.9,0.2,0.7) = 0.60   stroke₂ = mean(0.1,0.1) = 0.10 │
│  Head 4:  stroke₁ = mean(0.6,0.5,0.8) = 0.63   stroke₂ = mean(0.2,0.3) = 0.25 │
│                                                                                  │
│  Step 4: Average across heads                                                   │
│  ════════════════════════════                                                   │
│                                                                                  │
│  final₁ = mean(0.67, 0.63, 0.60, 0.63) = 0.63  → σ(0.63) = 0.65 → commit ✓    │
│  final₂ = mean(0.15, 0.25, 0.10, 0.25) = 0.19  → σ(0.19) = 0.55 → keep        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Three-Head Training

| Component | Role | Training Signal |
|-----------|------|-----------------|
| **[VC]** | Visual Class (patches → visual embedding) | Implicit via SMM |
| **[INT]** | Integration (context + [VC] → decision) | **Head A + Teacher distill** |
| **Head A** | Recognition: h_INT → FFN → symbol | Cross-entropy |
| **SMM (B)** | Stroke Modification: multi-head selection gate | BCE on stroke labels |
| **Head C** | Position: h_t → action attention → (parent, rel) | Cross-entropy on action |

```
Loss = L_symbol + λ₁·L_position + λ₂·L_stroke_select + λ₃·KL([INT] ∥ Teacher)

where:
  L_symbol = CE(FFN(h_t), target_symbol)       # Head A: what token?
  
  L_position = CE(action_probs, target_action)  # Head C (TPM): where to insert?
    q = concat(h_t, h_VC)                       # token-specific + spatial
    action_probs = softmax(q · K_actions)       # attention over action embeddings
    action = (parent_idx, relation_type)
  
  L_stroke_select: (score first, then pool)    # Head B (SMM)
    q = concat(h_INT, h_VC)                     # global context + visual
    patch_score_ij = q_h · k_ij
    stroke_score_i = mean(patch_scores)
    final_i = mean across heads
    L = BCE(σ(final_i), y_stroke_i)
  
  KL term: Teacher VLM guides [INT] for disambiguation

Consistent pattern across all heads:
  • SMM query: concat(h_INT, h_VC) - global + visual
  • TPM query: concat(h_t, h_VC)   - token + spatial
  • Keys from different embedding spaces (vocab, patches, actions)
  • Multi-head attention for robustness
```

---

## 2. Minimum Experiments

| Exp | Question | Method | Success Metric |
|-----|----------|--------|----------------|
| **1** | Trajectory patches work? | Recognition accuracy | ExpRate > 70%, 10× fewer patches |
| **2** | [INT] captures reasoning? | t-SNE + classification | Clusters by problem type, acc > 80% |
| **3** | Efficiency gains real? | Measure ops/memory | O(n²) vs O(n³) demonstrated |

---

## 3. Data & Augmentation Strategy

```
MathWriting (ready):
├── synthetic/              396K expressions with strokes
├── synthetic-bboxes.jsonl  Per-symbol bounding boxes ← KEY
└── symbols/                6,423 individual symbol strokes
```

### Compositional Augmentation (Simulates Edit Scenarios)

```
1. COMPOSE: Concatenate 2-4 MathWriting chunks → context
   "x^2+1" + "\frac{a}{b}" + "=0" → "x^2+1, \frac{a}{b}, =0"
   
2. COMMIT: Context becomes TEXT tokens (not patches)
   
3. EDIT: Replace one chunk with different MathWriting sample
   • Context: "x^2+1, [MASK], =0" (text, KV-cached)
   • Edit chunk: render "\sqrt{n}" + stroke artifacts → image
   • Conv2D: image → 4×4 patch grid (standard, no trajectory tracking)
   • Output: "\sqrt{n}" [EOS]

This proves O(n²): context=text, only edit chunk=image patches
```

### Stroke Artifacts (Incomplete Strokes as Distractor)

```
Image = ONLY unrecognized strokes (snowballing until commit)

If complete chunks A and B both in image → model outputs "A, B" (both)
So artifacts must be INCOMPLETE strokes, not full chunks

Training setup:
  Image: [complete chunk strokes] + [1-2 incomplete strokes from NEXT symbol]
  Target: complete chunk only (ignore incomplete)

Source options:
  1. MathWriting: take first k strokes of another symbol (k < total)
  2. stroke_corpus/: random incomplete shreds

Model learns: complete → output, incomplete → wait/ignore
```

---

## 4. Implementation Checklist

### Week 1: Recognition Baseline
- [ ] MathWriting synthetic data loader (chunks + bboxes)
- [ ] Chunk renderer: strokes → image (+ stroke artifacts overlay)
- [ ] Standard Conv2D patch embedding (4×4 grid)
- [ ] Train: context (text) + chunk (image) → LaTeX decoder

### Week 2: [INT] + Trajectory
- [ ] Add [INT] token with teacher distillation
- [ ] Extract trajectory embeddings from [INT]
- [ ] t-SNE visualization + classification

### Week 3: Paper
- [ ] Efficiency comparison table
- [ ] Figures: architecture, t-SNE, efficiency
- [ ] Write method section

---

## 5. Key Figures for Paper

| Figure | Content | Priority |
|--------|---------|----------|
| Fig 1 | Architecture (dual-token, teacher distillation) | Must |
| Fig 2 | **t-SNE of [INT] embeddings** (clusters by problem type) | ⭐ KEY |
| Fig 3 | Efficiency: patches, memory, O(n²) vs O(n³) | Must |
| Fig 4 | SMM and TPM module detail (multi-head attention flow) | Must |
| Table 1 | Quantitative: ExpRate, classification acc, speedup | Must |

### TikZ Diagrams (Publication-Ready)

```
context_tracker/docs/
├── architecture_diagram.pdf   # Overview: inputs → transformer → 3 heads
└── smm_tpm_detail.pdf         # Internal SMM + TPM: query/key projections, scoring
```

Compile with: `pdflatex architecture_diagram.tex`

---

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| Expression Recognition Rate | > 70% |
| Trajectory Classification | > 80% |
| Patch Reduction | 5-10× fewer than dense |
| Cumulative Cost | O(n²) proven |

---

*Target: 2-3 week prototype*
