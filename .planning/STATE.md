# Project State: DistriNews

**Last Updated:** 2025-01-30

---

## Project Reference

See: `.planning/PROJECT.md` (updated 2025-01-30)

**Core value:** Teach distributed ML training through hands-on implementation using only free resources
**Current focus:** Phase 2 — Baseline Training

---

## Current Position

```
Phase 1 ✓ ──► Phase 2 ◆ ──► Phase 3 ○ ──► Phase 4 ○ ──► Phase 5 ○ ──► Phase 6 ○ ──► Phase 7 ○
Foundation   Baseline    DDP       Inference   Benchmark   Document   Deploy

◆ = Current    ○ = Pending    ✓ = Complete
```

**Current Phase:** Phase 2 — Baseline Training
**Current Task:** Not started
**Blocked By:** None

---

## Phase Status

| Phase | Status | Plans | Progress |
|-------|--------|-------|----------|
| 1 - Foundation | ✓ Complete | 1/1 | 100% |
| 2 - Baseline | ◆ Current | 0/0 | 0% |
| 3 - DDP | ○ Pending | 0/0 | 0% |
| 4 - Inference | ○ Pending | 0/0 | 0% |
| 5 - Benchmark | ○ Pending | 0/0 | 0% |
| 6 - Documentation | ○ Pending | 0/0 | 0% |
| 7 - Deployment | ○ Pending | 0/0 | 0% |

---

## Recent Progress

### Phase 1: Foundation ✓ (2025-01-30)

**Files created:**
- `requirements.txt` — All dependencies with explanations
- `training/__init__.py` — Package marker
- `training/model.py` — NewsClassifier (DistilBERT wrapper)
- `training/dataset.py` — AGNewsDataset + tokenization

**Concepts learned:**
- What is DistilBERT and why we use it
- How tokenization works (text → numbers)
- What attention masks are for
- How PyTorch Dataset interface works

---

## Key Decisions Made

| Decision | When | Rationale |
|----------|------|-----------|
| Use Kaggle for training | Init | Free 2x T4 GPUs, 30 hrs/week |
| Use HF Spaces for deploy | Init | Free FastAPI hosting, git-based |
| CPU simulation for local dev | Init | Learn DDP without GPUs locally |
| DistilBERT over BERT | Init | Faster training, fits on T4 |
| Step-by-step learning mode | Init | User wants comprehensive explanations |
| max_length=128 | Phase 1 | Good balance of context vs speed |

---

## Open Issues

(None)

---

## Session Continuity

**Last Session:** 2025-01-30
**Context:**
- Phase 1 completed
- Model and dataset code written with comprehensive explanations
- Ready to begin Phase 2 (baseline training)

**Next Actions:**
1. Start Phase 2 — create training utilities
2. Build single-GPU training script
3. Test training loop on CPU locally

---

## Files Created

```
distrinews/
├── requirements.txt        ✓
├── training/
│   ├── __init__.py        ✓
│   ├── model.py           ✓
│   └── dataset.py         ✓
├── inference/              (empty)
├── checkpoints/            (empty)
└── benchmarks/             (empty)

.planning/
├── PROJECT.md             ✓
├── config.json            ✓
├── REQUIREMENTS.md        ✓
├── ROADMAP.md             ✓
├── STATE.md               ✓ (this file)
├── intel/                 ✓ (empty, for hooks)
├── research/
│   ├── CONCEPTS.md        ✓
│   ├── STACK.md           ✓
│   ├── ARCHITECTURE.md    ✓
│   ├── PITFALLS.md        ✓
│   └── SUMMARY.md         ✓
└── phases/
    └── 01-foundation/
        ├── 01-01-PLAN.md     ✓
        └── 01-01-SUMMARY.md  ✓
```

---

## Learning Notes

### Phase 1 Takeaways:
1. **Transformers use attention** — the model learns which words matter for each task
2. **Tokenization is essential** — models only understand numbers, not text
3. **Padding + attention mask** — handle variable-length inputs in batches
4. **PyTorch Dataset** — provides `__len__` and `__getitem__` for DataLoader

---
*State updated: 2025-01-30 after Phase 1 completion*
