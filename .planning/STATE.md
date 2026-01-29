# Project State: DistriNews

**Last Updated:** 2025-01-30

---

## Project Reference

See: `.planning/PROJECT.md` (updated 2025-01-30)

**Core value:** Teach distributed ML training through hands-on implementation using only free resources
**Current focus:** Phase 3 — Distributed Training (DDP)

---

## Current Position

```
Phase 1 ✓ ──► Phase 2 ✓ ──► Phase 3 ◆ ──► Phase 4 ○ ──► Phase 5 ○ ──► Phase 6 ○ ──► Phase 7 ○
Foundation   Baseline    DDP       Inference   Benchmark   Document   Deploy
   DONE        DONE      NEXT

◆ = Current    ○ = Pending    ✓ = Complete
```

**Current Phase:** Phase 3 — Distributed Training
**Current Task:** Not started
**Blocked By:** None

---

## Phase Status

| Phase | Status | Plans | Progress |
|-------|--------|-------|----------|
| 1 - Foundation | ✓ Complete | 1/1 | 100% |
| 2 - Baseline | ✓ Complete | 1/1 | 100% |
| 3 - DDP | ◆ Current | 0/0 | 0% |
| 4 - Inference | ○ Pending | 0/0 | 0% |
| 5 - Benchmark | ○ Pending | 0/0 | 0% |
| 6 - Documentation | ○ Pending | 0/0 | 0% |
| 7 - Deployment | ○ Pending | 0/0 | 0% |

---

## Recent Progress

### Phase 2: Baseline ✓ (2025-01-30)

**Files created:**
- `training/utils.py` — Helper functions (metrics, checkpointing)
- `training/train_single.py` — Complete single-GPU training script

**Concepts learned:**
- The 5-step training loop (forward, loss, backward, optimize, zero_grad)
- What backpropagation does (computes gradients)
- How loss guides learning (lower loss = better predictions)
- Why we need optimizer.zero_grad() (prevent gradient accumulation)
- model.train() vs model.eval() modes

### Phase 1: Foundation ✓ (2025-01-30)

**Files created:**
- `requirements.txt`, `training/model.py`, `training/dataset.py`

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
| AdamW with lr=2e-5 | Phase 2 | Standard for transformer fine-tuning |
| weight_decay=0.01 | Phase 2 | Prevents overfitting |

---

## Open Issues

(None)

---

## Session Continuity

**Last Session:** 2025-01-30
**Context:**
- Phase 1 and 2 completed
- Have working model, dataset, and single-GPU training
- Ready for the core learning: DDP distributed training

**Next Actions:**
1. Start Phase 3 — distributed training with DDP
2. Create train_ddp.py with DDP wrapper
3. Create run_ddp.sh launcher script
4. Test on Kaggle with 2 GPUs

---

## Files Created

```
distrinews/
├── requirements.txt        ✓
├── training/
│   ├── __init__.py        ✓
│   ├── model.py           ✓
│   ├── dataset.py         ✓
│   ├── utils.py           ✓
│   └── train_single.py    ✓
├── inference/              (empty)
├── checkpoints/            (empty)
└── benchmarks/             (empty)

.planning/
├── PROJECT.md             ✓
├── config.json            ✓
├── REQUIREMENTS.md        ✓
├── ROADMAP.md             ✓
├── STATE.md               ✓ (this file)
├── intel/                 ✓ (empty)
├── research/              ✓ (5 files)
└── phases/
    ├── 01-foundation/     ✓
    └── 02-baseline/       ✓
```

---

## Learning Notes

### Phase 1 Takeaways:
1. **Transformers use attention** — model learns which words matter
2. **Tokenization is essential** — models only understand numbers
3. **Padding + attention mask** — handle variable-length inputs
4. **PyTorch Dataset** — provides `__len__` and `__getitem__`

### Phase 2 Takeaways:
1. **Training loop = 5 steps** — forward, loss, backward, optimize, zero_grad
2. **Gradients point to loss** — positive = increase weight increases loss
3. **Learning rate controls step size** — too high = unstable, too low = slow
4. **AdamW is standard** — for transformer fine-tuning
5. **model.eval() + no_grad()** — for inference (faster, less memory)

---
*State updated: 2025-01-30 after Phase 2 completion*
