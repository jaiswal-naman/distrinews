# Project State: DistriNews

**Last Updated:** 2025-01-30

---

## Project Reference

See: `.planning/PROJECT.md` (updated 2025-01-30)

**Core value:** Teach distributed ML training through hands-on implementation using only free resources
**Current focus:** Phase 4 — Inference API

---

## Current Position

```
Phase 1 ✓ ──► Phase 2 ✓ ──► Phase 3 ✓ ──► Phase 4 ◆ ──► Phase 5 ○ ──► Phase 6 ○ ──► Phase 7 ○
Foundation   Baseline    DDP       Inference   Benchmark   Document   Deploy
   DONE        DONE      DONE        NEXT

◆ = Current    ○ = Pending    ✓ = Complete
```

**Current Phase:** Phase 4 — Inference API
**Current Task:** Not started
**Blocked By:** None

---

## Phase Status

| Phase | Status | Plans | Progress |
|-------|--------|-------|----------|
| 1 - Foundation | ✓ Complete | 1/1 | 100% |
| 2 - Baseline | ✓ Complete | 1/1 | 100% |
| 3 - DDP | ✓ Complete | 1/1 | 100% |
| 4 - Inference | ◆ Current | 0/0 | 0% |
| 5 - Benchmark | ○ Pending | 0/0 | 0% |
| 6 - Documentation | ○ Pending | 0/0 | 0% |
| 7 - Deployment | ○ Pending | 0/0 | 0% |

---

## Recent Progress

### Phase 3: DDP ✓ (2025-01-30)

**Files created:**
- `training/train_ddp.py` — Full DDP training script
- `training/run_ddp.sh` — torchrun launcher (Linux/Mac)
- `training/run_ddp.bat` — torchrun launcher (Windows)

**DDP concepts learned:**
- Process groups and init_process_group()
- RANK, LOCAL_RANK, WORLD_SIZE
- DistributedDataParallel wrapper
- DistributedSampler for data sharding
- set_epoch() for proper shuffling
- Rank 0 only logging/checkpointing
- Clean shutdown with destroy_process_group()

### Phase 2: Baseline ✓ (2025-01-30)
- Training loop, utils, checkpointing

### Phase 1: Foundation ✓ (2025-01-30)
- Model wrapper, dataset loader, dependencies

---

## Key Decisions Made

| Decision | When | Rationale |
|----------|------|-----------|
| Use Kaggle for training | Init | Free 2x T4 GPUs, 30 hrs/week |
| Use HF Spaces for deploy | Init | Free FastAPI hosting, git-based |
| DistilBERT over BERT | Init | Faster training, fits on T4 |
| AdamW with lr=2e-5 | Phase 2 | Standard for transformer fine-tuning |
| NCCL backend | Phase 3 | Fastest for GPU-to-GPU communication |

---

## Session Continuity

**Last Session:** 2025-01-30
**Context:**
- Phases 1, 2, 3 completed
- Have full DDP training pipeline
- Ready to build inference API

**Next Actions:**
1. Start Phase 4 — build FastAPI inference server
2. Create model_loader.py for checkpoint loading
3. Create app.py with /predict endpoint

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
│   ├── train_single.py    ✓
│   ├── train_ddp.py       ✓  ← THE MAIN LEARNING
│   ├── run_ddp.sh         ✓
│   └── run_ddp.bat        ✓
├── inference/              (empty - Phase 4)
├── checkpoints/            (empty - created during training)
└── benchmarks/             (empty - Phase 5)
```

---

## Learning Notes

### Phase 3 Takeaways (THE CORE):
1. **DDP = Same model, different data, synced gradients**
2. **init_process_group()** — All GPUs join a communication group
3. **LOCAL_RANK** — Which GPU this process uses
4. **DDP wrapper** — Adds gradient sync hooks to backward()
5. **DistributedSampler** — Ensures no data overlap between GPUs
6. **set_epoch()** — Different shuffle each epoch (uses epoch as seed)
7. **Rank 0 only** — Logging and checkpointing to avoid duplicates
8. **backward() does the magic** — AllReduce happens automatically

### Interview One-Liner:
> "DDP wraps the model and hooks into backward(). When loss.backward() is called,
> DDP performs AllReduce to average gradients across all GPUs. Each GPU then
> makes identical updates, keeping models synchronized."

---
*State updated: 2025-01-30 after Phase 3 completion*
