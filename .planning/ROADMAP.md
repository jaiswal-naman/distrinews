# Roadmap: DistriNews

**Version:** 1.0.0
**Created:** 2025-01-30
**Phases:** 7

---

## Overview

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
Foundation   Baseline    DDP       Inference   Benchmark   Document   Deploy
(model+data) (1 GPU)    (2 GPU)   (API)       (compare)   (README)   (HF Spaces)
```

---

## Phase 1: Foundation — Model & Dataset

**Goal:** Build the core model and dataset components that all training scripts will use.

**Requirements:** MOD-01, MOD-02, MOD-03, MOD-04, DAT-01, DAT-02, DAT-03, DAT-04, DEP-02

**Deliverables:**
- `training/model.py` — DistilBERT classifier wrapper
- `training/dataset.py` — AG News dataset loader + tokenization
- `requirements.txt` — All project dependencies

**Success Criteria:**
1. Model can be instantiated and moved to device
2. Dataset loads AG News and returns tokenized samples
3. Model produces correct output shape (batch, 4) for 4-class classification
4. Code has comments explaining WHY each design choice

**Why This Phase First:**
Everything depends on model and data. Build these right, and the rest is straightforward.

---

## Phase 2: Baseline Training — Single GPU

**Goal:** Create working single-GPU training to establish baseline metrics.

**Requirements:** SGT-01, SGT-02, SGT-03, SGT-04, SGT-05

**Deliverables:**
- `training/utils.py` — Logging, metrics, helpers
- `training/train_single.py` — Single-GPU training script

**Success Criteria:**
1. Training loop runs on CPU (for local testing)
2. Loss decreases over epochs
3. Accuracy tracked and logged
4. Epoch time measured
5. Checkpoint saved after training
6. Code has comments explaining training loop logic

**Why This Phase:**
Before distributed training, you need a working baseline. This also teaches the fundamentals.

---

## Phase 3: Distributed Training — DDP

**Goal:** Convert single-GPU training to multi-GPU DDP training.

**Requirements:** DDP-01 through DDP-09, CHK-01 through CHK-04, LCH-01 through LCH-03

**Deliverables:**
- `training/train_ddp.py` — DDP training script
- `training/run_ddp.sh` — torchrun launcher
- `checkpoints/` directory

**Success Criteria:**
1. Process group initializes with NCCL
2. Each GPU uses correct LOCAL_RANK
3. Model wrapped with DDP
4. DistributedSampler shards data correctly
5. set_epoch() called every epoch
6. Logs appear only from rank 0
7. Checkpoint saved only by rank 0
8. Clean shutdown with destroy_process_group
9. Code has comments explaining EVERY DDP concept

**Why This Phase:**
This is the core learning objective. Take your time to understand each component.

---

## Phase 4: Inference API

**Goal:** Create FastAPI server that loads trained model and serves predictions.

**Requirements:** API-01 through API-07

**Deliverables:**
- `inference/model_loader.py` — Model loading utilities
- `inference/app.py` — FastAPI application

**Success Criteria:**
1. Server starts and loads model at startup
2. POST /predict accepts {"text": "..."} and returns {"label": "...", "confidence": 0.xx}
3. GET /health returns healthy status
4. Invalid input returns clear error messages
5. Auto-detects GPU, falls back to CPU
6. Code has comments explaining API design choices

**Why This Phase:**
Shows the complete ML lifecycle: train → save → load → serve.

---

## Phase 5: Benchmarking

**Goal:** Run training on Kaggle, collect metrics, document speedup.

**Requirements:** BEN-01 through BEN-06

**Deliverables:**
- `benchmarks/single_gpu.md` — 1 GPU results
- `benchmarks/ddp_2gpu.md` — 2 GPU DDP results

**Success Criteria:**
1. Single-GPU training completed on Kaggle
2. DDP training completed on Kaggle (2 GPUs)
3. Time per epoch documented for both
4. Final accuracy documented for both
5. Speedup calculated and explained
6. Why speedup isn't exactly 2x explained

**Why This Phase:**
Numbers prove the concept. This is what you'll discuss in interviews.

---

## Phase 6: Documentation

**Goal:** Create comprehensive README and finalize code documentation.

**Requirements:** DOC-01 through DOC-07

**Deliverables:**
- `README.md` — Complete project documentation

**Success Criteria:**
1. README explains what the project is
2. README explains how DDP works (high-level)
3. README explains inference architecture
4. Setup instructions are clear and complete
5. Training commands documented (1 GPU vs 2 GPU)
6. Benchmark results included
7. Interview-ready summary section

**Why This Phase:**
Documentation makes your project portfolio-worthy. Recruiters and interviewers read READMEs.

---

## Phase 7: Deployment

**Goal:** Deploy inference API to Hugging Face Spaces.

**Requirements:** DEP-01, DEP-03

**Deliverables:**
- Live API on Hugging Face Spaces
- Deployment instructions in README

**Success Criteria:**
1. API accessible via public URL
2. /predict endpoint works remotely
3. Deployment steps documented
4. Public URL in README

**Why This Phase:**
Deployed = real. Anyone can test your model. Shows end-to-end capability.

---

## Phase Summary

| Phase | Name | Requirements | Key Output |
|-------|------|--------------|------------|
| 1 | Foundation | 9 | model.py, dataset.py |
| 2 | Baseline | 5 | train_single.py |
| 3 | DDP | 16 | train_ddp.py, run_ddp.sh |
| 4 | Inference | 7 | app.py |
| 5 | Benchmark | 6 | benchmark docs |
| 6 | Documentation | 7 | README.md |
| 7 | Deployment | 2 | Live API |

**Total:** 46 requirements across 7 phases

---

## Progress

- [ ] Phase 1: Foundation
- [ ] Phase 2: Baseline Training
- [ ] Phase 3: Distributed Training
- [ ] Phase 4: Inference API
- [ ] Phase 5: Benchmarking
- [ ] Phase 6: Documentation
- [ ] Phase 7: Deployment

---
*Roadmap created: 2025-01-30*
*Last updated: 2025-01-30*
