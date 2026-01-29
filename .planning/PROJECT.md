# DistriNews

## What This Is

An educational, production-style distributed machine learning project that teaches PyTorch Distributed Data Parallel (DDP) training through hands-on implementation. The learner builds a news classifier using DistilBERT on the AG News dataset, trains it across multiple GPUs using DDP, and deploys it as a FastAPI inference service — all using free resources.

This is a learning-first project: every piece of code includes explanations of WHY it exists and HOW it works.

## Core Value

**Teach distributed ML training concepts through building a real, deployable system — using only free resources.**

The learner must understand gradient synchronization, data sharding, and the difference between training scale and inference simplicity by the end.

## Requirements

### Validated

(None yet — ship to validate)

### Active

**Understanding & Documentation**
- [ ] Conceptual guide explaining DDP, gradient sync, data parallelism
- [ ] Architecture diagrams showing training vs inference
- [ ] Code comments explaining WHY, not just WHAT

**Training Infrastructure**
- [ ] DistilBERT model wrapper for text classification
- [ ] AG News dataset loader with tokenization
- [ ] Single-GPU training script (baseline)
- [ ] DDP training script for multi-GPU
- [ ] Training utilities (logging, metrics, checkpointing)
- [ ] Shell script launcher for torchrun

**Inference Infrastructure**
- [ ] Model loading logic (from checkpoint)
- [ ] FastAPI server with /predict endpoint
- [ ] Error handling and input validation

**Benchmarking**
- [ ] Single-GPU training benchmark documentation
- [ ] DDP 2-GPU training benchmark documentation
- [ ] Speedup analysis and explanation

**Deployment**
- [ ] Hugging Face Spaces deployment (free)
- [ ] Production-ready README

**Learning Materials**
- [ ] Step-by-step setup guide
- [ ] Kaggle notebook for free GPU access
- [ ] Troubleshooting guide

### Out of Scope

- Docker/containerization — adds complexity without learning value for DDP concepts
- Unit tests — focus on core training/inference understanding first
- Paid cloud services — must be 100% free
- Jupyter notebooks in main codebase — production-style .py files only
- Multi-node training — single machine, multi-GPU is enough for learning
- Mixed precision training — keep it simple for learning

## Context

**Learner Profile:**
- No prior distributed ML experience
- Wants to understand DDP deeply for interviews
- No GPU hardware locally
- Needs free cloud resources

**Technical Environment:**
- Local development on Windows (CPU)
- Remote training on Kaggle (2x T4 GPUs, free)
- Deployment on Hugging Face Spaces (free)

**Why This Matters:**
Distributed training is a core skill for ML engineering roles. Understanding DDP — how gradients synchronize, how data shards across GPUs, why we use DistributedSampler — separates junior from senior ML engineers.

## Constraints

- **Cost**: $0 budget — all resources must be free tier
- **Hardware**: No local GPUs — must work on CPU locally, use free cloud GPUs
- **Complexity**: Every concept must be explained — this is educational
- **Structure**: Strict project structure per spec — no shortcuts

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Kaggle for GPU training | Free 2x T4 GPUs, 30 hrs/week | — Pending |
| Hugging Face Spaces for deployment | Free FastAPI hosting, easy setup | — Pending |
| CPU simulation for local dev | Learn DDP mechanics without GPUs | — Pending |
| DistilBERT over BERT | Fits single GPU, faster training, same concepts | — Pending |

---
*Last updated: 2025-01-30 after initialization*
