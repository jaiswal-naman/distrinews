# Requirements: DistriNews

**Defined:** 2025-01-30
**Core Value:** Teach distributed ML training through hands-on implementation using only free resources

---

## v1 Requirements

### Model (MOD)

- [ ] **MOD-01**: DistilBERT model wrapper for 4-class text classification
- [ ] **MOD-02**: Model can be moved to specified device (CPU or GPU)
- [ ] **MOD-03**: Model can be wrapped with DDP for distributed training
- [ ] **MOD-04**: Model forward pass returns logits for classification

### Dataset (DAT)

- [ ] **DAT-01**: Load AG News dataset from HuggingFace datasets
- [ ] **DAT-02**: Tokenize text using DistilBERT tokenizer
- [ ] **DAT-03**: Create PyTorch Dataset that returns tokenized samples
- [ ] **DAT-04**: Support DistributedSampler for data sharding

### Single-GPU Training (SGT)

- [ ] **SGT-01**: Train model on single GPU as baseline
- [ ] **SGT-02**: Log training loss per epoch
- [ ] **SGT-03**: Log training accuracy per epoch
- [ ] **SGT-04**: Track epoch training time
- [ ] **SGT-05**: Save checkpoint after training

### Distributed Training (DDP)

- [ ] **DDP-01**: Initialize distributed process group with NCCL backend
- [ ] **DDP-02**: Each process uses its assigned GPU (LOCAL_RANK)
- [ ] **DDP-03**: Model wrapped with DistributedDataParallel
- [ ] **DDP-04**: Data sharded with DistributedSampler
- [ ] **DDP-05**: sampler.set_epoch() called every epoch
- [ ] **DDP-06**: Gradients synchronized automatically via AllReduce
- [ ] **DDP-07**: Only rank 0 logs to console
- [ ] **DDP-08**: Only rank 0 saves checkpoint
- [ ] **DDP-09**: Proper cleanup with destroy_process_group

### Checkpointing (CHK)

- [ ] **CHK-01**: Save model state_dict (not DDP wrapper)
- [ ] **CHK-02**: Save tokenizer alongside model
- [ ] **CHK-03**: Checkpoint saved to checkpoints/ directory
- [ ] **CHK-04**: Checkpoint filename: distilbert_agnews.pt

### Inference API (API)

- [ ] **API-01**: FastAPI server with /predict endpoint
- [ ] **API-02**: Input: JSON with "text" field
- [ ] **API-03**: Output: JSON with "label" and "confidence"
- [ ] **API-04**: Model loaded once at startup
- [ ] **API-05**: Auto-detect GPU, fallback to CPU
- [ ] **API-06**: Input validation with error messages
- [ ] **API-07**: Health check endpoint (/health)

### Benchmarking (BEN)

- [ ] **BEN-01**: Document single-GPU training time per epoch
- [ ] **BEN-02**: Document DDP 2-GPU training time per epoch
- [ ] **BEN-03**: Document final accuracy for both configurations
- [ ] **BEN-04**: Calculate and explain observed speedup
- [ ] **BEN-05**: Results in benchmarks/single_gpu.md
- [ ] **BEN-06**: Results in benchmarks/ddp_2gpu.md

### Documentation (DOC)

- [ ] **DOC-01**: README with project overview
- [ ] **DOC-02**: README explains how DDP works
- [ ] **DOC-03**: README explains inference architecture
- [ ] **DOC-04**: README with setup instructions
- [ ] **DOC-05**: README with training commands (1 GPU vs 2 GPU)
- [ ] **DOC-06**: Code comments explaining WHY, not just WHAT
- [ ] **DOC-07**: Interview-ready summary in README

### Launcher (LCH)

- [ ] **LCH-01**: Shell script to launch DDP training (run_ddp.sh)
- [ ] **LCH-02**: Uses torchrun with --nproc_per_node
- [ ] **LCH-03**: Configurable number of GPUs

### Deployment (DEP)

- [ ] **DEP-01**: Deployable to Hugging Face Spaces
- [ ] **DEP-02**: Requirements.txt with all dependencies
- [ ] **DEP-03**: Instructions for deployment in README

---

## v2 Requirements (Deferred)

### Enhanced Features

- **ENH-01**: Mixed precision training (fp16)
- **ENH-02**: Learning rate scheduler
- **ENH-03**: Gradient accumulation support
- **ENH-04**: TensorBoard logging
- **ENH-05**: Model evaluation on test set during training
- **ENH-06**: Early stopping

### Additional Deployment

- **DEP-04**: Docker container for inference
- **DEP-05**: Kubernetes deployment config
- **DEP-06**: CI/CD pipeline

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-node training | Single machine is enough for learning DDP concepts |
| Jupyter notebooks | Production-style .py files only per spec |
| Unit tests | Focus on core training/inference for v1 |
| OAuth/authentication | Beyond scope of ML training project |
| Database storage | Not needed for inference API |
| Model quantization | Optimization can wait for v2 |
| A/B testing | Beyond scope |
| Async training | DDP is synchronous, keep it simple |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MOD-01 | Phase 1 | Pending |
| MOD-02 | Phase 1 | Pending |
| MOD-03 | Phase 1 | Pending |
| MOD-04 | Phase 1 | Pending |
| DAT-01 | Phase 1 | Pending |
| DAT-02 | Phase 1 | Pending |
| DAT-03 | Phase 1 | Pending |
| DAT-04 | Phase 1 | Pending |
| SGT-01 | Phase 2 | Pending |
| SGT-02 | Phase 2 | Pending |
| SGT-03 | Phase 2 | Pending |
| SGT-04 | Phase 2 | Pending |
| SGT-05 | Phase 2 | Pending |
| DDP-01 | Phase 3 | Pending |
| DDP-02 | Phase 3 | Pending |
| DDP-03 | Phase 3 | Pending |
| DDP-04 | Phase 3 | Pending |
| DDP-05 | Phase 3 | Pending |
| DDP-06 | Phase 3 | Pending |
| DDP-07 | Phase 3 | Pending |
| DDP-08 | Phase 3 | Pending |
| DDP-09 | Phase 3 | Pending |
| CHK-01 | Phase 3 | Pending |
| CHK-02 | Phase 3 | Pending |
| CHK-03 | Phase 3 | Pending |
| CHK-04 | Phase 3 | Pending |
| API-01 | Phase 4 | Pending |
| API-02 | Phase 4 | Pending |
| API-03 | Phase 4 | Pending |
| API-04 | Phase 4 | Pending |
| API-05 | Phase 4 | Pending |
| API-06 | Phase 4 | Pending |
| API-07 | Phase 4 | Pending |
| BEN-01 | Phase 5 | Pending |
| BEN-02 | Phase 5 | Pending |
| BEN-03 | Phase 5 | Pending |
| BEN-04 | Phase 5 | Pending |
| BEN-05 | Phase 5 | Pending |
| BEN-06 | Phase 5 | Pending |
| LCH-01 | Phase 3 | Pending |
| LCH-02 | Phase 3 | Pending |
| LCH-03 | Phase 3 | Pending |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |
| DOC-03 | Phase 6 | Pending |
| DOC-04 | Phase 6 | Pending |
| DOC-05 | Phase 6 | Pending |
| DOC-06 | All | Pending |
| DOC-07 | Phase 6 | Pending |
| DEP-01 | Phase 7 | Pending |
| DEP-02 | Phase 1 | Pending |
| DEP-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 46 total
- Mapped to phases: 46
- Unmapped: 0 ✓

---
*Requirements defined: 2025-01-30*
*Last updated: 2025-01-30 after initial definition*
