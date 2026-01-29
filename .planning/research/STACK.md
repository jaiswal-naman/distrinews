# Technology Stack

Every technology choice for DistriNews, with explanations of WHY.

---

## Core Dependencies

### PyTorch (torch)
**Version:** 2.1+ (2024/2025 stable)
**Why:**
- Industry standard for ML research and production
- Built-in distributed training support (DDP)
- Excellent debugging (eager execution)
- Required for DistributedDataParallel

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```

### Transformers (HuggingFace)
**Version:** 4.36+
**Why:**
- Pre-trained DistilBERT model
- Tokenizers optimized for transformers
- Industry standard for NLP

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
```

### Datasets (HuggingFace)
**Version:** 2.16+
**Why:**
- Easy access to AG News dataset
- Efficient data loading with memory mapping
- Consistent API

```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
```

### FastAPI
**Version:** 0.109+
**Why:**
- Modern, fast Python web framework
- Automatic API documentation
- Async support
- Type hints / validation with Pydantic

```python
from fastapi import FastAPI
app = FastAPI()
```

### Uvicorn
**Version:** 0.27+
**Why:**
- ASGI server for FastAPI
- Production-ready
- Easy to use

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Full requirements.txt

```
# Core ML
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0

# Inference API
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0

# Utilities
tqdm>=4.66.0
numpy>=1.24.0
```

---

## Why DistilBERT (Not BERT or GPT)

| Model | Parameters | Memory | Training Time | Accuracy |
|-------|-----------|--------|---------------|----------|
| BERT-base | 110M | ~4GB | Slower | 94.5% |
| **DistilBERT** | **66M** | **~2.5GB** | **Faster** | **93.5%** |
| GPT-2 | 124M | ~5GB | Much slower | 92% |

**DistilBERT is ideal because:**
1. **Fits on free GPU** - Kaggle T4 has 16GB, DistilBERT uses ~2.5GB
2. **Trains fast** - Important for learning and iterating
3. **Still transformer-based** - Teaches same concepts as BERT
4. **Good accuracy** - Only 1% below BERT
5. **Production realistic** - Used in real deployments

---

## Why AG News Dataset

| Dataset | Samples | Classes | Task |
|---------|---------|---------|------|
| IMDB | 50K | 2 | Sentiment |
| **AG News** | **127K** | **4** | **Classification** |
| Yelp | 650K | 5 | Sentiment |

**AG News is ideal because:**
1. **Right size** - Large enough to see DDP speedup, small enough to train on free GPUs
2. **Multi-class** - More interesting than binary (4 classes)
3. **Real data** - Actual news articles, not synthetic
4. **Standard benchmark** - Commonly used, easy to compare

---

## Why NCCL Backend

| Backend | Speed | GPU Support | CPU Support |
|---------|-------|-------------|-------------|
| **NCCL** | **Fastest** | **Yes** | No |
| Gloo | Medium | Yes | Yes |
| MPI | Medium | Yes | Yes |

**NCCL is ideal because:**
- Optimized for NVIDIA GPUs (which Kaggle uses)
- Utilizes fast GPU interconnects
- Industry standard for distributed GPU training

---

## Why torchrun (Not torch.distributed.launch)

`torch.distributed.launch` is deprecated. `torchrun` is the modern replacement.

```bash
# Old (deprecated)
python -m torch.distributed.launch --nproc_per_node=2 train.py

# New (use this)
torchrun --nproc_per_node=2 train.py
```

**torchrun benefits:**
- Automatic error handling
- Better process management
- Elastic training support (not used here, but good to know)

---

## Why Hugging Face Spaces for Deployment

| Platform | Free Tier | GPU | Ease | FastAPI |
|----------|-----------|-----|------|---------|
| **HF Spaces** | **Yes** | **CPU** | **Easy** | **Yes** |
| AWS Lambda | Limited | No | Medium | Yes |
| GCP Run | Limited | No | Medium | Yes |
| Render | Yes | No | Easy | Yes |

**Hugging Face Spaces is ideal because:**
1. **Free** - No credit card required
2. **Git-based** - Push code, auto-deploys
3. **FastAPI support** - Via Docker or Gradio backend
4. **ML-focused** - Designed for ML model serving
5. **Community** - Easy to share your work

---

## Directory-to-Dependency Mapping

```
distrinews/
├── training/
│   ├── train_ddp.py      → torch, torch.distributed, transformers
│   ├── model.py          → transformers (DistilBertForSequenceClassification)
│   ├── dataset.py        → datasets, transformers (tokenizer)
│   └── utils.py          → torch, tqdm
│
├── inference/
│   ├── app.py            → fastapi, uvicorn
│   └── model_loader.py   → torch, transformers
```

---

## Version Pinning Strategy

For reproducibility, we pin major.minor, not patch:

```
torch>=2.1.0,<2.3.0
transformers>=4.36.0,<4.40.0
```

**Why:**
- Patch updates (2.1.0 → 2.1.1) are safe, include bug fixes
- Minor updates (2.1 → 2.2) may change behavior
- Major updates (2.x → 3.x) break things

---

## Local Development vs Kaggle

| Environment | PyTorch | CUDA | Purpose |
|-------------|---------|------|---------|
| Local (your PC) | CPU-only | No | Code development, testing |
| Kaggle | GPU (T4) | Yes | Actual DDP training |

**Workflow:**
1. Write code locally (CPU)
2. Test with small batches locally
3. Upload to Kaggle for real training
4. Download checkpoint
5. Deploy to HF Spaces

---

*This stack is battle-tested for learning distributed ML. Don't substitute components unless you have a specific reason.*
