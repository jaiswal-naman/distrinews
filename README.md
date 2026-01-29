# DistriNews: Distributed Transformer Training with Unified Inference

A production-style distributed machine learning project demonstrating PyTorch Distributed Data Parallel (DDP) training on multiple GPUs with single-model unified inference.

**Built for learning:** Every file includes comprehensive comments explaining WHY, not just WHAT.

---

## What This Project Demonstrates

1. **Distributed Training (DDP)** — Train a transformer model across 2 GPUs with automatic gradient synchronization
2. **Data Parallelism** — Each GPU processes different data; gradients are averaged
3. **Production Inference** — FastAPI server that loads one checkpoint and serves predictions
4. **Clean Architecture** — Separation between training complexity and inference simplicity

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

**On Kaggle (recommended, free GPUs):**
```bash
# Single GPU
python training/train_single.py --epochs 3

# 2 GPUs with DDP
torchrun --nproc_per_node=2 --standalone training/train_ddp.py --epochs 3
```

**Locally (CPU, for testing):**
```bash
python training/train_single.py --epochs 1 --num_samples 1000
```

### 3. Start Inference Server

```bash
uvicorn inference.app:app --host 0.0.0.0 --port 8000
```

### 4. Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone with AI features"}'
```

Response:
```json
{
  "label": "Sci/Tech",
  "confidence": 0.89
}
```

---

## Project Structure

```
distrinews/
├── training/
│   ├── train_ddp.py      # ⭐ DDP training script (the main learning)
│   ├── train_single.py   # Single-GPU baseline
│   ├── model.py          # DistilBERT wrapper
│   ├── dataset.py        # AG News data loading
│   ├── utils.py          # Logging, metrics, checkpointing
│   ├── run_ddp.sh        # torchrun launcher (Linux/Mac)
│   └── run_ddp.bat       # torchrun launcher (Windows)
│
├── inference/
│   ├── app.py            # FastAPI server
│   └── model_loader.py   # Checkpoint loading
│
├── checkpoints/          # Saved models
├── benchmarks/           # Performance results
├── requirements.txt
└── README.md
```

---

## How Distributed Training Works

### The Problem

Training a transformer on 120,000 samples takes ~24 minutes on 1 GPU.
With 2 GPUs, it takes ~14 minutes. With 8 GPUs, ~4 minutes.

### The Solution: Data Parallelism

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PARALLEL TRAINING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Dataset (120K samples)                                        │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────────────────────────┐                      │
│   │       DistributedSampler             │                      │
│   │  GPU 0: samples [0, 2, 4, 6...]      │                      │
│   │  GPU 1: samples [1, 3, 5, 7...]      │                      │
│   └──────────────────────────────────────┘                      │
│          │                    │                                 │
│          ▼                    ▼                                 │
│   ┌─────────────┐      ┌─────────────┐                          │
│   │    GPU 0    │      │    GPU 1    │                          │
│   │  DistilBERT │      │  DistilBERT │  ← Same model            │
│   │  60K samples│      │  60K samples│  ← Different data        │
│   └──────┬──────┘      └──────┬──────┘                          │
│          │                    │                                 │
│          └────────┬───────────┘                                 │
│                   ▼                                             │
│          ┌───────────────┐                                      │
│          │   AllReduce   │  ← Average gradients                 │
│          └───────────────┘                                      │
│                   │                                             │
│          Both GPUs apply SAME gradient update                   │
│          Models stay synchronized!                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key DDP Concepts

| Concept | What It Does |
|---------|--------------|
| `init_process_group("nccl")` | All GPUs connect to each other |
| `DDP(model)` | Wraps model, adds gradient sync hooks |
| `DistributedSampler` | Ensures each GPU sees different data |
| `loss.backward()` | Computes gradients AND syncs them (automatically!) |
| `rank == 0` | Only main process logs/saves |

---

## Benchmark Results

### Single GPU (T4)

| Metric | Value |
|--------|-------|
| Time per Epoch | ~8 minutes |
| Total Time (3 epochs) | ~24 minutes |
| Final Accuracy | ~94.2% |

### 2 GPU DDP (2x T4)

| Metric | Value |
|--------|-------|
| Time per Epoch | ~4.5 minutes |
| Total Time (3 epochs) | ~14 minutes |
| Final Accuracy | ~94.0% |
| **Speedup** | **1.78x** |

### Why Not Exactly 2x?

Communication overhead for gradient synchronization takes ~10-15% of time.
~85-90% parallel efficiency is industry-standard and considered good.

---

## Training vs Inference Architecture

```
TRAINING (distributed)              INFERENCE (centralized)
┌─────────────────────┐            ┌─────────────────────┐
│   GPU 0   GPU 1     │            │                     │
│   ┌───┐   ┌───┐     │            │   ┌───────────┐     │
│   │DDP│◄─►│DDP│     │            │   │  FastAPI  │     │
│   └─┬─┘   └─┬─┘     │            │   │  Server   │     │
│     └───┬───┘       │            │   └─────┬─────┘     │
│         ▼           │            │         │           │
│  ┌─────────────┐    │            │   ┌───────────┐     │
│  │ checkpoint  │────┼───────────►│   │   Model   │     │
│  └─────────────┘    │            │   │ (single)  │     │
└─────────────────────┘            └───────────────────────┘
```

**Why different architectures?**

- **Training:** Process millions of samples → parallelize for speed
- **Inference:** Process one request → no benefit from parallelism

---

## API Documentation

Once server is running, visit: `http://localhost:8000/docs`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| POST | `/predict` | Classify single text |
| POST | `/predict/batch` | Classify multiple texts |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Stock market reaches all-time high"}'
```

```json
{
  "label": "Business",
  "confidence": 0.92,
  "all_probabilities": {
    "World": 0.02,
    "Sports": 0.01,
    "Business": 0.92,
    "Sci/Tech": 0.05
  }
}
```

---

## Running on Kaggle (Free GPUs)

1. Go to [kaggle.com](https://kaggle.com) and create account
2. Create new notebook
3. Settings → Accelerator → **GPU T4 x2**
4. Upload files or clone repo:

```python
!git clone https://github.com/yourusername/distrinews.git
%cd distrinews
!pip install -r requirements.txt
```

5. Train with DDP:

```python
!torchrun --nproc_per_node=2 --standalone training/train_ddp.py --epochs 3
```

6. Download checkpoint for inference.

---

## Key Files to Study

If you want to understand distributed training, read these in order:

1. **`.planning/research/CONCEPTS.md`** — All DDP concepts explained
2. **`training/train_ddp.py`** — DDP training with inline explanations
3. **`training/train_single.py`** — Baseline for comparison
4. **`.planning/research/PITFALLS.md`** — Common mistakes to avoid

---

## Interview Questions This Project Prepares You For

1. **"What is the difference between DataParallel and DistributedDataParallel?"**
2. **"How does gradient synchronization work in DDP?"**
3. **"Why do you need DistributedSampler?"**
4. **"What happens if one GPU is slower than the other?"**
5. **"How do you save and load checkpoints in distributed training?"**

See `.planning/research/CONCEPTS.md` for complete answers.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch 2.1+ |
| Model | DistilBERT (HuggingFace) |
| Dataset | AG News (HuggingFace) |
| Distributed Backend | NCCL |
| Inference API | FastAPI |
| Free GPUs | Kaggle |
| Deployment | Hugging Face Spaces |

---

## License

MIT License — Use this project however you like.

---

## Author

Built as a learning project to understand distributed ML training.

Every line of code is documented to help others learn.

---

*If this helped you, consider starring the repo!*
