# System Architecture

How all the pieces fit together.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DistriNews Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PHASE 1: TRAINING (Kaggle)                        │    │
│  │                                                                     │    │
│  │   ┌──────────────┐                                                  │    │
│  │   │  AG News     │                                                  │    │
│  │   │  Dataset     │                                                  │    │
│  │   │  (127K)      │                                                  │    │
│  │   └──────┬───────┘                                                  │    │
│  │          │                                                          │    │
│  │          ▼                                                          │    │
│  │   ┌──────────────────────────────────────────┐                      │    │
│  │   │        DistributedSampler                │                      │    │
│  │   │   Splits data between GPUs               │                      │    │
│  │   └─────────────┬────────────┬───────────────┘                      │    │
│  │                 │            │                                      │    │
│  │          ┌──────┘            └──────┐                               │    │
│  │          ▼                          ▼                               │    │
│  │   ┌─────────────┐            ┌─────────────┐                        │    │
│  │   │   GPU 0     │            │   GPU 1     │                        │    │
│  │   │   (T4)      │◄──NCCL────►│   (T4)      │  Gradient Sync         │    │
│  │   │             │            │             │                        │    │
│  │   │ DistilBERT  │            │ DistilBERT  │  Same model            │    │
│  │   │   + DDP     │            │   + DDP     │                        │    │
│  │   └──────┬──────┘            └─────────────┘                        │    │
│  │          │                                                          │    │
│  │          │ (rank 0 only)                                            │    │
│  │          ▼                                                          │    │
│  │   ┌──────────────┐                                                  │    │
│  │   │ Checkpoint   │  distilbert_agnews.pt                            │    │
│  │   │ (weights)    │                                                  │    │
│  │   └──────┬───────┘                                                  │    │
│  │          │                                                          │    │
│  └──────────┼──────────────────────────────────────────────────────────┘    │
│             │                                                               │
│             │  Download from Kaggle                                         │
│             ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PHASE 2: INFERENCE (HF Spaces)                    │    │
│  │                                                                     │    │
│  │   ┌──────────────┐                                                  │    │
│  │   │ Checkpoint   │  Upload to deployment                            │    │
│  │   │ (weights)    │                                                  │    │
│  │   └──────┬───────┘                                                  │    │
│  │          │                                                          │    │
│  │          ▼                                                          │    │
│  │   ┌──────────────┐      ┌──────────────────────────────────┐        │    │
│  │   │  FastAPI     │      │         REST API                 │        │    │
│  │   │  Server      │◄────►│  POST /predict                   │        │    │
│  │   │              │      │  {"text": "..."} → {"label":...} │        │    │
│  │   │ DistilBERT   │      │                                  │        │    │
│  │   │  (single)    │      └──────────────────────────────────┘        │    │
│  │   └──────────────┘                                                  │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Training

```
Step 1: Load Dataset
────────────────────────────────────────────────────────
AG News (HuggingFace) → 120K train / 7.6K test samples


Step 2: Tokenize
────────────────────────────────────────────────────────
Raw text: "Wall Street gains as tech stocks rally"
    │
    ▼
Tokenizer (DistilBERT)
    │
    ▼
Tokens: [101, 2813, 2395, 5765, 2004, 6627, 15768, 8254, 102, 0, 0...]
         [CLS]                  text tokens                    [SEP] [PAD]


Step 3: Distribute Data
────────────────────────────────────────────────────────
120K samples
    │
    ├── GPU 0: samples [0, 2, 4, 6, 8, ...] (60K)
    │
    └── GPU 1: samples [1, 3, 5, 7, 9, ...] (60K)


Step 4: Forward Pass (parallel)
────────────────────────────────────────────────────────
GPU 0: batch → DistilBERT → logits → loss_0
GPU 1: batch → DistilBERT → logits → loss_1


Step 5: Backward Pass + Sync
────────────────────────────────────────────────────────
GPU 0: loss_0.backward() → gradients_0 ─┐
                                        ├──► AllReduce ──► averaged_gradients
GPU 1: loss_1.backward() → gradients_1 ─┘


Step 6: Optimizer Step
────────────────────────────────────────────────────────
GPU 0: weights = weights - lr * averaged_gradients
GPU 1: weights = weights - lr * averaged_gradients
                              ↑
                    Same update, models stay identical


Step 7: Save Checkpoint (rank 0 only)
────────────────────────────────────────────────────────
model.module.state_dict() → distilbert_agnews.pt
```

---

## Data Flow: Inference

```
Step 1: User Request
────────────────────────────────────────────────────────
POST /predict
{
    "text": "Apple announces new iPhone with AI features"
}


Step 2: Tokenize
────────────────────────────────────────────────────────
text → tokenizer → input_ids + attention_mask


Step 3: Model Forward (single GPU or CPU)
────────────────────────────────────────────────────────
input_ids → DistilBERT → logits [0.1, 0.05, 0.8, 0.05]
                                  ↓    ↓     ↓    ↓
                                World Sports Biz  Sci


Step 4: Post-process
────────────────────────────────────────────────────────
logits → softmax → probabilities → argmax → class label


Step 5: Response
────────────────────────────────────────────────────────
{
    "label": "Sci/Tech",
    "confidence": 0.80
}
```

---

## File Architecture

```
distrinews/
│
├── training/                    # Everything for distributed training
│   │
│   ├── train_ddp.py            # MAIN ENTRY POINT
│   │   │
│   │   ├── Initializes distributed environment
│   │   ├── Creates model, optimizer, dataloader
│   │   ├── Training loop with DDP
│   │   └── Saves checkpoint (rank 0 only)
│   │
│   ├── model.py                # Model definition
│   │   │
│   │   └── NewsClassifier
│   │       ├── Wraps DistilBertForSequenceClassification
│   │       └── Handles num_labels=4 for AG News
│   │
│   ├── dataset.py              # Data loading
│   │   │
│   │   ├── load_agnews_dataset()
│   │   │   └── Returns HuggingFace dataset
│   │   │
│   │   └── AGNewsDataset (torch Dataset)
│   │       ├── __init__: stores tokenizer, data
│   │       ├── __len__: returns dataset size
│   │       └── __getitem__: returns tokenized sample
│   │
│   ├── utils.py                # Utilities
│   │   │
│   │   ├── setup_distributed() → rank, world_size, device
│   │   ├── cleanup_distributed()
│   │   ├── get_logger() → rank-aware logging
│   │   └── compute_accuracy()
│   │
│   └── run_ddp.sh              # Launch script
│       │
│       └── torchrun --nproc_per_node=2 train_ddp.py
│
├── inference/                   # Everything for serving
│   │
│   ├── app.py                  # FastAPI application
│   │   │
│   │   ├── POST /predict
│   │   │   ├── Input: {"text": "..."}
│   │   │   └── Output: {"label": "...", "confidence": 0.xx}
│   │   │
│   │   └── GET /health
│   │       └── Returns {"status": "healthy"}
│   │
│   └── model_loader.py         # Model loading utilities
│       │
│       └── ModelLoader
│           ├── __init__: loads checkpoint + tokenizer
│           └── predict(text): returns label + confidence
│
├── checkpoints/                 # Saved models
│   └── distilbert_agnews.pt    # Trained weights
│
├── benchmarks/                  # Performance documentation
│   ├── single_gpu.md           # 1 GPU results
│   └── ddp_2gpu.md             # 2 GPU DDP results
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Process Architecture (During Training)

```
┌────────────────────────────────────────────────────────────────────┐
│                        KAGGLE NOTEBOOK                              │
│                                                                    │
│  torchrun --nproc_per_node=2 train_ddp.py                          │
│                    │                                               │
│                    ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    PROCESS SPAWNER                           │   │
│  │                                                             │   │
│  │   Sets environment variables:                               │   │
│  │   - MASTER_ADDR=localhost                                   │   │
│  │   - MASTER_PORT=12355                                       │   │
│  │   - WORLD_SIZE=2                                            │   │
│  │                                                             │   │
│  └─────────────┬───────────────────────────┬───────────────────┘   │
│                │                           │                       │
│                ▼                           ▼                       │
│  ┌─────────────────────────┐ ┌─────────────────────────┐           │
│  │      PROCESS 0          │ │      PROCESS 1          │           │
│  │                         │ │                         │           │
│  │  RANK=0                 │ │  RANK=1                 │           │
│  │  LOCAL_RANK=0           │ │  LOCAL_RANK=1           │           │
│  │                         │ │                         │           │
│  │  ┌───────────────────┐  │ │  ┌───────────────────┐  │           │
│  │  │  train_ddp.py     │  │ │  │  train_ddp.py     │  │           │
│  │  │                   │  │ │  │                   │  │           │
│  │  │  GPU 0 (T4)       │  │ │  │  GPU 1 (T4)       │  │           │
│  │  │  DistilBERT+DDP   │  │ │  │  DistilBERT+DDP   │  │           │
│  │  └───────────────────┘  │ │  └───────────────────┘  │           │
│  │                         │ │                         │           │
│  │  ✓ Logs to console      │ │  ✗ Silent              │           │
│  │  ✓ Saves checkpoint     │ │  ✗ No saving           │           │
│  │                         │ │                         │           │
│  └──────────┬──────────────┘ └──────────┬──────────────┘           │
│             │                           │                          │
│             └───────────┬───────────────┘                          │
│                         │                                          │
│                         ▼                                          │
│              ┌─────────────────────┐                               │
│              │    NCCL Backend     │                               │
│              │  (gradient sync)    │                               │
│              └─────────────────────┘                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture (Hugging Face Spaces)

```
┌────────────────────────────────────────────────────────────────────┐
│                     HUGGING FACE SPACES                             │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Docker Container                          │  │
│  │                                                              │  │
│  │   ┌────────────────────────────────────────────────────┐     │  │
│  │   │                  FastAPI App                        │     │  │
│  │   │                                                    │     │  │
│  │   │   ┌──────────────┐     ┌──────────────────────┐    │     │  │
│  │   │   │  /predict    │     │   ModelLoader        │    │     │  │
│  │   │   │  endpoint    │────►│                      │    │     │  │
│  │   │   └──────────────┘     │  - DistilBERT        │    │     │  │
│  │   │                        │  - Tokenizer         │    │     │  │
│  │   │   ┌──────────────┐     │  - Loaded at startup │    │     │  │
│  │   │   │  /health     │     └──────────────────────┘    │     │  │
│  │   │   │  endpoint    │                                 │     │  │
│  │   │   └──────────────┘                                 │     │  │
│  │   │                                                    │     │  │
│  │   └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │   Files:                                                     │  │
│  │   - app.py                                                   │  │
│  │   - model_loader.py                                          │  │
│  │   - distilbert_agnews.pt                                     │  │
│  │   - requirements.txt                                         │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   Public URL: https://huggingface.co/spaces/[username]/distrinews  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Diagram

```
                    ┌─────────────────────────────────────────┐
                    │              User Journey                │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ 1. Develop    │  │ 2. Train      │  │ 3. Deploy     │
           │   Locally     │  │   on Kaggle   │  │   to HF       │
           └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                   │                  │                  │
                   ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │  Your PC      │  │  Kaggle       │  │  HF Spaces    │
           │  (CPU only)   │  │  (2x T4 GPU)  │  │  (CPU)        │
           │               │  │               │  │               │
           │  - Write code │  │  - Run DDP    │  │  - Serve API  │
           │  - Test logic │  │  - Train      │  │  - Public URL │
           │  - Debug      │  │  - Benchmark  │  │               │
           └───────────────┘  └───────┬───────┘  └───────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  Checkpoint   │
                              │  .pt file     │
                              └───────────────┘
```

---

*Understand this architecture before writing code. Every file has a clear purpose.*
