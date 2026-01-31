# DistriNews: Complete Distributed ML Mastery Guide

> **Your comprehensive guide to mastering Distributed Machine Learning, from concepts to production deployment.**

This documentation will transform you from a distributed ML beginner to someone who can confidently discuss, implement, and deploy distributed training systems.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Complete File Reference](#3-complete-file-reference)
4. [Distributed Training Fundamentals](#4-distributed-training-fundamentals)
5. [PyTorch DDP Masterclass](#5-pytorch-ddp-masterclass)
6. [Step-by-Step Workflow](#6-step-by-step-workflow)
7. [DevOps/MLOps Pipeline](#7-devopsmlops-pipeline)
8. [Deployment Guide](#8-deployment-guide)
9. [Critical Concepts & Interview Prep](#9-critical-concepts--interview-prep)
10. [Troubleshooting & Best Practices](#10-troubleshooting--best-practices)
11. [Performance Benchmarks](#11-performance-benchmarks)
12. [Quick Reference Cards](#12-quick-reference-cards)

---

## 1. Project Overview

### What is DistriNews?

DistriNews is a **production-grade educational project** that teaches PyTorch Distributed Data Parallel (DDP) training through hands-on implementation. It trains a DistilBERT model on the AG News dataset across multiple GPUs and deploys it as a REST API.

### Why This Project Matters

```mermaid
mindmap
  root((Distributed ML Skills))
    Training
      Data Parallelism
      Gradient Synchronization
      Multi-GPU Coordination
    MLOps
      Model Versioning
      Checkpoint Management
      Reproducibility
    DevOps
      Docker Containerization
      API Development
      Cloud Deployment
    Interview Prep
      DDP vs DataParallel
      AllReduce Operations
      Scaling Strategies
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Educational Focus** | Every file has extensive inline comments explaining concepts |
| **Free Resources** | Uses Kaggle (free GPU) and HuggingFace Spaces (free deployment) |
| **Production-Ready** | Proper error handling, configuration, and deployment patterns |
| **Complete Lifecycle** | Training → Inference → Deployment |
| **Interview Prep** | Includes common DDP interview questions with answers |

### Tech Stack

```mermaid
graph TB
    subgraph "Training Stack"
        A[PyTorch 2.1+] --> B[DDP]
        C[Transformers] --> D[DistilBERT]
        E[Datasets] --> F[AG News]
    end

    subgraph "Inference Stack"
        G[FastAPI] --> H[REST API]
        I[Uvicorn] --> J[ASGI Server]
        K[Pydantic] --> L[Validation]
    end

    subgraph "Deployment Stack"
        M[Docker] --> N[Container]
        O[HuggingFace Spaces] --> P[Hosting]
    end

    B --> H
    D --> H
```

---

## 2. Architecture Deep Dive

### High-Level System Architecture

```mermaid
flowchart TB
    subgraph Training["Training Phase (Kaggle)"]
        direction TB
        D1[(AG News Dataset<br/>120K samples)]
        D1 --> DS[DistributedSampler]
        DS --> |"Split data"| GPU0
        DS --> |"Split data"| GPU1

        subgraph GPU0["GPU 0 (Rank 0)"]
            M0[DistilBERT Model]
            DL0[DataLoader]
            DL0 --> M0
        end

        subgraph GPU1["GPU 1 (Rank 1)"]
            M1[DistilBERT Model]
            DL1[DataLoader]
            DL1 --> M1
        end

        GPU0 <--> |"AllReduce<br/>Gradients"| GPU1
        M0 --> CP[(Checkpoint<br/>.pt file)]
    end

    subgraph Inference["Inference Phase (HuggingFace Spaces)"]
        direction TB
        CP --> ML[Model Loader]
        ML --> API[FastAPI Server]
        API --> |"/predict"| User([User])
        User --> |"POST text"| API
    end

    style Training fill:#e1f5fe
    style Inference fill:#e8f5e9
```

### Data Flow During Training

```mermaid
sequenceDiagram
    participant DS as DistributedSampler
    participant GPU0 as GPU 0
    participant GPU1 as GPU 1
    participant NCCL as NCCL Backend

    Note over DS: Epoch Start
    DS->>GPU0: Samples [0,2,4,6...]
    DS->>GPU1: Samples [1,3,5,7...]

    par Parallel Processing
        GPU0->>GPU0: Forward Pass (Batch A)
        GPU1->>GPU1: Forward Pass (Batch B)
    end

    par Parallel Backward
        GPU0->>GPU0: Compute Gradients A
        GPU1->>GPU1: Compute Gradients B
    end

    Note over NCCL: AllReduce Operation
    GPU0->>NCCL: Send Gradients A
    GPU1->>NCCL: Send Gradients B
    NCCL->>GPU0: Averaged Gradients
    NCCL->>GPU1: Averaged Gradients

    par Parallel Update
        GPU0->>GPU0: optimizer.step()
        GPU1->>GPU1: optimizer.step()
    end

    Note over GPU0,GPU1: Models remain synchronized!
```

### Inference Request Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant ML as ModelLoader
    participant T as Tokenizer
    participant M as DistilBERT

    U->>API: POST /predict {"text": "..."}
    API->>API: Validate Request (Pydantic)
    API->>ML: predict(text)
    ML->>T: tokenize(text)
    T-->>ML: input_ids, attention_mask
    ML->>M: forward(input_ids, attention_mask)
    M-->>ML: logits [4 classes]
    ML->>ML: softmax → probabilities
    ML-->>API: {label, confidence}
    API-->>U: JSON Response
```

### Project Structure Overview

```mermaid
graph TB
    subgraph Root["DistriNews Project"]
        README[README.md]
        REQ[requirements.txt]

        subgraph Training["training/"]
            TM[model.py<br/>NewsClassifier]
            TD[dataset.py<br/>AG News Loading]
            TU[utils.py<br/>Helpers]
            TS[train_single.py<br/>1 GPU Baseline]
            TDD[train_ddp.py<br/>DDP Training]
            TR[run_ddp.sh<br/>Launcher]
        end

        subgraph Inference["inference/"]
            IA[app.py<br/>FastAPI Server]
            IML[model_loader.py<br/>Checkpoint Loading]
        end

        subgraph Deployment["deployment/"]
            DF[Dockerfile]
            DR[requirements.txt]
            DRM[README.md]
        end

        subgraph Checkpoints["checkpoints/"]
            CP[distilbert_agnews.pt]
        end

        subgraph Benchmarks["benchmarks/"]
            B1[single_gpu.md]
            B2[ddp_2gpu.md]
        end
    end

    TDD --> CP
    CP --> IML
    IML --> IA
```

---

## 3. Complete File Reference

### Training Module (`training/`)

#### `model.py` - The Neural Network

**Purpose:** Wraps DistilBERT for 4-class news classification

```mermaid
classDiagram
    class NewsClassifier {
        +int num_labels = 4
        +str pretrained_model
        +DistilBertForSequenceClassification model
        +dict LABEL_MAP
        +forward(input_ids, attention_mask) Tensor
        +predict(input_ids, attention_mask) tuple
        +get_label_name(class_idx) str
    }

    NewsClassifier --> DistilBertForSequenceClassification : wraps
```

**Key Concepts in This File:**
- **DistilBERT**: Smaller, faster version of BERT (66M vs 110M params)
- **Classification Head**: Linear layer that maps hidden states → 4 classes
- **Logits vs Probabilities**: Raw scores vs softmax-normalized values

**Critical Code Pattern:**
```python
class NewsClassifier(nn.Module):
    def __init__(self, num_labels=4):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Shape: (batch_size, 4)
```

---

#### `dataset.py` - Data Pipeline

**Purpose:** Loads AG News dataset and handles tokenization

```mermaid
flowchart LR
    subgraph Input
        T[Raw Text<br/>"Apple announces..."]
    end

    subgraph Tokenization
        TOK[Tokenizer]
        T --> TOK
        TOK --> TID[Token IDs<br/>[101, 8271, 8756...]]
        TOK --> AM[Attention Mask<br/>[1, 1, 1, 1, 0, 0...]]
    end

    subgraph Padding
        TID --> PAD[Pad to max_length=128]
        AM --> PAD
    end

    subgraph Output
        PAD --> BATCH[Ready for Model]
    end
```

**Key Classes/Functions:**

| Function/Class | Purpose |
|---------------|---------|
| `load_agnews(split)` | Downloads AG News from HuggingFace |
| `get_tokenizer()` | Loads DistilBERT tokenizer |
| `AGNewsDataset` | PyTorch Dataset wrapper |
| `create_dataloader()` | Creates DataLoader with optional sampler |

**Tokenization Deep Dive:**
```
Original:     "Apple announces new iPhone"
Tokenized:    [CLS] apple announces new iphone [SEP] [PAD] [PAD] ...
Token IDs:    [101, 8271, 8756, 2047, 18059, 102, 0, 0, ...]
Attention:    [1, 1, 1, 1, 1, 1, 0, 0, ...]
              └─real tokens──┘ └─padding─┘
```

---

#### `train_ddp.py` - The Heart of Distributed Training

**Purpose:** Multi-GPU training with PyTorch DDP

```mermaid
flowchart TB
    subgraph Setup["1. Setup Phase"]
        ENV[Read Environment<br/>RANK, LOCAL_RANK, WORLD_SIZE]
        ENV --> INIT[init_process_group<br/>backend='nccl']
        INIT --> DEV[Set device<br/>cuda:local_rank]
    end

    subgraph DataPrep["2. Data Preparation"]
        LOAD[Load Dataset]
        LOAD --> SAMP[Create DistributedSampler]
        SAMP --> DL[Create DataLoader]
    end

    subgraph ModelPrep["3. Model Preparation"]
        CREATE[Create Model]
        CREATE --> MOVE[Move to Device]
        MOVE --> WRAP[Wrap with DDP]
    end

    subgraph Training["4. Training Loop"]
        EPOCH[For each epoch]
        EPOCH --> SETEP[sampler.set_epoch]
        SETEP --> TRAIN[Train batches]
        TRAIN --> EVAL[Evaluate]
        EVAL --> BARRIER[dist.barrier]
        BARRIER --> EPOCH
    end

    subgraph Cleanup["5. Cleanup"]
        SAVE[Save checkpoint<br/>rank 0 only]
        SAVE --> DESTROY[destroy_process_group]
    end

    Setup --> DataPrep --> ModelPrep --> Training --> Cleanup
```

**Critical DDP Functions:**

| Function | What It Does | Why It Matters |
|----------|--------------|----------------|
| `setup_distributed()` | Initializes process group | Enables GPU communication |
| `cleanup_distributed()` | Destroys process group | Releases resources |
| `is_main_process()` | Checks if rank == 0 | Prevents duplicate logs/saves |
| `train_one_epoch()` | Training loop with DDP | Automatic gradient sync |

**The DDP Magic - What Happens During `loss.backward()`:**

```mermaid
sequenceDiagram
    participant Code as Your Code
    participant DDP as DDP Wrapper
    participant NCCL as NCCL
    participant GPUs as All GPUs

    Code->>DDP: loss.backward()
    DDP->>DDP: Compute local gradients
    DDP->>NCCL: Trigger AllReduce hooks
    NCCL->>GPUs: Average all gradients
    GPUs-->>DDP: Return averaged gradients
    DDP-->>Code: backward() complete
    Note over Code: All GPUs now have identical gradients!
```

---

#### `train_single.py` - Single GPU Baseline

**Purpose:** Baseline training for comparison (no DDP)

**Key Differences from DDP:**

| Aspect | Single GPU | DDP |
|--------|------------|-----|
| Process Group | None | `init_process_group()` |
| Model Wrapper | None | `DDP(model)` |
| Sampler | Regular shuffle | `DistributedSampler` |
| Data Split | All data | Partitioned |
| Gradient Sync | None | Automatic AllReduce |

---

#### `utils.py` - Shared Utilities

**Key Utilities:**

| Utility | Purpose |
|---------|---------|
| `AverageMeter` | Tracks running averages (loss, accuracy) |
| `compute_accuracy()` | Calculates classification accuracy |
| `save_checkpoint()` | Saves model + optimizer state |
| `load_checkpoint()` | Loads with device mapping |
| `format_time()` | Converts seconds to readable format |

---

### Inference Module (`inference/`)

#### `app.py` - FastAPI Server

**Purpose:** REST API for serving predictions

```mermaid
flowchart TB
    subgraph Endpoints
        ROOT["GET /<br/>Redirect to /docs"]
        HEALTH["GET /health<br/>Server status"]
        PREDICT["POST /predict<br/>Single prediction"]
        BATCH["POST /predict/batch<br/>Batch predictions"]
    end

    subgraph Models["Pydantic Models"]
        PR[PredictRequest<br/>text: str]
        PResp[PredictResponse<br/>label, confidence, probs]
        HR[HealthResponse<br/>status, model_loaded]
        BR[BatchPredictRequest<br/>texts: list]
    end

    PR --> PREDICT
    PREDICT --> PResp
    BR --> BATCH
    HEALTH --> HR
```

**API Design Patterns:**

1. **Model Loading at Startup:**
```python
@app.on_event("startup")
async def load_model():
    global model_loader
    model_loader = ModelLoader(checkpoint_path)
```

2. **Input Validation with Pydantic:**
```python
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
```

3. **Proper Error Handling:**
```python
if model_loader is None:
    raise HTTPException(status_code=503, detail="Model not loaded")
```

---

#### `model_loader.py` - Model Loading & Inference

**Purpose:** Handles checkpoint loading and prediction logic

```mermaid
classDiagram
    class ModelLoader {
        +str checkpoint_path
        +torch.device device
        +DistilBertTokenizer tokenizer
        +NewsClassifier model
        +__init__(checkpoint_path)
        +_load_checkpoint()
        +predict(text) dict
        +predict_batch(texts) list
    }
```

**Key Pattern - CPU/GPU Flexibility:**
```python
# Auto-detect device
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint with device mapping
checkpoint = torch.load(
    checkpoint_path,
    map_location=self.device  # Allows loading GPU checkpoint on CPU
)
```

---

### Deployment Module (`deployment/`)

#### `Dockerfile` - Container Configuration

```dockerfile
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y gcc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY inference/ ./inference/
COPY training/model.py ./training/model.py
COPY checkpoints/ ./checkpoints/

# Configure
EXPOSE 7860
ENV MODEL_CHECKPOINT=/app/checkpoints/distilbert_agnews.pt

# Start server
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Docker Layer Optimization:**

```mermaid
graph TB
    subgraph Layers["Docker Layers (cached)"]
        L1[Base Image: python:3.10-slim]
        L2[System deps: gcc]
        L3[Python deps: requirements.txt]
        L4[Source code: inference/, training/]
        L5[Model: checkpoints/]
    end

    L1 --> L2 --> L3 --> L4 --> L5

    style L1 fill:#90EE90
    style L2 fill:#90EE90
    style L3 fill:#90EE90
    style L4 fill:#FFB6C1
    style L5 fill:#FFB6C1
```

---

## 4. Distributed Training Fundamentals

### The Problem: Why Distributed Training?

```mermaid
graph LR
    subgraph Problem["Single GPU Problem"]
        D[(1M Samples)]
        G1[GPU 0]
        D --> G1
        G1 --> T[86 hours!]
    end

    subgraph Solution["Multi-GPU Solution"]
        D2[(1M Samples)]
        G2[GPU 0<br/>500K samples]
        G3[GPU 1<br/>500K samples]
        D2 --> G2
        D2 --> G3
        G2 --> T2[43 hours]
        G3 --> T2
    end
```

### Types of Parallelism

```mermaid
graph TB
    subgraph DP["Data Parallelism"]
        D1[(Data)]
        D1 --> |Split| G1A[GPU 0<br/>Same Model]
        D1 --> |Split| G2A[GPU 1<br/>Same Model]
        G1A <--> |Sync Gradients| G2A
    end

    subgraph MP["Model Parallelism"]
        D2[(Data)]
        D2 --> G1B[GPU 0<br/>Layer 1-6]
        G1B --> G2B[GPU 1<br/>Layer 7-12]
    end

    subgraph PP["Pipeline Parallelism"]
        D3[(Data)]
        D3 --> |Batch 1| G1C[GPU 0<br/>Stage 1]
        G1C --> |Batch 1| G2C[GPU 1<br/>Stage 2]
        D3 --> |Batch 2| G1C
    end
```

**This Project Uses: Data Parallelism (DDP)**

### Data Parallelism Explained

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PARALLELISM                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   STEP 1: Same model copied to each GPU                            │
│   ┌─────────────┐      ┌─────────────┐                              │
│   │    GPU 0    │      │    GPU 1    │                              │
│   │   Model A   │  ==  │   Model A   │  ← Identical weights         │
│   └─────────────┘      └─────────────┘                              │
│                                                                     │
│   STEP 2: Different data to each GPU                               │
│   ┌─────────────┐      ┌─────────────┐                              │
│   │  Batch [0]  │      │  Batch [1]  │  ← Unique batches            │
│   │  Batch [2]  │      │  Batch [3]  │                              │
│   │  Batch [4]  │      │  Batch [5]  │                              │
│   └─────────────┘      └─────────────┘                              │
│                                                                     │
│   STEP 3: Compute gradients independently                          │
│   ┌─────────────┐      ┌─────────────┐                              │
│   │ Gradients_0 │      │ Gradients_1 │                              │
│   └──────┬──────┘      └──────┬──────┘                              │
│          │                    │                                     │
│          └────────┬───────────┘                                     │
│                   ▼                                                 │
│   STEP 4: AllReduce (average gradients)                            │
│          ┌───────────────────┐                                      │
│          │ Avg(Grad_0,Grad_1)│                                      │
│          └─────────┬─────────┘                                      │
│                    │                                                │
│          ┌────────┴────────┐                                        │
│          ▼                 ▼                                        │
│   STEP 5: Update with same gradients                               │
│   ┌─────────────┐      ┌─────────────┐                              │
│   │   Model A'  │  ==  │   Model A'  │  ← Still identical!          │
│   └─────────────┘      └─────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why AllReduce is Mathematically Correct

```mermaid
graph TB
    subgraph "Mathematical Equivalence"
        subgraph "Single GPU"
            SG["Batch size 64<br/>Gradient = g"]
        end

        subgraph "2 GPUs with DDP"
            G0["GPU 0: Batch 32<br/>Gradient = g₀"]
            G1["GPU 1: Batch 32<br/>Gradient = g₁"]
            AVG["Average: (g₀ + g₁) / 2"]
            G0 --> AVG
            G1 --> AVG
        end
    end

    SG --> EQ["g ≈ (g₀ + g₁) / 2"]
    AVG --> EQ
```

---

## 5. PyTorch DDP Masterclass

### DDP Components Overview

```mermaid
graph TB
    subgraph "DDP Components"
        PG[Process Group<br/>dist.init_process_group]
        DDP_W[DDP Wrapper<br/>DistributedDataParallel]
        DS[DistributedSampler]
        NCCL_B[NCCL Backend]
        TR[torchrun Launcher]
    end

    TR --> |"Sets env vars"| PG
    PG --> |"Enables"| NCCL_B
    NCCL_B --> |"Communication for"| DDP_W
    DS --> |"Shards data for"| DDP_W
```

### Component 1: Process Group

**What:** Enables communication between GPUs
**When:** Called ONCE at start of training

```python
# Initialize - all GPUs can now talk to each other
dist.init_process_group(
    backend="nccl",      # NVIDIA's GPU communication library
    init_method="env://", # Read config from environment
    world_size=2,        # Total GPUs
    rank=rank            # This GPU's ID
)
```

```mermaid
sequenceDiagram
    participant TorchRun
    participant GPU0 as Process 0
    participant GPU1 as Process 1
    participant NCCL

    TorchRun->>GPU0: Spawn with RANK=0
    TorchRun->>GPU1: Spawn with RANK=1

    GPU0->>NCCL: init_process_group()
    GPU1->>NCCL: init_process_group()

    Note over NCCL: Processes discover each other

    NCCL-->>GPU0: Group ready
    NCCL-->>GPU1: Group ready

    Note over GPU0,GPU1: Can now communicate!
```

### Component 2: DistributedDataParallel Wrapper

**What:** Wraps model to enable automatic gradient sync
**When:** After moving model to GPU, before training

```python
model = NewsClassifier()
model = model.to(device)
model = DDP(model, device_ids=[local_rank])  # The magic wrapper
```

**What DDP Does Internally:**

```mermaid
flowchart TB
    subgraph "DDP Initialization"
        WRAP[DDP wraps model]
        WRAP --> BROADCAST[Broadcast weights<br/>from rank 0 to all]
        BROADCAST --> HOOKS[Register gradient hooks<br/>on each parameter]
        HOOKS --> BUCKETS[Create gradient buckets<br/>for efficient AllReduce]
    end

    subgraph "During backward()"
        GRAD[Compute local gradient]
        GRAD --> HOOK[Hook triggered]
        HOOK --> ALLREDUCE[AllReduce gradient]
        ALLREDUCE --> STORE[Store averaged gradient]
    end
```

### Component 3: DistributedSampler

**What:** Ensures each GPU sees unique data
**When:** Creating DataLoader

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total GPUs
    rank=rank,                 # This GPU's ID
    shuffle=True
)
dataloader = DataLoader(dataset, sampler=sampler)
```

**How Data is Partitioned:**

```
Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Without Sampler (BAD):
  GPU 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  ← Same!
  GPU 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  ← Same!

With DistributedSampler (GOOD):
  GPU 0: [0, 2, 4, 6, 8]  ← Unique
  GPU 1: [1, 3, 5, 7, 9]  ← Unique
```

### Component 4: torchrun Launcher

**What:** Spawns processes and sets environment variables
**When:** Command line

```bash
torchrun --nproc_per_node=2 train_ddp.py
```

**Environment Variables Set by torchrun:**

| Variable | Meaning | Example |
|----------|---------|---------|
| `RANK` | Global process ID | 0, 1 |
| `LOCAL_RANK` | GPU ID on this machine | 0, 1 |
| `WORLD_SIZE` | Total processes | 2 |
| `MASTER_ADDR` | IP of rank 0 | localhost |
| `MASTER_PORT` | Communication port | 29500 |

### Component 5: NCCL Backend

**What:** NVIDIA's optimized GPU communication library
**Why:** Much faster than CPU-based communication

```mermaid
graph LR
    subgraph "Communication Backends"
        NCCL["NCCL<br/>GPU ↔ GPU<br/>Fastest"]
        GLOO["Gloo<br/>CPU/GPU<br/>Flexible"]
        MPI["MPI<br/>HPC<br/>Traditional"]
    end

    NCCL --> |"Use for"| GPU[GPU Training]
    GLOO --> |"Use for"| CPU[CPU Training]
    MPI --> |"Use for"| HPC[HPC Clusters]
```

---

## 6. Step-by-Step Workflow

### Complete Training Workflow

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Local Development"]
        DEV[Write/Edit Code]
        DEV --> TEST[Test on CPU<br/>with small data]
        TEST --> |"Works?"| COMMIT[Commit to Git]
    end

    subgraph Phase2["Phase 2: Training on Kaggle"]
        UPLOAD[Upload to Kaggle]
        UPLOAD --> ENABLE[Enable 2x T4 GPUs]
        ENABLE --> RUN["Run torchrun<br/>--nproc_per_node=2"]
        RUN --> MONITOR[Monitor Training]
        MONITOR --> DOWNLOAD[Download Checkpoint]
    end

    subgraph Phase3["Phase 3: Test Inference Locally"]
        LOCAL[Start Local Server]
        LOCAL --> TESTAPI[Test /predict endpoint]
        TESTAPI --> VERIFY[Verify predictions]
    end

    subgraph Phase4["Phase 4: Deploy to HF Spaces"]
        CREATESPACE[Create HF Space]
        CREATESPACE --> PUSHCODE[Push Docker config]
        PUSHCODE --> UPLOADCP[Upload checkpoint]
        UPLOADCP --> WAIT[Wait for build]
        WAIT --> LIVE[API Live!]
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4
```

### Detailed Training Steps

#### Step 1: Setup Environment (Kaggle)

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Output: GPUs: 2
```

#### Step 2: Launch DDP Training

```bash
torchrun --nproc_per_node=2 --standalone training/train_ddp.py \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-5
```

#### Step 3: What Happens During Launch

```mermaid
sequenceDiagram
    participant User
    participant TorchRun
    participant Process0 as Process 0
    participant Process1 as Process 1

    User->>TorchRun: torchrun --nproc_per_node=2 train.py

    TorchRun->>TorchRun: Detect 2 GPUs
    TorchRun->>TorchRun: Generate MASTER_ADDR, MASTER_PORT

    par Spawn Processes
        TorchRun->>Process0: Fork with RANK=0, LOCAL_RANK=0
        TorchRun->>Process1: Fork with RANK=1, LOCAL_RANK=1
    end

    Process0->>Process0: Read env vars
    Process1->>Process1: Read env vars

    Process0->>Process0: init_process_group()
    Process1->>Process1: init_process_group()

    Note over Process0,Process1: Both connected, training begins
```

#### Step 4: Training Progress

```
============================================================
DDP TRAINING CONFIGURATION
============================================================
World size (GPUs):    2
Batch size per GPU:   32
Effective batch size: 64
Learning rate:        2e-05
Epochs:               3
Device:               cuda:0
============================================================

[Rank 0] Initialized process group (world_size=2)
[Rank 1] Initialized process group (world_size=2)

Train samples: 120,000 (each GPU sees 60,000)

Epoch [1/3] - 4m 45s
  Train Loss: 0.4000 | Train Acc: 86.00%
  Test Loss:  0.2900 | Test Acc:  91.50%

Epoch [2/3] - 4m 30s
  Train Loss: 0.2000 | Train Acc: 92.50%
  Test Loss:  0.2300 | Test Acc:  93.20%

Epoch [3/3] - 4m 25s
  Train Loss: 0.1400 | Train Acc: 95.00%
  Test Loss:  0.2100 | Test Acc:  94.00%

============================================================
TRAINING COMPLETE
============================================================
Total time: 13m 45s
Best test accuracy: 94.00%

Checkpoint saved to: checkpoints/distilbert_agnews.pt
```

#### Step 5: Download and Test Checkpoint

```python
# Test locally
from inference.model_loader import ModelLoader

loader = ModelLoader("checkpoints/distilbert_agnews.pt")
result = loader.predict("Apple announces new iPhone with AI features")
print(result)
# {'label': 'Sci/Tech', 'confidence': 0.89, ...}
```

---

## 7. DevOps/MLOps Pipeline

### Complete MLOps Flow

```mermaid
flowchart TB
    subgraph Development
        CODE[Code Changes]
        CODE --> LINT[Lint & Format]
        LINT --> TEST[Unit Tests]
        TEST --> COMMIT[Git Commit]
    end

    subgraph Training
        COMMIT --> KAGGLE[Kaggle Notebook]
        KAGGLE --> TRAIN[DDP Training]
        TRAIN --> CHECKPOINT[Model Checkpoint]
        CHECKPOINT --> DOWNLOAD[Download .pt file]
    end

    subgraph Validation
        DOWNLOAD --> LOCALTEST[Local Inference Test]
        LOCALTEST --> METRICS[Check Accuracy]
        METRICS --> |"Pass"| APPROVE[Approve for Deploy]
    end

    subgraph Deployment
        APPROVE --> BUILDIMAGE[Build Docker Image]
        BUILDIMAGE --> PUSH[Push to HF Spaces]
        PUSH --> DEPLOY[Auto Deploy]
        DEPLOY --> MONITOR[Monitor Logs]
    end

    subgraph Production
        MONITOR --> API[Live API]
        API --> USERS[Users]
        USERS --> |"Feedback"| CODE
    end
```

### Model Versioning Strategy

```mermaid
graph TB
    subgraph Versioning
        V1[v1.0.0<br/>Initial model]
        V2[v1.1.0<br/>+More epochs]
        V3[v2.0.0<br/>New architecture]

        V1 --> V2 --> V3
    end

    subgraph Artifacts
        CP1[distilbert_agnews_v1.0.0.pt]
        CP2[distilbert_agnews_v1.1.0.pt]
        CP3[distilbert_agnews_v2.0.0.pt]
    end

    V1 --> CP1
    V2 --> CP2
    V3 --> CP3
```

### Checkpoint Contents

```python
# What's saved in checkpoint
checkpoint = {
    'epoch': 3,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.14,
    'accuracy': 94.0,
    'config': {
        'model_name': 'distilbert-base-uncased',
        'num_labels': 4,
        'max_length': 128
    }
}
torch.save(checkpoint, 'checkpoints/distilbert_agnews.pt')
```

### Environment Configuration

```mermaid
graph TB
    subgraph Environments
        DEV[Development<br/>Local CPU]
        TRAIN[Training<br/>Kaggle 2xT4]
        PROD[Production<br/>HF Spaces CPU]
    end

    subgraph Config
        DEV --> |"Small data<br/>Debug mode"| DEVC[dev.env]
        TRAIN --> |"Full data<br/>2 GPUs"| TRAINC[train.env]
        PROD --> |"CPU inference<br/>Port 7860"| PRODC[prod.env]
    end
```

---

## 8. Deployment Guide

### Hugging Face Spaces Deployment

```mermaid
flowchart TB
    subgraph Prepare["1. Prepare Files"]
        DF[Dockerfile]
        REQ[requirements.txt]
        CODE[inference/ code]
        MODEL[checkpoint .pt]
    end

    subgraph Create["2. Create Space"]
        HF[HuggingFace.co]
        HF --> NEW[New Space]
        NEW --> DOCKER[Select Docker SDK]
    end

    subgraph Upload["3. Upload Files"]
        GIT[Git Clone Space]
        GIT --> COPY[Copy files]
        COPY --> PUSH[Git Push]
    end

    subgraph Build["4. Automatic Build"]
        PUSH --> BUILD[Docker Build]
        BUILD --> RUN[Start Container]
        RUN --> HEALTH[Health Check]
    end

    subgraph Live["5. Live API"]
        HEALTH --> URL[Public URL]
        URL --> DOCS[/docs Swagger UI]
    end

    Prepare --> Create --> Upload --> Build --> Live
```

### Required Files for Deployment

```
deployment/
├── Dockerfile              # Container config
├── requirements.txt        # Python deps (CPU PyTorch!)
└── README.md              # HF Space metadata

inference/
├── app.py                 # FastAPI server
└── model_loader.py        # Model loading

training/
├── model.py               # Model class definition
└── __init__.py

checkpoints/
└── distilbert_agnews.pt   # Trained weights (~250MB)
```

### Deployment requirements.txt (CPU Version)

```txt
# CPU PyTorch for smaller Docker image
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.1.0

# ML
transformers>=4.36.0

# API
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
```

### HF Spaces README.md Format

```markdown
---
title: DistriNews
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
license: mit
---

# DistriNews - News Classification API

Classify news articles into 4 categories using DistilBERT.

## API Usage

```bash
curl -X POST "https://your-space.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone"}'
```
```

### Testing Deployed API

```bash
# Health check
curl https://your-space.hf.space/health

# Single prediction
curl -X POST https://your-space.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Stock market reaches all-time high"}'

# Batch prediction
curl -X POST https://your-space.hf.space/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Sports news", "Tech news"]}'
```

---

## 9. Critical Concepts & Interview Prep

### Concept Map

```mermaid
mindmap
  root((DDP Mastery))
    Core Concepts
      Data Parallelism
      Gradient Synchronization
      AllReduce Operation
    PyTorch Components
      init_process_group
      DistributedDataParallel
      DistributedSampler
    Infrastructure
      NCCL Backend
      torchrun Launcher
      Process/Rank/World
    Best Practices
      Only Rank 0 Saves
      set_epoch for Shuffle
      Proper Cleanup
```

### Interview Question 1: DDP vs DataParallel

**Q: What's the difference between `DataParallel` and `DistributedDataParallel`?**

```mermaid
graph TB
    subgraph DP["DataParallel (DP)"]
        DPM[Master GPU]
        DPW1[Worker GPU]
        DPW2[Worker GPU]
        DPM --> |"Scatter data"| DPW1
        DPM --> |"Scatter data"| DPW2
        DPW1 --> |"Gather gradients"| DPM
        DPW2 --> |"Gather gradients"| DPM
        DPM --> |"Update"| DPM
        DPM --> |"Broadcast"| DPW1
        DPM --> |"Broadcast"| DPW2
    end

    subgraph DDP["DistributedDataParallel (DDP)"]
        DDPG1[GPU 0]
        DDPG2[GPU 1]
        DDPG1 <--> |"AllReduce"| DDPG2
    end
```

**Answer:**
> "DataParallel uses a single process with Python threads. It has a bottleneck because one GPU must gather gradients, update weights, and broadcast back. DistributedDataParallel uses multiple processes (one per GPU) with AllReduce for gradient sync. DDP is faster because:
> 1. No single-GPU bottleneck
> 2. Overlaps gradient computation with communication
> 3. Each GPU updates independently with the same gradients
> DDP achieves ~1.8x speedup with 2 GPUs vs DP's ~1.2x."

### Interview Question 2: Gradient Synchronization

**Q: How does gradient synchronization work in DDP?**

**Answer:**
> "During `backward()`, DDP automatically performs AllReduce to average gradients across all GPUs. Here's the process:
> 1. Each GPU computes local gradients from its batch
> 2. DDP hooks trigger AllReduce operations
> 3. NCCL communicates gradients between GPUs
> 4. Each GPU receives the average of all gradients
> 5. `optimizer.step()` applies identical updates to all model copies
> This happens automatically - you just call `loss.backward()` like normal."

### Interview Question 3: DistributedSampler

**Q: Why do you need DistributedSampler?**

**Answer:**
> "Without DistributedSampler, each GPU would load the same batches, wasting computation. DistributedSampler:
> 1. Partitions the dataset so each GPU sees unique samples
> 2. Ensures no overlap between GPUs
> 3. Handles shuffling correctly with `set_epoch()`
>
> For 120K samples with 2 GPUs: GPU 0 gets samples [0,2,4...], GPU 1 gets [1,3,5...].
> `set_epoch()` is critical - it changes the random seed each epoch so different combinations are seen."

### Interview Question 4: Rank 0 Pattern

**Q: Why should only rank 0 save checkpoints?**

**Answer:**
> "In DDP, all processes run identical code. If all save checkpoints:
> 1. File corruption from concurrent writes
> 2. Wasted disk I/O
> 3. Race conditions
>
> By convention, rank 0 (the 'main' process) handles:
> - Logging/printing (avoid duplicate lines)
> - Checkpoint saving
> - TensorBoard logging
> - Dataset downloading
>
> Use `if rank == 0:` guards for these operations."

### Interview Question 5: Scaling Efficiency

**Q: If you have 2 GPUs, why isn't the speedup exactly 2x?**

**Answer:**
> "Real-world speedup is ~1.75-1.85x due to:
> 1. **Communication overhead**: AllReduce takes time (~10-15% of backward)
> 2. **Synchronization barriers**: Faster GPUs wait for slower ones
> 3. **Effective batch size**: Doubled batch can affect convergence
> 4. **Memory bandwidth**: PCIe contention for data transfer
>
> 85-95% parallel efficiency is considered good. Our 1.78x/2.0 = 89% efficiency is excellent."

### Key Formulas

| Metric | Formula |
|--------|---------|
| Speedup | `Time_1GPU / Time_NGPU` |
| Parallel Efficiency | `Speedup / N_GPUs × 100%` |
| Effective Batch Size | `batch_per_gpu × world_size` |
| Samples per GPU | `total_samples / world_size` |

---

## 10. Troubleshooting & Best Practices

### Common Errors and Solutions

```mermaid
flowchart TB
    subgraph Errors
        E1["NCCL error:<br/>unhandled system error"]
        E2["RuntimeError:<br/>address already in use"]
        E3["Timeout waiting<br/>for process"]
        E4["Duplicate logs<br/>appearing"]
        E5["Model divergence<br/>between GPUs"]
    end

    subgraph Solutions
        S1["Check CUDA/NCCL versions<br/>Ensure all GPUs visible"]
        S2["Kill zombie processes<br/>Change MASTER_PORT"]
        S3["Check all GPUs healthy<br/>Increase timeout"]
        S4["Add rank==0 guards<br/>for print statements"]
        S5["Check set_epoch called<br/>Verify AllReduce working"]
    end

    E1 --> S1
    E2 --> S2
    E3 --> S3
    E4 --> S4
    E5 --> S5
```

### Best Practices Checklist

```
✅ DDP Setup
  □ init_process_group() called ONCE at start
  □ Model moved to device BEFORE DDP wrap
  □ DDP wrapper created: model = DDP(model, device_ids=[local_rank])
  □ destroy_process_group() in finally block

✅ Data Loading
  □ DistributedSampler used for training
  □ sampler.set_epoch(epoch) called each epoch
  □ shuffle=False in DataLoader (sampler handles it)

✅ Logging & Saving
  □ All print/log statements guarded by rank==0
  □ Checkpoint saved only by rank 0
  □ model.module used for saving (unwrap DDP)

✅ Synchronization
  □ dist.barrier() before checkpoint save
  □ All GPUs reach barrier (avoid deadlock)

✅ Cleanup
  □ try/finally pattern for cleanup
  □ destroy_process_group() always called
```

### Debug Checklist

```bash
# Check GPU availability
nvidia-smi

# Check PyTorch sees GPUs
python -c "import torch; print(torch.cuda.device_count())"

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Test basic DDP
torchrun --nproc_per_node=2 -c "import torch.distributed as dist; dist.init_process_group('nccl')"
```

---

## 11. Performance Benchmarks

### Benchmark Comparison

```mermaid
xychart-beta
    title "Training Time Comparison"
    x-axis ["1 GPU", "2 GPU DDP"]
    y-axis "Time (minutes)" 0 --> 30
    bar [24, 14]
```

### Detailed Results

| Metric | 1 GPU | 2 GPU DDP | Improvement |
|--------|-------|-----------|-------------|
| Total Time | 24 min | 14 min | **1.74x faster** |
| Time/Epoch | 8 min | 4.5 min | **1.78x faster** |
| Throughput | 250 samples/s | 444 samples/s | **1.78x higher** |
| Final Accuracy | 94.2% | 94.0% | Same (within noise) |
| GPU Memory | 4.2 GB | 4.2 GB each | Same per GPU |
| GPU Utilization | 85% | 80% each | Slightly lower (sync overhead) |

### Scaling Projection

```mermaid
graph LR
    subgraph "Projected Speedup"
        G1["1 GPU<br/>1.0x"]
        G2["2 GPU<br/>1.78x"]
        G4["4 GPU<br/>~3.4x"]
        G8["8 GPU<br/>~6.5x"]
    end

    G1 --> G2 --> G4 --> G8
```

**Note:** Speedup is sub-linear due to communication overhead. Typical efficiency: 85-92%.

---

## 12. Quick Reference Cards

### DDP Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│                    DDP QUICK REFERENCE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LAUNCH:                                                    │
│    torchrun --nproc_per_node=N train.py                    │
│                                                             │
│  SETUP:                                                     │
│    dist.init_process_group(backend="nccl")                 │
│    model = DDP(model.to(device), device_ids=[local_rank])  │
│    sampler = DistributedSampler(dataset)                   │
│                                                             │
│  TRAINING LOOP:                                             │
│    for epoch in range(epochs):                             │
│        sampler.set_epoch(epoch)  # Critical!               │
│        for batch in dataloader:                            │
│            loss = model(batch)                             │
│            loss.backward()  # Auto gradient sync           │
│            optimizer.step()                                │
│                                                             │
│  CLEANUP:                                                   │
│    dist.destroy_process_group()                            │
│                                                             │
│  GUARDS:                                                    │
│    if rank == 0:  # Logging, saving                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Environment Variables Reference

```
┌─────────────────────────────────────────────────────────────┐
│               ENVIRONMENT VARIABLES                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RANK         = Global process ID (0, 1, 2, ...)           │
│  LOCAL_RANK   = GPU ID on this machine (0, 1)              │
│  WORLD_SIZE   = Total number of processes                  │
│  MASTER_ADDR  = IP address of rank 0                       │
│  MASTER_PORT  = Port for communication                     │
│                                                             │
│  Example (2 GPUs, 1 machine):                              │
│    Process 0: RANK=0, LOCAL_RANK=0, WORLD_SIZE=2           │
│    Process 1: RANK=1, LOCAL_RANK=1, WORLD_SIZE=2           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### API Endpoints Reference

```
┌─────────────────────────────────────────────────────────────┐
│                    API ENDPOINTS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GET  /health                                               │
│       → {"status": "healthy", "model_loaded": true}        │
│                                                             │
│  POST /predict                                              │
│       ← {"text": "News article..."}                        │
│       → {"label": "Sci/Tech", "confidence": 0.89, ...}     │
│                                                             │
│  POST /predict/batch                                        │
│       ← {"texts": ["Article 1", "Article 2"]}              │
│       → [{"label": "Sports", ...}, {"label": "World", ...}]│
│                                                             │
│  GET  /docs                                                 │
│       → Swagger UI documentation                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### File Learning Order

```
┌─────────────────────────────────────────────────────────────┐
│              RECOMMENDED LEARNING ORDER                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. .planning/research/CONCEPTS.md   ← Theory first        │
│  2. training/model.py                ← Model architecture  │
│  3. training/dataset.py              ← Data pipeline       │
│  4. training/train_single.py         ← Baseline training   │
│  5. training/train_ddp.py            ← DDP (main goal!)    │
│  6. inference/model_loader.py        ← Loading models      │
│  7. inference/app.py                 ← API serving         │
│  8. deployment/Dockerfile            ← Containerization    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: Your Path to Distributed ML Mastery

```mermaid
journey
    title Your Distributed ML Journey
    section Understanding
      Read CONCEPTS.md: 5: You
      Study single GPU baseline: 4: You
      Learn DDP components: 4: You
    section Implementation
      Run single GPU training: 5: You
      Run DDP training: 5: You
      Compare benchmarks: 4: You
    section Production
      Build inference API: 4: You
      Deploy to HuggingFace: 5: You
      Test production API: 5: You
    section Mastery
      Explain DDP in interview: 5: You
      Debug DDP issues: 4: You
      Scale to more GPUs: 4: You
```

### Key Takeaways

1. **DDP = Same model, different data, synchronized gradients**
2. **AllReduce automatically averages gradients during backward()**
3. **DistributedSampler ensures no data overlap between GPUs**
4. **Only rank 0 should log and save checkpoints**
5. **~85-95% parallel efficiency is the industry standard**
6. **torchrun handles process spawning and environment setup**
7. **NCCL is the fastest backend for GPU communication**

### You Can Now Confidently:

- [ ] Explain distributed training in interviews
- [ ] Implement DDP training from scratch
- [ ] Debug common DDP issues
- [ ] Deploy ML models as production APIs
- [ ] Discuss scaling strategies and trade-offs
- [ ] Understand MLOps pipelines end-to-end

---

*This documentation was created for the DistriNews project - a comprehensive learning resource for distributed machine learning.*

**Total Lines of Educational Code:** ~3,500
**Documentation:** ~2,000+ lines
**Learning Time Invested:** Worth every minute

**Now go build something amazing with distributed ML!** 🚀
