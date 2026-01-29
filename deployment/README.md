---
title: DistriNews
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# DistriNews - News Classification API

Classify news articles into 4 categories: World, Sports, Business, Sci/Tech.

## Model

- **Architecture:** DistilBERT (66M parameters)
- **Dataset:** AG News (120,000 samples)
- **Training:** PyTorch DDP on 2 GPUs
- **Accuracy:** ~94%

## API Usage

### Classify Text

```bash
curl -X POST https://your-space.hf.space/predict \
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

### Health Check

```bash
curl https://your-space.hf.space/health
```

## Classes

| Class | Description |
|-------|-------------|
| World | International news |
| Sports | Sports news |
| Business | Business and finance |
| Sci/Tech | Science and technology |

## Source Code

[GitHub Repository](https://github.com/yourusername/distrinews)

## Training Details

This model was trained using **PyTorch Distributed Data Parallel (DDP)** across 2 GPUs, demonstrating:
- Data parallelism
- Gradient synchronization
- Efficient multi-GPU training

See the repository for full training code and documentation.
