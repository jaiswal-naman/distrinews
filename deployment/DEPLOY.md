# Deploying to Hugging Face Spaces

This guide walks you through deploying DistriNews to Hugging Face Spaces (free).

---

## Prerequisites

1. **Trained model checkpoint** — `checkpoints/distilbert_agnews.pt`
2. **Hugging Face account** — Create at [huggingface.co](https://huggingface.co)

---

## Step-by-Step Deployment

### 1. Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name:** `distrinews`
   - **License:** MIT
   - **SDK:** Docker
   - **Hardware:** CPU basic (free)
3. Click "Create Space"

### 2. Clone Your Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/distrinews
cd distrinews
```

### 3. Copy Deployment Files

Copy these files from the `deployment/` folder:
- `Dockerfile`
- `requirements.txt`
- `README.md`

Copy these folders from the project:
- `inference/`
- `training/model.py` and `training/__init__.py`

Copy your trained checkpoint:
- `checkpoints/distilbert_agnews.pt`

Your Space should look like:
```
distrinews/
├── Dockerfile
├── README.md
├── requirements.txt
├── inference/
│   ├── __init__.py
│   ├── app.py
│   └── model_loader.py
├── training/
│   ├── __init__.py
│   └── model.py
└── checkpoints/
    └── distilbert_agnews.pt
```

### 4. Push to Hugging Face

```bash
git add .
git commit -m "Deploy DistriNews API"
git push
```

### 5. Wait for Build

Hugging Face will:
1. Build your Docker image
2. Start the container
3. Your API will be live!

Check build logs at: `https://huggingface.co/spaces/YOUR_USERNAME/distrinews`

### 6. Test Your API

Once deployed, test with:

```bash
# Replace with your Space URL
curl -X POST https://YOUR_USERNAME-distrinews.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple announces new iPhone"}'
```

---

## Alternative: Gradio Interface (Simpler)

If you prefer a web UI instead of API, you can use Gradio:

```python
# app.py (replace inference/app.py)
import gradio as gr
from inference.model_loader import ModelLoader

# Load model
loader = ModelLoader("checkpoints/distilbert_agnews.pt")

def predict(text):
    result = loader.predict(text)
    return f"{result['label']} ({result['confidence']:.1%})"

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="News Text", lines=3),
    outputs=gr.Textbox(label="Prediction"),
    title="DistriNews",
    description="Classify news into World, Sports, Business, or Sci/Tech"
)

demo.launch()
```

For Gradio, change Space SDK to "Gradio" instead of "Docker".

---

## Troubleshooting

### "Model not found" error

Make sure `checkpoints/distilbert_agnews.pt` is in your Space repository.
Large files (>10MB) need Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add checkpoints/distilbert_agnews.pt
git commit -m "Add model checkpoint"
git push
```

### Build fails

Check the build logs for errors. Common issues:
- Missing dependencies in requirements.txt
- File path issues in Dockerfile

### API not responding

The app must run on port 7860 for HF Spaces.
Check that uvicorn is started with `--port 7860`.

---

## Estimated Costs

| Component | Cost |
|-----------|------|
| Hugging Face Space (CPU) | **Free** |
| Storage (< 50GB) | **Free** |

For more resources, you can upgrade to paid tiers, but free is enough for this project.

---

*Your API is now live! Share the URL in your portfolio and resume.*
