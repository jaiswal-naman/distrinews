"""
inference/app.py - FastAPI Inference Server

This is the REST API that serves model predictions.

=============================================================================
CONCEPT: Why FastAPI?
=============================================================================

FastAPI is a modern Python web framework for building APIs:

1. FAST: One of the fastest Python frameworks (uses async)
2. AUTOMATIC DOCS: Generates OpenAPI/Swagger documentation
3. TYPE HINTS: Uses Python type hints for validation
4. EASY: Simple, intuitive API

Other options:
- Flask: Older, more verbose, no automatic validation
- Django: Overkill for a simple API
- Starlette: FastAPI is built on this

=============================================================================
CONCEPT: API Design
=============================================================================

Our API has two endpoints:

1. GET /health
   - Returns {"status": "healthy"}
   - Used by load balancers to check if server is alive

2. POST /predict
   - Input: {"text": "Some news article..."}
   - Output: {"label": "Business", "confidence": 0.92, ...}
   - The actual prediction endpoint

WHY POST FOR PREDICT?
- GET is for retrieving data (no side effects)
- POST is for sending data to process
- Text could be long (GET has URL length limits)

=============================================================================
CONCEPT: Model Loading Strategy
=============================================================================

We load the model ONCE at startup, not per request.

BAD (slow):
    @app.post("/predict")
    def predict(text):
        model = load_model()  # 5 seconds every request!
        return model.predict(text)

GOOD (fast):
    model = load_model()  # Once at startup

    @app.post("/predict")
    def predict(text):
        return model.predict(text)  # Milliseconds

This is why we use a global ModelLoader instance.

=============================================================================
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import our model loader
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.model_loader import ModelLoader


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================
# Pydantic models define the structure of API requests and responses.
# FastAPI uses them for:
# 1. Automatic validation (reject bad inputs)
# 2. Automatic documentation (OpenAPI/Swagger)
# 3. Type hints (IDE autocomplete)

class PredictRequest(BaseModel):
    """
    Request body for /predict endpoint.

    FIELDS:
    -------
    text : str
        The news text to classify.
        Must be non-empty.

    EXAMPLE:
    --------
    {
        "text": "Apple announces new iPhone with AI features"
    }
    """
    text: str = Field(
        ...,  # ... means required
        min_length=1,
        max_length=10000,
        description="The news text to classify",
        example="Apple announces new iPhone with revolutionary AI features"
    )


class PredictResponse(BaseModel):
    """
    Response body for /predict endpoint.

    FIELDS:
    -------
    label : str
        Predicted class: World, Sports, Business, or Sci/Tech

    confidence : float
        Probability of predicted class (0-1)

    all_probabilities : dict, optional
        Probabilities for all classes

    EXAMPLE:
    --------
    {
        "label": "Sci/Tech",
        "confidence": 0.89,
        "all_probabilities": {
            "World": 0.02,
            "Sports": 0.01,
            "Business": 0.08,
            "Sci/Tech": 0.89
        }
    }
    """
    label: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., description="Confidence score (0-1)")
    all_probabilities: Optional[dict] = Field(
        None,
        description="Probabilities for all classes"
    )


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = Field(..., description="Server health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class BatchPredictRequest(BaseModel):
    """Request body for /predict/batch endpoint."""
    texts: list = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to classify (max 100)",
        example=["News article 1...", "News article 2..."]
    )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Create FastAPI app with metadata
app = FastAPI(
    title="DistriNews API",
    description="""
    News classification API powered by DistilBERT.

    This model was trained using PyTorch Distributed Data Parallel (DDP)
    on the AG News dataset.

    **Classes:**
    - World: International news
    - Sports: Sports news
    - Business: Business and finance news
    - Sci/Tech: Science and technology news
    """,
    version="1.0.0",
    contact={
        "name": "DistriNews Project",
        "url": "https://github.com/yourusername/distrinews"
    }
)


# =============================================================================
# MODEL LOADING (at startup)
# =============================================================================
# We use a global variable to hold the model loader.
# It's loaded once when the server starts.

model_loader: Optional[ModelLoader] = None


@app.on_event("startup")
async def load_model():
    """
    Load the model when the server starts.

    WHY @app.on_event("startup")?
    ----------------------------
    This function runs ONCE when the server starts, before handling any requests.
    Perfect for:
    - Loading ML models
    - Connecting to databases
    - Initializing caches

    If loading fails, the server won't start (which is what we want).
    """
    global model_loader

    # Get checkpoint path from environment variable or use default
    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT",
        "checkpoints/distilbert_agnews.pt"
    )

    print(f"\n{'='*60}")
    print("Starting DistriNews API Server")
    print(f"{'='*60}")

    try:
        model_loader = ModelLoader(checkpoint_path)
        print(f"\nAPI ready to serve predictions!")
        print(f"{'='*60}\n")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("The server will start but /predict will fail.")
        print("Train a model first or set MODEL_CHECKPOINT env variable.")
        print(f"{'='*60}\n")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to docs."""
    return {"message": "Welcome to DistriNews API. Visit /docs for documentation."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check if the server is healthy and model is loaded.

    USAGE:
    ------
    Used by load balancers and orchestration systems (Kubernetes, etc.)
    to check if the server can handle requests.

    RETURNS:
    --------
    - status: "healthy" if server is running
    - model_loaded: true if model is ready for predictions
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader is not None
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Classify a news article.

    PARAMETERS:
    -----------
    request : PredictRequest
        JSON body with "text" field

    RETURNS:
    --------
    PredictResponse with:
        - label: Predicted class (World, Sports, Business, Sci/Tech)
        - confidence: Probability of predicted class
        - all_probabilities: Probabilities for all classes

    EXAMPLE:
    --------
    Request:
    ```json
    {"text": "Apple announces new iPhone with AI features"}
    ```

    Response:
    ```json
    {
        "label": "Sci/Tech",
        "confidence": 0.89,
        "all_probabilities": {
            "World": 0.02,
            "Sports": 0.01,
            "Business": 0.08,
            "Sci/Tech": 0.89
        }
    }
    ```
    """
    # Check if model is loaded
    if model_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first or check MODEL_CHECKPOINT."
        )

    # Make prediction
    try:
        result = model_loader.predict(request.text)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Classify multiple news articles at once.

    More efficient than multiple /predict calls for bulk processing.

    PARAMETERS:
    -----------
    request : BatchPredictRequest
        JSON body with "texts" field (list of strings)

    RETURNS:
    --------
    List of predictions, same format as /predict

    EXAMPLE:
    --------
    Request:
    ```json
    {"texts": ["Article 1...", "Article 2..."]}
    ```

    Response:
    ```json
    [
        {"label": "Sports", "confidence": 0.95, ...},
        {"label": "Business", "confidence": 0.88, ...}
    ]
    ```
    """
    if model_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    try:
        results = model_loader.predict_batch(request.texts)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# =============================================================================
# RUN SERVER
# =============================================================================
# This allows running the server directly: python inference/app.py
# For production, use: uvicorn inference.app:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("Starting DistriNews API Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health  - Health check")
    print("  POST /predict - Classify text")
    print("  GET  /docs    - API documentation")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(
        "inference.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (dev only)
    )
