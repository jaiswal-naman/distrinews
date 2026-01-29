"""
inference/model_loader.py - Model Loading for Inference

This module handles loading a trained model checkpoint for inference.

=============================================================================
CONCEPT: Training vs Inference Model Loading
=============================================================================

DURING TRAINING:
- Model is wrapped with DDP
- Optimizer state is needed
- We save model.module.state_dict()

DURING INFERENCE:
- No DDP wrapper needed
- No optimizer needed
- Just load weights into base model

The checkpoint we saved contains:
{
    "model_state_dict": {...},     # Model weights
    "optimizer_state_dict": {...}, # Not needed for inference
    "epoch": 3,                    # Not needed for inference
    "loss": 0.5,                   # For reference
    "accuracy": 94.5               # For reference
}

We only need model_state_dict for inference!

=============================================================================
CONCEPT: Device Selection for Inference
=============================================================================

Training: Always use GPU (speed matters for millions of samples)
Inference: Can use CPU or GPU

WHY CPU INFERENCE IS OFTEN FINE:
- Single request at a time
- Latency is dominated by network, not compute
- CPU is cheaper to deploy
- Works everywhere (no GPU needed)

We auto-detect GPU but work fine on CPU.

=============================================================================
"""

import os
import torch
from transformers import DistilBertTokenizer

# Import from training module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import NewsClassifier


class ModelLoader:
    """
    Loads a trained model checkpoint and provides prediction interface.

    USAGE:
    ------
    loader = ModelLoader("checkpoints/distilbert_agnews.pt")
    result = loader.predict("Breaking news about technology...")
    print(result)  # {"label": "Sci/Tech", "confidence": 0.95}

    WHY A CLASS?
    -----------
    1. Load model ONCE at startup, reuse for all requests
    2. Encapsulate model + tokenizer together
    3. Clean interface for the API layer
    """

    # Class labels for AG News
    LABELS = ["World", "Sports", "Business", "Sci/Tech"]

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/distilbert_agnews.pt",
        device: str = None
    ):
        """
        Initialize the model loader.

        PARAMETERS:
        -----------
        checkpoint_path : str
            Path to the saved checkpoint file.
            Default: checkpoints/distilbert_agnews.pt

        device : str, optional
            Device to run inference on: "cuda", "cpu", or None (auto-detect)

        WHAT HAPPENS:
        ------------
        1. Auto-detect device (GPU if available, else CPU)
        2. Load tokenizer
        3. Create model architecture
        4. Load trained weights into model
        5. Set model to evaluation mode
        """
        # =====================================================================
        # Step 1: Device selection
        # =====================================================================
        if device is None:
            # Auto-detect: use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Inference device: {self.device}")

        # =====================================================================
        # Step 2: Load tokenizer
        # =====================================================================
        # Must use the SAME tokenizer as training!
        # Different tokenizer = different token IDs = garbage predictions
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        print(f"Tokenizer loaded: distilbert-base-uncased")

        # =====================================================================
        # Step 3: Create model (architecture only, no weights yet)
        # =====================================================================
        self.model = NewsClassifier(num_labels=4)
        print(f"Model architecture created")

        # =====================================================================
        # Step 4: Load trained weights
        # =====================================================================
        self._load_checkpoint(checkpoint_path)

        # =====================================================================
        # Step 5: Set evaluation mode
        # =====================================================================
        # CRITICAL: model.eval() disables dropout and changes batch norm behavior
        # Without this, predictions will be inconsistent!
        self.model.eval()
        print(f"Model set to evaluation mode")

        print(f"Model ready for inference!")

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load weights from checkpoint file.

        WHY SEPARATE METHOD?
        -------------------
        Keeps __init__ clean and allows reloading if needed.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Please train the model first with train_ddp.py or train_single.py"
            )

        print(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        # map_location ensures it works even if trained on GPU but inferring on CPU
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load weights into model
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Move model to device
        self.model = self.model.to(self.device)

        # Print checkpoint info
        print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
        print(f"  Training accuracy: {checkpoint.get('accuracy', '?'):.2f}%")

    def predict(self, text: str) -> dict:
        """
        Make a prediction for a single text input.

        PARAMETERS:
        -----------
        text : str
            The news text to classify

        RETURNS:
        --------
        dict with:
            - label: Predicted class name (World, Sports, Business, Sci/Tech)
            - confidence: Probability of predicted class (0-1)
            - all_probabilities: Dict of all class probabilities

        EXAMPLE:
        --------
        >>> loader.predict("Apple announces new iPhone")
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
        # =====================================================================
        # Step 1: Tokenize input
        # =====================================================================
        # Same parameters as training!
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # =====================================================================
        # Step 2: Run inference
        # =====================================================================
        # torch.no_grad() disables gradient computation
        # - Faster (no need to track operations for backprop)
        # - Less memory (no gradient tensors stored)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        # =====================================================================
        # Step 3: Convert logits to probabilities
        # =====================================================================
        # Softmax converts raw scores to probabilities (sum to 1)
        probabilities = torch.softmax(logits, dim=-1)

        # Get the prediction
        confidence, predicted_idx = torch.max(probabilities, dim=-1)

        # Convert to Python types
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

        # Get all probabilities
        all_probs = probabilities.squeeze().tolist()

        return {
            "label": self.LABELS[predicted_idx],
            "confidence": round(confidence, 4),
            "all_probabilities": {
                label: round(prob, 4)
                for label, prob in zip(self.LABELS, all_probs)
            }
        }

    def predict_batch(self, texts: list) -> list:
        """
        Make predictions for multiple texts at once.

        More efficient than calling predict() multiple times
        because we batch the tokenization and inference.

        PARAMETERS:
        -----------
        texts : list of str
            List of news texts to classify

        RETURNS:
        --------
        list of dict: Same format as predict(), one per input text
        """
        # Tokenize all texts at once
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        confidences, predicted_indices = torch.max(probabilities, dim=-1)

        # Build results
        results = []
        for i in range(len(texts)):
            idx = predicted_indices[i].item()
            conf = confidences[i].item()
            probs = probabilities[i].tolist()

            results.append({
                "label": self.LABELS[idx],
                "confidence": round(conf, 4),
                "all_probabilities": {
                    label: round(prob, 4)
                    for label, prob in zip(self.LABELS, probs)
                }
            })

        return results


# =============================================================================
# TESTING CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Model Loader")
    print("=" * 60)

    # Check if checkpoint exists
    checkpoint_path = "checkpoints/distilbert_agnews.pt"

    if not os.path.exists(checkpoint_path):
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Creating a dummy checkpoint for testing...")

        # Create dummy checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        model = NewsClassifier(num_labels=4)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "epoch": 0,
            "loss": 0.0,
            "accuracy": 0.0
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Dummy checkpoint saved to {checkpoint_path}")
        print("(This is UNTRAINED - predictions will be random)")

    # Load model
    print("\n1. Loading model...")
    loader = ModelLoader(checkpoint_path)

    # Test prediction
    print("\n2. Testing single prediction...")
    test_texts = [
        "Apple announces new iPhone with revolutionary AI features",
        "Lakers defeat Celtics in overtime thriller",
        "Stock market reaches all-time high amid economic optimism",
        "Scientists discover high water in universe"
    ]

    for text in test_texts:
        result = loader.predict(text)
        print(f"\n   Text: {text[:50]}...")
        print(f"   Predicted: {result['label']} ({result['confidence']:.2%})")

    # Test batch prediction
    print("\n3. Testing batch prediction...")
    results = loader.predict_batch(test_texts)
    print(f"   Processed {len(results)} texts in batch")

    print("\n" + "=" * 60)
    print("Model loader tests passed!")
    print("=" * 60)
