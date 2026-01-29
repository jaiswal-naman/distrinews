"""
training/model.py - News Classification Model

This file contains the model wrapper for our news classifier.
We use DistilBERT, a smaller/faster version of BERT.

=============================================================================
CONCEPT: What is DistilBERT?
=============================================================================

DistilBERT is a "distilled" version of BERT:
- BERT: 110 million parameters, slower, slightly more accurate
- DistilBERT: 66 million parameters, 60% faster, 97% of BERT's accuracy

"Distillation" means training a smaller model to mimic a larger one.
The smaller model learns to produce similar outputs as the larger model.

For our learning purposes, DistilBERT is perfect:
- Fast enough to train on free GPUs
- Small enough to fit in memory
- Accurate enough for real applications
- Uses the same transformer architecture as BERT/GPT

=============================================================================
CONCEPT: What are Transformers?
=============================================================================

Transformers are the architecture behind ChatGPT, BERT, and modern NLP.

Key idea: "Attention" - the model learns which words are important for each task.

Example: "The movie was terrible but the acting was great"
- For sentiment: "terrible" and "great" get high attention
- For topic: "movie" and "acting" get high attention

We don't implement transformers from scratch - HuggingFace provides them.

=============================================================================
CONCEPT: Classification with Transformers
=============================================================================

Transformers output a vector for each input token. For classification:

1. Input: "Sports news about football"
2. Tokenize: [CLS] Sports news about football [SEP]
3. Transformer processes all tokens
4. Take the [CLS] token's output (represents whole sentence)
5. Pass through classification head (linear layer)
6. Output: 4 numbers (one per class)

The 4 numbers are "logits" - raw scores before softmax.
Higher logit = model thinks this class is more likely.

=============================================================================
"""

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertConfig


class NewsClassifier(nn.Module):
    """
    Wrapper around DistilBERT for 4-class news classification.

    WHY A WRAPPER?
    --------------
    We could use DistilBertForSequenceClassification directly, but wrapping it:
    1. Makes our code cleaner (hide HuggingFace details)
    2. Makes it easy to swap models later (just change this file)
    3. Lets us add custom logic if needed
    4. Follows software engineering best practices

    WHAT THIS CLASS DOES:
    --------------------
    1. Loads pre-trained DistilBERT weights
    2. Adds a classification head for 4 classes (AG News categories)
    3. Provides a clean forward() method

    AG NEWS CLASSES:
    ---------------
    0 = World
    1 = Sports
    2 = Business
    3 = Sci/Tech
    """

    # Class-level constant: maps class indices to human-readable labels
    # This is useful for inference later
    LABEL_MAP = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }

    def __init__(self, num_labels: int = 4, pretrained_model: str = "distilbert-base-uncased"):
        """
        Initialize the news classifier.

        PARAMETERS:
        -----------
        num_labels : int
            Number of output classes. Default 4 for AG News.

        pretrained_model : str
            Which pre-trained model to load. Default is DistilBERT base uncased.
            "uncased" means text is lowercased before processing.

        WHAT HAPPENS HERE:
        -----------------
        1. super().__init__() - Required for PyTorch modules
        2. Load pre-trained DistilBERT with classification head
        3. The classification head is randomly initialized (we'll train it)
        4. The transformer weights are pre-trained (we'll fine-tune them)
        """
        super().__init__()

        # Store config for reference
        self.num_labels = num_labels
        self.pretrained_model = pretrained_model

        # Load the pre-trained model with a classification head
        # DistilBertForSequenceClassification = DistilBERT + Linear layer for classification
        #
        # WHAT "from_pretrained" DOES:
        # 1. Downloads model weights (cached after first download)
        # 2. Loads the transformer architecture
        # 3. Initializes weights from pre-training
        # 4. Adds a classification head (randomly initialized)
        #
        # The pre-trained weights come from training on massive text corpora.
        # This gives the model understanding of language before we train on AG News.
        self.model = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert input tokens to class predictions.

        PARAMETERS:
        -----------
        input_ids : torch.Tensor
            Token IDs from the tokenizer. Shape: (batch_size, sequence_length)
            Each number represents a word/subword from the vocabulary.

        attention_mask : torch.Tensor
            Mask indicating real tokens (1) vs padding (0). Shape: (batch_size, sequence_length)
            This tells the model to ignore padding tokens.

        RETURNS:
        --------
        logits : torch.Tensor
            Raw classification scores. Shape: (batch_size, num_labels)
            NOT probabilities - need softmax for that.

        WHAT HAPPENS INSIDE:
        -------------------
        1. input_ids and attention_mask go into DistilBERT
        2. DistilBERT processes tokens through 6 transformer layers
        3. The [CLS] token's representation is extracted
        4. A linear layer maps [CLS] representation to 4 class scores
        5. We return these scores (logits)

        EXAMPLE:
        --------
        >>> model = NewsClassifier()
        >>> input_ids = torch.tensor([[101, 2054, 2003, 102, 0]])  # "what is" + padding
        >>> attention_mask = torch.tensor([[1, 1, 1, 1, 0]])  # last token is padding
        >>> logits = model(input_ids, attention_mask)
        >>> logits.shape
        torch.Size([1, 4])  # 1 sample, 4 classes
        """
        # Call the underlying HuggingFace model
        # It returns a SequenceClassifierOutput object with multiple attributes
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract just the logits (raw scores)
        # outputs.logits has shape (batch_size, num_labels)
        return outputs.logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Make a prediction with probability scores.

        This is a convenience method for inference. Returns both the
        predicted class and the confidence score.

        PARAMETERS:
        -----------
        input_ids, attention_mask: Same as forward()

        RETURNS:
        --------
        Tuple of:
            - predicted_class (int): Index of predicted class (0-3)
            - confidence (float): Probability of predicted class (0-1)

        WHY SOFTMAX?
        ------------
        Logits are raw scores (can be any number).
        Softmax converts them to probabilities:
        - All values become 0-1
        - All values sum to 1

        Example:
            logits = [2.0, 1.0, 0.5, 0.1]
            softmax → [0.52, 0.19, 0.12, 0.08]  # sums to ~1.0
        """
        # Get logits
        logits = self.forward(input_ids, attention_mask)

        # Convert to probabilities with softmax
        # dim=-1 means apply softmax across the last dimension (classes)
        probabilities = torch.softmax(logits, dim=-1)

        # Get the highest probability class
        confidence, predicted_class = torch.max(probabilities, dim=-1)

        return predicted_class.item(), confidence.item()

    def get_label_name(self, class_idx: int) -> str:
        """
        Convert class index to human-readable label.

        PARAMETERS:
        -----------
        class_idx : int
            Class index (0-3)

        RETURNS:
        --------
        str: Human-readable label (World, Sports, Business, Sci/Tech)
        """
        return self.LABEL_MAP.get(class_idx, "Unknown")


# =============================================================================
# TESTING CODE
# =============================================================================
# This runs when you execute: python training/model.py
# It verifies the model works correctly.

if __name__ == "__main__":
    print("=" * 60)
    print("Testing NewsClassifier")
    print("=" * 60)

    # Create model
    print("\n1. Creating model...")
    model = NewsClassifier(num_labels=4)
    print(f"   Model created: {model.pretrained_model}")
    print(f"   Number of labels: {model.num_labels}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy data
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_length = 32

    # Create fake input (normally this comes from tokenizer)
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length))  # 30522 = vocab size
    dummy_attention_mask = torch.ones(batch_size, seq_length)

    # Forward pass
    logits = model(dummy_input_ids, dummy_attention_mask)
    print(f"   Input shape: {dummy_input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected: torch.Size([{batch_size}, 4])")

    # Verify output shape
    assert logits.shape == (batch_size, 4), f"Wrong output shape: {logits.shape}"
    print("   ✓ Output shape correct!")

    # Test predict method
    print("\n3. Testing predict method...")
    pred_class, confidence = model.predict(dummy_input_ids[:1], dummy_attention_mask[:1])
    label = model.get_label_name(pred_class)
    print(f"   Predicted class: {pred_class} ({label})")
    print(f"   Confidence: {confidence:.4f}")

    # Test device movement
    print("\n4. Testing device movement...")
    if torch.cuda.is_available():
        model = model.cuda()
        print("   ✓ Model moved to GPU")
    else:
        print("   (No GPU available, staying on CPU)")

    print("\n" + "=" * 60)
    print("All tests passed! Model is working correctly.")
    print("=" * 60)
