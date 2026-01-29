"""
training/dataset.py - AG News Dataset Loading and Preprocessing

This file handles loading the AG News dataset and preparing it for training.

=============================================================================
CONCEPT: What is Tokenization?
=============================================================================

Transformers don't understand text - they understand numbers.
Tokenization converts text to numbers the model can process.

Example:
    Text: "Hello world"
    Tokens: ["[CLS]", "hello", "world", "[SEP]"]
    Token IDs: [101, 7592, 2088, 102]

SPECIAL TOKENS:
- [CLS] (ID 101): "Classification" token, added at start
- [SEP] (ID 102): "Separator" token, added at end
- [PAD] (ID 0): "Padding" token, fills short sequences

WHY PADDING?
Batches need uniform shapes. If one sentence has 10 words and another has 5:
    Sentence 1: [101, ...(10 words)..., 102, 0, 0, 0, 0, 0]  # padded to 20
    Sentence 2: [101, ...(5 words)..., 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # padded to 20

Now both have length 20 and can be batched together.

=============================================================================
CONCEPT: What is Attention Mask?
=============================================================================

The attention mask tells the model which tokens are real vs padding.

    tokens:         [CLS] hello world [SEP] [PAD] [PAD]
    attention_mask: [1,   1,    1,    1,    0,    0   ]

1 = "pay attention to this token"
0 = "ignore this token (it's just padding)"

Without this, the model would try to learn from padding tokens, which is wrong.

=============================================================================
CONCEPT: PyTorch Dataset
=============================================================================

PyTorch's DataLoader needs data in a specific format. A Dataset class provides:

1. __len__(): How many samples? (e.g., 120,000 for AG News train)
2. __getitem__(idx): Get sample at index idx

DataLoader uses these to:
- Shuffle data each epoch (calls __getitem__ in random order)
- Create batches (calls __getitem__ multiple times, stacks results)
- Support DistributedSampler (sampler calls __getitem__ for specific indices)

=============================================================================
CONCEPT: AG News Dataset
=============================================================================

AG News is a news classification dataset:
- 120,000 training samples
- 7,600 test samples
- 4 classes: World, Sports, Business, Sci/Tech

Each sample has:
- text: News headline + description
- label: 0, 1, 2, or 3

Example:
    text: "Wall Street rallies on strong earnings reports"
    label: 2 (Business)

=============================================================================
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DistilBertTokenizer
from typing import Dict, Optional


def load_agnews(split: str = "train") -> list:
    """
    Load AG News dataset from HuggingFace.

    PARAMETERS:
    -----------
    split : str
        Which split to load: "train" or "test"

    RETURNS:
    --------
    list: List of samples, each with 'text' and 'label' keys

    WHAT HAPPENS:
    ------------
    1. HuggingFace downloads dataset (cached after first time)
    2. We convert to list for easier handling
    3. Each sample has 'text' and 'label'

    WHY HuggingFace Datasets?
    ------------------------
    - One line to load any of 1000s of datasets
    - Handles downloading and caching
    - Memory-efficient (loads on demand)
    - Standardized format
    """
    print(f"Loading AG News {split} split...")

    # Load from HuggingFace Hub
    # This downloads the dataset if not cached
    dataset = load_dataset("ag_news", split=split)

    # Convert to list of dicts
    # Each dict has: {"text": "...", "label": 0-3}
    samples = []
    for item in dataset:
        samples.append({
            "text": item["text"],
            "label": item["label"]
        })

    print(f"Loaded {len(samples):,} samples")
    return samples


def get_tokenizer(model_name: str = "distilbert-base-uncased") -> DistilBertTokenizer:
    """
    Load the tokenizer for DistilBERT.

    PARAMETERS:
    -----------
    model_name : str
        Which model's tokenizer to load. Must match the model!

    RETURNS:
    --------
    DistilBertTokenizer: Tokenizer instance

    WHY MATCH MODEL AND TOKENIZER?
    -----------------------------
    Each model has its own vocabulary. DistilBERT's vocabulary maps:
        "hello" → 7592
        "world" → 2088
        etc.

    If you use the wrong tokenizer, word IDs won't match what the model expects!

    WHAT "uncased" MEANS:
    --------------------
    "uncased" = text is lowercased before tokenization
    "Hello World" → "hello world" → [7592, 2088]

    "cased" would keep capitalization, but uncased is more common for classification.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    return tokenizer


class AGNewsDataset(Dataset):
    """
    PyTorch Dataset for AG News.

    This class wraps the raw data and provides the interface that
    PyTorch DataLoader expects.

    WHY A CUSTOM DATASET CLASS?
    --------------------------
    1. PyTorch DataLoader needs __len__ and __getitem__
    2. We need to tokenize text on-the-fly (or could pre-tokenize)
    3. DistributedSampler needs this interface to shard data
    4. Clean separation of data loading from training logic

    DESIGN CHOICE: Tokenize in __getitem__
    -------------------------------------
    We tokenize each sample when it's requested, not upfront.
    Pros:
        - Lower memory usage (don't store all tokenized data)
        - Can change tokenization parameters easily
    Cons:
        - Slower (tokenize every epoch)
        - For production, might want to pre-tokenize and cache

    For learning, on-the-fly tokenization is clearer to understand.
    """

    def __init__(
        self,
        data: list,
        tokenizer: DistilBertTokenizer,
        max_length: int = 128
    ):
        """
        Initialize the dataset.

        PARAMETERS:
        -----------
        data : list
            List of samples from load_agnews()

        tokenizer : DistilBertTokenizer
            Tokenizer from get_tokenizer()

        max_length : int
            Maximum sequence length. Longer texts are truncated.
            Default 128 is good balance of speed vs context.

        WHY max_length=128?
        ------------------
        - Longer = more context but slower training and more memory
        - 128 is enough for news headlines + some description
        - DistilBERT can handle up to 512, but 128 is faster
        - For AG News, 128 captures most of the signal
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        DataLoader calls this to know how many batches to create.
        DistributedSampler calls this to divide data among GPUs.

        RETURNS:
        --------
        int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample by index.

        This is called by DataLoader (or DistributedSampler) to get
        specific samples. The sampler decides WHICH indices to request.

        PARAMETERS:
        -----------
        idx : int
            Index of sample to retrieve (0 to len-1)

        RETURNS:
        --------
        dict with:
            - input_ids: Token IDs, shape (max_length,)
            - attention_mask: Real tokens vs padding, shape (max_length,)
            - labels: Class label, shape () [scalar]

        WHY THESE THREE FIELDS?
        ----------------------
        - input_ids: The actual text, converted to numbers
        - attention_mask: Tells model which tokens are real
        - labels: What we're trying to predict

        The model uses input_ids and attention_mask to make predictions.
        We compare predictions to labels to compute loss.

        TOKENIZER PARAMETERS:
        --------------------
        - padding="max_length": Pad all sequences to max_length
        - truncation=True: Cut off text longer than max_length
        - max_length: Maximum number of tokens
        - return_tensors="pt": Return PyTorch tensors
        """
        # Get raw sample
        sample = self.data[idx]
        text = sample["text"]
        label = sample["label"]

        # Tokenize the text
        # This converts "Hello world" to {"input_ids": [101, 7592, 2088, 102, 0, ...], ...}
        encoding = self.tokenizer(
            text,
            padding="max_length",      # Pad to max_length
            truncation=True,           # Truncate if too long
            max_length=self.max_length,
            return_tensors="pt"        # Return PyTorch tensors
        )

        # Remove the batch dimension (tokenizer adds it)
        # Shape goes from (1, max_length) to (max_length,)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Convert label to tensor
        labels = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_dataloader(
    dataset: AGNewsDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    sampler=None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from a dataset.

    This is a convenience function. In DDP training, we'll create
    DataLoaders differently (with DistributedSampler).

    PARAMETERS:
    -----------
    dataset : AGNewsDataset
        The dataset to load from

    batch_size : int
        How many samples per batch. Default 32.
        Larger = faster training, more memory.
        Smaller = slower training, less memory, sometimes better generalization.

    shuffle : bool
        Randomize order each epoch? True for training, False for evaluation.

    num_workers : int
        Number of parallel data loading processes. 0 = main process only.
        Higher = faster loading, more memory.
        Set to 0 for debugging, 2-4 for training.

    sampler : optional
        Custom sampler (e.g., DistributedSampler). If provided, shuffle must be False.

    RETURNS:
    --------
    DataLoader: Iterator that yields batches

    WHY DataLoader?
    --------------
    DataLoader handles:
    - Batching (combine multiple samples)
    - Shuffling (randomize order)
    - Parallel loading (num_workers)
    - Pinned memory (faster GPU transfer)

    Without DataLoader, you'd write a lot of boilerplate code.
    """
    # Note: if sampler is provided, shuffle must be False
    # (the sampler handles shuffling)
    if sampler is not None:
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=torch.cuda.is_available()  # Faster GPU transfer
    )


# =============================================================================
# TESTING CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing AG News Dataset")
    print("=" * 60)

    # Test tokenizer
    print("\n1. Testing tokenizer...")
    tokenizer = get_tokenizer()

    # Show tokenization example
    example_text = "Wall Street rallies on strong earnings reports"
    tokens = tokenizer.tokenize(example_text)
    token_ids = tokenizer.encode(example_text)
    print(f"\n   Example text: '{example_text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Token IDs: {token_ids}")

    # Test dataset loading
    print("\n2. Loading AG News dataset...")
    # Load only 1000 samples for testing (faster)
    train_data = load_agnews("train")[:1000]
    print(f"   Loaded {len(train_data)} samples (subset for testing)")

    # Show sample
    print(f"\n   Sample 0:")
    print(f"   Text: {train_data[0]['text'][:100]}...")
    print(f"   Label: {train_data[0]['label']}")

    # Test AGNewsDataset
    print("\n3. Testing AGNewsDataset...")
    dataset = AGNewsDataset(train_data, tokenizer, max_length=128)
    print(f"   Dataset length: {len(dataset)}")

    # Get one sample
    sample = dataset[0]
    print(f"\n   Sample 0 after processing:")
    print(f"   input_ids shape: {sample['input_ids'].shape}")
    print(f"   attention_mask shape: {sample['attention_mask'].shape}")
    print(f"   labels: {sample['labels']}")

    # Verify shapes
    assert sample["input_ids"].shape == (128,), f"Wrong input_ids shape"
    assert sample["attention_mask"].shape == (128,), f"Wrong attention_mask shape"
    print("   ✓ Shapes correct!")

    # Test DataLoader
    print("\n4. Testing DataLoader...")
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

    # Get one batch
    batch = next(iter(dataloader))
    print(f"\n   Batch shapes:")
    print(f"   input_ids: {batch['input_ids'].shape}")
    print(f"   attention_mask: {batch['attention_mask'].shape}")
    print(f"   labels: {batch['labels'].shape}")

    # Verify batch shapes
    assert batch["input_ids"].shape == (4, 128), f"Wrong batch input_ids shape"
    assert batch["labels"].shape == (4,), f"Wrong batch labels shape"
    print("   ✓ Batch shapes correct!")

    # Show label distribution
    print("\n5. Label distribution (first 1000 samples):")
    from collections import Counter
    labels = [s["label"] for s in train_data]
    label_counts = Counter(labels)
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    for label, count in sorted(label_counts.items()):
        print(f"   {label_names[label]}: {count}")

    print("\n" + "=" * 60)
    print("All tests passed! Dataset is working correctly.")
    print("=" * 60)
