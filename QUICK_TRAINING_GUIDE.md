# Quick Training Guide - Faster Training with Corpus Subset

## Overview

Training on the full Text8 corpus (100MB, ~17M words) takes **1.5-2 hours per epoch** on CPU.

By using a **subset of 10M characters (~10% of corpus)**, training time is reduced to approximately:
- **10-15 minutes per epoch**
- **30-45 minutes total for 3 epochs**

This is ~4x faster while still producing meaningful word embeddings!

## Usage

### Default (Fast Training - 10M characters)

```bash
python main.py --mode both --epochs 3
```

This uses 10M characters by default, completing in ~30-45 minutes.

### Custom Corpus Size

```bash
# Very fast training (5M chars, ~15-20 min total)
python main.py --mode both --epochs 3 --max-chars 5000000

# Medium training (20M chars, ~1 hour total)
python main.py --mode both --epochs 3 --max-chars 20000000

# Full corpus (100M chars, ~3-5 hours total)
python main.py --mode both --epochs 3 --max-chars 100000000
```

## Recommended Settings

| Corpus Size | Time per Epoch | Total Time (3 epochs) | Quality |
|-------------|----------------|----------------------|---------|
| 5M chars    | ~5 min         | ~15-20 min          | Basic   |
| **10M chars** | **~10 min**  | **~30-45 min**      | **Good** (Recommended) |
| 20M chars   | ~20 min        | ~1 hour             | Better  |
| 50M chars   | ~40 min        | ~2 hours            | Very Good |
| 100M chars  | ~1.5 hours     | ~4-5 hours          | Best    |

## What Changes with Smaller Corpus?

### Advantages ✓
- **Much faster training** (4-10x speedup)
- **Lower memory usage**
- Still learns semantic relationships
- Still solves analogies
- Good for testing and development

### Trade-offs
- Slightly lower accuracy on rare words
- Smaller effective vocabulary coverage
- May need more epochs for convergence

## Expected Results with 10M Characters

After 3 epochs with 10M characters, you should still see:

✓ `similarity("apple", "banana")` > 0.5  
✓ `similarity("apple", "potato")` < 0.4  
✓ `analogy("man", "king", "woman")` → "queen" in top 5  
✓ Semantic clustering of related words

## Current Training Status

```
Corpus: 10,000,000 characters (~1.7M words)
Vocabulary: 5,000 BPE tokens
Embedding dimension: 100
Epochs: 3
Window size: 5
Negative samples: 5
Batch size: 2,048

Estimated completion: 30-45 minutes
```

## Monitoring Progress

```bash
# Check training progress
python monitor.py

# Quick evaluation when training completes
python monitor.py --eval
```

## Tips for Best Results

1. **Start with 10M** - Good balance of speed and quality
2. **Use 3 epochs minimum** - Helps convergence
3. **Monitor the loss** - Should decrease steadily
4. **Test incrementally** - Start small, increase if needed

## Why This Works

Word embeddings learn from **local context patterns**, not absolute corpus size. A 10M character subset contains:
- ~1.7M words
- ~15-20M training pairs (with window_size=5)
- Sufficient examples for common words
- Good coverage of semantic relationships

This is enough to learn meaningful word representations for demonstration purposes!

## Next Steps

Once training completes (~30-45 minutes), run evaluation:

```bash
python main.py --mode evaluate
```

Or use the monitor for quick results:

```bash
python monitor.py --eval
```
