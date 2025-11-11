# Word Embedding Model From Scratch

A complete implementation of word embeddings from scratch using Skip-gram with Negative Sampling and BPE tokenization.

## Features

- ✅ **BPE Tokenization**: Custom Byte-Pair Encoding implementation for efficient vocabulary learning
- ✅ **Skip-gram Model**: Word2Vec-style skip-gram architecture with negative sampling
- ✅ **From Scratch**: No pre-trained models or external embedding libraries (Gensim, etc.)
- ✅ **CPU Training**: Optimized NumPy operations for efficient CPU-based training
- ✅ **Semantic Understanding**: Learns word relationships and analogies

## Project Structure

```
embedding_model/
├── bpe_tokenizer.py      # BPE tokenization implementation
├── skipgram_model.py     # Skip-gram model with negative sampling
├── training.py           # Training pipeline
├── evaluation.py         # Similarity and analogy evaluation
├── main.py              # Main execution script
├── download_data.py     # Text8 corpus downloader
├── data/                # Corpus data
│   └── text8           # 100MB Wikipedia text
└── models/              # Saved models
    ├── tokenizer.pkl   # Trained BPE tokenizer
    └── embeddings.npy  # Word embeddings
```

## Installation

```bash
# Create virtual environment (optional)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install numpy requests
```

## Quick Start

### 1. Download Dataset

```bash
python download_data.py
```

This downloads the Text8 corpus (100MB of cleaned Wikipedia text).

### 2. Train and Evaluate

```bash
# Train and evaluate in one go (uses 10M chars by default, ~30-45 min)
python main.py --mode both --epochs 3

# Or separately
python main.py --mode train --epochs 3
python main.py --mode evaluate

# For faster training with smaller corpus (15-20 min)
python main.py --mode both --epochs 3 --max-chars 5000000

# For full corpus training (3-5 hours)
python main.py --mode both --epochs 3 --max-chars 100000000
```

### 3. Custom Configuration

```bash
python main.py --mode both \
    --vocab-size 5000 \
    --embedding-dim 100 \
    --epochs 3 \
    --window-size 5 \
    --negative-samples 5 \
    --batch-size 2048 \
    --learning-rate 0.025
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vocab-size` | 5000 | BPE vocabulary size |
| `--embedding-dim` | 100 | Embedding dimensionality |
| `--epochs` | 3 | Training epochs |
| `--window-size` | 5 | Context window size |
| `--negative-samples` | 5 | Negative samples per positive |
| `--batch-size` | 2048 | Training batch size |
| `--learning-rate` | 0.025 | Initial learning rate |
| `--max-chars` | 10000000 | Max characters from corpus (10M = ~30-45 min) |

## Architecture

### 1. BPE Tokenizer

- Learns 5,000 most frequent character-pair merges
- Handles out-of-vocabulary words via subword tokenization
- Efficient vocabulary representation

### 2. Skip-gram Model

**Embeddings:**
- Target embeddings: `[vocab_size, embedding_dim]`
- Context embeddings: `[vocab_size, embedding_dim]`
- Initialized with uniform random values in `[-0.5/dim, 0.5/dim]`

**Training:**
- Negative sampling with unigram^0.75 distribution
- Binary cross-entropy loss
- SGD optimizer with linear learning rate decay (0.025 → 0.0001)
- Vectorized batch updates with NumPy

## Evaluation

The model is evaluated on:

### 1. Similarity Tests
```python
similarity("apple", "banana") > similarity("apple", "potato")
similarity("king", "queen") ≈ high
```

### 2. Nearest Neighbors
```python
nearest_neighbors("apple", k=10)
# Expected: banana, orange, fruit, etc.
```

### 3. Word Analogies
```python
analogy("man", "king", "woman") → "queen"
# man:king :: woman:?
```

## Expected Results

After 3 epochs of training (~1.5 hours on CPU):

- ✅ `similarity("apple", "banana")` > 0.5
- ✅ `similarity("apple", "potato")` < 0.4
- ✅ `analogy("man", "king", "woman")` returns "queen" in top 3
- ✅ Semantic clustering of similar words

## Training Performance

**Default (10M characters - Recommended):**
- **Corpus**: 10M chars (~1.7M words, 10% of Text8)
- **Tokens**: ~3M after BPE tokenization
- **Training pairs**: ~15M (window_size=5)
- **Training time**: ~10-15 minutes per epoch on modern CPU
- **Total time**: ~30-45 minutes for 3 epochs
- **Memory usage**: ~500MB-1GB RAM

**Full Corpus (100M characters - Optional):**
- **Corpus**: Text8 full (100MB, ~17M words)
- **Tokens**: ~30M after BPE tokenization
- **Training pairs**: ~150M (window_size=5)
- **Training time**: ~1.5-2 hours per epoch
- **Total time**: ~4-5 hours for 3 epochs
- **Memory usage**: ~2-3GB RAM

## Implementation Details

### BPE Tokenization
1. Start with character-level tokens
2. Iteratively merge most frequent pairs
3. Learn 5,000 merge rules
4. Handle OOV words via subword decomposition

### Skip-gram Training
1. Generate (center, context) pairs with sliding window
2. For each positive pair, sample K negative examples
3. Compute loss: `-log(σ(pos)) - Σ log(σ(-neg))`
4. Update embeddings via SGD with gradients

### Negative Sampling
- Distribution: `P(w) ∝ count(w)^0.75`
- Smoothed unigram distribution
- 5 negative samples per positive pair

## Success Criteria

- [x] Implementation from scratch (no Gensim/pre-trained models)
- [x] BPE tokenization
- [x] Skip-gram with negative sampling
- [x] CPU-only training
- [x] Semantic similarity validation
- [x] Analogy solving capability
- [x] ~5-6 hours total development time

## Advanced Usage

### Using the Trained Embeddings

```python
import numpy as np
from bpe_tokenizer import BPETokenizer
from evaluation import EmbeddingEvaluator

# Load
tokenizer = BPETokenizer()
tokenizer.load('models/tokenizer.pkl')
embeddings = np.load('models/embeddings.npy')

# Evaluate
evaluator = EmbeddingEvaluator(embeddings, tokenizer)

# Similarity
sim = evaluator.similarity("king", "queen")
print(f"Similarity: {sim:.4f}")

# Nearest neighbors
neighbors = evaluator.nearest_neighbors("apple", k=10)
for word, score in neighbors:
    print(f"{word}: {score:.4f}")

# Analogy
results = evaluator.analogy("man", "king", "woman", k=5)
for word, score in results:
    print(f"{word}: {score:.4f}")
```

### Training on Custom Corpus

```python
from training import train_embeddings

model, tokenizer = train_embeddings(
    corpus_path='path/to/your/corpus.txt',
    tokenizer_path='models/custom_tokenizer.pkl',
    embedding_path='models/custom_embeddings.npy',
    vocab_size=5000,
    embedding_dim=100,
    epochs=3
)
```

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` (e.g., 1024 or 512)
- Reduce `vocab_size` (e.g., 3000)
- Reduce `embedding_dim` (e.g., 50)

**Slow Training:**
- Reduce `epochs` to 2 or 1
- Use smaller corpus subset
- Increase `batch_size` if memory allows

**Poor Results:**
- Train for more epochs (4-5)
- Increase `window_size` (e.g., 7-10)
- Tune `learning_rate` (try 0.05 or 0.01)

## References

- Mikolov et al. (2013): "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al. (2013): "Distributed Representations of Words and Phrases and their Compositionality"
- Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units"

## License

MIT License - Feel free to use for learning and research purposes.
