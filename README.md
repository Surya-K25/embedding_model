# Custom Word Embedding Model

A complete implementation of a custom word embedding model using **Skip-gram architecture** with **BPE tokenization**, demonstrating semantic understanding through similarity comparisons, analogy solving, and visualization.

## 🎯 Project Overview

This project creates word embeddings from scratch using the Text8 corpus (~100MB), implementing:
- **BPE (Byte Pair Encoding)** tokenization with 12K vocabulary
- **Word2Vec Skip-gram** model with 128-dimensional embeddings
- **Semantic similarity** calculations using cosine distance
- **Analogy solving** through vector arithmetic
- **2D visualizations** using t-SNE and UMAP

## 🏗️ Architecture

### 1. Data Pipeline
```
Text8 Corpus (100MB) 
    ↓
Preprocessing (lowercase, chunking)
    ↓
BPE Tokenization (12K vocab)
    ↓
Word2Vec Skip-gram Training
    ↓
128-dimensional Word Vectors
```

### 2. Model Architecture

**Word2Vec Skip-gram Configuration:**
- **Vector Size**: 128 dimensions
- **Context Window**: 5 words (±5 around target)
- **Negative Sampling**: 10 negative samples
- **Minimum Frequency**: 5 occurrences
- **Training Epochs**: 10
- **Architecture**: Skip-gram (predicts context from center word)

**Why Skip-gram?**
- Better performance on rare words
- Produces more accurate word vectors for smaller datasets
- Captures more nuanced semantic relationships

### 3. Tokenization Strategy

**BPE (Byte Pair Encoding):**
- Vocabulary size: 12,000 tokens
- Handles subword units for better coverage
- Reduces out-of-vocabulary issues
- Balances between word-level and character-level

## 📁 Project Structure

```
new_embeds/
├── src/
│   ├── data_loader.py      # Downloads and preprocesses Text8 corpus
│   ├── tokenizer.py         # BPE tokenization using Hugging Face
│   ├── trainer.py           # Word2Vec Skip-gram training
│   ├── evaluator.py         # Similarity, analogies, K-NN
│   ├── visualizer.py        # t-SNE and UMAP visualizations
│   └── main.py              # Complete end-to-end pipeline
├── data/
│   ├── text8                # Raw corpus
│   └── text8_processed.txt  # Preprocessed corpus
├── models/
│   ├── bpe_tokenizer.json   # Trained BPE tokenizer
│   ├── vocabulary.json      # Token vocabulary
│   ├── word2vec_skipgram.model  # Trained Word2Vec model
│   └── training_info.json   # Training metadata
├── output/
│   ├── tsne_embeddings.png      # t-SNE visualization
│   ├── umap_embeddings.png      # UMAP visualization
│   ├── semantic_clusters.png    # Clustered word groups
│   └── results_summary.json     # Evaluation results
├── env/                     # Python virtual environment
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd c:\Apps\new_embeds
```

2. **Activate the virtual environment:**
```bash
.\env\Scripts\activate
```

3. **Install dependencies** (already done):
```bash
pip install -r requirements.txt
```

### Running the Complete Pipeline

**Run the main demo script:**
```bash
python src/main.py
```

This will:
1. Download the Text8 corpus (~31MB download)
2. Preprocess and chunk the corpus
3. Train BPE tokenizer (or load existing)
4. Tokenize the entire corpus
5. Train Word2Vec model (15-45 minutes on CPU)
6. Run similarity tests
7. Solve analogies
8. Find nearest neighbors
9. Create visualizations
10. Save all results

### Running Individual Modules

**Test data loading:**
```bash
python src/data_loader.py
```

**Test tokenization:**
```bash
python src/tokenizer.py
```

**Test training:**
```bash
python src/trainer.py
```

**Test evaluation:**
```bash
python src/evaluator.py
```

**Test visualization:**
```bash
python src/visualizer.py
```

## 📊 Features & Demonstrations

### 1. Semantic Similarity

Calculate cosine similarity between word pairs:

```python
from evaluator import EmbeddingEvaluator
from gensim.models import Word2Vec

model = Word2Vec.load("models/word2vec_skipgram.model")
evaluator = EmbeddingEvaluator(model)

# High similarity (related concepts)
print(evaluator.cosine_similarity("king", "queen"))      # ~0.75-0.85
print(evaluator.cosine_similarity("man", "woman"))       # ~0.70-0.80

# Lower similarity (unrelated)
print(evaluator.cosine_similarity("king", "computer"))   # ~0.20-0.40
```

**Expected Results:**
- High similarity: Related concepts (0.7-0.9)
- Medium similarity: Somewhat related (0.4-0.7)
- Low similarity: Unrelated (0.0-0.4)

### 2. Analogy Solving

Solve analogies using vector arithmetic: `a:b :: c:?` → `b - a + c`

```python
# man:king :: woman:?
results = evaluator.solve_analogy("man", "king", "woman", topn=5)
# Expected: queen (or similar royal female terms)

# walk:walked :: swim:?
results = evaluator.solve_analogy("walk", "walked", "swim", topn=5)
# Expected: swam, swimming
```

**Example Analogies Tested:**
- **Gender-Royalty**: man:king :: woman:queen ✓
- **Past Tense**: walk:walked :: talk:talked ✓
- **Comparatives**: good:better :: bad:worse ✓
- **Geography**: France:Paris :: England:London ✓

### 3. K-Nearest Neighbors

Find most similar words:

```python
similar_words = evaluator.most_similar("king", topn=10)
# Returns: queen, prince, royal, throne, monarch, etc.
```

### 4. Visualizations

**t-SNE Projection:**
- Reduces 128D embeddings to 2D
- Preserves local structure
- Shows semantic clusters

**UMAP Projection:**
- Alternative dimensionality reduction
- Better preserves global structure
- Faster than t-SNE

**Semantic Clusters:**
- Groups related words by category
- Color-coded visualization
- Shows relationships between concepts

## 🔬 Technical Details

### Data Preprocessing

1. **Download**: Text8 corpus from Matt Mahoney's website
2. **Chunking**: Split into 1000-word chunks for structure
3. **Cleaning**: Text8 is already lowercase and cleaned
4. **Statistics**: ~17M words, ~253K unique tokens

### BPE Tokenization

```python
tokenizer = BPETokenizer(vocab_size=12000)
tokenizer.train(corpus_file)

# Results:
# - Vocabulary: ~12,000 subword tokens
# - Special tokens: <UNK>, <PAD>, <BOS>, <EOS>
# - Min frequency: 5 occurrences
```

### Word2Vec Training

```python
trainer = EmbeddingTrainer(
    vector_size=128,      # Embedding dimensions
    window=5,             # Context window size
    min_count=5,          # Minimum word frequency
    negative=10,          # Negative samples
    epochs=10,            # Training iterations
    sg=1,                 # Skip-gram (not CBOW)
    hs=0                  # Use negative sampling
)
```

**Training Process:**
1. Build vocabulary from tokenized corpus
2. Initialize random embeddings
3. For each word, predict context words
4. Update embeddings using negative sampling
5. Repeat for multiple epochs

### Evaluation Metrics

**Cosine Similarity:**
$$
\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

**Vector Arithmetic for Analogies:**
$$
\text{target} = \vec{b} - \vec{a} + \vec{c}
$$

Then find nearest neighbor to target vector.

## 📈 Performance Metrics

### Training Performance
- **Dataset Size**: ~17M words, ~100MB
- **Vocabulary Size**: ~12K tokens (after BPE)
- **Training Time**: 15-45 minutes (CPU-dependent)
- **Model Size**: ~6-8 MB

### Evaluation Results

**Similarity Tests** (Expected):
- Related pairs: 0.65-0.85 similarity
- Unrelated pairs: 0.15-0.40 similarity

**Analogy Accuracy**:
- Top-1 accuracy: 40-60% (typical for smaller corpus)
- Top-5 accuracy: 60-80%

**Coverage**:
- Vocabulary coverage: ~95% of corpus
- OOV handling: Via BPE subword tokens

## 🎓 Key Concepts Demonstrated

### 1. Distributional Semantics
*"You shall know a word by the company it keeps"* - Words appearing in similar contexts have similar meanings.

### 2. Skip-gram Architecture
- **Input**: Center word
- **Output**: Context words (within window)
- **Learning**: Adjusts embeddings to predict context

### 3. Negative Sampling
- Instead of full softmax over vocabulary
- Sample K negative examples per positive
- Faster training, similar results

### 4. Vector Space Properties
- **Similarity**: Cosine distance captures semantic similarity
- **Arithmetic**: Vector operations encode relationships
- **Clusters**: Related words cluster in embedding space

## 🛠️ Customization

### Adjust Vocabulary Size
```python
tokenizer = BPETokenizer(vocab_size=15000)  # Increase to 15K
```

### Change Embedding Dimensions
```python
trainer = EmbeddingTrainer(vector_size=150)  # Increase to 150D
```

### Modify Training Parameters
```python
trainer = EmbeddingTrainer(
    epochs=15,        # More epochs
    window=7,         # Larger context window
    negative=15       # More negative samples
)
```

### Use Different Corpus
```python
loader = DataLoader()
# Point to your own corpus file
corpus_file = "path/to/your/corpus.txt"
```

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution**: Run the full pipeline first: `python src/main.py`

### Issue: Training takes too long
**Solution**: 
- Reduce epochs: `epochs=5`
- Reduce vocabulary: `vocab_size=8000`
- Use smaller corpus subset

### Issue: Out of memory
**Solution**:
- Reduce vector size: `vector_size=100`
- Process corpus in batches
- Close other applications

### Issue: Poor analogy results
**Solution**:
- Increase training data
- Increase epochs: `epochs=15-20`
- Adjust window size: `window=7-10`
- Use larger embeddings: `vector_size=150-200`

## 📚 References

### Algorithms
- **Word2Vec**: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- **BPE**: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016)
- **t-SNE**: van der Maaten & Hinton, "Visualizing Data using t-SNE" (2008)
- **UMAP**: McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" (2018)

### Libraries Used
- **Gensim**: Word2Vec implementation
- **Hugging Face Tokenizers**: BPE tokenization
- **scikit-learn**: t-SNE and evaluation metrics
- **UMAP**: Dimensionality reduction
- **Matplotlib**: Visualizations
- **NumPy**: Numerical operations

### Datasets
- **Text8**: Cleaned Wikipedia corpus from Matt Mahoney
- URL: http://mattmahoney.net/dc/text8.zip

## 🎯 Success Criteria

✅ **Data Loading**: Text8 corpus downloaded and preprocessed  
✅ **Tokenization**: BPE tokenizer trained with 12K vocabulary  
✅ **Training**: Word2Vec model trained with Skip-gram  
✅ **Similarity**: Cosine similarity calculations working  
✅ **Analogies**: Vector arithmetic solving analogies  
✅ **Nearest Neighbors**: K-NN search functional  
✅ **Visualizations**: t-SNE and UMAP plots generated  
✅ **Performance**: Training completes in <2 hours on CPU  
✅ **Documentation**: Comprehensive README and comments  

## 📝 License

This project is for educational purposes. The Text8 corpus and all libraries used are open-source.

## 👏 Acknowledgments

- Matt Mahoney for the Text8 corpus
- Gensim team for excellent Word2Vec implementation
- Hugging Face for tokenizers library
- scikit-learn and UMAP developers

---

**Happy Embedding! 🚀**

For questions or issues, please refer to the module documentation in each source file.
