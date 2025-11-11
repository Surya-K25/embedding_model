# Configuration file for Word Embedding Model

# Data Configuration
DATA_DIR = "data"
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
CORPUS_CHUNK_SIZE = 1000  # Words per line in preprocessed corpus

# Tokenization Configuration
VOCAB_SIZE = 12000  # BPE vocabulary size (10K-15K range)
MIN_TOKEN_FREQUENCY = 5  # Minimum frequency for tokens

# Model Configuration
VECTOR_SIZE = 128  # Embedding dimensions (100-150 range)
WINDOW_SIZE = 5  # Context window size
MIN_WORD_COUNT = 5  # Minimum word frequency
NEGATIVE_SAMPLES = 10  # Negative sampling (5-10 range)
TRAINING_EPOCHS = 10  # Training epochs (5-10 range)
USE_SKIPGRAM = 1  # 1 for Skip-gram, 0 for CBOW
USE_HIERARCHICAL_SOFTMAX = 0  # 0 for negative sampling, 1 for hierarchical softmax

# Output Configuration
MODEL_DIR = "models"
OUTPUT_DIR = "output"

# Visualization Configuration
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
VISUALIZATION_WORDS = 100  # Number of words to visualize

# Random Seeds
RANDOM_SEED = 42
