"""
Training Module
Implements Word2Vec Skip-gram model training using Gensim.
"""

from gensim.models import Word2Vec
from pathlib import Path
import time
import json
import multiprocessing


class EmbeddingTrainer:
    """Trains Word2Vec embeddings using Skip-gram architecture."""
    
    def __init__(
        self,
        vector_size=128,
        window=5,
        min_count=5,
        negative=10,
        epochs=10,
        workers=None,
        sg=1,  # Skip-gram
        hs=0   # Use negative sampling (not hierarchical softmax)
    ):
        """
        Initialize embedding trainer with Word2Vec parameters.
        
        Args:
            vector_size: Dimensionality of embeddings (100-150)
            window: Context window size
            min_count: Minimum word frequency
            negative: Number of negative samples (5-10)
            epochs: Number of training epochs (5-10)
            workers: Number of worker threads (None = use all cores)
            sg: 1 for Skip-gram, 0 for CBOW
            hs: 1 for hierarchical softmax, 0 for negative sampling
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.workers = workers or multiprocessing.cpu_count()
        self.sg = sg
        self.hs = hs
        self.model = None
        self.training_time = 0
    
    def train(self, tokenized_sentences, output_dir="models"):
        """
        Train Word2Vec model on tokenized sentences.
        
        Args:
            tokenized_sentences: List of tokenized sentences (list of token lists)
            output_dir: Directory to save trained model
        
        Returns:
            Trained Word2Vec model
        """
        print("=" * 60)
        print("Training Word2Vec Skip-gram Model")
        print("=" * 60)
        print(f"Vector size: {self.vector_size}")
        print(f"Window size: {self.window}")
        print(f"Minimum count: {self.min_count}")
        print(f"Negative samples: {self.negative}")
        print(f"Epochs: {self.epochs}")
        print(f"Workers: {self.workers}")
        print(f"Training sentences: {len(tokenized_sentences)}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            hs=self.hs,
            negative=self.negative,
            epochs=self.epochs,
            seed=42,
            compute_loss=True
        )
        
        self.training_time = time.time() - start_time
        
        print("=" * 60)
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Vocabulary size: {len(self.model.wv)}")
        print("=" * 60)
        
        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / "word2vec_skipgram.model"
        self.model.save(str(model_path))
        print(f"Model saved to {model_path}")
        
        # Save word vectors separately (for faster loading)
        wv_path = output_path / "word2vec_vectors.kv"
        self.model.wv.save(str(wv_path))
        print(f"Word vectors saved to {wv_path}")
        
        # Save training info
        info = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'negative': self.negative,
            'epochs': self.epochs,
            'sg': self.sg,
            'hs': self.hs,
            'vocab_size': len(self.model.wv),
            'training_time_seconds': self.training_time,
            'training_sentences': len(tokenized_sentences)
        }
        
        info_path = output_path / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Training info saved to {info_path}")
        
        return self.model
    
    def load_model(self, model_path):
        """
        Load a trained Word2Vec model.
        
        Args:
            model_path: Path to saved model file
        
        Returns:
            Loaded Word2Vec model
        """
        print(f"Loading model from {model_path}...")
        self.model = Word2Vec.load(str(model_path))
        print(f"Model loaded. Vocabulary size: {len(self.model.wv)}")
        return self.model
    
    def get_word_vector(self, word):
        """
        Get vector for a word.
        
        Args:
            word: Word string
        
        Returns:
            Numpy array of word vector, or None if not in vocabulary
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        try:
            return self.model.wv[word]
        except KeyError:
            return None
    
    def get_vocabulary(self):
        """Get list of all words in vocabulary."""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        return list(self.model.wv.index_to_key)
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        return len(self.model.wv)
    
    def continue_training(self, new_sentences, epochs=5):
        """
        Continue training on additional data.
        
        Args:
            new_sentences: Additional tokenized sentences
            epochs: Number of additional epochs
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        print(f"Continuing training for {epochs} epochs on {len(new_sentences)} sentences...")
        self.model.build_vocab(new_sentences, update=True)
        self.model.train(
            new_sentences,
            total_examples=len(new_sentences),
            epochs=epochs
        )
        print("Continued training completed.")


def main():
    """Demo function to test training."""
    from data_loader import DataLoader
    from tokenizer import BPETokenizer
    
    # Load and prepare data
    print("Loading data...")
    loader = DataLoader()
    loader.download_text8()
    corpus_file = loader.preprocess_corpus()
    
    # Tokenize
    print("\nTokenizing corpus...")
    tokenizer = BPETokenizer(vocab_size=12000)
    
    # Try to load existing tokenizer, or train new one
    tokenizer_path = Path("models/bpe_tokenizer.json")
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
    else:
        tokenizer.train(corpus_file)
    
    tokenized_sentences = tokenizer.tokenize_corpus(corpus_file)
    
    # Train embeddings
    print("\nTraining Word2Vec model...")
    trainer = EmbeddingTrainer(
        vector_size=128,
        window=5,
        min_count=5,
        negative=10,
        epochs=10
    )
    
    model = trainer.train(tokenized_sentences)
    
    # Test the model
    print("\n" + "=" * 60)
    print("Model Testing")
    print("=" * 60)
    
    # Test vocabulary
    vocab = trainer.get_vocabulary()
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample words: {vocab[:20]}")
    
    # Test word vector
    test_word = "king" if "king" in vocab else vocab[100]
    vector = trainer.get_word_vector(test_word)
    print(f"\nVector for '{test_word}':")
    print(f"  Shape: {vector.shape}")
    print(f"  First 10 dimensions: {vector[:10]}")
    
    # Test similarity
    if "king" in vocab and "queen" in vocab:
        similarity = model.wv.similarity("king", "queen")
        print(f"\nSimilarity between 'king' and 'queen': {similarity:.4f}")


if __name__ == "__main__":
    main()
