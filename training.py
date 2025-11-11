"""Training script for skip-gram word embeddings."""

import numpy as np
from collections import Counter
import time
import os

from bpe_tokenizer import BPETokenizer
from skipgram_model import SkipGram

def load_corpus(corpus_path, max_chars=10000000):
    """
    Load and return the text corpus.
    
    Args:
        corpus_path: Path to corpus file
        max_chars: Maximum number of characters to load (default: 10M, ~10% of Text8)
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read(max_chars)
    return text

def generate_training_pairs(token_ids, window_size=5):
    """
    Generate (center, context) pairs for training.
    
    Args:
        token_ids: List of token IDs
        window_size: Maximum distance between center and context words
    
    Returns:
        Lists of center and context IDs
    """
    center_ids = []
    context_ids = []
    
    for i, center in enumerate(token_ids):
        # Random window size for each center word
        window = np.random.randint(1, window_size + 1)
        
        # Get context words within window
        start = max(0, i - window)
        end = min(len(token_ids), i + window + 1)
        
        for j in range(start, end):
            if j != i:  # Don't include the center word itself
                center_ids.append(center)
                context_ids.append(token_ids[j])
    
    return center_ids, context_ids

def train_embeddings(corpus_path, tokenizer_path, embedding_path, 
                     vocab_size=5000, embedding_dim=100, epochs=3,
                     window_size=5, negative_samples=5, batch_size=2048,
                     initial_lr=0.025, max_corpus_chars=10000000):
    """
    Train word embeddings using skip-gram with negative sampling.
    
    Args:
        corpus_path: Path to text corpus
        tokenizer_path: Path to save/load tokenizer
        embedding_path: Path to save embeddings
        vocab_size: Size of BPE vocabulary
        embedding_dim: Dimensionality of embeddings
        epochs: Number of training epochs
        window_size: Context window size
        negative_samples: Number of negative samples
        batch_size: Training batch size
        initial_lr: Initial learning rate
        max_corpus_chars: Maximum characters to load from corpus (default: 10M)
    """
    
    # Load corpus
    print("Loading corpus...")
    print(f"  Using subset: first {max_corpus_chars:,} characters")
    text = load_corpus(corpus_path, max_chars=max_corpus_chars)
    print(f"Corpus size: {len(text)} characters, ~{len(text.split())} words")
    
    # Train or load tokenizer
    if os.path.exists(tokenizer_path):
        print(f"\nLoading tokenizer from {tokenizer_path}...")
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.load(tokenizer_path)
    else:
        print(f"\nTraining BPE tokenizer...")
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        # Use fewer merges to keep vocabulary manageable
        num_merges = min(1000, vocab_size // 2)
        tokenizer.train_bpe(text, num_merges=num_merges)
        tokenizer.save(tokenizer_path)
    
    # Tokenize corpus
    print("\nTokenizing corpus...")
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(tokens)
    print(f"Number of tokens: {len(token_ids)}")
    
    # Count token frequencies for negative sampling
    token_counts = Counter(token_ids)
    
    # Initialize model
    print(f"\nInitializing Skip-gram model...")
    print(f"  Vocabulary size: {len(tokenizer.vocab)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Window size: {window_size}")
    print(f"  Negative samples: {negative_samples}")
    
    model = SkipGram(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=embedding_dim,
        learning_rate=initial_lr,
        window_size=window_size,
        negative_samples=negative_samples
    )
    
    # Prepare negative sampling distribution
    model.prepare_negative_sampling(token_counts)
    
    # Generate training pairs
    print("\nGenerating training pairs...")
    center_ids, context_ids = generate_training_pairs(token_ids, window_size)
    print(f"Number of training pairs: {len(center_ids)}")
    
    # Convert to numpy arrays
    center_ids = np.array(center_ids, dtype=np.int32)
    context_ids = np.array(context_ids, dtype=np.int32)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    total_pairs = len(center_ids)
    num_batches = (total_pairs + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Shuffle training data
        indices = np.random.permutation(total_pairs)
        center_ids_shuffled = center_ids[indices]
        context_ids_shuffled = context_ids[indices]
        
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_pairs)
            
            # Get batch
            batch_center = center_ids_shuffled[start_idx:end_idx]
            batch_context = context_ids_shuffled[start_idx:end_idx]
            
            # Sample negatives
            batch_negatives = model.sample_negatives(len(batch_center))
            
            # Train on batch
            loss = model.train_batch(batch_center, batch_context, batch_negatives)
            epoch_loss += loss
            
            # Update learning rate
            progress = (epoch * num_batches + batch_idx) / (epochs * num_batches)
            model.update_learning_rate(progress)
            
            # Print progress
            if (batch_idx + 1) % 1000 == 0 or batch_idx == num_batches - 1:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}/{num_batches} | "
                      f"Loss: {avg_loss:.4f} | LR: {model.learning_rate:.6f}")
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / num_batches
        
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        
        # Save embeddings after each epoch
        epoch_embedding_path = embedding_path.replace('.npy', f'_epoch{epoch + 1}.npy')
        model.save_embeddings(epoch_embedding_path)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    
    # Save final embeddings
    model.save_embeddings(embedding_path)
    
    return model, tokenizer


if __name__ == '__main__':
    # Configuration
    corpus_path = 'data/text8'
    tokenizer_path = 'models/tokenizer.pkl'
    embedding_path = 'models/embeddings.npy'
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train (using 10M chars = ~10% of corpus for faster training)
    model, tokenizer = train_embeddings(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        embedding_path=embedding_path,
        vocab_size=5000,
        embedding_dim=100,
        epochs=3,
        window_size=5,
        negative_samples=5,
        batch_size=2048,
        initial_lr=0.025,
        max_corpus_chars=10000000  # 10M chars (~10 minutes total training)
    )
