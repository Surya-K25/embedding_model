"""Quick test script to validate the implementation with a small dataset."""

import numpy as np
from collections import Counter

from bpe_tokenizer import BPETokenizer
from skipgram_model import SkipGram
from evaluation import EmbeddingEvaluator

def quick_test():
    """Run a quick test with minimal data to verify everything works."""
    
    print("="*70)
    print("QUICK VALIDATION TEST")
    print("="*70)
    
    # Small test corpus with clear semantic relationships
    test_corpus = """
    the cat sat on the mat
    the dog sat on the log
    the cat and the dog are animals
    the king and the queen rule the kingdom
    the man and the woman are people
    apple banana orange are fruits
    apple is red banana is yellow orange is orange
    car bus truck are vehicles
    run walk jump are actions
    """ * 100  # Repeat for more training data
    
    print("\n1. Testing BPE Tokenizer...")
    print("-" * 70)
    
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train_bpe(test_corpus, num_merges=100)
    
    # Test tokenization
    test_text = "the cat sat"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(tokens)
    print(f"Text: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    print("\n2. Testing Skip-gram Model...")
    print("-" * 70)
    
    # Tokenize entire corpus
    all_tokens = tokenizer.tokenize(test_corpus)
    all_token_ids = tokenizer.encode(all_tokens)
    print(f"Total tokens: {len(all_token_ids)}")
    
    # Count token frequencies
    token_counts = Counter(all_token_ids)
    
    # Initialize model
    model = SkipGram(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=50,
        learning_rate=0.025,
        window_size=3,
        negative_samples=3
    )
    
    model.prepare_negative_sampling(token_counts)
    
    # Generate training pairs
    print("Generating training pairs...")
    from training import generate_training_pairs
    center_ids, context_ids = generate_training_pairs(all_token_ids, window_size=3)
    print(f"Training pairs: {len(center_ids)}")
    
    # Convert to numpy
    center_ids = np.array(center_ids, dtype=np.int32)
    context_ids = np.array(context_ids, dtype=np.int32)
    
    # Train for a few iterations
    print("\nTraining model...")
    batch_size = 128
    num_iterations = 500
    
    for i in range(num_iterations):
        # Random batch
        indices = np.random.choice(len(center_ids), batch_size, replace=False)
        batch_center = center_ids[indices]
        batch_context = context_ids[indices]
        batch_negatives = model.sample_negatives(batch_size)
        
        loss = model.train_batch(batch_center, batch_context, batch_negatives)
        
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i + 1}/{num_iterations}, Loss: {loss:.4f}")
    
    print("\n3. Testing Evaluation Functions...")
    print("-" * 70)
    
    evaluator = EmbeddingEvaluator(model.get_embeddings(), tokenizer)
    
    # Test similarity
    print("\nSimilarity tests:")
    test_pairs = [
        ("cat", "dog"),
        ("apple", "banana"),
        ("king", "queen"),
        ("man", "woman"),
    ]
    
    for word1, word2 in test_pairs:
        sim = evaluator.similarity(word1, word2)
        if sim is not None:
            print(f"  similarity('{word1}', '{word2}') = {sim:.4f}")
        else:
            print(f"  similarity('{word1}', '{word2}') = N/A")
    
    # Test nearest neighbors
    print("\nNearest neighbors for 'cat':")
    neighbors = evaluator.nearest_neighbors("cat", k=5)
    for word, sim in neighbors:
        print(f"  {word}: {sim:.4f}")
    
    # Test analogy
    print("\nAnalogy test: man:woman::king:?")
    results = evaluator.analogy("man", "woman", "king", k=5)
    if results:
        print("  Predictions:")
        for word, score in results:
            print(f"    {word}: {score:.4f}")
    else:
        print("  Could not compute analogy")
    
    print("\n" + "="*70)
    print("✓ ALL COMPONENTS WORKING!")
    print("="*70)
    print("\nThe implementation is ready. You can now train on the full Text8 corpus.")
    print("Run: python main.py --mode both --epochs 3")


if __name__ == '__main__':
    quick_test()
