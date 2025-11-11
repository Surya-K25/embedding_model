"""Evaluation functions for word embeddings."""

import numpy as np
from bpe_tokenizer import BPETokenizer

class EmbeddingEvaluator:
    def __init__(self, embeddings, tokenizer):
        """
        Initialize evaluator.
        
        Args:
            embeddings: Numpy array of word embeddings (vocab_size, embedding_dim)
            tokenizer: BPETokenizer instance
        """
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        
        # Normalize embeddings for efficient cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = embeddings / (norms + 1e-10)
    
    def get_word_embedding(self, word):
        """
        Get embedding for a word.
        
        Args:
            word: Input word
        
        Returns:
            Embedding vector or None if word not in vocabulary
        """
        # Tokenize the word
        tokens = self.tokenizer.tokenize_word(word)
        
        if not tokens:
            return None
        
        # Average embeddings of all subword tokens
        embeddings = []
        for token in tokens:
            token_id = self.tokenizer.vocab.get(token)
            if token_id is not None and token_id < len(self.embeddings):
                embeddings.append(self.embeddings[token_id])
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def get_word_embedding_normalized(self, word):
        """Get normalized embedding for a word."""
        embedding = self.get_word_embedding(word)
        if embedding is None:
            return None
        norm = np.linalg.norm(embedding)
        return embedding / (norm + 1e-10)
    
    def similarity(self, word1, word2):
        """
        Compute cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
        
        Returns:
            Cosine similarity score (-1 to 1) or None if word not found
        """
        emb1 = self.get_word_embedding_normalized(word1)
        emb2 = self.get_word_embedding_normalized(word2)
        
        if emb1 is None or emb2 is None:
            return None
        
        return np.dot(emb1, emb2)
    
    def nearest_neighbors(self, word, k=10):
        """
        Find k nearest neighbors of a word.
        
        Args:
            word: Input word
            k: Number of neighbors to return
        
        Returns:
            List of (word, similarity) tuples
        """
        word_emb = self.get_word_embedding_normalized(word)
        
        if word_emb is None:
            return []
        
        # Compute similarities with all words
        word_similarities = []
        
        for token, token_id in self.tokenizer.vocab.items():
            if token in ['<PAD>', '<UNK>', '</w>'] or token_id >= len(self.embeddings):
                continue
            
            # Get normalized embedding
            emb = self.normalized_embeddings[token_id]
            similarity = np.dot(word_emb, emb)
            
            # Reconstruct word from token (remove </w> marker)
            word_repr = token.replace('</w>', '')
            
            word_similarities.append((word_repr, similarity, token))
        
        # Sort by similarity
        word_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out the query word itself and return top k
        results = []
        for word_repr, sim, token in word_similarities:
            if len(results) >= k:
                break
            # Skip if it's the same word
            if word_repr != word:
                results.append((word_repr, sim))
        
        return results
    
    def analogy(self, word_a, word_b, word_c, k=5):
        """
        Solve word analogy: a:b :: c:?
        
        Args:
            word_a: First word of analogy (e.g., "man")
            word_b: Second word of analogy (e.g., "king")
            word_c: Third word of analogy (e.g., "woman")
            k: Number of candidates to return
        
        Returns:
            List of (word, score) tuples
        """
        emb_a = self.get_word_embedding_normalized(word_a)
        emb_b = self.get_word_embedding_normalized(word_b)
        emb_c = self.get_word_embedding_normalized(word_c)
        
        if emb_a is None or emb_b is None or emb_c is None:
            return []
        
        # Compute target vector: b - a + c
        target = emb_b - emb_a + emb_c
        target = target / (np.linalg.norm(target) + 1e-10)
        
        # Find nearest neighbors to target
        word_similarities = []
        
        for token, token_id in self.tokenizer.vocab.items():
            if token in ['<PAD>', '<UNK>', '</w>'] or token_id >= len(self.embeddings):
                continue
            
            emb = self.normalized_embeddings[token_id]
            similarity = np.dot(target, emb)
            
            word_repr = token.replace('</w>', '')
            word_similarities.append((word_repr, similarity))
        
        # Sort by similarity
        word_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out input words and return top k
        results = []
        for word_repr, sim in word_similarities:
            if len(results) >= k:
                break
            if word_repr not in [word_a, word_b, word_c]:
                results.append((word_repr, sim))
        
        return results
    
    def evaluate_similarity_pairs(self, word_pairs):
        """
        Evaluate similarity for multiple word pairs.
        
        Args:
            word_pairs: List of (word1, word2) tuples
        
        Returns:
            List of (word1, word2, similarity) tuples
        """
        results = []
        for word1, word2 in word_pairs:
            sim = self.similarity(word1, word2)
            results.append((word1, word2, sim))
        return results
    
    def evaluate_analogies(self, analogy_questions):
        """
        Evaluate multiple analogy questions.
        
        Args:
            analogy_questions: List of (a, b, c, expected_d) tuples
        
        Returns:
            List of results with predictions
        """
        results = []
        
        for a, b, c, expected_d in analogy_questions:
            predictions = self.analogy(a, b, c, k=5)
            
            # Check if expected answer is in top predictions
            predicted_words = [word for word, _ in predictions]
            rank = predicted_words.index(expected_d) + 1 if expected_d in predicted_words else None
            
            results.append({
                'question': f"{a}:{b}::{c}:?",
                'expected': expected_d,
                'predictions': predictions,
                'rank': rank,
                'correct': rank == 1 if rank else False
            })
        
        return results


def print_evaluation_results(evaluator, test_words, analogy_questions):
    """Print comprehensive evaluation results."""
    
    print("\n" + "="*70)
    print("WORD EMBEDDING EVALUATION")
    print("="*70)
    
    # 1. Similarity tests
    print("\n1. SIMILARITY TESTS")
    print("-" * 70)
    
    similarity_pairs = [
        ("apple", "banana"),
        ("apple", "potato"),
        ("apple", "orange"),
        ("king", "queen"),
        ("man", "woman"),
        ("run", "walk"),
        ("good", "bad"),
    ]
    
    for word1, word2 in similarity_pairs:
        sim = evaluator.similarity(word1, word2)
        if sim is not None:
            print(f"  similarity('{word1}', '{word2}') = {sim:.4f}")
        else:
            print(f"  similarity('{word1}', '{word2}') = N/A (word not found)")
    
    # 2. Nearest neighbors
    print("\n2. NEAREST NEIGHBORS")
    print("-" * 70)
    
    for word in test_words:
        print(f"\n  Top 10 words similar to '{word}':")
        neighbors = evaluator.nearest_neighbors(word, k=10)
        if neighbors:
            for i, (neighbor, sim) in enumerate(neighbors, 1):
                print(f"    {i:2d}. {neighbor:15s} (similarity: {sim:.4f})")
        else:
            print(f"    Word '{word}' not found in vocabulary")
    
    # 3. Analogy tests
    print("\n3. ANALOGY TESTS")
    print("-" * 70)
    
    results = evaluator.evaluate_analogies(analogy_questions)
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    for result in results:
        print(f"\n  {result['question']}")
        print(f"    Expected: {result['expected']}")
        print(f"    Predictions:")
        for i, (word, score) in enumerate(result['predictions'][:5], 1):
            marker = "✓" if word == result['expected'] else " "
            print(f"      {marker} {i}. {word:15s} (score: {score:.4f})")
        if result['rank']:
            print(f"    Rank: {result['rank']}")
        else:
            print(f"    Expected answer not in top 5")
    
    print(f"\n  Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    
    # 4. Validation checks
    print("\n4. VALIDATION CHECKS")
    print("-" * 70)
    
    apple_banana = evaluator.similarity("apple", "banana")
    apple_potato = evaluator.similarity("apple", "potato")
    
    if apple_banana and apple_potato:
        check1 = apple_banana > apple_potato
        print(f"  ✓ similarity('apple', 'banana') [{apple_banana:.4f}] > "
              f"similarity('apple', 'potato') [{apple_potato:.4f}]: {check1}")
    
    # Check analogy
    man_king_woman = evaluator.analogy("man", "king", "woman", k=3)
    if man_king_woman:
        predicted_words = [word for word, _ in man_king_woman]
        check2 = "queen" in predicted_words
        rank = predicted_words.index("queen") + 1 if check2 else None
        print(f"  {'✓' if check2 else '✗'} analogy('man', 'king', 'woman') contains 'queen': {check2}")
        if rank:
            print(f"    'queen' is at rank {rank}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    import os
    
    # Load tokenizer and embeddings
    tokenizer_path = 'models/tokenizer.pkl'
    embedding_path = 'models/embeddings.npy'
    
    if not os.path.exists(tokenizer_path) or not os.path.exists(embedding_path):
        print("Please run training.py first to train the model.")
        exit(1)
    
    print("Loading tokenizer and embeddings...")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    embeddings = np.load(embedding_path)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create evaluator
    evaluator = EmbeddingEvaluator(embeddings, tokenizer)
    
    # Test words for nearest neighbors
    test_words = ["apple", "king", "run", "red"]
    
    # Analogy questions: (a, b, c, expected_d)
    analogy_questions = [
        ("man", "king", "woman", "queen"),
        ("man", "woman", "king", "queen"),
        ("big", "bigger", "small", "smaller"),
        ("good", "best", "bad", "worst"),
    ]
    
    # Print results
    print_evaluation_results(evaluator, test_words, analogy_questions)
