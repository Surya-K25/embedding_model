"""
Evaluation Module
Implements similarity calculations, analogy solving, and K-nearest neighbors.
"""

import numpy as np
from typing import List, Tuple, Dict
from gensim.models import Word2Vec


class EmbeddingEvaluator:
    """Evaluates word embeddings for similarity, analogies, and nearest neighbors."""
    
    def __init__(self, model):
        """
        Initialize evaluator with trained Word2Vec model.
        
        Args:
            model: Trained Word2Vec model or KeyedVectors
        """
        # Handle both Word2Vec model and KeyedVectors
        if hasattr(model, 'wv'):
            self.wv = model.wv
        else:
            self.wv = model
        
        self.vocab = list(self.wv.index_to_key)
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
        
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            return self.wv.similarity(word1, word2)
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return 0.0
    
    def similarity_batch(self, word_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """
        Calculate similarity for multiple word pairs.
        
        Args:
            word_pairs: List of (word1, word2) tuples
        
        Returns:
            Dictionary mapping word pairs to similarity scores
        """
        results = {}
        for word1, word2 in word_pairs:
            sim = self.cosine_similarity(word1, word2)
            results[(word1, word2)] = sim
        return results
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find K most similar words to a given word.
        
        Args:
            word: Query word
            topn: Number of similar words to return
        
        Returns:
            List of (word, similarity_score) tuples
        """
        try:
            return self.wv.most_similar(word, topn=topn)
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return []
    
    def solve_analogy(
        self,
        word_a: str,
        word_b: str,
        word_c: str,
        topn: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy: a is to b as c is to ?
        Uses vector arithmetic: b - a + c
        
        Args:
            word_a: First word in analogy (e.g., "man")
            word_b: Second word in analogy (e.g., "king")
            word_c: Third word in analogy (e.g., "woman")
            topn: Number of results to return
        
        Returns:
            List of (word, similarity_score) tuples
        """
        try:
            # Use Gensim's built-in analogy solver
            # positive=[word_b, word_c] means add these vectors
            # negative=[word_a] means subtract this vector
            # Result: word_b - word_a + word_c
            results = self.wv.most_similar(
                positive=[word_b, word_c],
                negative=[word_a],
                topn=topn
            )
            return results
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return []
    
    def solve_analogy_manual(
        self,
        word_a: str,
        word_b: str,
        word_c: str,
        topn: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Manually solve analogy using vector arithmetic and cosine similarity.
        
        Args:
            word_a: First word in analogy
            word_b: Second word in analogy
            word_c: Third word in analogy
            topn: Number of results to return
        
        Returns:
            List of (word, similarity_score) tuples
        """
        try:
            # Get vectors
            vec_a = self.wv[word_a]
            vec_b = self.wv[word_b]
            vec_c = self.wv[word_c]
            
            # Compute target vector: b - a + c
            target_vec = vec_b - vec_a + vec_c
            
            # Normalize
            target_vec = target_vec / np.linalg.norm(target_vec)
            
            # Find most similar words to target vector
            results = self.wv.similar_by_vector(target_vec, topn=topn + 3)
            
            # Filter out input words
            filtered = [
                (word, score) for word, score in results
                if word not in [word_a, word_b, word_c]
            ]
            
            return filtered[:topn]
            
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return []
    
    def evaluate_similarity_tests(self, test_cases: List[Tuple[str, str, str]]) -> Dict:
        """
        Evaluate similarity on test cases with expected relationships.
        
        Args:
            test_cases: List of (word1, word2, relationship) tuples
        
        Returns:
            Dictionary with results
        """
        results = []
        
        for word1, word2, relationship in test_cases:
            sim = self.cosine_similarity(word1, word2)
            results.append({
                'word1': word1,
                'word2': word2,
                'relationship': relationship,
                'similarity': sim
            })
        
        return {
            'test_cases': results,
            'average_similarity': np.mean([r['similarity'] for r in results])
        }
    
    def evaluate_analogies(self, analogy_tests: List[Tuple[str, str, str, str]]) -> Dict:
        """
        Evaluate analogy solving on test cases.
        
        Args:
            analogy_tests: List of (word_a, word_b, word_c, expected_word_d) tuples
        
        Returns:
            Dictionary with results and accuracy
        """
        results = []
        correct = 0
        
        for word_a, word_b, word_c, expected_d in analogy_tests:
            predictions = self.solve_analogy(word_a, word_b, word_c, topn=5)
            
            # Check if expected word is in top predictions
            predicted_words = [word for word, _ in predictions]
            is_correct = expected_d in predicted_words[:1]  # Check top-1 accuracy
            
            if is_correct:
                correct += 1
            
            results.append({
                'analogy': f"{word_a}:{word_b} :: {word_c}:?",
                'expected': expected_d,
                'predictions': predictions,
                'correct': is_correct
            })
        
        accuracy = correct / len(analogy_tests) if analogy_tests else 0
        
        return {
            'test_cases': results,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(analogy_tests)
        }
    
    def distance(self, word1: str, word2: str) -> float:
        """
        Calculate distance between two words (1 - cosine_similarity).
        
        Args:
            word1: First word
            word2: Second word
        
        Returns:
            Distance score
        """
        return 1 - self.cosine_similarity(word1, word2)
    
    def doesnt_match(self, words: List[str]) -> str:
        """
        Find the word that doesn't match others in the list.
        
        Args:
            words: List of words
        
        Returns:
            The word that doesn't match
        """
        try:
            return self.wv.doesnt_match(words)
        except KeyError as e:
            print(f"Word not in vocabulary: {e}")
            return ""
    
    def word_in_vocab(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.wv
    
    def get_vector(self, word: str) -> np.ndarray:
        """Get vector for a word."""
        try:
            return self.wv[word]
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return None


def main():
    """Demo function to test evaluation."""
    from pathlib import Path
    from gensim.models import Word2Vec
    
    # Load trained model
    model_path = Path("models/word2vec_skipgram.model")
    
    if not model_path.exists():
        print("Model not found. Please train the model first.")
        return
    
    print("Loading model...")
    model = Word2Vec.load(str(model_path))
    
    # Initialize evaluator
    evaluator = EmbeddingEvaluator(model)
    
    print("\n" + "=" * 60)
    print("SIMILARITY TESTS")
    print("=" * 60)
    
    # Test similarity
    test_pairs = [
        ("king", "queen", "royalty"),
        ("man", "woman", "gender"),
        ("king", "man", "gender-royalty"),
        ("france", "paris", "country-capital"),
        ("good", "bad", "antonyms"),
    ]
    
    for word1, word2, relation in test_pairs:
        if evaluator.word_in_vocab(word1) and evaluator.word_in_vocab(word2):
            sim = evaluator.cosine_similarity(word1, word2)
            print(f"{word1:15s} <-> {word2:15s} ({relation:20s}): {sim:.4f}")
    
    print("\n" + "=" * 60)
    print("ANALOGY TESTS")
    print("=" * 60)
    
    # Test analogies
    analogy_tests = [
        ("man", "king", "woman", "queen"),
        ("man", "woman", "king", "queen"),
        ("walk", "walking", "swim", "swimming"),
        ("good", "better", "bad", "worse"),
    ]
    
    for word_a, word_b, word_c, expected in analogy_tests:
        if all(evaluator.word_in_vocab(w) for w in [word_a, word_b, word_c]):
            print(f"\n{word_a}:{word_b} :: {word_c}:? (expected: {expected})")
            results = evaluator.solve_analogy(word_a, word_b, word_c, topn=5)
            for word, score in results:
                marker = "✓" if word == expected else " "
                print(f"  {marker} {word:20s} {score:.4f}")
    
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS")
    print("=" * 60)
    
    # Test nearest neighbors
    test_words = ["king", "computer", "happy"]
    
    for word in test_words:
        if evaluator.word_in_vocab(word):
            print(f"\nMost similar to '{word}':")
            similar = evaluator.most_similar(word, topn=5)
            for similar_word, score in similar:
                print(f"  {similar_word:20s} {score:.4f}")


if __name__ == "__main__":
    main()
