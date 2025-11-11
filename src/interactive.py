"""
Quick Start Script
Provides an interactive interface to use the trained embedding model.
"""

import sys
from pathlib import Path
from gensim.models import Word2Vec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluator import EmbeddingEvaluator
from visualizer import EmbeddingVisualizer


def load_model():
    """Load the trained model."""
    model_path = Path("models/word2vec_skipgram.model")
    
    if not model_path.exists():
        print("❌ Model not found!")
        print("Please run 'python src/main.py' first to train the model.")
        sys.exit(1)
    
    print("Loading model...")
    model = Word2Vec.load(str(model_path))
    print(f"✓ Model loaded. Vocabulary: {len(model.wv)} words\n")
    
    return model


def print_menu():
    """Display the main menu."""
    print("\n" + "=" * 60)
    print("  WORD EMBEDDING MODEL - INTERACTIVE INTERFACE")
    print("=" * 60)
    print("1. Calculate similarity between two words")
    print("2. Solve an analogy (a:b :: c:?)")
    print("3. Find similar words (K-nearest neighbors)")
    print("4. Explore word vector")
    print("5. Test word in vocabulary")
    print("6. Batch similarity tests")
    print("7. Create visualization")
    print("0. Exit")
    print("=" * 60)


def similarity_mode(evaluator):
    """Interactive similarity calculation."""
    print("\n--- SIMILARITY CALCULATOR ---")
    word1 = input("Enter first word: ").strip().lower()
    word2 = input("Enter second word: ").strip().lower()
    
    if not evaluator.word_in_vocab(word1):
        print(f"❌ '{word1}' not in vocabulary")
        return
    if not evaluator.word_in_vocab(word2):
        print(f"❌ '{word2}' not in vocabulary")
        return
    
    similarity = evaluator.cosine_similarity(word1, word2)
    distance = evaluator.distance(word1, word2)
    
    print(f"\nResults:")
    print(f"  {word1} <-> {word2}")
    print(f"  Cosine Similarity: {similarity:.4f}")
    print(f"  Distance: {distance:.4f}")
    
    # Interpretation
    if similarity > 0.7:
        print("  → Very similar / closely related")
    elif similarity > 0.5:
        print("  → Moderately similar / related")
    elif similarity > 0.3:
        print("  → Somewhat similar / loosely related")
    else:
        print("  → Not very similar / unrelated")


def analogy_mode(evaluator):
    """Interactive analogy solver."""
    print("\n--- ANALOGY SOLVER ---")
    print("Format: A is to B as C is to ?")
    print("Example: man is to king as woman is to ?")
    
    word_a = input("Enter word A: ").strip().lower()
    word_b = input("Enter word B: ").strip().lower()
    word_c = input("Enter word C: ").strip().lower()
    
    if not all(evaluator.word_in_vocab(w) for w in [word_a, word_b, word_c]):
        missing = [w for w in [word_a, word_b, word_c] if not evaluator.word_in_vocab(w)]
        print(f"❌ Words not in vocabulary: {', '.join(missing)}")
        return
    
    print(f"\nSolving: {word_a}:{word_b} :: {word_c}:?")
    results = evaluator.solve_analogy(word_a, word_b, word_c, topn=10)
    
    if results:
        print("\nTop predictions:")
        for i, (word, score) in enumerate(results, 1):
            print(f"  {i:2d}. {word:<20s} (similarity: {score:.4f})")
    else:
        print("❌ Could not solve analogy")


def nearest_neighbors_mode(evaluator):
    """Interactive K-NN search."""
    print("\n--- K-NEAREST NEIGHBORS ---")
    word = input("Enter word: ").strip().lower()
    
    if not evaluator.word_in_vocab(word):
        print(f"❌ '{word}' not in vocabulary")
        return
    
    try:
        k = int(input("How many similar words? (default: 10): ").strip() or "10")
    except ValueError:
        k = 10
    
    print(f"\nMost similar to '{word}':")
    similar = evaluator.most_similar(word, topn=k)
    
    for i, (similar_word, score) in enumerate(similar, 1):
        print(f"  {i:2d}. {similar_word:<20s} {score:.4f}")


def vector_mode(evaluator):
    """Explore word vector."""
    print("\n--- WORD VECTOR EXPLORER ---")
    word = input("Enter word: ").strip().lower()
    
    if not evaluator.word_in_vocab(word):
        print(f"❌ '{word}' not in vocabulary")
        return
    
    vector = evaluator.get_vector(word)
    
    print(f"\nVector for '{word}':")
    print(f"  Dimensions: {vector.shape[0]}")
    print(f"  First 10 values: {vector[:10]}")
    print(f"  Mean: {vector.mean():.4f}")
    print(f"  Std: {vector.std():.4f}")
    print(f"  Min: {vector.min():.4f}")
    print(f"  Max: {vector.max():.4f}")


def vocab_test_mode(evaluator):
    """Test if word is in vocabulary."""
    print("\n--- VOCABULARY TEST ---")
    word = input("Enter word: ").strip().lower()
    
    if evaluator.word_in_vocab(word):
        print(f"✓ '{word}' IS in the vocabulary")
        
        # Show some info
        similar = evaluator.most_similar(word, topn=5)
        print(f"\nTop 5 similar words:")
        for w, score in similar:
            print(f"  - {w} ({score:.4f})")
    else:
        print(f"❌ '{word}' is NOT in the vocabulary")
        
        # Suggest alternatives
        print("\nTrying to find similar words in vocabulary...")
        vocab = evaluator.vocab[:1000]  # Search first 1000 words
        suggestions = [v for v in vocab if word in v or v in word][:5]
        
        if suggestions:
            print("Did you mean one of these?")
            for s in suggestions:
                print(f"  - {s}")


def batch_similarity_mode(evaluator):
    """Batch similarity tests."""
    print("\n--- BATCH SIMILARITY TEST ---")
    print("Enter word pairs (one per line, separated by comma)")
    print("Example: king,queen")
    print("Enter empty line to finish.")
    
    pairs = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        
        parts = line.split(',')
        if len(parts) == 2:
            pairs.append((parts[0].strip().lower(), parts[1].strip().lower()))
        else:
            print("  Invalid format. Use: word1,word2")
    
    if not pairs:
        print("No pairs entered.")
        return
    
    print(f"\n{'Word 1':<15} {'Word 2':<15} {'Similarity':<12}")
    print("-" * 45)
    
    for word1, word2 in pairs:
        if evaluator.word_in_vocab(word1) and evaluator.word_in_vocab(word2):
            sim = evaluator.cosine_similarity(word1, word2)
            print(f"{word1:<15} {word2:<15} {sim:.4f}")
        else:
            missing = []
            if not evaluator.word_in_vocab(word1):
                missing.append(word1)
            if not evaluator.word_in_vocab(word2):
                missing.append(word2)
            print(f"{word1:<15} {word2:<15} N/A ({', '.join(missing)} not in vocab)")


def visualization_mode(model):
    """Create custom visualization."""
    print("\n--- VISUALIZATION CREATOR ---")
    print("1. Visualize specific words")
    print("2. Visualize top N frequent words")
    print("3. Visualize word groups (clusters)")
    
    choice = input("Choose option (1-3): ").strip()
    
    visualizer = EmbeddingVisualizer(model)
    
    if choice == "1":
        print("\nEnter words to visualize (comma-separated):")
        words_input = input("> ").strip()
        words = [w.strip().lower() for w in words_input.split(',')]
        
        method = input("Method (tsne/umap): ").strip().lower() or "tsne"
        
        if method == "tsne":
            visualizer.tsne_visualization(
                words=words,
                output_file="output/custom_tsne.png"
            )
        else:
            visualizer.umap_visualization(
                words=words,
                output_file="output/custom_umap.png"
            )
    
    elif choice == "2":
        try:
            n = int(input("Number of words to visualize: ").strip() or "100")
        except ValueError:
            n = 100
        
        method = input("Method (tsne/umap): ").strip().lower() or "tsne"
        
        if method == "tsne":
            visualizer.tsne_visualization(
                n_words=n,
                output_file="output/custom_tsne.png"
            )
        else:
            visualizer.umap_visualization(
                n_words=n,
                output_file="output/custom_umap.png"
            )
    
    elif choice == "3":
        print("\nEnter word groups (format: GroupName: word1,word2,word3)")
        print("Enter empty line to finish.")
        
        word_groups = {}
        while True:
            line = input("> ").strip()
            if not line:
                break
            
            if ':' in line:
                group_name, words_str = line.split(':', 1)
                words = [w.strip().lower() for w in words_str.split(',')]
                word_groups[group_name.strip()] = words
        
        if word_groups:
            method = input("Method (tsne/umap): ").strip().lower() or "tsne"
            visualizer.visualize_word_clusters(
                word_groups,
                method=method,
                output_file="output/custom_clusters.png"
            )
        else:
            print("No word groups entered.")


def main():
    """Main interactive loop."""
    model = load_model()
    evaluator = EmbeddingEvaluator(model)
    
    while True:
        print_menu()
        choice = input("\nEnter choice: ").strip()
        
        if choice == "1":
            similarity_mode(evaluator)
        elif choice == "2":
            analogy_mode(evaluator)
        elif choice == "3":
            nearest_neighbors_mode(evaluator)
        elif choice == "4":
            vector_mode(evaluator)
        elif choice == "5":
            vocab_test_mode(evaluator)
        elif choice == "6":
            batch_similarity_mode(evaluator)
        elif choice == "7":
            visualization_mode(model)
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
