"""
Main Demo Script
End-to-end pipeline demonstrating the custom word embedding model.
Includes data loading, tokenization, training, evaluation, and visualization.
"""

import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from tokenizer import BPETokenizer
from trainer import EmbeddingTrainer
from evaluator import EmbeddingEvaluator
from visualizer import EmbeddingVisualizer


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def load_or_download_data():
    """Step 1: Load or download the corpus."""
    print_section("STEP 1: DATA LOADING")
    
    loader = DataLoader()
    
    # Download Text8 corpus
    loader.download_text8()
    
    # Preprocess corpus
    corpus_file = loader.preprocess_corpus()
    
    # Show statistics
    stats = loader.get_corpus_stats()
    print("\nCorpus Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:,.0f}")
        else:
            print(f"  {key}: {value}")
    
    # Show sample
    sentences = loader.load_corpus_sentences()
    print("\nSample sentences:")
    for i, sent in enumerate(sentences[:2], 1):
        preview = ' '.join(sent.split()[:15]) + '...'
        print(f"  {i}. {preview}")
    
    return corpus_file, loader


def train_or_load_tokenizer(corpus_file):
    """Step 2: Train or load BPE tokenizer."""
    print_section("STEP 2: BPE TOKENIZATION")
    
    tokenizer = BPETokenizer(vocab_size=12000)
    tokenizer_path = Path("models/bpe_tokenizer.json")
    
    if tokenizer_path.exists():
        print("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training new BPE tokenizer...")
        tokenizer.train(corpus_file)
    
    # Show vocabulary info
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
    tokenizer.show_vocab_sample(20)
    
    # Test tokenization
    test_text = "the king and queen lived in a beautiful castle"
    print(f"\nTokenization test:")
    print(f"  Text: {test_text}")
    encoded = tokenizer.encode(test_text)
    print(f"  Encoded: {encoded[:10]}...")
    tokens = tokenizer.tokenizer.encode(test_text).tokens
    print(f"  Tokens: {tokens}")
    
    return tokenizer


def tokenize_corpus(tokenizer, corpus_file):
    """Step 3: Tokenize the corpus."""
    print_section("STEP 3: CORPUS TOKENIZATION")
    
    tokenized_file = Path("data/text8_tokenized.txt")
    
    if tokenized_file.exists():
        print("Loading existing tokenized corpus...")
        with open(tokenized_file, 'r', encoding='utf-8') as f:
            tokenized_sentences = [line.strip().split() for line in f if line.strip()]
    else:
        print("Tokenizing corpus...")
        tokenized_sentences = tokenizer.tokenize_corpus(
            corpus_file,
            output_file=tokenized_file
        )
    
    print(f"Total tokenized sentences: {len(tokenized_sentences)}")
    print(f"Sample: {tokenized_sentences[0][:15]}")
    
    return tokenized_sentences


def train_or_load_model(tokenized_sentences):
    """Step 4: Train or load Word2Vec model."""
    print_section("STEP 4: WORD2VEC TRAINING")
    
    model_path = Path("models/word2vec_skipgram.model")
    
    trainer = EmbeddingTrainer(
        vector_size=128,
        window=5,
        min_count=5,
        negative=10,
        epochs=10
    )
    
    if model_path.exists():
        print("Loading existing model...")
        model = trainer.load_model(model_path)
    else:
        print("Training new Word2Vec model...")
        print("This may take 15-45 minutes depending on your CPU...")
        model = trainer.train(tokenized_sentences)
    
    # Show model info
    vocab = trainer.get_vocabulary()
    print(f"\nModel vocabulary size: {len(vocab)}")
    print(f"Sample words: {vocab[:30]}")
    
    return model, trainer


def run_similarity_tests(evaluator):
    """Step 5a: Run similarity tests."""
    print_section("STEP 5A: SIMILARITY TESTS")
    
    # Define test cases
    test_pairs = [
        # High similarity (related concepts)
        ("king", "queen", "HIGH - royalty"),
        ("man", "woman", "HIGH - gender"),
        ("good", "better", "HIGH - comparative"),
        ("walk", "walking", "HIGH - verb forms"),
        
        # Medium similarity (somewhat related)
        ("king", "man", "MEDIUM - gender-royalty"),
        ("good", "bad", "MEDIUM - antonyms"),
        
        # Low similarity (unrelated)
        ("king", "computer", "LOW - unrelated"),
        ("happy", "tree", "LOW - unrelated"),
    ]
    
    print("\nSimilarity Scores:")
    print(f"{'Word 1':<15} {'Word 2':<15} {'Expected':<20} {'Score':<10}")
    print("-" * 70)
    
    results = []
    for word1, word2, expected in test_pairs:
        if evaluator.word_in_vocab(word1) and evaluator.word_in_vocab(word2):
            sim = evaluator.cosine_similarity(word1, word2)
            results.append((word1, word2, expected, sim))
            print(f"{word1:<15} {word2:<15} {expected:<20} {sim:.4f}")
        else:
            missing = word1 if not evaluator.word_in_vocab(word1) else word2
            print(f"{word1:<15} {word2:<15} {expected:<20} N/A ('{missing}' not in vocab)")
    
    return results


def run_analogy_tests(evaluator):
    """Step 5b: Run analogy tests."""
    print_section("STEP 5B: ANALOGY TESTS")
    
    # Define analogy test cases
    analogy_tests = [
        ("man", "king", "woman", "queen", "Gender-Royalty"),
        ("man", "woman", "king", "queen", "Gender-Royalty Reverse"),
        ("walk", "walked", "talk", "talked", "Past Tense"),
        ("good", "better", "bad", "worse", "Comparative"),
        ("france", "paris", "england", "london", "Country-Capital"),
        ("big", "bigger", "small", "smaller", "Comparative Size"),
    ]
    
    print("\nAnalogy Solving Results:")
    correct_count = 0
    total_count = 0
    
    for word_a, word_b, word_c, expected, description in analogy_tests:
        print(f"\n{description}:")
        print(f"  {word_a}:{word_b} :: {word_c}:? (expected: {expected})")
        
        if all(evaluator.word_in_vocab(w) for w in [word_a, word_b, word_c]):
            results = evaluator.solve_analogy(word_a, word_b, word_c, topn=5)
            
            if results:
                total_count += 1
                top_prediction = results[0][0]
                
                if top_prediction == expected or (results and expected in [w for w, _ in results[:3]]):
                    correct_count += 1
                    print(f"  ✓ Correct! Top predictions:")
                else:
                    print(f"  ✗ Incorrect. Top predictions:")
                
                for i, (word, score) in enumerate(results, 1):
                    marker = "★" if word == expected else " "
                    print(f"    {marker} {i}. {word:<20} (similarity: {score:.4f})")
        else:
            missing = [w for w in [word_a, word_b, word_c] if not evaluator.word_in_vocab(w)]
            print(f"  ✗ Cannot solve: {missing} not in vocabulary")
    
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"\n{'─' * 70}")
        print(f"Analogy Accuracy: {correct_count}/{total_count} = {accuracy:.1%}")


def run_nearest_neighbor_tests(evaluator):
    """Step 5c: Run K-nearest neighbor tests."""
    print_section("STEP 5C: K-NEAREST NEIGHBORS")
    
    # Test words
    test_words = ["king", "computer", "happy", "one", "big", "red"]
    
    for word in test_words:
        if evaluator.word_in_vocab(word):
            print(f"\nMost similar to '{word}':")
            similar = evaluator.most_similar(word, topn=8)
            
            for i, (similar_word, score) in enumerate(similar, 1):
                print(f"  {i}. {similar_word:<20} {score:.4f}")
        else:
            print(f"\n'{word}' not in vocabulary")


def create_visualizations(model):
    """Step 6: Create visualizations."""
    print_section("STEP 6: VISUALIZATIONS")
    
    visualizer = EmbeddingVisualizer(model)
    
    # 1. General t-SNE visualization
    print("\n1. Creating t-SNE visualization (top 100 words)...")
    try:
        visualizer.tsne_visualization(
            n_words=100,
            output_file="output/tsne_embeddings.png"
        )
        print("   ✓ Saved to output/tsne_embeddings.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 2. UMAP visualization
    print("\n2. Creating UMAP visualization (top 100 words)...")
    try:
        visualizer.umap_visualization(
            n_words=100,
            output_file="output/umap_embeddings.png"
        )
        print("   ✓ Saved to output/umap_embeddings.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Word clusters
    print("\n3. Creating semantic clusters visualization...")
    
    word_groups = {
        'Royalty': ['king', 'queen', 'prince', 'princess', 'royal', 'throne'],
        'Family': ['father', 'mother', 'brother', 'sister', 'family', 'parent'],
        'Animals': ['dog', 'cat', 'horse', 'animal', 'bird', 'fish'],
        'Numbers': ['one', 'two', 'three', 'four', 'five', 'number'],
        'Colors': ['red', 'blue', 'green', 'white', 'black', 'color']
    }
    
    try:
        visualizer.visualize_word_clusters(
            word_groups,
            method='tsne',
            output_file="output/semantic_clusters.png"
        )
        print("   ✓ Saved to output/semantic_clusters.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")


def save_results_summary(trainer, similarity_results):
    """Save a summary of results."""
    print_section("SAVING RESULTS SUMMARY")
    
    summary = {
        'model_info': {
            'vector_size': trainer.vector_size,
            'window': trainer.window,
            'min_count': trainer.min_count,
            'negative': trainer.negative,
            'epochs': trainer.epochs,
            'vocab_size': trainer.get_vocab_size(),
            'training_time_seconds': trainer.training_time
        },
        'similarity_tests': [
            {
                'word1': w1,
                'word2': w2,
                'expected': exp,
                'similarity': float(sim)
            }
            for w1, w2, exp, sim in similarity_results
        ]
    }
    
    output_file = Path("output/results_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to {output_file}")


def main():
    """Run the complete pipeline."""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "CUSTOM WORD EMBEDDING MODEL DEMO" + " " * 26 + "║")
    print("║" + " " * 15 + "Skip-gram with BPE Tokenization" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    
    start_time = time.time()
    
    try:
        # Step 1: Data Loading
        corpus_file, loader = load_or_download_data()
        
        # Step 2: Tokenization
        tokenizer = train_or_load_tokenizer(corpus_file)
        
        # Step 3: Tokenize corpus
        tokenized_sentences = tokenize_corpus(tokenizer, corpus_file)
        
        # Step 4: Train model
        model, trainer = train_or_load_model(tokenized_sentences)
        
        # Step 5: Evaluation
        evaluator = EmbeddingEvaluator(model)
        
        similarity_results = run_similarity_tests(evaluator)
        run_analogy_tests(evaluator)
        run_nearest_neighbor_tests(evaluator)
        
        # Step 6: Visualizations
        create_visualizations(model)
        
        # Save summary
        save_results_summary(trainer, similarity_results)
        
        # Final summary
        total_time = time.time() - start_time
        print_section("PIPELINE COMPLETED")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nOutputs saved to:")
        print(f"  - models/          (trained model and tokenizer)")
        print(f"  - output/          (visualizations and results)")
        print(f"  - data/            (corpus and tokenized data)")
        
        print("\n" + "=" * 70)
        print("SUCCESS! Your custom word embedding model is ready to use.")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
