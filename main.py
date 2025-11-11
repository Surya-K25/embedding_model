"""Main script to train and evaluate word embeddings from scratch."""

import os
import argparse
import numpy as np

from training import train_embeddings
from evaluation import EmbeddingEvaluator, print_evaluation_results
from bpe_tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser(description='Train word embeddings from scratch')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'both'],
                       help='Mode: train, evaluate, or both')
    parser.add_argument('--corpus', type=str, default='data/text8',
                       help='Path to corpus file')
    parser.add_argument('--vocab-size', type=int, default=5000,
                       help='BPE vocabulary size')
    parser.add_argument('--embedding-dim', type=int, default=100,
                       help='Embedding dimensionality')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--window-size', type=int, default=5,
                       help='Context window size')
    parser.add_argument('--negative-samples', type=int, default=5,
                       help='Number of negative samples')
    parser.add_argument('--batch-size', type=int, default=2048,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.025,
                       help='Initial learning rate')
    parser.add_argument('--max-chars', type=int, default=10000000,
                       help='Maximum characters to use from corpus (default: 10M)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # File paths
    tokenizer_path = 'models/tokenizer.pkl'
    embedding_path = 'models/embeddings.npy'
    
    # Training
    if args.mode in ['train', 'both']:
        print("\n" + "="*70)
        print("TRAINING WORD EMBEDDINGS")
        print("="*70)
        
        model, tokenizer = train_embeddings(
            corpus_path=args.corpus,
            tokenizer_path=tokenizer_path,
            embedding_path=embedding_path,
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            window_size=args.window_size,
            negative_samples=args.negative_samples,
            batch_size=args.batch_size,
            initial_lr=args.learning_rate,
            max_corpus_chars=args.max_chars
        )
    
    # Evaluation
    if args.mode in ['evaluate', 'both']:
        print("\n" + "="*70)
        print("EVALUATING WORD EMBEDDINGS")
        print("="*70)
        
        # Load tokenizer and embeddings
        if not os.path.exists(tokenizer_path) or not os.path.exists(embedding_path):
            print("Error: Model not found. Please train first.")
            return
        
        print("\nLoading tokenizer and embeddings...")
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
        
        embeddings = np.load(embedding_path)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Create evaluator
        evaluator = EmbeddingEvaluator(embeddings, tokenizer)
        
        # Define test cases
        test_words = ["apple", "banana", "king", "queen", "run", "walk"]
        
        analogy_questions = [
            ("man", "king", "woman", "queen"),
            ("man", "woman", "king", "queen"),
            ("big", "bigger", "small", "smaller"),
            ("good", "best", "bad", "worst"),
            ("walk", "walking", "run", "running"),
        ]
        
        # Print evaluation results
        print_evaluation_results(evaluator, test_words, analogy_questions)
        
        print("\n" + "="*70)
        print("COMPLETED!")
        print("="*70)


if __name__ == '__main__':
    main()
