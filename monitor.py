"""
Monitor training progress and display current model status.
"""

import os
import time
from datetime import datetime

def check_files():
    """Check which files exist and their sizes."""
    files = {
        'Corpus': 'data/text8',
        'Tokenizer': 'models/tokenizer.pkl',
        'Embeddings (final)': 'models/embeddings.npy',
        'Embeddings (epoch 1)': 'models/embeddings_epoch1.npy',
        'Embeddings (epoch 2)': 'models/embeddings_epoch2.npy',
        'Embeddings (epoch 3)': 'models/embeddings_epoch3.npy',
    }
    
    print("\n" + "="*70)
    print("TRAINING PROGRESS MONITOR")
    print("="*70)
    print(f"\nChecked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("File Status:")
    print("-" * 70)
    
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_mb = size / (1024 * 1024)
            modified = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"✓ {name:25s} {size_mb:8.2f} MB  (modified: {modified.strftime('%H:%M:%S')})")
        else:
            print(f"✗ {name:25s} Not found")
    
    print("\n" + "="*70)
    
    # Estimate progress
    if os.path.exists('models/tokenizer.pkl'):
        print("\nTraining Stage: Tokenizer complete ✓")
        
        if os.path.exists('models/embeddings_epoch1.npy'):
            print("Training Stage: Epoch 1 complete ✓")
        else:
            print("Training Stage: Training epoch 1...")
            
        if os.path.exists('models/embeddings_epoch2.npy'):
            print("Training Stage: Epoch 2 complete ✓")
        else:
            print("Training Stage: Waiting for epoch 2...")
            
        if os.path.exists('models/embeddings_epoch3.npy'):
            print("Training Stage: Epoch 3 complete ✓")
        else:
            print("Training Stage: Waiting for epoch 3...")
            
        if os.path.exists('models/embeddings.npy'):
            print("\n🎉 TRAINING COMPLETE! Ready for evaluation.")
    else:
        print("\nTraining Stage: Building BPE tokenizer...")
    
    print("="*70)


def quick_evaluate():
    """Quick evaluation if embeddings exist."""
    if not os.path.exists('models/embeddings.npy'):
        print("\n⚠ Embeddings not ready yet. Wait for training to complete.")
        return
    
    print("\n" + "="*70)
    print("QUICK EVALUATION")
    print("="*70)
    
    try:
        import numpy as np
        from bpe_tokenizer import BPETokenizer
        from evaluation import EmbeddingEvaluator
        
        print("\nLoading model...")
        tokenizer = BPETokenizer()
        tokenizer.load('models/tokenizer.pkl')
        embeddings = np.load('models/embeddings.npy')
        
        evaluator = EmbeddingEvaluator(embeddings, tokenizer)
        
        print("\nSimilarity Tests:")
        print("-" * 70)
        
        pairs = [
            ("apple", "banana"),
            ("apple", "potato"),
            ("king", "queen"),
        ]
        
        for w1, w2 in pairs:
            sim = evaluator.similarity(w1, w2)
            if sim:
                print(f"  similarity('{w1}', '{w2}') = {sim:.4f}")
        
        print("\nNearest Neighbors for 'apple':")
        print("-" * 70)
        neighbors = evaluator.nearest_neighbors("apple", k=5)
        for word, score in neighbors:
            print(f"  {word}: {score:.4f}")
        
        print("\nAnalogy: man:king::woman:?")
        print("-" * 70)
        results = evaluator.analogy("man", "king", "woman", k=5)
        for word, score in results:
            print(f"  {word}: {score:.4f}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n⚠ Error during evaluation: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--eval':
        quick_evaluate()
    else:
        check_files()
        
        if os.path.exists('models/embeddings.npy'):
            print("\n💡 Tip: Run 'python monitor.py --eval' for quick evaluation")
            print("💡 Or run 'python main.py --mode evaluate' for full evaluation")
