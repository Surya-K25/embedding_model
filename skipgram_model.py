"""Skip-gram model with negative sampling implementation."""

import numpy as np
from collections import Counter

class SkipGram:
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.025, 
                 window_size=5, negative_samples=5):
        """
        Initialize Skip-gram model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimensionality of word embeddings
            learning_rate: Initial learning rate
            window_size: Context window size
            negative_samples: Number of negative samples per positive sample
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.window_size = window_size
        self.negative_samples = negative_samples
        
        # Initialize embedding matrices with small random values
        init_range = 0.5 / embedding_dim
        self.target_embeddings = np.random.uniform(
            -init_range, init_range, (vocab_size, embedding_dim)
        ).astype(np.float32)
        self.context_embeddings = np.random.uniform(
            -init_range, init_range, (vocab_size, embedding_dim)
        ).astype(np.float32)
        
        # For negative sampling
        self.sampling_probs = None
        
    def prepare_negative_sampling(self, token_counts):
        """
        Prepare probability distribution for negative sampling.
        Uses unigram distribution raised to the power of 0.75.
        
        Args:
            token_counts: Counter object with token frequencies
        """
        # Get frequency for each token ID
        freqs = np.zeros(self.vocab_size, dtype=np.float64)
        for token_id, count in token_counts.items():
            if 0 <= token_id < self.vocab_size:
                freqs[token_id] = count
        
        # Apply smoothing: P(w) ∝ count(w)^0.75
        freqs = np.power(freqs, 0.75)
        
        # Normalize to get probabilities
        self.sampling_probs = freqs / np.sum(freqs)
        
        print(f"Negative sampling distribution prepared")
    
    def sample_negatives(self, batch_size):
        """Sample negative examples."""
        return np.random.choice(
            self.vocab_size, 
            size=(batch_size, self.negative_samples),
            p=self.sampling_probs
        )
    
    def sigmoid(self, x):
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def train_batch(self, center_ids, context_ids, negative_ids):
        """
        Train on a batch of examples.
        
        Args:
            center_ids: Array of center word IDs (batch_size,)
            context_ids: Array of context word IDs (batch_size,)
            negative_ids: Array of negative sample IDs (batch_size, negative_samples)
        
        Returns:
            Average loss for the batch
        """
        batch_size = len(center_ids)
        
        # Get embeddings
        center_embeds = self.target_embeddings[center_ids]  # (batch_size, dim)
        context_embeds = self.context_embeddings[context_ids]  # (batch_size, dim)
        negative_embeds = self.context_embeddings[negative_ids]  # (batch_size, neg_samples, dim)
        
        # Positive samples: compute score and loss
        pos_score = np.sum(center_embeds * context_embeds, axis=1)  # (batch_size,)
        pos_loss = -np.log(self.sigmoid(pos_score) + 1e-10)
        
        # Negative samples: compute scores and loss
        neg_scores = np.einsum('ij,ikj->ik', center_embeds, negative_embeds)  # (batch_size, neg_samples)
        neg_loss = -np.sum(np.log(self.sigmoid(-neg_scores) + 1e-10), axis=1)
        
        # Total loss
        total_loss = np.mean(pos_loss + neg_loss)
        
        # Gradients for positive samples
        pos_grad = (self.sigmoid(pos_score) - 1)[:, np.newaxis]  # (batch_size, 1)
        
        # Update context embeddings (positive)
        context_grad = pos_grad * center_embeds
        np.add.at(self.context_embeddings, context_ids, 
                  -self.learning_rate * context_grad)
        
        # Update center embeddings (positive)
        center_grad_pos = pos_grad * context_embeds
        
        # Gradients for negative samples
        neg_grad = self.sigmoid(neg_scores)[:, :, np.newaxis]  # (batch_size, neg_samples, 1)
        
        # Update context embeddings (negative)
        negative_grad = neg_grad * center_embeds[:, np.newaxis, :]  # (batch_size, neg_samples, dim)
        for i in range(batch_size):
            np.add.at(self.context_embeddings, negative_ids[i], 
                     -self.learning_rate * negative_grad[i])
        
        # Update center embeddings (negative)
        center_grad_neg = np.sum(neg_grad * negative_embeds, axis=1)  # (batch_size, dim)
        
        # Combined gradient for center embeddings
        center_grad_total = center_grad_pos + center_grad_neg
        np.add.at(self.target_embeddings, center_ids, 
                 -self.learning_rate * center_grad_total)
        
        return total_loss
    
    def update_learning_rate(self, progress):
        """
        Decay learning rate linearly.
        
        Args:
            progress: Training progress from 0 to 1
        """
        min_lr = 0.0001
        self.learning_rate = max(
            min_lr,
            self.initial_lr * (1 - progress)
        )
    
    def get_embedding(self, token_id):
        """Get embedding vector for a token."""
        return self.target_embeddings[token_id]
    
    def get_embeddings(self):
        """Get all target embeddings."""
        return self.target_embeddings
    
    def save_embeddings(self, path):
        """Save embeddings to file."""
        np.save(path, self.target_embeddings)
        print(f"Embeddings saved to {path}")
    
    def load_embeddings(self, path):
        """Load embeddings from file."""
        self.target_embeddings = np.load(path)
        print(f"Embeddings loaded from {path}")


if __name__ == '__main__':
    # Test the model
    print("Testing Skip-gram model...")
    
    vocab_size = 100
    embedding_dim = 50
    
    model = SkipGram(vocab_size, embedding_dim)
    
    # Create fake token counts for negative sampling
    token_counts = Counter({i: np.random.randint(10, 100) for i in range(vocab_size)})
    model.prepare_negative_sampling(token_counts)
    
    # Create a small batch
    batch_size = 32
    center_ids = np.random.randint(0, vocab_size, batch_size)
    context_ids = np.random.randint(0, vocab_size, batch_size)
    negative_ids = model.sample_negatives(batch_size)
    
    # Train on batch
    loss = model.train_batch(center_ids, context_ids, negative_ids)
    print(f"Batch loss: {loss:.4f}")
    
    # Get embedding
    embedding = model.get_embedding(0)
    print(f"Embedding shape: {embedding.shape}")
