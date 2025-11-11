"""Simplified BPE tokenizer for word embedding model."""

from collections import Counter, defaultdict
import pickle
import re

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merge_rules = []
        self.vocab = {}
        self.id_to_token = {}
        
    def get_stats(self, word_freqs):
        """Count frequency of adjacent character pairs."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair, word_freqs):
        """Merge all occurrences of the most frequent pair."""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in word_freqs:
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = word_freqs[word]
        
        return new_word_freqs
    
    def train_bpe(self, text, num_merges=None):
        """
        Train BPE on the given text.
        
        Args:
            text: Input text (space-separated words)
            num_merges: Number of merge operations (default: vocab_size - 256)
        """
        if num_merges is None:
            num_merges = self.vocab_size - 256
        
        # Split text into words and get frequencies
        words = text.split()
        word_counts = Counter(words)
        
        # Initialize vocabulary with character-level tokens
        word_freqs = {}
        for word, freq in word_counts.items():
            # Add space between each character and </w> at the end
            word_freqs[' '.join(list(word)) + ' </w>'] = freq
        
        print(f"Training BPE with {num_merges} merges...")
        
        # Learn merge rules
        for i in range(num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            self.merge_rules.append(best_pair)
            
            if (i + 1) % 500 == 0:
                print(f"  Completed {i + 1}/{num_merges} merges")
        
        # Build vocabulary - limit to vocab_size most frequent tokens
        token_freqs = Counter()
        for word, freq in word_freqs.items():
            for token in word.split():
                token_freqs[token] += freq
        
        # Add special tokens
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # Add most frequent tokens up to vocab_size
        most_common = token_freqs.most_common(self.vocab_size - 2)
        for i, (token, _) in enumerate(most_common, start=2):
            self.vocab[token] = i
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of merge rules: {len(self.merge_rules)}")
        
    def tokenize_word(self, word):
        """Apply BPE merges to a single word."""
        # Start with character-level representation
        word = ' '.join(list(word)) + ' </w>'
        
        # Apply merge rules in order
        for pair in self.merge_rules:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            word = word.replace(bigram, replacement)
        
        return word.split()
    
    def tokenize(self, text):
        """Tokenize text using learned BPE."""
        words = text.split()
        tokens = []
        
        for word in words:
            tokens.extend(self.tokenize_word(word))
        
        return tokens
    
    def encode(self, tokens):
        """Convert tokens to integer IDs."""
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def decode(self, ids):
        """Convert integer IDs back to tokens."""
        tokens = [self.id_to_token.get(id, '<UNK>') for id in ids]
        # Join and remove </w> markers
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text
    
    def save(self, path):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'merge_rules': self.merge_rules,
            'vocab': self.vocab,
            'id_to_token': self.id_to_token
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path):
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.merge_rules = data['merge_rules']
        self.vocab = data['vocab']
        self.id_to_token = data['id_to_token']
        print(f"Tokenizer loaded from {path}")


if __name__ == '__main__':
    # Test the tokenizer
    print("Testing BPE Tokenizer...")
    
    # Small test corpus
    test_text = "the cat sat on the mat the dog sat on the log"
    
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train_bpe(test_text, num_merges=50)
    
    # Test tokenization
    test_sentence = "the cat sat"
    tokens = tokenizer.tokenize(test_sentence)
    print(f"\nOriginal: {test_sentence}")
    print(f"Tokens: {tokens}")
    
    ids = tokenizer.encode(tokens)
    print(f"IDs: {ids}")
    
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
