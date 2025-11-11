"""
Tokenization Module
Implements BPE (Byte Pair Encoding) tokenization using Hugging Face's tokenizers library.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence
from pathlib import Path
import json


class BPETokenizer:
    """Byte Pair Encoding tokenizer for word embeddings."""
    
    def __init__(self, vocab_size=12000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (10,000-15,000)
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.token_to_id = {}
        self.id_to_token = {}
    
    def train(self, corpus_file, output_dir="models"):
        """
        Train BPE tokenizer on corpus.
        
        Args:
            corpus_file: Path to text corpus file
            output_dir: Directory to save tokenizer
        """
        print(f"Training BPE tokenizer with vocab size {self.vocab_size}...")
        
        # Initialize tokenizer with BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        
        # Set normalizer to lowercase
        self.tokenizer.normalizer = Sequence([Lowercase()])
        
        # Use whitespace pre-tokenizer
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
            show_progress=True,
            min_frequency=5  # Minimum frequency for tokens
        )
        
        # Train on corpus
        corpus_file = str(corpus_file)
        self.tokenizer.train([corpus_file], trainer)
        
        # Build token mappings
        self._build_vocab_mappings()
        
        # Save tokenizer
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tokenizer_path = output_path / "bpe_tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))
        
        # Save vocabulary separately
        vocab_path = output_path / "vocabulary.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'vocab_size': len(self.token_to_id)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Tokenizer saved to {tokenizer_path}")
        print(f"Vocabulary saved to {vocab_path}")
        print(f"Actual vocabulary size: {len(self.token_to_id)}")
    
    def load(self, tokenizer_path):
        """
        Load a trained tokenizer.
        
        Args:
            tokenizer_path: Path to saved tokenizer JSON file
        """
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._build_vocab_mappings()
        print(f"Tokenizer loaded. Vocabulary size: {len(self.token_to_id)}")
    
    def _build_vocab_mappings(self):
        """Build token<->id mappings from trained tokenizer."""
        vocab = self.tokenizer.get_vocab()
        self.token_to_id = vocab
        self.id_to_token = {id_: token for token, id_ in vocab.items()}
    
    def encode(self, text):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
        
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids):
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        
        return self.tokenizer.decode(token_ids)
    
    def tokenize_corpus(self, corpus_file, output_file=None):
        """
        Tokenize entire corpus and optionally save.
        
        Args:
            corpus_file: Path to corpus text file
            output_file: Optional path to save tokenized corpus
        
        Returns:
            List of tokenized sentences (each as list of token strings)
        """
        print("Tokenizing corpus...")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tokenized_sentences = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Encode and get tokens
                encoding = self.tokenizer.encode(line)
                tokens = encoding.tokens
                tokenized_sentences.append(tokens)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for tokens in tokenized_sentences:
                    f.write(' '.join(tokens) + '\n')
            print(f"Tokenized corpus saved to {output_file}")
        
        print(f"Tokenized {len(tokenized_sentences)} sentences")
        return tokenized_sentences
    
    def get_vocab_size(self):
        """Get the vocabulary size."""
        return len(self.token_to_id)
    
    def get_token(self, token_id):
        """Get token string from ID."""
        return self.id_to_token.get(token_id, "<UNK>")
    
    def get_id(self, token):
        """Get ID from token string."""
        return self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0))
    
    def show_vocab_sample(self, n=20):
        """Display a sample of the vocabulary."""
        print(f"\nVocabulary sample (first {n} tokens):")
        for i, (token, id_) in enumerate(list(self.token_to_id.items())[:n]):
            print(f"  {id_:4d}: {token}")


def main():
    """Demo function to test tokenization."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    loader.download_text8()
    corpus_file = loader.preprocess_corpus()
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=12000)
    tokenizer.train(corpus_file)
    
    # Show vocabulary sample
    tokenizer.show_vocab_sample(30)
    
    # Test encoding/decoding
    test_text = "the king and queen lived in a beautiful castle"
    print(f"\nTest text: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    tokens = tokenizer.tokenizer.encode(test_text).tokens
    print(f"Tokens: {tokens}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Tokenize corpus
    tokenized = tokenizer.tokenize_corpus(
        corpus_file,
        output_file="data/text8_tokenized.txt"
    )
    
    print(f"\nFirst sentence tokens: {tokenized[0][:20]}")


if __name__ == "__main__":
    main()
