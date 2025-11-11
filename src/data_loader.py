"""
Data Loading Module
Downloads and prepares the Text8 corpus for training.
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import re


class DataLoader:
    """Handles downloading and preprocessing of text corpus."""
    
    TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
    
    def __init__(self, data_dir="data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_file = self.data_dir / "text8"
        self.processed_file = self.data_dir / "text8_processed.txt"
    
    def download_text8(self):
        """Download Text8 corpus if not already present."""
        zip_path = self.data_dir / "text8.zip"
        
        if self.raw_file.exists():
            print(f"Text8 corpus already exists at {self.raw_file}")
            return
        
        print("Downloading Text8 corpus (~31MB)...")
        response = requests.get(self.TEXT8_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Clean up zip file
        zip_path.unlink()
        print(f"Text8 corpus downloaded and extracted to {self.raw_file}")
    
    def preprocess_corpus(self, max_lines=None):
        """
        Preprocess the corpus: lowercase, handle punctuation, etc.
        
        Args:
            max_lines: Maximum number of lines to process (None for all)
        
        Returns:
            Path to processed file
        """
        if self.processed_file.exists():
            print(f"Processed corpus already exists at {self.processed_file}")
            return self.processed_file
        
        print("Preprocessing corpus...")
        
        with open(self.raw_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Text8 is already lowercase and cleaned, but we'll do some additional processing
        # Split into sentences (approximate by spaces - text8 has no punctuation)
        # This helps with training structure
        
        # Add some structure: split long sequences into manageable chunks
        words = text.split()
        chunk_size = 1000  # Words per line
        
        processed_lines = []
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i+chunk_size]
            processed_lines.append(' '.join(chunk))
            
            if max_lines and len(processed_lines) >= max_lines:
                break
        
        # Write processed corpus
        with open(self.processed_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
        
        print(f"Corpus preprocessed and saved to {self.processed_file}")
        print(f"Total lines: {len(processed_lines)}")
        print(f"Total words: ~{len(words)}")
        
        return self.processed_file
    
    def load_corpus_text(self):
        """
        Load the preprocessed corpus text.
        
        Returns:
            Full corpus text as string
        """
        if not self.processed_file.exists():
            raise FileNotFoundError(
                f"Processed corpus not found at {self.processed_file}. "
                "Run preprocess_corpus() first."
            )
        
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_corpus_sentences(self):
        """
        Load corpus as list of sentences.
        
        Returns:
            List of sentence strings
        """
        if not self.processed_file.exists():
            raise FileNotFoundError(
                f"Processed corpus not found at {self.processed_file}. "
                "Run preprocess_corpus() first."
            )
        
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def get_corpus_stats(self):
        """Get basic statistics about the corpus."""
        if not self.processed_file.exists():
            raise FileNotFoundError("Processed corpus not found.")
        
        sentences = self.load_corpus_sentences()
        all_words = []
        for sent in sentences:
            all_words.extend(sent.split())
        
        unique_words = set(all_words)
        
        stats = {
            'total_sentences': len(sentences),
            'total_words': len(all_words),
            'unique_words': len(unique_words),
            'avg_words_per_sentence': len(all_words) / len(sentences) if sentences else 0
        }
        
        return stats


def main():
    """Demo function to test data loading."""
    loader = DataLoader()
    
    # Download corpus
    loader.download_text8()
    
    # Preprocess
    loader.preprocess_corpus()
    
    # Show stats
    stats = loader.get_corpus_stats()
    print("\nCorpus Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,.0f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    
    # Show sample
    sentences = loader.load_corpus_sentences()
    print("\nFirst 3 sentences (sample):")
    for i, sent in enumerate(sentences[:3], 1):
        preview = ' '.join(sent.split()[:20]) + '...'
        print(f"  {i}. {preview}")


if __name__ == "__main__":
    main()
