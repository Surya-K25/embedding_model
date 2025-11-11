"""
Visualization Module
Implements t-SNE and UMAP visualizations for word embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from pathlib import Path
from typing import List, Optional


class EmbeddingVisualizer:
    """Visualizes word embeddings in 2D space."""
    
    def __init__(self, model):
        """
        Initialize visualizer with trained Word2Vec model.
        
        Args:
            model: Trained Word2Vec model or KeyedVectors
        """
        # Handle both Word2Vec model and KeyedVectors
        if hasattr(model, 'wv'):
            self.wv = model.wv
        else:
            self.wv = model
        
        self.vocab = list(self.wv.index_to_key)
    
    def prepare_data(self, words: Optional[List[str]] = None, n_words: int = 100):
        """
        Prepare word vectors for visualization.
        
        Args:
            words: Specific words to visualize (None = use top-N frequent words)
            n_words: Number of words to visualize if words=None
        
        Returns:
            Tuple of (word_list, vector_matrix)
        """
        if words is None:
            # Use most frequent words
            words = self.vocab[:n_words]
        else:
            # Filter out words not in vocabulary
            words = [w for w in words if w in self.wv]
        
        if not words:
            raise ValueError("No valid words to visualize")
        
        # Get vectors
        vectors = np.array([self.wv[word] for word in words])
        
        return words, vectors
    
    def tsne_visualization(
        self,
        words: Optional[List[str]] = None,
        n_words: int = 100,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42,
        output_file: Optional[str] = None,
        title: str = "t-SNE Word Embeddings Visualization"
    ):
        """
        Create t-SNE visualization of word embeddings.
        
        Args:
            words: Specific words to visualize
            n_words: Number of words if words=None
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            random_state: Random seed
            output_file: Path to save figure (None = display only)
            title: Plot title
        """
        print(f"Creating t-SNE visualization...")
        
        # Prepare data
        word_list, vectors = self.prepare_data(words, n_words)
        print(f"Visualizing {len(word_list)} words")
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(word_list) - 1),
            n_iter=n_iter,
            random_state=random_state
        )
        
        coords_2d = tsne.fit_transform(vectors)
        
        # Create visualization
        self._plot_embeddings(
            coords_2d,
            word_list,
            title=title,
            output_file=output_file
        )
    
    def umap_visualization(
        self,
        words: Optional[List[str]] = None,
        n_words: int = 100,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        output_file: Optional[str] = None,
        title: str = "UMAP Word Embeddings Visualization"
    ):
        """
        Create UMAP visualization of word embeddings.
        
        Args:
            words: Specific words to visualize
            n_words: Number of words if words=None
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random seed
            output_file: Path to save figure (None = display only)
            title: Plot title
        """
        print(f"Creating UMAP visualization...")
        
        # Prepare data
        word_list, vectors = self.prepare_data(words, n_words)
        print(f"Visualizing {len(word_list)} words")
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(word_list) - 1),
            min_dist=min_dist,
            random_state=random_state
        )
        
        coords_2d = reducer.fit_transform(vectors)
        
        # Create visualization
        self._plot_embeddings(
            coords_2d,
            word_list,
            title=title,
            output_file=output_file
        )
    
    def _plot_embeddings(
        self,
        coords_2d: np.ndarray,
        words: List[str],
        title: str,
        output_file: Optional[str] = None,
        figsize: tuple = (16, 12)
    ):
        """
        Plot 2D embeddings with word labels.
        
        Args:
            coords_2d: 2D coordinates (n_words, 2)
            words: List of word labels
            title: Plot title
            output_file: Path to save figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Scatter plot
        plt.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            alpha=0.6,
            s=100,
            c=range(len(words)),
            cmap='viridis'
        )
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(
                word,
                xy=(coords_2d[i, 0], coords_2d[i, 1]),
                xytext=(5, 2),
                textcoords='offset points',
                ha='left',
                fontsize=9,
                alpha=0.8
            )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        
        plt.show()
    
    def visualize_word_clusters(
        self,
        word_groups: dict,
        method: str = 'tsne',
        output_file: Optional[str] = None
    ):
        """
        Visualize specific groups of related words.
        
        Args:
            word_groups: Dictionary mapping group names to word lists
            method: 'tsne' or 'umap'
            output_file: Path to save figure
        """
        # Flatten word groups
        all_words = []
        group_labels = []
        
        for group_name, words in word_groups.items():
            valid_words = [w for w in words if w in self.wv]
            all_words.extend(valid_words)
            group_labels.extend([group_name] * len(valid_words))
        
        if not all_words:
            print("No valid words found in vocabulary")
            return
        
        print(f"Visualizing {len(all_words)} words from {len(word_groups)} groups")
        
        # Get vectors
        vectors = np.array([self.wv[word] for word in all_words])
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            title = "t-SNE: Word Clusters"
        else:  # umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            title = "UMAP: Word Clusters"
        
        coords_2d = reducer.fit_transform(vectors)
        
        # Create plot with color-coded groups
        plt.figure(figsize=(16, 12))
        
        # Plot each group with different color
        unique_groups = list(word_groups.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        
        for i, group_name in enumerate(unique_groups):
            # Get indices for this group
            indices = [j for j, label in enumerate(group_labels) if label == group_name]
            group_coords = coords_2d[indices]
            group_words = [all_words[j] for j in indices]
            
            # Plot points
            plt.scatter(
                group_coords[:, 0],
                group_coords[:, 1],
                alpha=0.6,
                s=150,
                c=[colors[i]],
                label=group_name
            )
            
            # Add labels
            for j, word in enumerate(group_words):
                plt.annotate(
                    word,
                    xy=(group_coords[j, 0], group_coords[j, 1]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='left',
                    fontsize=9,
                    alpha=0.9
                )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Cluster visualization saved to {output_file}")
        
        plt.show()


def main():
    """Demo function to test visualization."""
    from pathlib import Path
    from gensim.models import Word2Vec
    
    # Load trained model
    model_path = Path("models/word2vec_skipgram.model")
    
    if not model_path.exists():
        print("Model not found. Please train the model first.")
        return
    
    print("Loading model...")
    model = Word2Vec.load(str(model_path))
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(model)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION TESTS")
    print("=" * 60)
    
    # 1. General t-SNE visualization
    print("\n1. Creating t-SNE visualization of top 100 words...")
    visualizer.tsne_visualization(
        n_words=100,
        output_file="output/tsne_top100.png"
    )
    
    # 2. UMAP visualization
    print("\n2. Creating UMAP visualization of top 100 words...")
    visualizer.umap_visualization(
        n_words=100,
        output_file="output/umap_top100.png"
    )
    
    # 3. Word clusters visualization
    print("\n3. Creating word clusters visualization...")
    
    word_groups = {
        'Royalty': ['king', 'queen', 'prince', 'princess', 'throne', 'crown'],
        'Family': ['father', 'mother', 'brother', 'sister', 'son', 'daughter'],
        'Animals': ['dog', 'cat', 'horse', 'bird', 'fish', 'lion'],
        'Colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
        'Numbers': ['one', 'two', 'three', 'four', 'five', 'six']
    }
    
    visualizer.visualize_word_clusters(
        word_groups,
        method='tsne',
        output_file="output/word_clusters_tsne.png"
    )


if __name__ == "__main__":
    main()
