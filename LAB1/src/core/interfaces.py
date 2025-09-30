from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Tokenizer(ABC):
    """
    Abstract Base Class for all Tokenizers.
    Defines the required 'tokenize' method.
    """
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Converts a string of text into a list of tokens.
        """
        pass

class Vectorizer(ABC):
    """
    Abstract Base Class for all Vectorizers (e.g., Count, TF-IDF).
    Defines the required 'fit' and 'transform' methods.
    """
    @abstractmethod
    def fit(self, corpus: List[str]) -> 'Vectorizer':
        """
        Learns the vocabulary from a list of documents (corpus).
        Returns the fitted Vectorizer instance.
        """
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms a list of documents into a document-term matrix
        based on the learned vocabulary.
        """
        pass

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        A convenience method that performs fit and then transform on the same data.
        """
        return self.fit(corpus).transform(corpus)

# Helper function to simulate module initialization
def dummy_init_files():
    """Placeholder function to simulate __init__.py for module imports."""
    pass

if __name__ == '__main__':
    # Simple test for interfaces
    print("Tokenizer and Vectorizer interfaces defined successfully.")
