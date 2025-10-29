"""
Lab 5 Prerequisite: Vectorizer Classes
(Simulated from Lab 3)

Note: This is a simplified implementation for the lab.
In a real-world scenario, sklearn.feature_extraction.text.TfidfVectorizer
would be used, as it is highly optimized.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
from src.tokenization.tokenizer import BaseTokenizer, RegexTokenizer
import numpy as np

class BaseVectorizer(ABC):
    """Abstract base class for vectorizers."""
    
    @abstractmethod
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fits the vectorizer to the documents and transforms them into a matrix.
        """
        pass
    
    @abstractmethod
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforms new documents into a matrix using the fitted vocabulary.
        """
        pass

class TfidfVectorizer(BaseVectorizer):
    """
    Implements TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
    """
    
    def __init__(self, tokenizer: BaseTokenizer):
        """
        Initializes the vectorizer.
        
        Args:
            tokenizer (BaseTokenizer): An instance of a tokenizer class.
        """
        self.tokenizer = tokenizer
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count: int = 0

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fits the vectorizer to the text data and transforms it into a TF-IDF matrix.
        
        Args:
            documents (List[str]): A list of text documents.
            
        Returns:
            np.ndarray: A (n_documents, n_vocabulary) TF-IDF matrix.
        """
        self._doc_count = len(documents)
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]
        
        # Build vocabulary
        vocab = set()
        for doc in tokenized_docs:
            vocab.update(doc)
        
        self._vocabulary = {term: i for i, term in enumerate(sorted(list(vocab)))}
        
        # Calculate IDF (Inverse Document Frequency)
        df = {term: 0 for term in self._vocabulary}
        for doc in tokenized_docs:
            for term in set(doc):  # Use set for document frequency
                if term in df:
                    df[term] += 1
                    
        for term, count in df.items():
            # Using the standard "smooth" IDF formula
            self._idf[term] = np.log((self._doc_count + 1) / (count + 1)) + 1
            
        # Calculate TF-IDF
        tfidf_matrix = np.zeros((self._doc_count, len(self._vocabulary)))
        
        for i, doc in enumerate(tokenized_docs):
            tf = {term: 0 for term in self._vocabulary}
            for term in doc:
                if term in tf:
                    tf[term] += 1
            
            # Normalize TF
            doc_len = len(doc)
            if doc_len > 0:
                for term, count in tf.items():
                    tf[term] = count / doc_len
            
            # Calculate TF-IDF score for each term in the doc
            for term, term_index in self._vocabulary.items():
                tfidf_matrix[i, term_index] = tf[term] * self._idf.get(term, 0)
                
        return tfidf_matrix

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforms new documents into a TF-IDF matrix using the existing
        vocabulary and IDF scores.
        
        Args:
            documents (List[str]): A list of new text documents.
            
        Returns:
            np.ndarray: A (n_documents, n_vocabulary) TF-IDF matrix.
        """
        if not self._vocabulary:
            raise ValueError("Vectorizer has not been fitted yet. Call fit_transform first.")
            
        tokenized_docs = [self.tokenizer.tokenize(doc) for doc in documents]
        tfidf_matrix = np.zeros((len(documents), len(self._vocabulary)))
        
        for i, doc in enumerate(tokenized_docs):
            tf = {term: 0 for term in self._vocabulary}
            for term in doc:
                if term in tf:
                    tf[term] += 1
            
            # Normalize TF
            doc_len = len(doc)
            if doc_len > 0:
                for term, count in tf.items():
                    tf[term] = count / doc_len
            
            # Calculate TF-IDF score
            for term, term_index in self._vocabulary.items():
                tfidf_matrix[i, term_index] = tf.get(term, 0) * self._idf.get(term, 0)
                
        return tfidf_matrix

if __name__ == '__main__':
    # Example usage
    print("Testing TfidfVectorizer:")
    docs = [
        "this is the first document",
        "this document is the second document"
    ]
    
    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    print(f"Vocabulary: {vectorizer._vocabulary}")
    print(f"IDF Scores: {vectorizer._idf}")
    print(f"TF-IDF Matrix:\n{tfidf_matrix}")
    
    new_docs = ["this is a new document"]
    new_matrix = vectorizer.transform(new_docs)
    print(f"New Doc Matrix:\n{new_matrix}")
