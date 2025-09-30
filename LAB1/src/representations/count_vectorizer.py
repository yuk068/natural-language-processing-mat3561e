from typing import List, Dict
from src.core.interfaces import Vectorizer, Tokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer # Using RegexTokenizer as default
from src.preprocessing.simple_tokenizer import SimpleTokenizer # Not strictly needed here, but good practice

class CountVectorizer(Vectorizer):
    """
    Implements the Bag-of-Words model by counting word occurrences.
    It uses a Tokenizer instance to preprocess the documents.
    """
    def __init__(self, tokenizer: Tokenizer):
        """
        Initializes the CountVectorizer with a specific tokenizer.
        
        Args:
            tokenizer: An instance of a class inheriting from Tokenizer.
        """
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}
        self.is_fitted = False

    def fit(self, corpus: List[str]) -> 'CountVectorizer':
        """
        Learns the vocabulary from a list of documents.
        
        Args:
            corpus: A list of documents (strings).
            
        Returns:
            The fitted CountVectorizer instance.
        """
        unique_tokens = set()
        
        # 1. Collect all unique tokens from the corpus
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)
            
        # 2. Create the vocabulary dictionary (token to index mapping)
        # Sorting ensures a consistent and reproducible index order.
        sorted_tokens = sorted(list(unique_tokens))
        self.vocabulary_ = {token: index for index, token in enumerate(sorted_tokens)}
        self.is_fitted = True
        
        return self

    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms a list of documents into a list of count vectors.
        
        Args:
            documents: A list of documents (strings) to transform.
            
        Returns:
            A list of lists of integers (the document-term matrix).
        """
        if not self.is_fitted:
            print("WARNING: Vectorizer is not fitted. Run .fit() or .fit_transform() first.")
            return []
            
        document_term_matrix: List[List[int]] = []
        vocab_size = len(self.vocabulary_)
        
        # 1. Process each document
        for document in documents:
            # Create a zero vector for the current document
            vector = [0] * vocab_size
            
            # Tokenize the document
            tokens = self.tokenizer.tokenize(document)
            
            # 2. Count token occurrences and update the vector
            for token in tokens:
                if token in self.vocabulary_:
                    # Increment count at the token's corresponding index
                    index = self.vocabulary_[token]
                    vector[index] += 1
            
            document_term_matrix.append(vector)
            
        return document_term_matrix

if __name__ == '__main__':
    # Simple self-test (Evaluation example from Lab 2)
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=tokenizer)
    
    dtm = vectorizer.fit_transform(corpus)
    
    print("--- CountVectorizer Self-Test ---")
    print("Corpus:")
    for doc in corpus:
        print(f"  - {doc}")
    print("\nLearned Vocabulary (Token: Index):")
    print(vectorizer.vocabulary_)
    print("\nDocument-Term Matrix (DTM):")
    for doc_vector in dtm:
        print(f"  - {doc_vector}")
