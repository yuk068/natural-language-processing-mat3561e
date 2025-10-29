"""
Lab 5 Prerequisite: Tokenizer Classes
(Simulated from Lab 2)
"""

import re
from abc import ABC, abstractmethod
from typing import List

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Converts a string of text into a list of tokens.
        """
        pass

class RegexTokenizer(BaseTokenizer):
    """
    A tokenizer that splits text into tokens using a regular expression.
    """
    
    def __init__(self, pattern: str = r"\b\w+\b"):
        """
        Initializes the tokenizer.
        
        Args:
            pattern (str): The regex pattern to use for finding tokens.
                           Default pattern finds all sequences of word characters.
        """
        self.pattern = re.compile(pattern)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by:
        1. Converting text to lowercase.
        2. Finding all matches for the regex pattern.
        
        Args:
            text (str): The input string to tokenize.
            
        Returns:
            List[str]: A list of lowercase tokens.
        """
        return self.pattern.findall(text.lower())

if __name__ == '__main__':
    # Example usage
    print("Testing RegexTokenizer:")
    tokenizer = RegexTokenizer()
    text = "Hello, world! This is a test."
    tokens = tokenizer.tokenize(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    
    text_2 = "Punctuation... should be! handled?"
    tokens_2 = tokenizer.tokenize(text_2)
    print(f"Text: '{text_2}'")
    print(f"Tokens: {tokens_2}")
