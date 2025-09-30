import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    """
    A simple tokenizer that splits text based on whitespace and
    handles basic punctuation by separating it from words.
    """
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text: lowercases, separates punctuation, and splits by space.
        Example: "Hello, world!" -> ["hello", ",", "world", "!"]
        
        Args:
            text: The input string.
            
        Returns:
            A list of tokens.
        """
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Handle basic punctuation by inserting a space before and after it.
        # This handles: comma (,), period (.), question mark (?), exclamation mark (!), and colon (:)
        # We use a space on both sides to ensure it is split by subsequent whitespace splitting.
        text = re.sub(r'([,.:?!])', r' \1 ', text)
        
        # 3. Split the text into tokens based on whitespace.
        # This correctly handles multiple spaces and the spaces added around punctuation.
        tokens = text.split()
        
        return tokens

if __name__ == '__main__':
    # Simple self-test
    tokenizer = SimpleTokenizer()
    test_sentence = "Hello, world! This is a test."
    print(f"Input: {test_sentence}")
    print(f"Output: {tokenizer.tokenize(test_sentence)}")
