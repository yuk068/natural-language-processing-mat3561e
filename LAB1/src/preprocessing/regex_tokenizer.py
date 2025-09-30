import re
from typing import List
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    """
    A more robust tokenizer using a single regular expression.
    It captures words, numbers, and most punctuation as separate tokens.
    """
    # Regex pattern:
    # \w+                   -> Matches one or more word characters (letters, numbers, underscore)
    # [^\w\s]               -> Matches any single non-word, non-whitespace character (punctuation)
    # The '|' (OR) ensures that we match the longest possible token first.
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text: lowercases and uses a robust regex to find tokens.
        
        Args:
            text: The input string.
            
        Returns:
            A list of tokens.
        """
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Use regex to find all matches
        tokens = self.TOKEN_PATTERN.findall(text)
        
        return tokens

if __name__ == '__main__':
    # Simple self-test
    tokenizer = RegexTokenizer()
    test_sentence = "NLP is fascinating... isn't it? Let's see how it handles 123 numbers!"
    print(f"Input: {test_sentence}")
    print(f"Output: {tokenizer.tokenize(test_sentence)}")
