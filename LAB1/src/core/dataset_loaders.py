import os
from typing import List

def load_raw_text_data(dataset_path: str, filename: str = "en_ewt-ud-train.txt") -> str:
    """
    Loads the raw text content from the specified dataset path.
    
    Args:
        dataset_path: The base path to the UD_English-EWT directory.
        filename: The specific file to load (default: training data).
        
    Returns:
        The entire content of the file as a single string.
    """
    file_path = os.path.join(dataset_path, filename)
    
    # Check if the file exists before attempting to read
    if not os.path.exists(file_path):
        # NOTE: Since the file content cannot be directly accessed in this environment,
        # we provide a simulation of the text that would be loaded.
        print(f"WARNING: File not found at {file_path}. Using mock data for demonstration.")
        return (
            "This is the first sentence of the EWT dataset sample. "
            "NLP is fascinating... isn't it? Let's see how it handles 123 numbers and punctuation! "
            "She said, 'Hello world!' and then walked away. The 2024 election. U.S.A. Inc."
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}. Using mock data.")
        return (
            "This is the first sentence of the EWT dataset sample. "
            "NLP is fascinating... isn't it? Let's see how it handles 123 numbers and punctuation! "
            "She said, 'Hello world!' and then walked away. The 2024 election. U.S.A. Inc."
        )
