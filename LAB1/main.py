import os
import sys
import datetime
from typing import List, Dict, Any

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.core.dataset_loaders import load_raw_text_data
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

# --- Paths ---
LOG_DIR = "log"
RESULTS_DIR = "results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"run_{timestamp}.log")

# Redirect print to also write to log file
class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file_path)


# --- Config ---
DATASET_BASE_PATH = "data/UD_English-EWT"
DATASET_FILENAME = "en_ewt-ud-train.txt"

def run_lab1_evaluation():
    """Performs Task 1 & 2 evaluation: sample sentences test."""
    print("=" * 70)
    print("LAB 1: TEXT TOKENIZATION EVALUATION")
    print("=" * 70)

    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    test_sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",
        "U.S.A. Inc. paid $100."
    ]

    print("\n--- Test Simple Tokenizer & Regex Tokenizer (Basic Examples) ---")
    for i, sentence in enumerate(test_sentences):
        print(f"\n[S{i+1}] Original: '{sentence}'")

        simple_tokens = simple_tokenizer.tokenize(sentence)
        print(f"SimpleTokenizer: {simple_tokens}")

        regex_tokens = regex_tokenizer.tokenize(sentence)
        print(f"RegexTokenizer: {regex_tokens}")

    dataset_path = os.path.join(os.path.dirname(__file__), DATASET_BASE_PATH)
    raw_text = load_raw_text_data(dataset_path, DATASET_FILENAME)

    sample_text = raw_text[:500]
    print("\n" + "=" * 70)
    print("TASK 3: TOKENIZING SAMPLE TEXT FROM UD_ENGLISH-EWT")
    print("=" * 70)
    print(f"Original Sample (first 100 chars): '{sample_text[:100]}...'")

    simple_tokens_ud = simple_tokenizer.tokenize(sample_text)
    regex_tokens_ud = regex_tokenizer.tokenize(sample_text)

    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens_ud[:20]}")
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens_ud[:20]}")

    # Save token outputs
    with open(os.path.join(RESULTS_DIR, f"lab1_tokens_{timestamp}.txt"), "w", encoding="utf-8") as f:
        f.write("SimpleTokenizer:\n")
        f.write(" ".join(simple_tokens_ud) + "\n\n")
        f.write("RegexTokenizer:\n")
        f.write(" ".join(regex_tokens_ud) + "\n")

    return raw_text, simple_tokenizer, regex_tokenizer


def run_lab2_evaluation(raw_text: str, tokenizer: RegexTokenizer):
    """Performs Lab 2 evaluation: Count Vectorization test."""
    print("\n" + "=" * 70)
    print("LAB 2: COUNT VECTORIZATION EVALUATION")
    print("=" * 70)

    corpus_basic = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    vectorizer_basic = CountVectorizer(tokenizer=tokenizer)
    dtm_basic = vectorizer_basic.fit_transform(corpus_basic)

    print("\n--- Test CountVectorizer (Basic Corpus) ---")
    print("Corpus:", corpus_basic)
    print("\nLearned Vocabulary (Token: Index):")
    print(vectorizer_basic.vocabulary_)

    # Save vocabulary
    with open(os.path.join(RESULTS_DIR, f"lab2_vocab_basic_{timestamp}.txt"), "w", encoding="utf-8") as f:
        for token, idx in vectorizer_basic.vocabulary_.items():
            f.write(f"{token}: {idx}\n")

    print("\nDocument-Term Matrix (DTM) - Row format:")
    for doc_vector in dtm_basic:
        print(f"  {doc_vector}")

    print("\n" + "=" * 70)
    print("TASK 4: COUNT VECTORIZATION ON UD_ENGLISH-EWT SAMPLE")
    print("=" * 70)

    documents = [doc.strip() for doc in raw_text.split('\n\n') if doc.strip()][:5]
    if not documents:
        documents = [raw_text[i:i+100] for i in range(0, 300, 100) if raw_text[i:i+100]]

    print(f"Corpus size (Number of Documents): {len(documents)}")
    print(f"First document sample: '{documents[0][:50]}...'")

    vectorizer_ud = CountVectorizer(tokenizer=tokenizer)
    dtm_ud = vectorizer_ud.fit_transform(documents)

    print(f"\nLearned Vocabulary Size: {len(vectorizer_ud.vocabulary_)}")
    print(f"Sample Vocabulary (first 10 tokens): {dict(list(vectorizer_ud.vocabulary_.items())[:10])}")
    print(f"\nDocument-Term Matrix (DTM) Shape: {len(dtm_ud)} documents x {len(dtm_ud[0]) if dtm_ud else 0} features")

    # Save DTM results
    with open(os.path.join(RESULTS_DIR, f"lab2_dtm_{timestamp}.txt"), "w", encoding="utf-8") as f:
        for i, doc_vector in enumerate(dtm_ud):
            non_zero_entries = [f"{token}:{doc_vector[index]}" for token, index in vectorizer_ud.vocabulary_.items() if doc_vector[index] > 0]
            f.write(f"[D{i+1}] {non_zero_entries}\n")

    print("\nFirst 3 Document Vectors (DTM):")
    for i, doc_vector in enumerate(dtm_ud[:3]):
        non_zero_entries = [f"{token}:{doc_vector[index]}" for token, index in vectorizer_ud.vocabulary_.items() if doc_vector[index] > 0]
        print(f"  [D{i+1}] Non-Zero Counts ({len(non_zero_entries)} unique tokens): {non_zero_entries}")


if __name__ == '__main__':
    raw_text_data, simple_tok, regex_tok = run_lab1_evaluation()
    run_lab2_evaluation(raw_text_data, regex_tok)
    print(f"\nLogs saved to: {log_file_path}")
    print(f"Results saved in: {RESULTS_DIR}/")
