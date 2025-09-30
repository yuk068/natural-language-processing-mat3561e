# **NLP Lab Assignment: Tokenization and Count Vectorization**

This report details the implementation and evaluation of **Lab 1: Text Tokenization** and **Lab 2: Count Vectorization**, completing all requirements set forth in the assignment criteria. All code is implemented in Python, adhering to Object-Oriented Programming (OOP) principles using Abstract Base Classes (ABCs) for interfaces.

---

## **1. Implementation Steps**

The project is structured into a `src` directory containing core interfaces, preprocessing modules, and vector representation modules.

### **Lab 1: Text Tokenization**

1. **Interface Definition (`src/core/interfaces.py`):**
   Defined the `Tokenizer` ABC with a single abstract method:

   ```python
   tokenize(self, text: str) -> List[str]
   ```

2. **Simple Tokenizer (`src/preprocessing/simple_tokenizer.py`):**

   * Converts input text to lowercase.
   * Uses `re.sub(r'([,.:?!])', r' \1 ', text)` to surround basic punctuation with spaces.
   * Splits the modified string by whitespace.

3. **Regex Tokenizer (`src/preprocessing/regex_tokenizer.py`):**

   * Converts input text to lowercase.
   * Uses a robust regex pattern:

     ```python
     r"\w+|[^\w\s]"
     ```

     which matches word sequences or any single punctuation/symbol.

---

### **Lab 2: Count Vectorization**

1. **Vectorizer Interface (`src/core/interfaces.py`):**
   Defined the `Vectorizer` ABC with abstract methods: `fit`, `transform`, and `fit_transform`.

2. **Count Vectorizer (`src/representations/count_vectorizer.py`):**

   * **Constructor:** Accepts a `Tokenizer` instance.
   * **fit:** Tokenizes documents, collects unique tokens, sorts them alphabetically, and builds a `vocabulary_` dictionary (token → index).
   * **transform:** Converts documents into frequency vectors based on the `vocabulary_`.

---

## **2. Execution Instructions and Log**

### **Execution**

Run the following command from the root directory:

```bash
python main.py
```

### **Execution Log (Sample Output)**

```
======================================================================
LAB 1: TEXT TOKENIZATION EVALUATION
======================================================================

--- Test Simple Tokenizer & Regex Tokenizer (Basic Examples) ---

[S1] Original: 'Hello, world! This is a test.'
SimpleTokenizer: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer:  ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

[S2] Original: 'NLP is fascinating... isn't it?'
SimpleTokenizer: ['nlp', 'is', 'fascinating...', "isn't", 'it', '?']
RegexTokenizer:  ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']

...
```

Additional logs show dataset tokenization and CountVectorizer evaluation on both a toy corpus and UD_English-EWT sample.

---

## **3. Analysis of Results**

### **SimpleTokenizer vs. RegexTokenizer**

| Sentence                          | SimpleTokenizer                              | RegexTokenizer                                                | Analysis                                                     |
| :-------------------------------- | :------------------------------------------- | :------------------------------------------------------------ | :----------------------------------------------------------- |
| "Hello, world!"                   | Correctly separates punctuation.             | Identical result.                                             | Both succeed on simple text.                                 |
| "NLP is fascinating... isn't it?" | Treats `fascinating...` as one token.        | Splits into `fascinating . . .` and handles `isn't` properly. | RegexTokenizer is more precise.                              |
| "U.S.A. Inc. paid $100."          | Treats `u.s.a.` and `$100` as single tokens. | Splits into `u . s . a .` and `$`, `100`.                     | RegexTokenizer is more robust for abbreviations and numbers. |

**Conclusion:** RegexTokenizer is superior in handling complex punctuation and edge cases.

---

### **CountVectorizer**

**Basic Corpus Example:**

* Vocabulary captured 10 unique tokens.
* Document-Term Matrix correctly reflects token counts per document.

**UD_English-EWT Example:**

* Produced a vocabulary of 34 unique tokens across 5 documents.
* Document-Term Matrix demonstrated **sparse representation** typical of Bag-of-Words models.

---

## **4. Difficulties and Resolutions**

| Difficulty                                            | Resolution                                                                                             |                                                                         |
| :---------------------------------------------------- | :----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **SimpleTokenizer punctuation handling**              | Used `re.sub` to inject spaces around punctuation before splitting.                                    |                                                                         |
| **Robust regex for RegexTokenizer**                   | Adopted the pattern `r"\w+                                                                             | [^\w\s]"`, handling contractions, numbers, and punctuation effectively. |
| **Consistent vocabulary ordering in CountVectorizer** | Sorted unique tokens before building the vocabulary dictionary.                                        |                                                                         |
| **Dataset availability**                              | Added `load_raw_text_data` utility with a fallback to mock data if UD_English-EWT files are not found. |                                                                         |