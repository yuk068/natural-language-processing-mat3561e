# Spark NLP Pipeline (PySpark)

This repository contains a Python implementation of a Spark NLP Pipeline.
It mirrors the Scala assignment and supports tokenization, stopword removal, TF-IDF vectorization,
and optional Word2Vec and LogisticRegression stages.

---

## Requirements

* Conda environment with:

  * Python 3.10
  * pyspark (3.5.1 recommended)
  * loguru (optional, for richer logging)
  * numpy, pandas, matplotlib (optional, for analysis and visualization)

Install all recommended packages inside your conda environment:

```bash
pip install pyspark==3.5.1 numpy pandas matplotlib loguru jupyterlab ipykernel
```

---

## Project Structure

```
src/
├── data/                       # place c4-train.00000-of-01024-30K.json here
├── results/                    # outputs generated here
├── log/                        # metrics/logs generated here
├── pipeline.py                 # main Python script
└── README.md
```

---

## Implementation Steps

1. **Data Loading**:
   The script reads a gzipped JSON file from the C4 dataset (`c4-train.00000-of-01024-30K.json.gz`).
   A user-defined limit (default 1000) controls how many records are processed.

2. **Preprocessing**:

   * The text column is automatically detected and standardized to `text`.
   * Tokenization is performed using either `RegexTokenizer` (default) or `Tokenizer` (whitespace).
   * Stop words are removed via `StopWordsRemover`.

3. **Feature Extraction**:

   * Default: `HashingTF` + `IDF` to create TF-IDF vectors.
   * Alternative: `Word2Vec` (with configurable vector size).

4. **Model Extension**:

   * A toy `LogisticRegression` stage can be added.
   * Synthetic labels are generated for demonstration (replaceable with real labels).

5. **Output**:

   * Transformed rows (default 20) are written to `results/`.
   * Metrics (runtime, vocabulary size, config details) are written to `log/`.
   * Filenames are timestamped to avoid overwriting.

---

## How to Run

Activate your conda environment:

```bash
conda activate spark-nlp
```

Run the pipeline with default settings:

```bash
python pipeline.py --data data/c4-train.00000-of-01024-30K.json.gz
```

### Options

* `--limit` : limit number of records (default 1000)
* `--tokenizer` : `regex` (default) or `simple`
* `--num-features` : number of features for HashingTF (default 20000)
* `--use-word2vec` : use Word2Vec instead of TF-IDF
* `--word2vec-size` : Word2Vec vector size (default 100)
* `--add-logreg` : add a toy LogisticRegression stage
* `--results` : custom path for results
* `--log` : custom path for metrics log
* `--n-output` : number of transformed rows to write (default 20)

---

## Example Runs (Assignment + Exercises)

```bash
# 1. Default run (RegexTokenizer + HashingTF+IDF, 20k features)
python pipeline.py --data data/c4-train.00000-of-01024-30K.json.gz

# 2. Switch Tokenizer (Simple whitespace tokenizer)
python pipeline.py --data data/c4-train.00000-of-01024-30K.json.gz --tokenizer simple

# 3. Adjust Feature Vector Size (e.g., 1000 features)
python pipeline.py --data data/c4-train.00000-of-01024-30K.json.gz --num-features 1000

# 4. Extend Pipeline with LogisticRegression
python pipeline.py --data data/c4-train.00000-of-01024-30K.json.gz --add-logreg

# 5. Try Word2Vec (default vector size = 100)
python pipeline.py --data data/c4-train.00000-of-01024-30K.json.gz --use-word2vec

# 6. Word2Vec with custom vector size (200)
python pipeline.py --data data/c4-train.00000-of-01024-30K.json.gz --use-word2vec --word2vec-size 200
```

Each run creates timestamped outputs under `results/` and logs under `log/`.

---

## Results and Observations

* **Vocabulary size**: The script logs how many unique tokens remain after tokenization and stopword removal.
* **Feature size vs. vocab size**: With small `num-features` (e.g., 1000), hash collisions occur. Logs explicitly note this.
* **Word2Vec embeddings**: Useful for semantic representation; runtime is longer than TF-IDF.
* **LogisticRegression**: Works as a toy example with synthetic labels. Accuracy is random, but pipeline integration is verified.

Sample outputs can be inspected in `results/pipeline_output_<timestamp>.txt`.

---

## Difficulties Encountered

1. **Data size**: The C4 dataset is large; limiting to 1000 records avoids memory issues.
2. **PySpark setup**: Matching the correct Spark + Python + Java versions was required.
3. **Logging and overwriting**: Fixed by adding timestamped filenames.
4. **Synthetic labels**: LogisticRegression needed labels; solved with a synthetic labeling function.