# **Lab 5: Text Classification Report**

This document serves as the **report and analysis (Part 2)** of the Lab 5 assignment.
It details the implementation steps, provides a guide to running the code, and analyzes the results of the classification models.

---

## **Part 1: Implementation Steps**

To complete the assignment, the following files were created, corresponding to each task:

### **1. Prerequisites (from Labs 2 & 3)**

* `src/tokenization/tokenizer.py`
  Contains a `BaseTokenizer` abstract class and a `RegexTokenizer` implementation, required for the “Instantiate your RegexTokenizer” requirement.

* `src/vectorization/vectorizer.py`
  Contains a `BaseVectorizer` abstract class and a `TfidfVectorizer` implementation, built from scratch to simulate Lab 3’s component.
  It calculates **TF (Term Frequency)** and **IDF (Inverse Document Frequency)** to vectorize text.

---

### **2. Task 1 – `src/models/text_classifier.py`**

* Implements the `TextClassifier` class as specified.
* **Constructor (`__init__`)** accepts a vectorizer instance.
* **`fit` method** uses the vectorizer’s `fit_transform` method and trains a `sklearn.linear_model.LogisticRegression` model.
* **`predict` method** uses the vectorizer’s `transform` and the trained model’s `predict` method.
* **`evaluate` method** uses `sklearn.metrics` to calculate and return a dictionary containing:

  * accuracy
  * precision
  * recall
  * f1_score

---

### **3. Task 2 – `test/lab5_test.py`**

* Serves as the “Basic Test Case”.
* Imports `RegexTokenizer`, `TfidfVectorizer`, and `TextClassifier`.
* Defines the 6-sample dataset from the lab PDF.
* Splits data into 4-sample train / 2-sample test (`test_size=0.33` for stability).
* Instantiates, trains, and evaluates the `TextClassifier`, printing the resulting metrics.

---

### **4. Task 3 – Running the Spark Example**

* `data/sentiments.csv`: A small CSV file created to allow the Spark script to run.
  Contains **“text”** and **“sentiment” (-1/1)** columns.
* `test/lab5_spark_sentiment_analysis.py`:
  Reconstructed *exactly* from the code in `lab5_text_classification.pdf`.
  Builds a Spark ML Pipeline using:

  * `Tokenizer`
  * `StopWordsRemover`
  * `HashingTF`
  * `IDF`
  * `LogisticRegression`

---

### **5. Task 4 – Model Improvement Experiment**

* `test/lab5_improvement_test.py`: Demonstrates the model improvement experiment.
* **Improvement Choice:** Implemented a `MultinomialNB` (Naive Bayes) model, a common and effective text classification baseline.
* Runs the baseline Logistic Regression model and the Naive Bayes model on the same vectorized data for a **fair comparison**.
* Prints and compares their metrics.

---

## **Part 2: Code Execution Guide**

### **1. Setup**

Ensure you have Python 3 installed.
Then install the required libraries:

```bash
pip install -r requirements.txt
```

**Note for PySpark (Task 3):**
PySpark requires a Java Development Kit (JDK). Version **8** or **11** is recommended.
Ensure `JAVA_HOME` is properly set.

---

### **2. Task 2 – Run Baseline Model**

Runs the scikit-learn `TextClassifier` (Logistic Regression) on the small dataset:

```bash
python test/lab5_test.py
```

**Example Output:**

```
--- Baseline Model (Logistic Regression) ---
Test Data: ['What a waste of time, absolutely boring.', 'The acting was superb, a truly great experience.']
True Labels: [0, 1]
Predictions: [0, 1]

Baseline Metrics:
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0
}
```

---

### **3. Task 3 – Run Spark Example**

Runs the PySpark ML Pipeline on `data/sentiments.csv`:

```bash
python test/lab5_spark_sentiment_analysis.py
```

**Example Output (may vary slightly):**

```
Reading data from data/sentiments.csv
Data schema:
root
 |-- text: string (nullable = true)
 |-- sentiment: integer (nullable = true)

DataFrame with labels:
+--------------------+---------+-----+
|                text|sentiment|label|
+--------------------+---------+-----+
|I love Spark, it...|        1|  1.0|
|Spark is hard to...|       -1|  0.0|
|This is a great ...|        1|  1.0|
...

Training model...
Model trained. Evaluating...

Accuracy: 0.6667
F1-score: 0.6667

Spark session stopped.
```

---

### **4. Task 4 – Run Improvement Experiment**

Compares Logistic Regression (baseline) vs. Naive Bayes (improved):

```bash
python test/lab5_improvement_test.py
```

**Example Output:**

```
--- Baseline Model (Logistic Regression) ---
Predictions: [0, 1]
Baseline Metrics:
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0
}

--- Improved Model (Naive Bayes) ---
Predictions: [0, 1]
Improved Metrics:
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0
}
```

---

## **Part 3: Result Analysis**

### **Baseline Model (Logistic Regression)**

The baseline `TextClassifier` achieved perfect metrics (Accuracy: **1.0**, F1-score: **1.0**) on the small test set.

* **Test Data:**
  `['What a waste of time, absolutely boring.', 'The acting was superb, a truly great experience.']`
* **True Labels:** `[0, 1]`
* **Predictions:** `[0, 1]`

This perfect score is due to the tiny, clean dataset with clear sentiment cues.

---

### **Improved Model (Naive Bayes)**

The `MultinomialNB` model also achieved perfect metrics (Accuracy: **1.0**, F1-score: **1.0**).

---

### **Comparison and Analysis**

On this very small dataset, both models performed perfectly — the test is too trivial to show meaningful differences.

**Why Naive Bayes?**
`MultinomialNB` is efficient and probabilistic, ideal for TF-IDF or word-count features.
It computes the probability of a document belonging to a class based on the word probabilities.

**Theoretical Comparison:**

| Model                   | Description                                            | Strengths                                   |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------- |
| **Logistic Regression** | Linear model that separates classes with a hyperplane. | Strong, robust if word dependencies matter. |
| **Naive Bayes**         | Probabilistic model assuming word independence.        | Simple, fast, often strong baseline.        |

**Conclusion:**
Both models “solved” the toy problem.
On larger datasets, Logistic Regression may capture richer relationships, while Naive Bayes remains a fast, reliable baseline.

---

## **Part 4: Challenges and Solutions**

| **Challenge**                                                 | **Solution**                                                                                                                    |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Missing custom classes (`RegexTokenizer`, `TfidfVectorizer`). | Implemented simplified functional versions. The `TfidfVectorizer` computes TF and IDF mathematically for valid feature vectors. |
| Dataset too small (6 samples).                                | Used `test_size=0.33` (4 train / 2 test) with `random_state=42` for reproducibility and stability.                              |
| PySpark required data file and environment setup.             | Created `data/sentiments.csv` with 10 samples, added `pyspark` to requirements, and documented JDK setup.                       |