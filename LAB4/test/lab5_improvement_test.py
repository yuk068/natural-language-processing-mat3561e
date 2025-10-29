"""
Lab 5: Task 4 - Model Improvement Experiment

This script compares the baseline LogisticRegression model
with an alternative model: Multinomial Naive Bayes.
"""

import numpy as np
from src.tokenization.tokenizer import RegexTokenizer
from src.vectorization.vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import json

# 1. Dataset from the PDF
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

# 2. Split the data (using the same split as lab5_test.py for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

# --- Baseline: Logistic Regression ---

# 3. Instantiate baseline components
tokenizer_lr = RegexTokenizer()
vectorizer_lr = TfidfVectorizer(tokenizer=tokenizer_lr)
classifier_lr = TextClassifier(vectorizer=vectorizer_lr)

# 4. Train the baseline
classifier_lr.fit(X_train, y_train)

# 5. Make predictions
y_pred_lr = classifier_lr.predict(X_test)

# 6. Evaluate
metrics_lr = classifier_lr.evaluate(y_test, y_pred_lr)

# 7. Print baseline results
print("--- Baseline Model (Logistic Regression) ---")
print(f"Test Data: {X_test}")
print(f"True Labels: {y_test}")
print(f"Predictions: {y_pred_lr.tolist()}")
print("Baseline Metrics:")
print(json.dumps(metrics_lr, indent=2))


# --- Improvement: Naive Bayes ---

# 3. Instantiate improvement components
# We must use the SAME tokenizer and vectorizer logic for a fair comparison
tokenizer_nb = RegexTokenizer()
vectorizer_nb = TfidfVectorizer(tokenizer=tokenizer_nb)

# 4. Train the vectorizer and transform data
# Naive Bayes model is not wrapped in our TextClassifier,
# so we perform the steps manually.
X_train_vec = vectorizer_nb.fit_transform(X_train)
X_test_vec = vectorizer_nb.transform(X_test)

# Instantiate the Multinomial Naive Bayes model
model_nb = MultinomialNB()

# 5. Train the model
model_nb.fit(X_train_vec, y_train)

# 6. Make predictions
y_pred_nb = model_nb.predict(X_test_vec)

# 7. Evaluate
# We can re-use the evaluate method from our baseline classifier
# since it's just a static helper.
metrics_nb = classifier_lr.evaluate(y_test, y_pred_nb)

# 8. Print improvement results
print("\n--- Improved Model (Naive Bayes) ---")
print(f"Predictions: {y_pred_nb.tolist()}")
print("Improved Metrics:")
print(json.dumps(metrics_nb, indent=2))
