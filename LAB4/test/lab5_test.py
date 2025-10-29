"""
Lab 5: Task 2 - Basic Test Case
"""

import numpy as np
from src.tokenization.tokenizer import RegexTokenizer
from src.vectorization.vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split
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

print(f"Total dataset size: {len(texts)} samples")

# 2. Split the data
# The PDF suggests 80/20, but for 6 samples, that's 1.2 test samples.
# Let's use test_size=0.33 (2 samples) for a more stable test.
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 3. Instantiate components
tokenizer = RegexTokenizer()
vectorizer = TfidfVectorizer(tokenizer=tokenizer)

# 4. Instantiate TextClassifier
classifier = TextClassifier(vectorizer=vectorizer)

# 5. Train the classifier
classifier.fit(X_train, y_train)

# 6. Make predictions on the test data
y_pred = classifier.predict(X_test)

# 7. Evaluate the predictions
metrics = classifier.evaluate(y_test, y_pred)

# 8. Print the results
print("\n--- Baseline Model (Logistic Regression) ---")
print(f"Test Data: {X_test}")
print(f"True Labels: {y_test}")
print(f"Predictions: {y_pred.tolist()}")

print("\nBaseline Metrics:")
# Pretty-print the dictionary
print(json.dumps(metrics, indent=2))
