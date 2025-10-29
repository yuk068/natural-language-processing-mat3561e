"""
Lab 5: Task 1 - TextClassifier Implementation
"""

from typing import List, Dict, Any
from src.vectorization.vectorizer import BaseVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class TextClassifier:
    """
    A wrapper class for a scikit-learn text classification pipeline.
    
    This class handles vectorization and model training/prediction.
    """

    def __init__(self, vectorizer: BaseVectorizer):
        """
        Initializes the classifier.
        
        Args:
            vectorizer (BaseVectorizer): An instance of a fitted or unfitted
                                         vectorizer (e.g., TfidfVectorizer).
        """
        self.vectorizer = vectorizer
        # Initialize the Logistic Regression model
        # Using 'liblinear' as it's good for small datasets
        self._model = LogisticRegression(solver='liblinear', random_state=42)

    def fit(self, texts: List[str], labels: List[int]):
        """
        Trains the classifier.
        
        This method will:
        1. Fit the vectorizer to the texts and transform them into a feature matrix X.
        2. Train the LogisticRegression model on X and the provided labels.
        
        Args:
            texts (List[str]): A list of training documents.
            labels (List[int]): A list of corresponding training labels.
        """
        print("Fitting vectorizer and training model...")
        # Use the vectorizer to fit and transform the input texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train the model
        self._model.fit(X, labels)
        print("Model training complete.")

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Makes predictions on new texts.
        
        This method will:
        1. Use the *already fitted* vectorizer to transform the texts into a
           feature matrix X.
        2. Use the trained model to predict labels for X.
        
        Args:
            texts (List[str]): A list of new documents to classify.
            
        Returns:
            np.ndarray: A numpy array of predicted labels.
        """
        # Use the vectorizer to transform (NOT fit_transform) the input texts
        X = self.vectorizer.transform(texts)
        
        # Use the trained model to predict
        predictions = self._model.predict(X)
        return predictions

    def evaluate(self, y_true: List[int], y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates evaluation metrics for the model's predictions.
        
        Args:
            y_true (List[int]): The true (ground truth) labels.
            y_pred (np.ndarray): The labels predicted by the model.
            
        Returns:
            Dict[str, float]: A dictionary containing accuracy, precision,
                              recall, and f1-score.
        """
        # Calculate metrics
        # Use zero_division=0 to handle cases with no positive predictions
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        return metrics

if __name__ == '__main__':
    # Example usage (as a simple test)
    from src.tokenization.tokenizer import RegexTokenizer
    from src.vectorization.vectorizer import TfidfVectorizer

    # 1. Dataset
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring."
    ]
    labels = [1, 0, 1, 0] # 1 for positive, 0 for negative

    # 2. Instantiate components
    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    
    # 3. Instantiate classifier
    classifier = TextClassifier(vectorizer=vectorizer)
    
    # 4. Train
    classifier.fit(texts, labels)
    
    # 5. Predict (on new data)
    test_texts = [
        "I really loved this, amazing!",
        "So boring I fell asleep."
    ]
    predictions = classifier.predict(test_texts)
    
    print(f"\nTest texts: {test_texts}")
    print(f"Predictions: {predictions}") # Expected: [1, 0]
    
    # 6. Evaluate (on training data just for this example)
    train_preds = classifier.predict(texts)
    metrics = classifier.evaluate(labels, train_preds)
    print(f"\nTraining Metrics (for demonstration):\n{metrics}")
