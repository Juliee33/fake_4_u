"""
predict.py

This script loads the trained Linear SVM model and TF-IDF vectorizer
to perform inference (prediction) on new unseen news articles.

Purpose:
- Enable real-time fake news detection
- Serve as backend logic for the Streamlit application
- Ensure consistent preprocessing between training and deployment
- Provide prediction confidence for interpretability
"""

import os
import pickle
import numpy as np

# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------

# Get absolute project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_PATH = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODELS_PATH, "linear_svm.pkl")
VECTORIZER_FILE = os.path.join(MODELS_PATH, "vectorizer.pkl")


# ---------------------------------------------------------
# Load Model and Vectorizer
# ---------------------------------------------------------

def load_model():
    """
    Loads the trained Linear SVM model and TF-IDF vectorizer.

    Returns:
        tuple:
            model: Trained classifier
            vectorizer: Fitted TF-IDF transformer
    """
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def predict_news(text):
    """
    Predicts whether a news article is Real or Fake
    and returns a confidence score.

    Args:
        text (str): Raw news article text

    Returns:
        tuple:
            - str: Prediction label ("Real News" or "Fake News")
            - float: Confidence percentage (0–100%)
    """

    model, vectorizer = load_model()

    # Transform input text using trained TF-IDF
    text_vectorized = vectorizer.transform([text])

    # Predict class
    prediction = model.predict(text_vectorized)[0]

    # Get decision score from LinearSVC
    decision_score = model.decision_function(text_vectorized)[0]

    # Convert to probability using sigmoid function
    probability = 1 / (1 + np.exp(-decision_score))
    confidence = round(probability * 100, 2)

    if prediction == 1:
        return "Real News", confidence
    else:
        return "Fake News", confidence


# ---------------------------------------------------------
# Manual Testing (Command Line)
# ---------------------------------------------------------

if __name__ == "__main__":
    sample_text = input("Enter news text to classify:\n")

    prediction, confidence = predict_news(sample_text)

    print("\nPrediction:", prediction)
    print("Confidence:", confidence, "%")