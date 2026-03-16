"""
predict_distilbert.py

Prediction module for the DistilBERT Fake News Detection model.

This script loads the trained transformer model and performs
inference on input text.

Returns:
    prediction (str)  -> "Fake News" or "Real News"
    confidence (float) -> probability score from the model
"""

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------
# The trained model folder should contain:
# config.json
# pytorch_model.bin
# tokenizer files

# Dynamically find the correct project path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Build path to the model folder
model_path = os.path.join(BASE_DIR, "models", "distilbert_fake_news_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load trained DistilBERT classification model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Put model into evaluation mode (important for inference)
model.eval()


# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def predict_fake_news(text):
    """
    Predict whether a news article is Fake or Real using DistilBERT.

    Parameters
    ----------
    text : str
        Input news article or headline.

    Returns
    -------
    prediction : str
        "Fake News" or "Real News"

    confidence : float
        Model probability score (0-1)
    """

    # Tokenize input text for the transformer model
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Disable gradient calculation (faster inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits (raw model outputs)
    logits = outputs.logits

    # Convert logits → probabilities using softmax
    probabilities = torch.softmax(logits, dim=1)

    # Get highest probability
    confidence = torch.max(probabilities).item()

    # Get predicted class index
    prediction = torch.argmax(probabilities).item()

    # Convert class index to label
    if prediction == 0:
        return "Fake News", confidence
    else:
        return "Real News", confidence