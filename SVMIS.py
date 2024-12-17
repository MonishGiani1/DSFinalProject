from transformers import DistilBertTokenizer, DistilBertModel
from bs4 import BeautifulSoup
import requests
import torch
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import joblib
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Set device (Detects if you have an NVIDIA card otherwise uses CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# Load the pre-trained SVM model
svm_model = joblib.load('svm_model.pkl')  # Replace with your model path

# Preprocessing function
def clean_text(text):
    """
    Clean and preprocess the input text.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Function to extract text from a URL
def extract_text_from_url(url):
    """
    Extracts the main content text from a news article URL.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        print(f"Error fetching or parsing the URL: {e}")
        return None

# Embedding generation function
def get_distilbert_embeddings(texts, max_len=128):
    """
    Generate DistilBERT embeddings for a list of texts.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Main script
def main():
    print("\nWelcome to the Fake News Detector!")
    url = input("Please enter the URL of a news article: ")

    # Extract article text
    print("\nFetching and processing the article...")
    article_text = extract_text_from_url(url)
    if not article_text:
        print("Could not extract text from the URL. Please try again.")
        return

    # Preprocess and generate embeddings
    cleaned_text = clean_text(article_text)
    print("\nCleaned Text:\n", cleaned_text[:500], "...")  # Show first 500 characters of cleaned text
    embeddings = get_distilbert_embeddings([cleaned_text])

    # Make prediction
    prediction = svm_model.predict(embeddings)[0]
    probability = svm_model.predict_proba(embeddings)[0][1]  # Probability of being fake

    # Display result
    print("\nPrediction Results:")
    print(f"Prediction: {'Fake News' if prediction == 1 else 'True News'}")
    print(f"Confidence: {round(probability if prediction == 1 else 1 - probability, 2) * 100:.2f}%")

# Run the script
main()
