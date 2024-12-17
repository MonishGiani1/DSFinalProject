import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm.auto import tqdm
import joblib

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

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

# Set device (Detects if you have an NVIDIA card otherwise uses CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DistilBERT-based embedding generation
def get_distilbert_embeddings_batchwise(texts, batch_size=16, max_len=128):
    """
    Generate DistilBERT embeddings in batches for a list of texts.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating DistilBERT Embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

# Load datasets
true_file = 'True.csv'
fake_file = 'Fake.csv'
fake_df = pd.read_csv(fake_file)
true_df = pd.read_csv(true_file)

# Add labels: 1 for Fake, 0 for True
fake_df['label'] = 1
true_df['label'] = 0

# Combine datasets
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the combined dataset
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Apply preprocessing to the text column
shuffled_df['cleaned_text'] = shuffled_df['text'].apply(clean_text)

# Select only the cleaned text and label columns
processed_df = shuffled_df[['cleaned_text', 'label']]

# Split data into training and testing sets
train_data, test_data = train_test_split(processed_df, test_size=0.2, random_state=42)

# Get DistilBERT embeddings for training and testing data
X_train = get_distilbert_embeddings_batchwise(train_data["cleaned_text"].tolist(), batch_size=16, max_len=128)
X_test = get_distilbert_embeddings_batchwise(test_data["cleaned_text"].tolist(), batch_size=16, max_len=128)
y_train = train_data["label"].values
y_test = test_data["label"].values

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest model
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, rf_probabilities)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='b')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

joblib.dump(rf_model, 'random_forest_model.pkl')

