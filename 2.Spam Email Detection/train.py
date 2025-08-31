"""
Spam Email Detection Model Training Script
==========================================

This script trains multiple machine learning models for spam email detection
using TF-IDF vectorization and text preprocessing.

Models trained:
- Naive Bayes
- Random Forest
- Logistic Regression

The best performing model is saved for web application use.
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
import joblib
import time
from tqdm import tqdm
from joblib import Parallel, delayed

n_jobs = max(1, int(os.cpu_count() * 0.9))
warnings.filterwarnings("ignore")

# sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Text Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Cache objects for optimization
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess_text_optimized(text):
    text = text.lower()
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Keep only alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]

    # Stem words
    tokens = [STEMMER.stem(word) for word in tokens]
    return " ".join(tokens)


def load_dataset():
    """Load and prepare the email dataset."""
    print("Loading dataset...")
    data = pd.read_csv("combined_data.csv")

    # Use text and label columns
    data = data[["text", "label"]].copy()

    print(f"Dataset loaded: {data.shape[0]} samples")
    print(f"Label distribution:\n{data['label'].value_counts()}")

    return data


def preprocess_dataset(data):
    """Preprocess the entire dataset."""
    print(f"Processing {len(data)} texts...")
    start_time = time.time()
    # Parallel preprocessing with progress bar
    data["cleaned_text"] = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_text_optimized)(text)
        for text in tqdm(data["text"], desc="Preprocessing")
    )
    end_time = time.time()
    print(f"✅ Preprocessing completed in {end_time - start_time:.2f} seconds!")
    print(
        f"average text length after preprocessing: {int(data['cleaned_text'].str.len().mean())}"
    )

    # Remove empty rows
    data = data[data["cleaned_text"].str.strip() != ""]
    print(f"Final dataset shape: {data.shape}")

    return data[["cleaned_text", "label"]]


def vectorize_text(data):
    """Create TF-IDF vectors from preprocessed text."""
    print("Creating TF-IDF vectors...")

    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=500,
        max_df=0.9,
        sublinear_tf=True,
    )

    # Fit and transform the cleaned emails
    X = tfidf_vectorizer.fit_transform(data["cleaned_text"])
    y = data["label"]

    print(f"TF-IDF Vectorization completed.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")

    return X, y, tfidf_vectorizer


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results."""
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    # Initialize models
    models = {
        "NaiveBayes": MultinomialNB(alpha=0.1),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=20, random_state=42, n_jobs=n_jobs
        ),
        "LogisticRegression": LogisticRegression(
            random_state=42, solver="liblinear", C=1.0, n_jobs=n_jobs
        ),
    }

    results = {}
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        start_time = time.time()

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        end_time = time.time()
        print(f"{model_name} trained in {end_time - start_time:.2f} seconds.")

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{model_name} Evaluation:")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        results[model_name] = f1
        trained_models[model_name] = model

    return results, trained_models


def save_best_model(results, trained_models, tfidf_vectorizer):
    """Save the best performing model with vectorizer."""
    # Find best model
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    best_f1_score = results[best_model_name]

    print(f"\n" + "=" * 60)
    print("BEST MODEL SELECTION")
    print("=" * 60)
    print(f"Best model: {best_model_name}")
    print(f"F1 Score: {best_f1_score:.4f}")

    # Create saved_models directory
    os.makedirs("saved_models", exist_ok=True)

    # Save model artifacts
    model_artifacts = {
        "model": best_model,
        "vectorizer": tfidf_vectorizer,
        "model_name": best_model_name,
        "f1_score": best_f1_score,
        "feature_names": tfidf_vectorizer.get_feature_names_out(),
    }

    joblib.dump(model_artifacts, "saved_models/spam_detection_model.pkl")
    print("✅ Model saved successfully to 'saved_models/spam_detection_model.pkl'")

    return best_model_name, best_f1_score


def main():
    """Main training pipeline."""
    print("🚀 Starting Spam Email Detection Model Training")
    print("=" * 60)

    # Load dataset
    data = load_dataset()

    # Preprocess dataset
    processed_data = preprocess_dataset(data)

    # Create TF-IDF vectors
    X, y, tfidf_vectorizer = vectorize_text(processed_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Train models
    results, trained_models = train_models(X_train, X_test, y_train, y_test)

    # Save best model
    best_model_name, best_f1_score = save_best_model(
        results, trained_models, tfidf_vectorizer
    )

    print(
        f"\n🎉 Training completed! Best model: {best_model_name} (F1: {best_f1_score:.4f})"
    )


if __name__ == "__main__":
    main()
