#!/usr/bin/env python3
"""
Spam Email Detection Web Application
====================================

FastAPI-based web application for spam email detection with Persian UI.
Uses trained machine learning models to classify emails as spam or legitimate.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import re
import numpy as np
from typing import Dict, Any

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Cache NLTK objects
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# Initialize FastAPI app
app = FastAPI(
    title="سیستم تشخیص ایمیل اسپم",
    description="سیستم هوشمند تشخیص ایمیل‌های مشکوک و اسپم",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models
model_artifacts = None

def load_model():
    """Load the trained spam detection model."""
    global model_artifacts
    
    model_path = os.path.join("saved_models", "spam_detection_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model_artifacts = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        print(f"Model: {model_artifacts['model_name']}")
        print(f"F1 Score: {model_artifacts['f1_score']:.4f}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_text(text: str) -> str:
    """
    Preprocess email text for prediction.
    
    Args:
        text (str): Raw email text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert to lowercase
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

# Pydantic models
class EmailInput(BaseModel):
    email_content: str = Field(..., min_length=10, max_length=10000, description="Email content to analyze")
    
    @validator('email_content')
    def validate_email_content(cls, v):
        if not v.strip():
            raise ValueError('Email content cannot be empty')
        if len(v.strip()) < 10:
            raise ValueError('Email content too short (minimum 10 characters)')
        return v.strip()

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_name: str
    is_spam: bool
    message: str

# API Routes
@app.get("/")
async def serve_ui():
    """Serve the main Persian UI."""
    return FileResponse("static/spam_detection_ui.html")

@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(email_input: EmailInput) -> PredictionResponse:
    """
    Predict if an email is spam or legitimate.
    
    Args:
        email_input: EmailInput containing the email content
        
    Returns:
        PredictionResponse with prediction results
    """
    if model_artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess the email text
        cleaned_text = preprocess_text(email_input.email_content)
        
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Email content is empty after preprocessing")
        
        # Vectorize using the trained TF-IDF vectorizer
        text_vector = model_artifacts["vectorizer"].transform([cleaned_text])
        
        # Get prediction
        prediction = model_artifacts["model"].predict(text_vector)[0]
        
        # Get prediction probabilities
        if hasattr(model_artifacts["model"], "predict_proba"):
            probabilities = model_artifacts["model"].predict_proba(text_vector)[0]
            confidence = float(max(probabilities))
        else:
            # For models without predict_proba (like some Naive Bayes)
            decision_scores = model_artifacts["model"].decision_function(text_vector)[0] if hasattr(model_artifacts["model"], "decision_function") else 0.85
            confidence = float(1 / (1 + np.exp(-abs(decision_scores))))  # Sigmoid transformation
        
        # Convert prediction to human-readable format
        is_spam = bool(prediction == 1)
        prediction_text = "اسپم" if is_spam else "معتبر"
        
        # Create appropriate message
        if is_spam:
            message = "این ایمیل اسپم تشخیص داده شد"
        else:
            message = "این ایمیل معتبر تشخیص داده شد"
        
        return PredictionResponse(
            prediction=prediction_text,
            confidence=confidence,
            model_name=model_artifacts["model_name"],
            is_spam=is_spam,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    if model_artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": model_artifacts["model_name"],
        "f1_score": model_artifacts["f1_score"],
        "num_features": len(model_artifacts["feature_names"]),
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_artifacts is not None,
        "version": "1.0.0"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    print("🚀 Starting Spam Email Detection Web Application...")
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Load the trained model
    success = load_model()
    if not success:
        print("❌ Failed to load model. Please run train.py first!")
        # Don't exit here, let the app start but API calls will fail
    else:
        print("✅ Application ready!")

if __name__ == "__main__":
    print("Starting development server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["static"]
    )