import numpy as np
import joblib
from preprocess import extract_mfcc

# Load pretrained ANN model
emotion_model = joblib.load("backend/ml/speech/saved_model/speech_emotion_ann.pkl")

EMOTION_LABELS = ["Calm", "Happy", "Angry", "Sad", "Fear"]

def predict_emotion(audio_bytes):
    features = extract_mfcc(audio_bytes)
    prediction = emotion_model.predict(features)[0]
    return EMOTION_LABELS[prediction]
