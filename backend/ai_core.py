# backend/ai_core.py

import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------- ECG MODEL ----------------
ecg_model = tf.keras.models.load_model(
    "backend/ml/ecg/saved_model/ecg_cnn_model.keras"
)

ECG_LABELS = [
    "Normal",
    "Supraventricular",
    "Ventricular",
    "Fusion",
    "Unknown"
]

def predict_ecg(ecg_signal):
    """
    ecg_signal: list or numpy array of length 187
    """
    ecg_signal = np.array(ecg_signal).reshape(1, 187, 1)
    preds = ecg_model.predict(ecg_signal)
    return ECG_LABELS[np.argmax(preds)]


# ---------------- EEG MODEL ----------------
eeg_model = tf.keras.models.load_model(
    "backend/ml/eeg/saved_model/eeg_ann_model.keras"
)

def predict_eeg(eeg_features):
    """
    eeg_features: list or numpy array of length 178
    """
    eeg_features = np.array(eeg_features).reshape(1, -1)
    prob = eeg_model.predict(eeg_features)[0][0]
    return "Seizure" if prob > 0.5 else "Normal"


# ---------------- SPEECH EMOTION MODEL ----------------
# Lightweight ANN-based emotion classifier (placeholder logic)
# You can later replace this with a loaded joblib model

def recognize_speech_emotion(audio_bytes):
    """
    audio_bytes: raw audio bytes
    """
    # SAFE, STABLE default (no crash)
    return "Calm"


# ---------------- MULTIMODAL FUSION ----------------
def classify_neuro_state(ecg_result, eeg_result, emotion):
    """
    Simple rule-based fusion (exam-friendly)
    """
    if eeg_result == "Seizure":
        return "Critical"
    if ecg_result != "Normal":
        return "At Risk"
    if emotion in ["Fear", "Angry", "Sad"]:
        return "Anxious"
    return "Stable"
