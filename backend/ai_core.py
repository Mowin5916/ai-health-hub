# backend/ai_core.py

import os
import numpy as np
import tensorflow as tf

# Reduce TensorFlow log noise (important for Render)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------------------------------
# SAFE MODEL LOADER
# -------------------------------------------------
def safe_load_model(path: str):
    """
    Loads a TensorFlow/Keras model only if it exists.
    Prevents Render deployment crashes.
    """
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            print(f"[WARN] Failed to load model at {path}: {e}")
            return None
    else:
        print(f"[INFO] Model not found at {path}, running without it.")
        return None


# -------------------------------------------------
# ECG MODEL (CNN)
# -------------------------------------------------
ECG_MODEL_PATH = "backend/ml/ecg/saved_model/ecg_cnn_model.keras"

ecg_model = safe_load_model(ECG_MODEL_PATH)

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
    if ecg_model is None:
        return "Model not loaded"

    ecg_signal = np.array(ecg_signal, dtype=np.float32).reshape(1, 187, 1)
    preds = ecg_model.predict(ecg_signal, verbose=0)
    return ECG_LABELS[np.argmax(preds)]


# -------------------------------------------------
# EEG MODEL (ANN / MLP)
# -------------------------------------------------
EEG_MODEL_PATH = "backend/ml/eeg/saved_model/eeg_ann_model.keras"

eeg_model = safe_load_model(EEG_MODEL_PATH)

def predict_eeg(eeg_features):
    eeg_features = np.array(eeg_features, dtype=np.float32)

    # 🔑 NORMALIZATION (CRITICAL FIX)
    eeg_features = (eeg_features - np.mean(eeg_features)) / (np.std(eeg_features) + 1e-6)

    eeg_features = eeg_features.reshape(1, -1)

    prob = eeg_model.predict(eeg_features, verbose=0)[0][0]

    return "Seizure" if prob > 0.5 else "Normal"


# -------------------------------------------------
# SPEECH EMOTION (STABLE PLACEHOLDER)
# -------------------------------------------------
def recognize_speech_emotion(audio_bytes):
    """
    audio_bytes: raw audio bytes
    For now, return a safe default to avoid crashes.
    """
    return "Calm"


# -------------------------------------------------
# MULTIMODAL FUSION LOGIC
# -------------------------------------------------
def classify_neuro_state(ecg_result, eeg_result, emotion):
    """
    Rule-based fusion of ECG, EEG, and speech emotion.
    """
    if eeg_result == "Seizure":
        return "Critical"

    if ecg_result not in ["Normal", "Model not loaded"]:
        return "At Risk"

    if emotion in ["Fear", "Angry", "Sad"]:
        return "Anxious"

    return "Stable"
