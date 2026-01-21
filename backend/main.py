from fastapi import FastAPI
from pydantic import BaseModel
from backend.ai_core import predict_ecg, predict_eeg, classify_neuro_state

app = FastAPI(title="AI Health Hub – ANN & DL Backend")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000", "http://192.168.1.4:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- REQUEST SCHEMAS ----------

class ECGRequest(BaseModel):
    ecg_signal: list  # length = 187


class EEGRequest(BaseModel):
    eeg_features: list  # length = 178


class NeuroStateRequest(BaseModel):
    ecg_signal: list
    eeg_features: list
    emotion: str


# ---------- API ENDPOINTS ----------

@app.post("/ecg/predict")
def ecg_predict(req: ECGRequest):
    if len(req.ecg_signal) != 187:
        return {
            "error": "ECG signal must contain exactly 187 values"
        }

    result = predict_ecg(req.ecg_signal)
    return {"ecg_prediction": result}


@app.post("/eeg/predict")
def eeg_predict(req: EEGRequest):
    if len(req.eeg_features) != 178:
        return {
            "error": "EEG features must contain exactly 178 values"
        }

    result = predict_eeg(req.eeg_features)
    return {"eeg_prediction": result}


@app.post("/neuro/state")
def neuro_state(req: NeuroStateRequest):

    if len(req.ecg_signal) != 187:
        return {"error": "ECG signal must contain exactly 187 values"}

    if len(req.eeg_features) != 178:
        return {"error": "EEG features must contain exactly 178 values"}

    ecg_result = predict_ecg(req.ecg_signal)
    eeg_result = predict_eeg(req.eeg_features)

    state = classify_neuro_state(
        ecg_result,
        eeg_result,
        req.emotion
    )

    return {
        "ecg": ecg_result,
        "eeg": eeg_result,
        "emotion": req.emotion,
        "final_neuro_state": state
    }

