import librosa
import numpy as np

def extract_mfcc(audio_bytes, sr=22050):
    # Load audio from bytes
    audio, _ = librosa.load(audio_bytes, sr=sr)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Take mean across time
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean.reshape(1, -1)
