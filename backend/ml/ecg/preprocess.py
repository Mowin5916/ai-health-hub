import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_mitbih_data(path_train, path_test):
    train_df = pd.read_csv(path_train, header=None)
    test_df = pd.read_csv(path_test, header=None)

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    return X_train, X_test, y_train, y_test


def normalize_ecg(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def plot_ecg_sample(signal, label):
    plt.figure(figsize=(10, 3))
    plt.plot(signal)
    plt.title(f"ECG Heartbeat Sample – Class {label}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


# 🔽 THIS PART IS CRITICAL 🔽
if __name__ == "__main__":
    print("Loading ECG dataset...")

    X_train, X_test, y_train, y_test = load_mitbih_data(
        "backend/data/ecg/mitbih_train.csv",
        "backend/data/ecg/mitbih_test.csv"
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    X_train, X_test = normalize_ecg(X_train, X_test)

    print("Plotting ECG sample...")
    plot_ecg_sample(X_train[0], y_train[0])

    print("Done.")
