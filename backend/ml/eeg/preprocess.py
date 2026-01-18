import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_eeg_data(path):
    df = pd.read_csv(path)

    # Drop index column if present
    if 'Unnamed' in df.columns[0]:
        df = df.iloc[:, 1:]

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert to binary classification
    y = np.where(y == 1, 1, 0)

    return X, y


def preprocess_eeg(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
