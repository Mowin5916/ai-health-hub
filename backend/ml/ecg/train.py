import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_mitbih_data, normalize_ecg
from model import build_ecg_cnn
from tensorflow.keras.callbacks import EarlyStopping

# Load data
X_train, X_test, y_train, y_test = load_mitbih_data(
    "backend/data/ecg/mitbih_train.csv",
    "backend/data/ecg/mitbih_test.csv"
)

# Normalize
X_train, X_test = normalize_ecg(X_train, X_test)

# Reshape for CNN (samples, timesteps, channels)
X_train = X_train.reshape(-1, 187, 1)
X_test = X_test.reshape(-1, 187, 1)

# Build model
model = build_ecg_cnn(input_shape=(187, 1), num_classes=5)

# Train
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("backend/ml/ecg/saved_model/ecg_cnn_model.keras")

# Plot accuracy & loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("ECG CNN Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("ECG CNN Loss")

plt.show()
