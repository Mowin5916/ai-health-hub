import matplotlib.pyplot as plt
from preprocess import load_eeg_data, preprocess_eeg
from model import build_eeg_ann

# Load EEG data
X, y = load_eeg_data(
    "backend/data/eeg/Epileptic Seizure Recognition.csv"
)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_eeg(X, y)

# Build ANN model
model = build_eeg_ann(input_dim=X_train.shape[1])

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"EEG ANN Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("backend/ml/eeg/saved_model/eeg_ann_model.keras")

# Plot accuracy & loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("EEG ANN Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("EEG ANN Loss")

plt.show()
