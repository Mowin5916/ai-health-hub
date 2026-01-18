from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_eeg_ann(input_dim):
    model = Sequential()

    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
