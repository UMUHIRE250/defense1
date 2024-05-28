import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data Collection
data = pd.read_csv('beacon_data.csv')

# Assume the last column is the room label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 2. Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert room labels to numerical values
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Definition
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Number of unique rooms
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Model Training
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 5. Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Save the model and the scaler
model.save('path_to_saved_model/my_model.keras')

# Save the scaler
import joblib
joblib.dump(scaler, 'path_to_saved_model/scaler.save')
