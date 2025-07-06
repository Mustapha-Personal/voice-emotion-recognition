import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Load saved features
print("Loading dataset...")
emo_df = pd.read_csv('emotions_features.csv')

# Clean and prepare
emo_df = emo_df.fillna(0)
X = emo_df.drop(columns=['Emotion'])
Y = emo_df['Emotion']

# Sample only 1000 entries for now (adjust as needed)
X = X.sample(1000, random_state=42)
Y = Y.loc[X.index]

# Encode target
print("Encoding labels...")
encoder = OneHotEncoder()
Y_encoded = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# Split
print("Splitting dataset...")
x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42, shuffle=True)

# Scale
print("Scaling features...")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape for CNN
x_train_cnn = np.expand_dims(x_train, axis=2)
x_test_cnn = np.expand_dims(x_test, axis=2)

# Save scaler and encoder
joblib.dump(scaler, 'emotion_scaler.pkl')
joblib.dump(encoder, 'emotion_encoder.pkl')

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Define a small CNN
print("Building model...")
model = Sequential([
    L.Conv1D(64, kernel_size=5, padding='same', activation='relu', input_shape=(x_train_cnn.shape[1], 1)),
    L.BatchNormalization(),
    L.MaxPooling1D(pool_size=2),
    
    L.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    L.BatchNormalization(),
    L.MaxPooling1D(pool_size=2),
    
    L.Flatten(),
    L.Dense(64, activation='relu'),
    L.BatchNormalization(),
    L.Dropout(0.3),
    L.Dense(Y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
print("Training...")
history = model.fit(
    x_train_cnn, y_train,
    validation_data=(x_test_cnn, y_test),
    epochs=10, batch_size=64,
    callbacks=[early_stopping, lr_reduction, checkpoint],
    verbose=1
)

# Save model
model.save("cnn_model_full.keras")
print("✅ Training complete and model saved.")

# Save scaler and encoder
joblib.dump(scaler, 'emotion_scaler.pkl')
joblib.dump(encoder, 'emotion_encoder.pkl')

# Save settings for inference
inference_config = {
    'desired_length': x_train.shape[1],  # number of features per sample
    'frame_length': 2048,
    'hop_length': 512,
    'sample_rate': 22050,
    'duration': 2.5,
    'offset': 0.6
}
joblib.dump(inference_config, 'inference_config.pkl')
