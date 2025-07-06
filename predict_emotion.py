import numpy as np
import librosa
import joblib
from tensorflow import keras

# Load model and preprocessors
MODEL = keras.models.load_model("cnn_model_full.keras")
SCALER = joblib.load('emotion_scaler.pkl')
ENCODER = joblib.load('emotion_encoder.pkl')
CONFIG = joblib.load('inference_config.pkl')

# Extract config values
FRAME_LENGTH = CONFIG['frame_length']
HOP_LENGTH = CONFIG['hop_length']
SAMPLE_RATE = CONFIG['sample_rate']
DESIRED_LENGTH = CONFIG['desired_length']
DURATION = CONFIG['duration']
OFFSET = CONFIG['offset']

# Feature extractors
def zcr(data):
    return np.squeeze(librosa.feature.zero_crossing_rate(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))

def rmse(data):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH))

def mfcc(data):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    return np.ravel(mfcc_feature.T)

def extract_features(data):
    return np.hstack((zcr(data), rmse(data), mfcc(data)))

# Get feature vector for a file
def get_predict_feat(path):
    data, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET)
    feat = extract_features(data)

    # Pad/truncate
    if len(feat) > DESIRED_LENGTH:
        feat = feat[:DESIRED_LENGTH]
    else:
        feat = np.pad(feat, (0, DESIRED_LENGTH - len(feat)), 'constant')

    feat = SCALER.transform([feat])
    feat = np.expand_dims(feat, axis=2)
    return feat

# Predict emotion
def predict_emotion(path):
    feat = get_predict_feat(path)
    pred = MODEL.predict(feat)
    label = ENCODER.inverse_transform(pred)
    return label[0][0]
