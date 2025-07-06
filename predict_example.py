# predict_example.py
from predict_emotion import predict_emotion

path = "ravdess/Actor_01/03-01-05-01-02-02-01.wav"
emotion = predict_emotion(path)
print("Predicted Emotion:", emotion)
