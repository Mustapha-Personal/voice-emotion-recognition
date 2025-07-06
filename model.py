
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import audio access libraries
import librosa
import librosa.display
import IPython.display as ipd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import timeit
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization , GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as L


import joblib
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

ravdess_path = 'ravdess/audio_speech_actors/'
speech_signal_path = 'emotions'
cremad_path = 'AudioWAV/'
savee_path = 'ALL/'
toronto_path = 'toronto/speeches/'

ravdess_ls = os.listdir(ravdess_path)
print(ravdess_ls)

file_emotion = []
file_path = []
c = 0
for i in ravdess_ls:
    # as their are 24 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(ravdess_path  + i)
    for f in actor:
        part = f.split('.')[0].split('-')
# third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(ravdess_path + i + '/' + f)

emotions = pd.DataFrame(file_emotion, columns=['Emotions'])
# dataframe for path of files.
paths = pd.DataFrame(file_path, columns=['Path'])
ravdess_df = pd.concat([paths, emotions], axis=1)
# changing integers to actual emotions.
ravdess_df.Emotions.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust',
                             8:'surprise'},
                            inplace=True)

print(ravdess_df.head())


crema_ls = os.listdir(cremad_path)

file_emotion = []
file_path = []

for file in crema_ls:
    # storing file paths
    file_path.append(cremad_path+ file)
    # storing file emotions
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
cremad_df = pd.concat([path_df, emotion_df], axis=1)
print(cremad_df.head())


tess_ls = os.listdir(toronto_path)

file_emotion = []
file_path = []

for dir in tess_ls:
    directories = os.listdir(toronto_path + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(toronto_path + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
tess_df = pd.concat([path_df, emotion_df], axis=1)
print(tess_df.head())



savee_ls = os.listdir(savee_path)

file_emotion = []
file_path = []

for file in savee_ls:
    file_path.append(savee_path+ file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
savee_df = pd.concat([ path_df,emotion_df], axis=1)
print(savee_df.head())

# creating Dataframe using all the 4 dataframes we created so far.
concate = pd.concat([ravdess_df, cremad_df, tess_df, savee_df], axis = 0)
concate.to_csv("data_path.csv",index=False)
print(concate.head())

print(concate.Emotions.value_counts())


# Get the counts
emotion_counts = concate.Emotions.value_counts()

# Create a bar plot
sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
# Add labels and title
plt.xlabel('Emotions')
plt.ylabel('Counts')
plt.title('Distribution of Emotions')
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.show()


# Instead of using file_path[0], use concate['Path'].values[0]
file_path = concate['Path'].values[0]
label = concate['Emotions'].values[0]
# Then load
data, sr = librosa.load(file_path)
print(sr)
print(label)

ipd.Audio(data,rate=sr)

# CREATE LOG MEL SPECTROGRAM
plt.figure(figsize=(10, 5))
spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128,fmax=8000)
log_spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, y_axis='mel', sr=sr, x_axis='time')
plt.title('Mel Spectrogram ')
plt.colorbar(format='%+2.0f dB')


mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)

# MFCC
plt.figure(figsize=(16, 10))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()


# NOISE
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# STRETCH
def stretch(data, rate=0.8):
    return librosa.resample(data, orig_sr=sr, target_sr=int(sr * rate))

# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


# NORMAL AUDIO
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y=data, sr=sr)
ipd.Audio(data,rate=sr)

# AUDIO WITH NOISE
x = stretch(data)
plt.figure(figsize=(12,5))
librosa.display.waveshow(y=x, sr=sr)
ipd.Audio(x, rate=sr)


def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])

    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_features(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_features(data)
    audio=np.array(aud)

    noised_audio=noise(data)
    aud2=extract_features(noised_audio)
    audio=np.vstack((audio,aud2))

    pitched_audio=pitch(data,sr)
    aud3=extract_features(pitched_audio)
    audio=np.vstack((audio,aud3))

    pitched_audio1=pitch(data,sr)
    pitched_noised_audio=noise(pitched_audio1)
    aud4=extract_features(pitched_noised_audio)
    audio=np.vstack((audio,aud4))

    return audio


start = timeit.default_timer()
# Define a function to get features for a single audio file
def process_feature(path, emotion):
    features = get_features(path)
    X = []
    Y = []
    for ele in features:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)
    return X, Y

paths = concate.Path
emotions = concate.Emotions

# Run the loop in parallel
results = Parallel(n_jobs=-1)(delayed(process_feature)(path, emotion) for (path, emotion) in zip(paths, emotions))

# Collect the results
X = []
Y = []
for result in results:
    x, y = result
    X.extend(x)
    Y.extend(y)


stop = timeit.default_timer()

print('Time: ', stop - start)

features = pd.DataFrame(X)
features['Emotion'] = Y
features.to_csv('emotions_features.csv', index=False)
print(features.head())

emo_df = pd.read_csv('emotions_features.csv')
print(emo_df.head())

print(emo_df.isna().sum())

emo_df = emo_df.fillna(0)
emo_df.isna().sum()


X = emo_df.drop(columns = ['Emotion'])
Y = emo_df['Emotion']



en = OneHotEncoder()
Y = en.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state = 42 , test_size = 0.2 , shuffle = True)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#1. early stopping
early_stopping = EarlyStopping(monitor='val_accuracy',mode='max',patience=5,restore_best_weights=True)
#2. ReduceLROnPlateau
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
#3. ModelCheckpoint
model_checkpoint = ModelCheckpoint('best_model1_weights.keras', monitor='val_accuracy', save_best_only=True)


x_train_cnn = np.expand_dims(x_train, axis=2)
x_test_cnn = np.expand_dims(x_test, axis=2)

model = tf.keras.Sequential([
    L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(x_train_cnn.shape[1],1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),

    L.Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    Dropout(0.2),  # Add dropout layer after the second max pooling layer

    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),

    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    Dropout(0.2),  # Add dropout layer after the fourth max pooling layer

    L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    Dropout(0.2),  # Add dropout layer after the fifth max pooling layer

    L.Flatten(),
    L.Dense(512,activation='relu'),
    L.BatchNormalization(),
    L.Dense(7,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

history=model.fit(x_train_cnn, y_train, epochs=40, validation_data=(x_test_cnn, y_test), batch_size=64,callbacks=[early_stopping,lr_reduction,model_checkpoint])


model.save_weights("cnn_model.weights.h5")

print('model evaluation accuracy: ',model.evaluate(x_test_cnn , y_test))

# epochs = [i for i in range(40)]
# fig , ax = plt.subplots(1,2)
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# test_acc = history.history['val_accuracy']
# test_loss = history.history['val_loss']

# fig.set_size_inches(15,6)
# ax[0].plot(epochs , train_loss , label = 'Training Loss')
# ax[0].plot(epochs , test_loss , label = 'Testing Loss')
# ax[0].set_title('Training & Testing Loss')
# ax[0].legend()
# ax[0].set_xlabel("Epochs")

# ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
# ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
# ax[1].set_title('Training & Testing Accuracy')
# ax[1].legend()
# ax[1].set_xlabel("Epochs")
# plt.show()

# inference on test data:
pred = model.predict(x_test_cnn)
y_pred = en.inverse_transform(pred)
y_true = en.inverse_transform(y_test)

df_results = pd.DataFrame({
    'True Label': y_true.flatten(),
    'Predicted Label': y_pred.flatten()
})

df_results.head(15)

model.save("cnn_model_full.keras")  # saves architecture + weights + optimizer state

joblib.dump(scaler, 'emotion_scaler.pkl') #save scaler
joblib.dump(en, 'emotion_encoder.pkl')    #save encoder


model = load_model("cnn_model_full.keras")
print('Done')








# proper usage

import numpy as np
import librosa
import joblib
from tensorflow import keras

MODEL = keras.models.load_model("cnn_model_full.keras")
SCALER = joblib.load('emotion_scaler.pkl')
ENCODER = joblib.load('emotion_encoder.pkl')

def zcr(data, frame_length, hop_length):
    zcr_feature = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_feature)

def rmse(data, frame_length=2048, hop_length=512):
    rmse_feature = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_feature)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)


def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    
    desired_length = 2376
    current_length = len(res)
    
    if current_length > desired_length:
        # Truncate
        res = res[:desired_length]
    elif current_length < desired_length:
        # Pad with zeros
        res = np.pad(res, (0, desired_length - current_length), 'constant')
    
    result = np.reshape(res, (1, desired_length))
    i_result = SCALER.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    
    return final_result

def prediction(path1):
    res=get_predict_feat(path1)
    predictions=MODEL.predict(res)
    y_pred = ENCODER.inverse_transform(predictions)
    print(y_pred[0][0]) 


prediction("ravdess/Actor_01/03-01-05-01-02-02-01.wav")