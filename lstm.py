import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from features import save_file, open_file
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

labels = open_file("labels")
features_list = open_file("features_list")

'''
for i, f in enumerate(features_list):
    if np.isnan(f).sum() > 0 or np.isinf(f).sum() > 0:
        print(f'NaN or Inf found in sample {i}')
'''

# tensoy
features_list_temp = [torch.from_numpy(f) for f in features_list]

scaler = StandardScaler()
scaled_features_list = []
#pominiecie probek nan
for f in features_list_temp:
    if np.isnan(f).sum() == 0 and np.isinf(f).sum() == 0:
        scaled_f = scaler.fit_transform(f)
        scaled_features_list.append(scaled_f)
    else:
        scaled_features_list.append(np.zeros_like(f))


features_list_temp = [torch.from_numpy(f) for f in scaled_features_list]

# padding
padded_features = pad_sequence(features_list_temp, batch_first=True)

X = np.array(padded_features)
y = np.array(labels)

'''
print(f'Shape of X: {X.shape}')  
print(f'Shape of y: {y.shape}')  

print(np.isnan(X).sum(), np.isinf(X).sum())
print(np.isnan(y).sum(), np.isinf(y).sum())

print("Range of X values:", X.min(), X.max())
print("Range of y values:", y.min(), y.max())

'''
# skalowaniee
scaler = StandardScaler()
num_samples, max_timesteps, num_features = X.shape
X = X.reshape(-1, num_features)
X = scaler.fit_transform(X)
X = X.reshape(num_samples, max_timesteps, num_features)
save_file("scaler", scaler)

# model
model = Sequential()
model.add(LSTM(64, input_shape=(max_timesteps, num_features), return_sequences=True))
model.add(Dropout(0.2))
#1 hidden with 128 units, dropout 20%
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# trening
history = model.fit(X, y, epochs=50, batch_size=32)

#loss, mae = model.evaluate(X, y)
#print(f'Loss: {loss}, MAE: {mae}')

save_file("model_1", model)
