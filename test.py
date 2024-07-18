import numpy as np
from features import save_file, open_file, find_audio_files, extract_logfbank_features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from math import sqrt
from features import save_file, open_file
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


labels_test = open_file("labels_test")
features_list_test = open_file("features_list_test")
#print(labels_test)
#print(features_list_test)

#usuwanie próbek nan
features_test_new = []
labels_test_new = []
for i, f in enumerate(features_list_test):
    if not np.isnan(f).sum() > 0 or np.isinf(f).sum() > 0:
        features_test_new.append(f)
        labels_test_new.append((labels_test[i]))


# tens
features_list_temp_test = [torch.from_numpy(f) for f in features_test_new]

scaler = StandardScaler()
scaled_features_list_test = []
for f in features_list_temp_test:
    if np.isnan(f).sum() == 0 and np.isinf(f).sum() == 0:
        scaled_f = scaler.fit_transform(f)
        scaled_features_list_test.append(scaled_f)
    else:
        #print(f'Skipping sample with NaN or Inf values')
        scaled_features_list_test.append(np.zeros_like(f))

features_list_temp_test = [torch.from_numpy(f) for f in scaled_features_list_test]

# ppading
padded_features_test = pad_sequence(features_list_temp_test, batch_first=True)

X_test = np.array(padded_features_test)
y_test = np.array(labels_test_new)

# skalowanie
scaler = open_file("scaler")  #skaler ten który użyty został do treningu
num_samples, max_timesteps, num_features = X_test.shape
X_test = X_test.reshape(-1, num_features)
X_test = scaler.transform(X_test)  # skalowanie
X_test = X_test.reshape(num_samples, max_timesteps, num_features)
#print(f'Range of X_test values: {np.min(X_test)} to {np.max(X_test)}')

model_1 = open_file("model_1")

# Prognozy na zbiorze testowym
y_pred = model_1.predict(X_test)
y_pred = y_pred.flatten()

# Ocena modelu na zbiorze testowym
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rms = sqrt(mean_squared_error(y_test, y_pred))

print(f'Test MAE: {mae}')
print(f'Test MSE: {mse}')
print(f'Test RMSE: {rms}')

