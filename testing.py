#testowanie modelu na randomowych nagraniach

from features import extract_logfbank_features, save_file, open_file
from tensorflow.keras.preprocessing.sequence import pad_sequences

audio_1 = 'C:/Users/magda/Desktop/praktyki/projekt/Duza_intonacja.wav'
feat = extract_logfbank_features(audio_1)
model_1 = open_file("model_1")

max_timesteps = open_file("max_timesteps")

#padding
feat_p = pad_sequences([feat], maxlen=max_timesteps)

#predykcja
audio_1_predict = model_1.predict(feat_p)
print(audio_1_predict)
