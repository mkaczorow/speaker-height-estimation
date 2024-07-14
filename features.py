from python_speech_features import logfbank
import os
import pickle
import soundfile as sf
import numpy as np
import librosa
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

import crepe
import tensorflow

#Functions for saving files
def save_file(filename, data):
    with open(filename, 'wb') as writer:
        pickle.dump(data, writer)


def open_file(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def find_audio_files(directory):
    """
    Searches in a folder of audio files.

    :param directory: Path of folder.
    :return: List of paths with audio files.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files


def read_audio_file(file_path):
    """
    Reads an audio file.

    :param file_path: Path of audio file.
    :return: tuple (sample_rate, audio_data)
    """
    audio_data, sample_rate = sf.read(file_path, dtype='int16')
    return sample_rate, audio_data

def extract_logfbank_features(name):
    """
    Extracts features (logfbank) from audio file.

    :param name: path of auio files.
    :return: List of logfbank features.
    """
    rate, sig = read_audio_file(name)
    fbank_f = logfbank(sig, rate)
    fbank_f = (fbank_f - np.mean(fbank_f, axis=0)) / np.std(fbank_f, axis=0)

    return fbank_f

def extract_f0(name):
    """
    Extracts features (f0) from audio file.

    :param name: path of audio files.
    :return: Value of fundamental frequency (mean)
    """
    y, sr = librosa.load(name, sr=16000)
    #frequency = crepe.predict(y, sr, viterbi=True)

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=16000)
    indx = [i for i, vf in enumerate(voiced_flag) if vf]

    frequency = np.mean(f0[indx])
    frequency = round(frequency,2)
    return frequency

