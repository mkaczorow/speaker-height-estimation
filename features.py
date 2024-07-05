from python_speech_features import logfbank
import os
import pickle
import soundfile as sf

#Functions for saving files
def save_file(filename, data):
    with open(filename, 'wb') as writer:
        pickle.dump(data, writer)


def open_file(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

#path = "C:\\Users\\magda\\Desktop\\praktyki\\projekt\\TIMIT\\data\\TRAIN\\"

def find_audio_files(directory):
    """
    Searches in a folder of audio files.

    :param directory: Path of folder.
    :return: List of paths with audio files.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

#audio_files = find_audio_files(path)

def read_audio_file(file_path):
    """
    Reads an audio file.

    :param file_path: Path of audio file.
    :return: Tuple (sample_rate, audio_data)
    """
    audio_data, sample_rate = sf.read(file_path, dtype='int16')
    return sample_rate, audio_data

def extract_logfbank_features(audio_files):
    """
    Extracts features (logfbank) from list of audio files.

    :param audio_files: List of auio files.
    :return: List of logfbank features.
    """
    fbank_feat = []
    for filename in audio_files:
        rate, sig = read_audio_file(filename)
        fbank_f = logfbank(sig, rate)
        fbank_feat.append(fbank_f)

    return fbank_feat

#fbank_test = extract_logfbank_features(audio_files)
#print(fbank_test)
