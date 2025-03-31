'''
Module to load audio files and sample them according to
Wav2Vec2's required sample rate of 16khz
'''

import librosa

def load_and_resample_audio(folder_raw, sample_rate):
    """Loads and resamples an audio file to the target sample rate."""
    audio, sr = librosa.load(folder_raw, sr=sample_rate)
    return audio, sr