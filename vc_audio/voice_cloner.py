import os
import librosa
import numpy as np

def extract_mfcc(filepath, sr=16000, n_mfcc=40):
    """
    Load audio and extract MFCC features.

    Args:
        filepath (str): Path to the .wav audio file.
        sr (int): Target sampling rate.
        n_mfcc (int): Number of MFCC features to extract.

    Returns:
        np.ndarray: MFCC feature matrix.
    """
    y, sr = librosa.load(filepath, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_log_mel_spectrogram(filepath, sr=16000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Load audio and extract log-scaled mel spectrogram.

    Args:
        filepath (str): Path to the .wav file.
        sr (int): Target sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between frames.
        n_mels (int): Number of Mel bands to generate.

    Returns:
        np.ndarray: Log-scaled mel spectrogram.
    """
    y, sr = librosa.load(filepath, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                               hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec
