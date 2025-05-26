"""
Audio vocoder utilities for waveform and spectrogram processing in TTS pipelines.

Adapted from open-source TTS projects (e.g., Tacotron, WaveRNN, and others).
"""

import math
import numpy as np
import librosa
import scripts.params as p
from scipy.signal import lfilter
import soundfile as sf


def label_2_float(x, bits):
    """Convert integer label to float in [-1, 1] range."""
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    """Convert float in [-1, 1] to integer label."""
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def load_wav(path):
    """Load a waveform from a file using librosa."""
    return librosa.load(str(path), sr=p.sample_rate)[0]


def save_wav(x, path):
    """Save a waveform to a file using soundfile."""
    sf.write(path, x.astype(np.float32), p.sample_rate)


def split_signal(x):
    """Split a 16-bit signal into coarse and fine 8-bit components."""
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    """Combine coarse and fine 8-bit components into a 16-bit signal."""
    return coarse * 256 + fine - 2**15


def encode_16bits(x):
    """Encode a float waveform to 16-bit PCM."""
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


mel_basis = None


def linear_to_mel(spectrogram):
    """Convert a linear spectrogram to a mel spectrogram."""
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)


def build_mel_basis():
    """Build a mel filterbank matrix."""
    return librosa.filters.mel(p.sample_rate, p.n_fft, n_mels=p.num_mels, fmin=p.fmin)


def normalize(S):
    """Normalize a spectrogram to [0, 1]."""
    return np.clip((S - p.min_level_db) / -p.min_level_db, 0, 1)


def denormalize(S):
    """Denormalize a spectrogram from [0, 1] to dB scale."""
    return (np.clip(S, 0, 1) * -p.min_level_db) + p.min_level_db


def amp_to_db(x):
    """Convert amplitude to decibels."""
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    """Convert decibels to amplitude."""
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    """Compute a normalized linear spectrogram from a waveform."""
    D = stft(y)
    S = amp_to_db(np.abs(D)) - p.ref_level_db
    return normalize(S)


def melspectrogram(y):
    """Compute a normalized mel spectrogram from a waveform."""
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)


def stft(y):
    """Short-time Fourier transform using librosa."""
    return librosa.stft(y=y, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length)


def pre_emphasis(x):
    """Apply pre-emphasis filter to a waveform."""
    return lfilter([1, -p.preemphasis], [1], x)


def de_emphasis(x):
    """Apply de-emphasis filter to a waveform."""
    return lfilter([1], [1, -p.preemphasis], x)


def encode_mu_law(x, mu):
    """Mu-law encode a waveform."""
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    """Mu-law decode a waveform."""
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

