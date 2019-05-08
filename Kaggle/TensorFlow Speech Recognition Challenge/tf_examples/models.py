"""
Model definitions for simple speech recognition.
"""

import tensorflow as tf
from typing import Dict


def _next_power_of_two(x: int):
    """
    Calculates the smallest enclosing power of two for an input.
    Args:
        x: Positive integer number.
    Returns:
        Next largest power of two integer.
    """
    return 1 if x == 0 else 2 ** ((x - 1).bit_length())


def prepare_model_settings(label_count: int, sample_rate: int, clip_duration_ms: int, window_size_ms: int,
                           window_stride_ms: int, feature_bin_count: int, preprocess: str) -> Dict:
    """
    Calculates common settings needed for all models.

    Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        feature_bin_count: Number of frequency bins to use for analysis.
        preprocess: How the spectrogram is processed to produce features.

    Returns:
        Dictionary containing common settings.

    Raises:
        ValueError: If the preprocess mode isn't recognized.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    spectrogram_length = 0 if length_minus_window < 0 else 1 + int(length_minus_window / window_stride_samples)

    preprocess = preprocess.lower()
    if preprocess == "average":
        fft_bin_count = 1 + int(_next_power_of_two(window_size_samples) / 2)
        average_window_width = int(fft_bin_count / feature_bin_count)
        fingerprint_width = int(fft_bin_count / average_window_width) + 1
    elif preprocess == "mfcc":
        average_window_width = -1
        fingerprint_width = feature_bin_count
    else:
        raise ValueError("Unknown preprocess mode %s (should be 'mfcc' or 'average')" % preprocess)
    fingerprint_size = fingerprint_width * spectrogram_length

    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "fingerprint_width": fingerprint_width,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
        "preprocess": preprocess,
        "average_window_width": average_window_width,
    }



