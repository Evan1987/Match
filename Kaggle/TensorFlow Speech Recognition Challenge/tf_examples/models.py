"""
Model definitions for simple speech recognition.
"""

import tensorflow as tf
import abc
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


def create_model(fingerprint_input: tf.Tensor, model_settings: Dict, model_architecture: str, **kwargs):

    args = {
        "fingerprint_input": fingerprint_input,
        "model_settings": model_settings
        # "is_training": is_training,
        # "runtime_settings": runtime_settings
    }

    switch = {
        "single_fc": SingleFC(**args),
        "conv": Conv(**args),
        "low_latency_conv": Low_Latency_Conv(**args),
        # "low_latency_svdf": Low_Latency_SVDF(**args),   # todo: not finished now
        "tiny_conv": Tiny_Conv(**args)
    }

    try:
        return switch[model_architecture]()
    except KeyError:
        raise Exception("Unknown model architecture: %s, should be one of "
                        "'single_fc', 'conv', 'low_latency_conv', 'low_latency_svdf' or 'tiny_conv'"
                        % model_architecture)


class Model(metaclass=abc.ABCMeta):
    def __init__(self, fingerprint_input: tf.Tensor, model_settings: Dict, **kwargs):
        self.fingerprint_input = fingerprint_input
        self.model_settings = model_settings
        # self.runtime_settings = runtime_settings
        self.other_settings = kwargs   # future: to be Compatible with future functions
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")
        self.output = self._construct_model()

    @abc.abstractmethod
    def _construct_model(self):
        # will use dropout or not
        return ...

    def __call__(self):
        return self.output, self.keep_prob


class SingleFC(Model):
    """
    Very simple model with just one hidden fc layer.
    (fingerprint_input)
        v
    [FC]<-(weights, bias, None)
    As expected, it doesn't produce very accurate results, but it is very fast and simple.
    """
    def _construct_model(self):
        label_count = self.model_settings["label_count"]
        with tf.name_scope("fc"):
            w_init = tf.truncated_normal_initializer(stddev=1e-3)
            b_init = tf.zeros_initializer()
            logits = tf.layers.dense(inputs=self.fingerprint_input, units=label_count, activation=None,
                                     kernel_initializer=w_init, bias_initializer=b_init)
        return logits


class Conv(Model):
    """
    This is roughly the network labeled as 'cnn-trad-fpool3'
    in the 'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    (fingerprint_input)
        v
    [Conv2D]<-(weights, bias, relu, MaxPool)
        v
    [Conv2D]<-(weights, bias, relu, MaxPool)
        v
    [FC]<-(weights, bias)

    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same paper with slightly less accuracy,
    see 'low_latency_conv' below.
    """
    def _construct_model(self):
        label_count = self.model_settings["label_count"]
        input_frequency_size = self.model_settings["fingerprint_width"]  # width
        input_time_size = self.model_settings["spectrogram_length"]  # height

        w_init = tf.truncated_normal_initializer(stddev=1e-2)
        b_init = tf.zeros_initializer()
        with tf.name_scope("conv1"):
            fingerprint_4d = tf.reshape(self.fingerprint_input, shape=[-1, input_time_size, input_frequency_size, 1])
            first_conv = tf.layers.conv2d(inputs=fingerprint_4d, filters=64, kernel_size=[20, 8], strides=[1, 1],
                                          activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=b_init,
                                          data_format="channels_last", padding="same")
            first_conv = tf.layers.dropout(first_conv, rate=1.0 - self.keep_prob)
            first_maxpool = tf.layers.max_pooling2d(first_conv, pool_size=[2, 2], strides=[2, 2], padding="same")
        with tf.name_scope("conv2"):
            second_conv = tf.layers.conv2d(inputs=first_maxpool, filters=64, kernel_size=[10, 4], strides=[1, 1],
                                           activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=b_init,
                                           data_format="channels_last", padding="same")
            second_conv = tf.layers.dropout(second_conv, rate=1.0 - self.keep_prob)
        with tf.name_scope("fc"):
            fc_input = tf.layers.flatten(second_conv)
            logits = tf.layers.dense(inputs=fc_input, units=label_count, activation=None,
                                     kernel_initializer=w_init, bias_initializer=b_init)
        return logits


class Low_Latency_Conv(Model):
    """
    A conv model with low compute requirements
    This is roughly the network labeled as 'cnn-one-fstride4' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
    (fingerprint_input)
        v
    [Conv2D]<-(weights, bias, relu)
        v
    [FC]<-(weights, bias)
        v
    [FC]<-(weights, bias)
        v
    [FC]<-(weights, bias)
    This produces slightly lower quality results than the 'conv' model,
    but needs fewer weight parameters and computations.
    """
    def _construct_model(self):
        label_count = self.model_settings["label_count"]
        input_frequency_size = self.model_settings["fingerprint_width"]  # width
        input_time_size = self.model_settings["spectrogram_length"]  # height

        w_init = tf.truncated_normal_initializer(stddev=1e-2)
        b_init = tf.zeros_initializer()

        with tf.name_scope("conv"):
            fingerprint_4d = tf.reshape(self.fingerprint_input, shape=[-1, input_time_size, input_frequency_size, 1])
            first_conv = tf.layers.conv2d(inputs=fingerprint_4d, filters=186, kernel_size=[input_time_size, 8],
                                          strides=[1, 1], activation=tf.nn.relu, kernel_initializer=w_init,
                                          bias_initializer=b_init, data_format="channels_last", padding="valid")
            first_conv = tf.layers.dropout(first_conv, rate=1.0 - self.keep_prob)
            first_conv = tf.layers.flatten(first_conv)

        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(inputs=first_conv, units=128, activation=None,
                                  kernel_initializer=w_init, bias_initializer=b_init)
            fc1 = tf.layers.dropout(fc1, rate=1.0 - self.keep_prob)

        with tf.name_scope("fc2"):
            fc2 = tf.layers.dense(inputs=fc1, units=128, activation=None,
                                  kernel_initializer=w_init, bias_initializer=b_init)
            fc2 = tf.layers.dropout(fc2, rate=1.0 - self.keep_prob)

        with tf.name_scope("fc3"):
            logits = tf.layers.dense(inputs=fc2, units=label_count, activation=None,
                                     kernel_initializer=w_init, bias_initializer=b_init)

        return logits

# todo: unfinished yet
# class Low_Latency_SVDF(Model):
#     """
#     An SVDF model with low compute requirements.
#     This is based in the topology presented in the 'Compressing Deep Neural Networks using a Rank-Constrained Topology':
#     https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf
#     (fingerprint_input)
#         v
#     [SVDF]<-(weights, bias, relu)
#         v
#     [FC]<-(weights, bias)
#         v
#     [FC]<-(weights, bias)
#         v
#     [FC]<-(weights, bias)
#
#     This model produces lower recognition accuracy than the 'conv' model above,
#     but requires fewer weight parameters and, significantly fewer computations.
#     """
#     def _construct_model(self):
#         label_count = self.model_settings["label_count"]
#         input_frequency_size = self.model_settings["fingerprint_width"]  # width
#         input_time_size = self.model_settings["spectrogram_length"]  # height
#
#         input_shape = self.fingerprint_input.shape
#         if len(input_shape) != 2:
#             raise ValueError("Input to 'SVDF' must have rank == 2")
#         elif input_shape[-1].value is None:
#             raise ValueError('The last dimension of the inputs to `SVDF` should be defined. Found `None`.')
#         elif input_shape[-1].value % input_frequency_size != 0:
#             raise ValueError("Input feature dimension %d must be a multiple of frame size %d" %
#                              (input_shape[-1].value, input_frequency_size))
#
#         rank = 2
#         num_units = 1280
#         num_filters = rank * num_units
#         batch = 1
#         memory = tf.get_variable(dtype=tf.float32, initializer=tf.zeros_initializer(),
#                                  shape=[num_filters, batch, input_time_size],
#                                  trainable=False, name="runtime_memory")
#         first_time_flag = tf.get_variable(dtype=tf.int32, initializer=1, name="first_time_flag")
#
#
#         w_init = tf.truncated_normal_initializer(stddev=1e-2)
#         b_init = tf.zeros_initializer()


class Tiny_Conv(Model):
    """
    A convolutional model aimed at microcontrollers.
    Devices like DSPs and microcontrollers can have very small amounts of memory and limited processing power.
    This model is designed to use less than 20KB of working RAM, and fit within 32KB of read-only (flash) memory.

    (fingerprint_input)
        v
    [Conv2D]<-(weights, bias, relu)
        v
    [FC]<-(weights, bias)

    This doesn't produce particularly accurate results, but it's designed to be used as the first stage of a pipeline,
    running on a low-energy piece of hardware that can always be on, and then wake higher-power chips
    when a possible utterance has been found, so that more accurate analysis can be done.
    """
    def _construct_model(self):
        label_count = self.model_settings["label_count"]
        input_frequency_size = self.model_settings["fingerprint_width"]  # width
        input_time_size = self.model_settings["spectrogram_length"]  # height

        w_init = tf.truncated_normal_initializer(stddev=1e-2)
        b_init = tf.zeros_initializer()

        with tf.name_scope("conv"):
            fingerprint_4d = tf.reshape(self.fingerprint_input, shape=[-1, input_time_size, input_frequency_size, 1])
            conv = tf.layers.conv2d(inputs=fingerprint_4d, filters=8, kernel_size=[10, 8], strides=[2, 2],
                                    activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=b_init,
                                    data_format="channels_last", padding="SAME")
            conv = tf.layers.dropout(conv, rate=1.0 - self.keep_prob)
            conv = tf.layers.flatten(conv)

        with tf.name_scope("fc"):
            logit = tf.layers.dense(inputs=conv, units=label_count, activation=None,
                                    kernel_initializer=w_init, bias_initializer=b_init)
            return logit
