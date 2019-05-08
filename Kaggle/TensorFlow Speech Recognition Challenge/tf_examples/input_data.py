"""
A learning script based on tensorflow official examples:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands

The official competition site is on kaggle:
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/overview
"""

import os
import hashlib
import random
import re
import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from scipy.io import wavfile
from typing import List, Dict


MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # 128MB
SILENCE_LABEL = "_silence_"
SILENCE_INDEX = 0  # the word index used in AudioPrecessor
UNKNOWN_WORD_LABEL = "_unknown_"
UNKNOWN_WORD_INDEX = 1  # the word index used in AudioPrecessor
BACKGROUND_NOISE_DIR_NAME = "_background_noise_"
RANDOM_SEED = 59185


def prepare_words_list(wanted_words: List[str]):
    """
    Prepare common tokens to the custom word list
    :param wanted_words: List of custom words
    :return: List with standard silence and unknown token added
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename: str, validation_percentage: float, testing_percentage: float):
    """
    Determine which data partition the file should belong to.
    We want to keep files in the same training, validation or testing sets even if new ones added over time.
    This makes it less likely that testing samples will accidentally be reused in training when long run started.
    To keep this stability, hash of filename is taken and used to determine which set it should belong to.
    The determination only depends on name and set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example. (e.g. wavs from same person will be in same set)

    :param filename: File Path of the data sample.
    :param validation_percentage: How much of the data set to use for validation.
    :param testing_percentage: How much of the data set to use for testing.
    :return: String. One of 'training', 'validation' or 'testing'
    """
    base_name = os.path.basename(filename)  # extract the final component of the path -> os.path.split(filename)[1]
    speaker = re.sub("_nohash_.*$", "", base_name)  # -> base_name.split("_nohash_")[0]
    speaker_hashed = hashlib.sha1(speaker.encode(encoding="utf-8")).hexdigest()
    percentage_hash = (int(speaker_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS)

    # find partition by cum percentage [----validation----|--testing--|--------training--------]
    if percentage_hash < validation_percentage:
        result = "validation"
    elif percentage_hash < (validation_percentage + testing_percentage):
        result = "testing"
    else:
        result = "training"
    return result

# 这个函数需要audio_ops依赖，在tf1.4中，这个模块尚不完整，可以从1.5拷贝gen_audio_ops.py到 tensorflow/python/ops即可
# 但是也有另外一种实现方式不需要用到tf的io接口，参照本例之后。
# def load_wav_file(filename):
#     """Loads an audio file and returns a float PCM-encoded array of samples.
#
#        Args:
#          filename: Path to the .wav file to load.
#
#        Returns:
#          Numpy array holding the sample data as floats between -1.0 and 1.0.
#     """
#
#     with tf.Session(graph=tf.Graph()) as sess:
#         wav_filename_placeholder = tf.placeholder(tf.string, [])
#         wav_loader = io_ops.read_file(wav_filename_placeholder)
#         wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
#         return sess.run(
#             wav_decoder,
#             feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def load_wav_file(filename: str) -> (int, np.ndarray):
    """
    利用scipy.io API实现同等功能，输出时根据 PCM位数进行标准化
    :param filename: 文件路径
    :return:
        -sample_rate
        -Numpy array holding the sample data as floats between -1.0 and 1.0
    """
    sample_rate, data = wavfile.read(filename)
    if data.dtype == np.dtype("int16"):  # 16bit
        samples = data / (2 ** 15)  # 32768
    elif data.dtype == np.dtype("int32"):  # 32bit
        samples = data / (2 ** 31)  # 2147483648
    elif data.dtype == np.dtype("uint8"):  # 8bit
        samples = data / 255
    return sample_rate, samples


def save_wav_file(filename: str, rate: int, samples: np.ndarray, bit_length=16):
    """
    保存文件至本地目录
    :param filename: 目标路径
    :param rate: 采样率
    :param samples: 标准化后的数据 between -1.0 and 1.0
    :param bit_length: PCM 位数
    :return: None
    """
    data = None
    if bit_length == 16:
        data = (samples * (2 ** 15)).astype(np.int16)
    elif bit_length == 32:
        data = (samples * (2 ** 31)).astype(np.int32)
    elif bit_length == 8:
        data = (samples * 255).astype(np.uint8)
    if data:
        wavfile.write(filename, rate, data)


class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, *, data_dir: str, silence_percentage: float, unknown_percentage: float, wanted_words: List[str],
                 validation_percentage: float, testing_percantage: float, model_settings: Dict, summary_dir: str=None):

        assert validation_percentage + testing_percantage < 100, \
            "Invalid validation and testing percentage amount, their sum must be lower than 100"
        assert silence_percentage + unknown_percentage < 100, \
            "Invalid silence and unknown percentage amount, their sum must be lower than 100"

        self.data_dir = data_dir
        self.data_index: Dict[str, List[Dict[str, str]]] = dict.fromkeys(["validation", "testing", "training"], [])
        self.word_list = prepare_words_list(wanted_words)
        self.word_to_index = {word: index for index, word in enumerate(self.word_list)}  # _silence_: 0, _unknown_: 1,..
        self.prepare_data_index(silence_percentage, unknown_percentage, wanted_words,
                                validation_percentage, testing_percantage)
        self.background_data: List[np.ndarray] = self.prepare_background_data()
        self.graph = self._build_graph(model_settings)
        self.summary_writer_ = tf.summary.FileWriter(os.path.join(summary_dir, "data/"), self.graph) if summary_dir else None

    def prepare_data_index(self, silence_percentage: float, unknown_percentage: float, wanted_words: List[str],
                           validation_percentage: float, testing_percentage: float):
        """
        Prepares a list of the samples organized by set and label.

        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.

        Args:
            silence_percentage: How much of the resulting data should be background.
            unknown_percentage: How much should be audio outside the wanted classes.
            wanted_words: Labels of the classes we want to be able to recognize.
            validation_percentage: How much of the data set to use for validation.
            testing_percentage: How much of the data set to use for testing.

        Returns:
            to self's property
            Dictionary containing a list of file information for each set partition,
            and a lookup map for each class to determine its numeric index.

        Raises:
            Exception: If wanted word are not found.
        """
        random.seed(RANDOM_SEED)
        all_words = set()  # 收集已经获取到文件的word，目录下全部的word
        unknown_index = dict.fromkeys(["validation", "testing", "training"], [])  # 非关注词的划分集合

        # 遍历目录下的全部音频文件，并对其进行划分 -> validation, testing, training
        search_path = os.path.join(self.data_dir, "*", "*.wav")
        for wav_path in glob.glob(search_path):
            word = os.path.basename(os.path.dirname(wav_path)).lower()  # 解出文件所在的文件夹名称
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words.add(word)
            set_index = which_set(wav_path, validation_percentage, testing_percentage)

            info = {"label": word, "file": wav_path}
            if word in wanted_words:
                self.data_index[set_index].append(info)
            else:
                unknown_index[set_index].append(info)

        # wanted_words不能有缺失
        if not all_words:
            raise Exception("No .wavs found at %s" % search_path)
        for wanted_word in wanted_words:
            if wanted_word not in all_words:
                raise Exception("Not found %s in labels. Only found %s" % (wanted_word, ", ".join(all_words)))

        # 给剩余全部word（未在wanted中的）加上UNKNOWN_WORD_INDEX
        for word in all_words:
            if word not in wanted_words:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX

        # We need an arbitrary file to load as the input for the silence samples.
        # It will be multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index["training"][0]["file"]
        silence_info = {"label": SILENCE_LABEL, "file": silence_wav_path}

        # 为每一个partition_set增加silence样本和unknown样本
        for set_index in ["validation", "testing", "training"]:
            set_size = len(self.data_index[set_index])

            # total size after add silence samples and unknown samples
            expected_size = int(set_size / (100.0 - silence_percentage - unknown_percentage))

            # 增加silence样本
            silence_size = int(expected_size * silence_percentage / 100) + 1
            self.data_index[set_index].extend([silence_info] * silence_size)

            # 增加unknown样本
            random.shuffle(unknown_index[set_index])  # 先打散
            unknown_size = int(expected_size * unknown_percentage / 100) + 1
            self.data_index[set_index].extend(unknown_index[set_index][: unknown_size])

            # 最终打散各个分区
            random.shuffle(self.data_index[set_index])

    def prepare_background_data(self) -> List[np.ndarray]:
        """
        Searches a folder for background noise audio, and loads it into memory.

        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.

        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.

        Returns:
            List of raw PCM-encoded audio samples of background noise.

        Raises:
            Exception: If files aren't found in the folder.
        """
        background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return background_data

        for wav_path in os.listdir(background_dir):
            if not wav_path.endswith(".wav"):
                continue
            _, data = load_wav_file(os.path.join(background_dir, wav_path))
            background_data.append(data)
        if not background_data:
            raise Exception("No background wav files found in %s" % background_dir)

        return background_data

    def _build_graph(self, model_settings):
        """
        Builds a TensorFlow graph to apply the input distortions.

        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

            - wav_filename_placeholder_: Filename of the WAV to load.
            - foreground_volume_placeholder_: How loud the main clip should be.
            - time_shift_padding_placeholder_: Where to pad the clip.
            - time_shift_offset_placeholder_: How much to move the clip in time.
            - background_data_placeholder_: PCM sample data for background noise.
            - background_volume_placeholder_: Loudness of mixed-in background.
            - output_: Output 2D fingerprint of processed audio.

        Args:
            model_settings: Information about the current model being trained.

        Raises:
            ValueError: If the preprocessing mode isn't recognized.
        """
        graph = tf.Graph()
        tf.reset_default_graph()
        desired_samples = model_settings["desired_samples"]
        with graph.as_default():
            with tf.name_scope("input"):
                self.wav_filename_placeholder_ = tf.placeholder(dtype=tf.string, shape=[], name='wav_filename')
                self.foreground_volume = tf.placeholder(dtype=tf.float32, shape=[], name="foreground_volume")
                self.time_shift_padding = tf.placeholder(dtype=tf.int32, shape=[2, 2], name="time_shift_padding")
                self.time_shift_offset = tf.placeholder(dtype=tf.int32, shape=[], name="time_shift_offset")
                self.background = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="background")
                self.background_volume = tf.placeholder(dtype=tf.float32, shape=[], name="background_volume")

                wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
                wav_decoder = contrib_audio.decode_wav(contents=wav_loader, desired_channels=1,
                                                       desired_samples=desired_samples)

                # A breakpoint foreground data with no processings
                self.unprocessed_foreground = wav_decoder.audio * self.foreground_volume
                padded_foreground = tf.pad(self.unprocessed_foreground, paddings=self.time_shift_padding, mode="CONSTANT")
                sliced_foreground = tf.slice(input_=padded_foreground, begin=self.time_shift_offset,
                                             size=[desired_samples, -1])

                scaled_background = self.background * self.background_volume
                sliced_background = tf.slice(input_=scaled_background, begin=0, size=[desired_samples, -1])

                total = tf.clip_by_value(sliced_foreground + sliced_background, -1.0, 1.0)

                spectrogram = contrib_audio.audio_spectrogram(input=total,
                                                              window_size=model_settings["window_size_samples"],
                                                              stride=model_settings["window_stride_samples"],
                                                              magnitude_squared=True)
                tf.summary.image("spectrogram", tensor=tf.expand_dims(spectrogram, -1), max_outputs=1)

                # shrink the data
                if model_settings["preprocess"] == "average":
                    self.output_ = tf.nn.pool(input=tf.expand_dims(spectrogram, axis=-1),
                                              window_shape=[1, model_settings["average_window_width"]],
                                              strides=[1, model_settings["average_window_width"]],
                                              pooling_type="AVG",
                                              padding="SAME")
                    tf.summary.image("shrunk_spectrogram", self.output_, max_outputs=1)

                elif model_settings["preprocess"] == "mfcc":
                    self.output_ = contrib_audio.mfcc(spectrogram=spectrogram,
                                                      sample_rate=wav_decoder.sample_rate,
                                                      dct_coefficient_count=model_settings["fingerprint_width"])
                    tf.summary.image("mfcc", tf.expand_dims(self.output_, axis=-1), max_outputs=1)

                else:
                    raise ValueError("Unknown preprocess mode '%s' (should be 'mfcc' or 'average')" %
                                     model_settings["preprocess"])

                self.merged_summries = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope="input"))
        return graph

    def set_size(self, mode: str) -> int:
        assert mode in ["validation", "testing", "training"], \
            "Invalid mode %s (should be 'validation', 'testing' or 'training')" % mode
        return len(self.data_index[mode])

    def get_data(self, how_many: int, offset: int, model_settings, proba_with_background: float,
                 background_volume_range, time_shift: int, mode: str, sess: tf.Session) \
            -> (List[np.ndarray, List[int]]):
        """
        Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
            how_many: Desired number of samples to return. -1 means the entire contents of this partition.
            offset: Where to start when fetching deterministically.
            model_settings: Information about the current model being trained.
            proba_with_background: How many clips will have background noise, 0.0 to 1.0.
            background_volume_range: How loud the background noise will be.
            time_shift: How much to randomly shift the clips by in time.
            mode: Which partition to use, must be 'training', 'validation', or 'testing'.
            sess: TensorFlow session that was active when processor was created.
        Returns:
            List of sample data for the transformed samples
            List of label indexes
        Raises:
            ValueError: If background samples are too short.
        """
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        data: List[np.ndarray] = []
        labels: List[int] = []
        desired_samples: int = model_settings["desired_samples"]
        use_background: bool = self.background_data and (mode == "training")
        pick_deterministically: bool = (mode != "train")
        for i in range(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = random.randint(len(candidates) - 1)  # if not deterministic, i and offset are useless

            sample: Dict = candidates[sample_index]

            # If we're time shifting, set up the offset for this sample.
            time_shift_amount = np.random.randint(-time_shift, time_shift) if time_shift > 0 else 0
            if time_shift_amount > 0:  # before pad
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:  # after pad
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]

            # Choose a section of background noise to mix in.
            if use_background or sample["label"] == SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_data))
                background_sample = self.background_data[background_index]

                # 从background_sample中选取一个长度为desired_samples的切片
                if len(background_sample) <= desired_samples:
                    raise ValueError(
                        "Background sample is too short! Need more than %d samples but only %d were found" %
                        (desired_samples, len(background_sample)))
                offset = np.random.randint(0, len(background_sample) - desired_samples)
                background_sample = background_sample[offset: (offset + desired_samples)].reshape(-1, 1)

                if sample["label"] == SILENCE_LABEL:
                    background_volume = random.uniform(0, 1)
                else:
                    background_volume = 0. if random.uniform(0, 1) > proba_with_background else np.random.uniform(0, background_volume_range)
            else:
                background_sample = np.zeros((desired_samples, 1))
                background_volume = 0.

            foreground_volume = 0. if sample["label"] == SILENCE_LABEL else 1.

            feed_dict = {
                self.wav_filename_placeholder_: sample["file"],
                self.foreground_volume: foreground_volume,
                self.time_shift_padding: time_shift_padding,
                self.time_shift_offset: time_shift_offset,
                self.background: background_sample,
                self.background_volume: background_volume,
            }

            summary, data_tensor = sess.run([self.merged_summries, self.output_], feed_dict=feed_dict)
            if self.summary_writer_:
                self.summary_writer_.add_summary(summary)

            data.append(data_tensor.flatten())
            labels.append(self.word_to_index[sample["label"]])
        return data, labels

    def get_feature_for_wav(self, wav_filename: str, model_settings: Dict, sess: tf.Session) -> np.ndarray:
        """
        Applies the feature transformation process to the input_wav.

        Runs the feature generation process (generally producing a spectrogram from
        the input samples) on the WAV file. This can be useful for testing and
        verifying implementations being run on other platforms.

        Args:
            wav_filename: The path to the input audio file.
            model_settings: Information about the current model being trained.
            sess: TensorFlow session that was active when processor was created.

        Returns:
            Numpy data array containing the generated features.
        """
        desired_samples = model_settings["desired_samples"]
        feed_dict = {
            self.wav_filename_placeholder_: wav_filename,
            self.foreground_volume: 1.,
            self.time_shift_padding: [[0, 0], [0, 0]],
            self.time_shift_offset: [0, 0],
            self.background: np.zeros((desired_samples, 1)),
            self.background_volume: 0.,
        }

        return sess.run(self.output_, feed_dict=feed_dict)

    def get_unprocessed_data(self, how_many: int, mode: str, sess: tf.Session) -> (List[np.ndarray, List[int]]):
        """
        Retrieve sample data for the given partition, with no transformations.

        Args:
            how_many: Desired number of samples to return. -1 means the entire contents of this partition.
            mode: Which partition to use, must be 'training', 'validation', or 'testing'.
            sess: TensorFlow session that was active when processor was created.
        Returns:
            List of sample data for the samples, and list of labels in one-hot form.
        """
        candidates = self.data_index[mode]
        sample_count = len(candidates) if how_many == -1 else how_many
        data: List[np.ndarray] = []
        labels: List[int] = []

        for i in range(sample_count):
            sample_index = i if how_many == -1 else np.random.randint(len(candidates))
            sample = candidates[sample_index]
            foreground_volume = 0. if sample["label"] == SILENCE_LABEL else 1.
            feed_dict = {
                self.wav_filename_placeholder_: sample["file"],
                self.foreground_volume: foreground_volume,
            }
            data_tensor = sess.run(self.unprocessed_foreground, feed_dict=feed_dict)
            data.append(data_tensor.flatten())
            labels.append(self.word_to_index[sample["label"]])
        return data, labels

































