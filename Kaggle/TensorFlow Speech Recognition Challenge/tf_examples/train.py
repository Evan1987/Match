
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import models
import input_data
import constant
from tensorflow.contrib.quantize import create_training_graph
from typing import Dict


FLAGS: argparse.Namespace = None


class TrainGraph:
    def __init__(self, model_settings: Dict):
        self.graph = self._build_graph(model_settings)

    def _build_graph(self, model_settings: Dict):
        fingerprint_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']

        graph = tf.Graph()
        tf.reset_default_graph()
        with graph.as_default():
            with tf.name_scope("input"):
                self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
                self.fingerprint_input = tf.placeholder(dtype=tf.float32, shape=[None, fingerprint_size])

                # if FLAGS.quantize:
                #     fingerprint_min, fingerprint_max = input_data.get_features_range(model_settings)
                #     fingerprint_input = tf.fake_quant_with_min_max_args(self.fingerprint_input,
                #                                                         min=fingerprint_min,
                #                                                         max=fingerprint_max)
                # else:
                #     fingerprint_input = self.fingerprint_input

            with tf.name_scope("model"):
                logits, self.keep_prob = models.create_model(self.fingerprint_input, model_settings, FLAGS.model_architecture)

            # Optionally we can add runtime checks to spot when NaNs or other symptoms of
            # numerical errors start occurring during training.
            control_dependencies = []
            if FLAGS.check_nans:
                checks = tf.add_check_numerics_ops()
                control_dependencies.append(checks)

            with tf.name_scope("loss"):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=logits)
            with tf.name_scope("train"), tf.control_dependencies(control_dependencies):
                self.lr = tf.placeholder(dtype=tf.float32, shape=[])
                self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(cross_entropy)
            with tf.name_scope("eval"):
                self.pred = tf.argmax(logits, axis=1)
                self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), dtype=tf.float32))
                self.confusion_matrix = tf.confusion_matrix(
                    labels=self.labels, predictions=self.pred, num_classes=label_count
                )
                tf.summary.scalar("accuracy", self.acc)

            self.global_step = tf.train.get_or_create_global_step()
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)

            self.saver = tf.train.Saver(tf.global_variables())
            self.merged_summries = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope="eval"))

        return graph


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    model_settings = models.prepare_model_settings(
        label_count=len(FLAGS.wanted_words.split(",")) + 2,  # add 'unknown' and 'silence'
        sample_rate=FLAGS.sample_rate,
        clip_duration_ms=FLAGS.clip_duration_ms,
        window_size_ms=FLAGS.window_size_ms,
        window_stride_ms=FLAGS.window_stride_ms,
        feature_bin_count=FLAGS.feature_bin_count,
        preprocess=FLAGS.preprocess
    )

    audio_processor = input_data.AudioProcessor(
        data_dir=FLAGS.data_dir,
        silence_percentage=FLAGS.silence_percentage,
        unknown_percentage=FLAGS.unknown_percentage,
        wanted_words=FLAGS.wanted_words.split(","),
        validation_percentage=FLAGS.validation_percenstage,
        testing_percantage=FLAGS.testing_percentage,
        model_settings=model_settings,
        summary_dir=FLAGS.summary_dir
    )

    fingerprint_size = model_settings["fingerprint_size"]
    label_count = model_settings["label_count"]
    time_shift_samples = int(FLAGS.time_shift_ms * FLAGS.sample_rate / 1000)

    # Set learning rate epoch-wised. It's often effective to set higher learning rate at the start of training followed
    # by lower one towards the end.
    epoch_list = list(map(int, FLAGS.epoches.split(",")))
    lr_list = list(map(float, FLAGS.learning_rate.split(",")))
    assert len(epoch_list) == len(lr_list), "--epochs and --learning_rate must be same length"











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=constant.TRAIN_PATH, help="where the train data placed")
    parser.add_argument("--background_volume", type=float, default=0.1, help="the volume of background noise, 0 to 1")
    parser.add_argument("--proba_with_background", type=float, default=0.8, help="how many training samples should be mixed with background noise, 0 to 1")
    parser.add_argument("--silence_percentage", type=float, default=10.0, help="how much of the training data should be silence, 0 to 100")
    parser.add_argument("--unknown_percentage", type=float, default=10.0, help="how much of the training data should be unknown words, 0 to 100")
    parser.add_argument("--time_shift_ms", type=float, default=100.0, help="range to randomly shift the training audio by ms")
    parser.add_argument("--testing_percentage", type=int, default=10, help="how much of wavs would be used as test set, 0 to 100")
    parser.add_argument("--validation_percentage", type=int, default=10, help="how much of wavs would be used as validation set, 0 to 100")
    parser.add_argument("--sample_rate", type=int, default=constant.SAMPLE_RATE, help="expected sample rate of wav")
    parser.add_argument("--clip_duration_ms", type=int, default=1000, help="expected duration in ms of wavs")
    parser.add_argument("--window_size_ms", type=float, default=30.0, help="how long each spectrogram timeslice is")
    parser.add_argument("--window_stride_ms", type=float, default=10.0, help="how far to move in time between spectrogram timeslices")
    parser.add_argument("--feature_bin_count", type=int, default=40, help="how many bins to use for the MFCC fingerprint")
    parser.add_argument("--epochs", type=str, default="15000,3000", help="how many training loops to run")
    parser.add_argument("--eval_step_interval", type=int, default=400, help="how often to eval the training results")
    parser.add_argument("--learning_rate", type=str, default="0.001, 0.0001", help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--summary_dir", type=str, default=constant.SUMMARY_PATH, help="where to save summary logs")
    parser.add_argument("--wanted_words", type=str, default="yes,no,up,down,left,right,on,off,stop,go", help="words to use with sep ','")
    parser.add_argument("--train_dir", type=str, default=constant.TRAIN_LOG_PATH, help="where to write train event and ckpt")
    parser.add_argument("--save_step_interval", type=int, default=100, help="how often to save model ckpt")
    parser.add_argument("--start_checkpoint", type=str, default="", help="If specified, restore pretrained model before any training")
    parser.add_argument("--model_architecture", type=str, choices=["single_fc", "conv", "low_latency_conv", "low_latency_svdf", "tiny_conv"], default="conv", help="what model arch to use")
    parser.add_argument("--check_nans", type=bool, default=False, help="whether to check for invalid number during processing")
    # parser.add_argument("--quantize", type=bool, default=False, help="whether to train the model for 8-bit deployment")
    parser.add_argument("--preprocess", type=str, default="mfcc", choices=["mfcc", "average"], help="spectrogram processing mode, 'mfcc' or 'average'")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0] + unparsed])


