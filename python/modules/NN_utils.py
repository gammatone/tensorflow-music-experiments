#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-09 15:15:53
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
NN_utils.py
Module gathering useful classes and functions for NNs.
"""

import os

import tensorflow as tf
from tensorflow.keras.models import Sequential


def optimizer_call(lr=0.001):
	"""
	Function called when creating an optimizer.
	Returns an optimizer function.
	"""
	return tf.keras.optimizers.RMSprop(lr=lr)

def create_RNN_OutDense(in_seq_length, out_seq_length, n_features, stateful=False, batch_size=None):
	"""
	Function called when creating a RNN model.
	Returns a keras model.
	"""
	rnn_model = tf.keras.models.Sequential([
			tf.keras.layers.LSTM(64, batch_input_shape=(batch_size, in_seq_length, n_features),
								stateful=stateful, return_sequences=False),
			# tf.keras.layers.LSTM(32, stateful=stateful, return_sequences=False),
			tf.keras.layers.Dense(out_seq_length),
		])
	return rnn_model

def create_RNN_InDense(in_seq_length, out_seq_length, n_features, stateful=False, batch_size=None):
	"""
	Function called when creating a RNN model.
	Returns a keras model.

	Notes about stateful/stateless mode if using an RNN model:   
            (1) Basically a stateful model doesn't reset its cell states at each prediction call.
            The stateful mode can be useful when evaluating a one-to-one sample model while
            keeping trace of the previous input samples.
            For more info about stateful/stateless RNN mode, please see:
            - https://stackoverflow.com/questions/43881364/why-can-model-not-even-predict-sine
            - http://philipperemy.github.io/keras-stateful-lstm/
            - https://fairyonice.github.io/Understand-Keras's-RNN-behind-the-scenes-with-a-sin-wave-example.html
            - https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html
            - https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
            (2) A related topic is the backpropagation techniques for RNN model training.
            Truncated backprop:
            - https://magenta.tensorflow.org/blog/2017/06/01/waybackprop/
	"""
	# if not stateful:
	rnn_model = tf.keras.models.Sequential([
			# tf.keras.layers.TimeDistributed(tf.keras.layers.InputLayer(), ),
			tf.keras.layers.Input(batch_input_shape=(batch_size, in_seq_length, n_features)),
			tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4)),
			tf.keras.layers.LSTM(16, stateful=stateful, return_sequences=True),
			# tf.keras.layers.LSTM(8, stateful=stateful, return_sequences=True),
			tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='relu')),
			tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
		])
	# else:
	# 	rnn_model = tf.keras.models.Sequential([
	# 			tf.keras.layers.Dense(16, batch_input_shape=(batch_size, in_seq_length, n_features)),
	# 			tf.keras.layers.LSTM(64, stateful=stateful, return_sequences=True),
	# 			tf.keras.layers.Dense(16, activation='relu'),
	# 			tf.keras.layers.Dense(1),
	# 		])

	return rnn_model

def create_RNN_AE(in_seq_length, out_seq_length, n_features, stateful=False, batch_size=None):
	"""
	Function called when creating a RNN model.
	Returns a keras model.
	"""
	rnn_model = tf.keras.models.Sequential([
			tf.keras.layers.LSTM(128, batch_input_shape=(batch_size, in_seq_length, n_features),
								stateful=stateful, return_sequences=False),
			tf.keras.layers.RepeatVector(out_seq_length),
			tf.keras.layers.LSTM(128, stateful=stateful, return_sequences=True),
			tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
		])
	return rnn_model

def create_feedfwdNN(in_seq_length, out_length, n_features):
	"""
	Function called when creating a feed-forward NN model.
	Returns a keras model.
	"""
	rnn_model = tf.keras.models.Sequential([
			tf.keras.layers.Conv1D(32, 512, input_shape=(in_seq_length, n_features)),
			tf.keras.layers.MaxPool1D(8),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Conv1D(64, 32),
			# tf.keras.layers.MaxPool1D(2),
			tf.keras.layers.ReLU(),
			# tf.keras.layers.Conv1D(64, 128),
			# tf.keras.layers.MaxPool1D(2),
			# tf.keras.layers.ReLU(),
			tf.keras.layers.Flatten(),

			# tf.keras.layers.Flatten(input_shape=(in_feature_length, n_features)),

			# tf.keras.layers.Dense(4096, activation="relu"),
			tf.keras.layers.Dense(1024, activation="relu"),
			# tf.keras.layers.Dense(512, activation="relu"),
			# tf.keras.layers.Dense(512, activation="relu"),
			tf.keras.layers.Dense(out_length),
		])
	return rnn_model
