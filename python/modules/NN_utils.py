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

def create_RNN(in_seq_length, out_seq_length, n_features, stateful=False, batch_size=None):
	"""
	Function called when creating a RNN model.
	Returns a keras model.
	"""
	rnn_model = tf.keras.models.Sequential([
			tf.keras.layers.LSTM(32, batch_input_shape=(batch_size, in_seq_length, n_features),
								stateful=stateful, return_sequences=False),
			tf.keras.layers.Dense(out_seq_length),
		])
	return rnn_model

def create_feedfwdNN(in_feature_length, out_length, n_features):
	"""
	Function called when creating a feed-forward NN model.
	Returns a keras model.
	"""
	rnn_model = tf.keras.models.Sequential([
			# tf.keras.layers.Conv1D(8, 3),
			# tf.keras.layers.MaxPool1D(2),
			# tf.keras.layers.ReLU(),
			# tf.keras.layers.Conv1D(16, 3),
			# tf.keras.layers.MaxPool1D(2),
			# tf.keras.layers.ReLU(),
			tf.keras.layers.Flatten(input_shape=(in_feature_length, n_features)),

			tf.keras.layers.Dense(256, activation="relu"),
			tf.keras.layers.Dense(512, activation="relu"),
			tf.keras.layers.Dense(256, activation="relu"),
			tf.keras.layers.Dense(out_length),
		])
	return rnn_model
