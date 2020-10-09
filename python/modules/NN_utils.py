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
from tensorflow.keras.layers import Dense, LSTM


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
			tf.keras.layers.LSTM(4, batch_input_shape=(batch_size, in_seq_length, n_features), stateful=stateful, return_sequences=False),
			tf.keras.layers.Dense(out_seq_length),
		])
	return rnn_model
