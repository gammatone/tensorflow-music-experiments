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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPool1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Add, Flatten, Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import TimeDistributed


def optimizer_call(lr=0.001):
	"""
	Function called when creating an optimizer.
	Returns an optimizer function.
	"""
	return tf.keras.optimizers.RMSprop(lr=lr)

def vanilla_LSTM(in_seq_length, out_seq_length, n_features,
				stateful=False,
				batch_size=None):
	"""
	Returns a vanilla LSTM net declared with TF functional API.

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
	in_NN = Input(batch_input_shape=(batch_size, in_seq_length, n_features))
	x = TimeDistributed(Dense(4))(in_NN)
	x = LSTM(16, stateful=stateful, return_sequences=True)(x)
	x = TimeDistributed(Dense(4))(x)
	x = Activation('relu')(x)
	out_NN = TimeDistributed(Dense(4))(x)
	# Create model
	model = Model(inputs=in_NN, outputs=out_NN)
	return model

def vanilla_feedfwdNN(in_seq_length, out_seq_length, n_features):
	"""
	Returns a vanilla feed-forwad net declared with TF functional API.
	"""
	in_NN = Input(shape=(in_seq_length, n_features))
	x = Conv1D(32, 512)(in_NN)
	x = MaxPool1D(8)(x)
	x = Activation('relu')(x)
	x = Flatten()(x)
	x = Dense(1024)(x)
	x = Activation('relu')(x)
	out_NN = Dense(out_seq_length)(x)
	# Create model
	model = Model(inputs=in_NN, outputs=out_NN)
	return model
