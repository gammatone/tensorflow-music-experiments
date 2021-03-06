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
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Add, Flatten, Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import TimeDistributed, RepeatVector
from tensorflow.keras.layers import Lambda


def optimizer_call(lr=0.001):
    """
    Function called when creating an optimizer.
    Returns an optimizer function.
    """
    return tf.keras.optimizers.RMSprop(lr=lr)

import tensorflow.keras.backend as K

def custom_loss(y_true, y_pred):
    """
    Custom loss function to be called when training model 
    """ 
    # fft_true = Lambda(lambda v: tf.math.log(1e-5 + tf.math.abs(tf.cast(tf.signal.rfft(v), tf.float32))))(y_true)
    # fft_pred = Lambda(lambda v: tf.math.log(1e-5 + tf.math.abs(tf.cast(tf.signal.rfft(v), tf.float32))))(y_pred)

    # fft_true = Lambda(lambda v: tf.math.abs(tf.cast(tf.signal.rfft(v), tf.float32)))(y_true)
    # fft_pred = Lambda(lambda v: tf.math.abs(tf.cast(tf.signal.rfft(v), tf.float32)))(y_pred)

    # # Compute MSE on ffts
    # mse = K.mean(K.square(fft_true - fft_pred), axis=1)
    # norm_mse = mse / K.mean(K.square(fft_true), axis=1)
    # custom_loss = norm_mse

    # Compute normalized MSE on temporal signal
    mse = K.mean(K.square(y_true - y_pred), axis=1)
    mse = mse / K.mean(K.square(y_true), axis=1)
    custom_loss = mse

    return custom_loss

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

    # x = TimeDistributed(Dense(4))(in_NN)
    # x = Activation('tanh')(x)

    x = LSTM(2, stateful=stateful, return_sequences=True)(in_NN)

    # x_signal = LSTM(2, stateful=stateful, return_sequences=True)(in_NN[:,:,0:1])
    # x_cutoff = LSTM(1, stateful=stateful, return_sequences=True)(in_NN[:,:,1:2])
    # x_res = LSTM(1, stateful=stateful, return_sequences=True)(in_NN[:,:,2:3])
    # x = tf.concat([x_signal, x_cutoff, x_res], 2)
    # x = LSTM(2, stateful=stateful, return_sequences=True)(x)

    # x = TimeDistributed(Dense(4))(x)
    # x = Activation('tanh')(x)

    out_NN = TimeDistributed(Dense(1))(x)

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
