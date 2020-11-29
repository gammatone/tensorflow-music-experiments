#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-11 17:43:20
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
training_utils.py
Module gathering useful classes and functions for Machine Learning model training.
"""
import os
import numpy as np

# Custom imports
from dataset_utils import get_dict_from_pkl, stack_array_from_dict_lastaxis
from array_utils import bufferize_array, unbufferize_array
from plot_utils import plot_by_key


class ModelHandler():
    """
    A class that will load pickle dictionary files containing train and validation data to perform model training.
    """
    def __init__(   self,
                    pkl_load_dir, input_keys, groundtruth_keys,
                    need_training=False,
                    train_filename=None, val_filename=None, eval_filename="evaluation.pkl",
                    ):
        """
        """
        # define pickle directories (where the pickle files are located)
        self.pkl_load_dir = pkl_load_dir
        # Training enable flag
        self.need_training = need_training
        if self.need_training:
            # Define train & val datasets pickle filepaths 
            self.pkl_train_filepath = os.path.join(os.path.join(self.pkl_load_dir, train_filename))
            self.pkl_val_filepath = os.path.join(os.path.join(self.pkl_load_dir, val_filename))
        # Define evaluation dataset pickle filepath 
        self.pkl_eval_filepath = os.path.join(os.path.join(self.pkl_load_dir, eval_filename))
        # Load pickle data
        self.load_pkl_data()

        # Define signal keys (i.e. key to load in the pickle dict for both input and groundtruth data)
        self.input_keys = input_keys
        self.groundtruth_keys = groundtruth_keys

        return

    def load_pkl_data(self):
        """
        Load pickle dictionary for evaluation set
        If need_training flag is ON, load pickle dictionaries for train and validation sets
        """
        if self.need_training:
            self.train_dict = get_dict_from_pkl(self.pkl_train_filepath)
            self.val_dict = get_dict_from_pkl(self.pkl_val_filepath)
        self.eval_dict = get_dict_from_pkl(self.pkl_eval_filepath)

    def prepare_datasets(self):
        """
        For evaluation set: stack data into a numpy array acoording to input and groundtruth keys
        If need_training flag is ON, Idem for both train and validation sets: 
        """
        if self.need_training:
            # Train set
            # Stack inputs (i.e. features) on last axis
            self.x_train = stack_array_from_dict_lastaxis(self.train_dict, self.input_keys)
            self.y_train = stack_array_from_dict_lastaxis(self.train_dict, self.groundtruth_keys)

            # Validation set
            # Stack inputs (i.e. features) on last axis
            self.x_val = stack_array_from_dict_lastaxis(self.val_dict, self.input_keys)
            self.y_val = stack_array_from_dict_lastaxis(self.val_dict, self.groundtruth_keys)

        # Evaluation set
        # Stack inputs (i.e. features) on last axis
        self.x_eval = stack_array_from_dict_lastaxis(self.eval_dict, self.input_keys)
        self.y_eval = stack_array_from_dict_lastaxis(self.eval_dict, self.groundtruth_keys)

    # Abstract
    # def train_model(self):
    #     return

    # Abstract
    # def evaluate_model(self):
    #     return



class KerasModelHandler(ModelHandler):
    """
    Model handler child class using Keras API 
    """
    def __init__(   self,
                    pkl_load_dir, input_keys, groundtruth_keys,
                    keras_optim, keras_loss,
                    **kwargs,
                    ):
        """
        Init parent class and add optimizer & loss
        """
        super(KerasModelHandler, self).__init__(pkl_load_dir, input_keys, groundtruth_keys, **kwargs)
        self.optimizer = keras_optim
        self.loss = keras_loss

    def train_model(self, keras_model, epochs=50, batch_size=32, val_freq=3, sample_weight=None):
        """
        Compile provided keras_model with selected optimizer andstart training
        Will work only if need_training flag is True. Else it will throw an error.
        """

        keras_model.compile(loss=self.loss, optimizer=self.optimizer)

        # train model
        training_state = keras_model.fit(self.x_train, self.y_train, shuffle=True,
                                    epochs=epochs, batch_size=batch_size,
                                    validation_data=(self.x_val, self.y_val),
                                    # validation_split=0.1,
                                    validation_freq=val_freq,
                                    sample_weight=sample_weight
                                    )
        return training_state

    def evaluate_model(self, keras_model, batch_size=None, start_end_idxs=None, need_plot=False):
        """
        Compute the average loss function score on the evaluation dataset.
        Plot the IO of the model if needed
        """
        keras_model.compile(loss=self.loss, optimizer=self.optimizer)
        keras_model.reset_states()
        if start_end_idxs is not None:
            self.x_eval = self.x_eval[start_end_idxs[0]:start_end_idxs[1]]
            self.y_eval = self.y_eval[start_end_idxs[0]:start_end_idxs[1]]
        if need_plot:
            # # Predict output of training dataset inputs from trained model
            # predicted_output = self.predict(keras_model, self.x_train, batch_size=batch_size)
            # array_dict = {}
            # # Append signals from train dict
            # for key in self.train_dict.keys():
            #     array_dict[key] = self.train_dict[key][:]
            # # Append predictions in dict
            # array_dict["predictions"] = predicted_output
            # plot_by_key(array_dict, array_dict.keys(), title="Model predictions on training set")
            # Predict output of evaluation dataset inputs from trained model
            predicted_output = self.predict(keras_model, self.x_eval, batch_size=batch_size)
            array_dict = {}
            # Append signals from evaluation dict
            for key in self.eval_dict.keys():
                array_dict[key] = self.eval_dict[key][:]
            # Append predictions in dict
            array_dict["predictions"] = predicted_output
            plot_by_key(array_dict, array_dict.keys(), title="Model predictions on evaluation set")
        return keras_model.evaluate(self.x_eval, self.y_eval, batch_size=batch_size)

    def predict(self, keras_model, x_inputs, batch_size=None):
        """
        """
        return keras_model.predict(x_inputs, batch_size=batch_size)



class KerasBufferizedNNHandler(KerasModelHandler):
    """
    Keras Model handler child class.
    Dedicated for NN model working with IO buffers of same or different length.
    Example: RNN models which work with input and output sequences will require to bufferize datasets before training and validation.

    # N.B. For RNN, friendly training format is:
    #        x.shape = (n_chunks, IN_SEQ_LENGTH, N_IN_FEATURES) ; y.shape = (n_chunks, OUT_SEQ_LENGTH, N_OUT_FEATURES))
    """
    def __init__(   self,
                    pkl_load_dir, input_keys, groundtruth_keys,
                    keras_optim, keras_loss,
                    input_length=100, hop_size=50, output_length=1,
                    **kwargs,
                    ):
        """
        Init parent class and add optimizer
        """
        super(KerasBufferizedNNHandler, self).__init__(pkl_load_dir, input_keys, groundtruth_keys,
                                                keras_optim, keras_loss,
                                                **kwargs)
        # RNN hyper-parameters
        self.INPUT_LENGTH = input_length
        self.HOP_SIZE = hop_size
        self.OUTPUT_LENGTH = output_length

    def prepare_datasets(self):
        """
        For both train and validation sets: Stack data into a numpy array acoording to input and groundtruth keys.
        Override parent method to bufferize data.
        """
        super(KerasBufferizedNNHandler, self).prepare_datasets()
        if self.need_training:
            # Train set
            # For RNN split the input data into chunks of length IN_SEQ_LENGTH and HOP_SIZE
            self.x_train = bufferize_array(self.x_train, self.INPUT_LENGTH, hop_size=self.HOP_SIZE)
            # For RNN split the groundtruth data into chunks of length OUT_SEQ_LENGTH and HOP_SIZE
            self.y_train = bufferize_array(self.y_train, self.OUTPUT_LENGTH, hop_size=self.HOP_SIZE,
                                                start_index=self.INPUT_LENGTH - self.OUTPUT_LENGTH)

            # Validation set
            # For RNN split the input data into chunks of length IN_SEQ_LENGTH and HOP_SIZE
            self.x_val = bufferize_array(self.x_val, self.INPUT_LENGTH, hop_size=self.HOP_SIZE)
            # For RNN split the groundtruth data into chunks of length OUT_SEQ_LENGTH and HOP_SIZE
            self.y_val = bufferize_array(self.y_val, self.OUTPUT_LENGTH, hop_size=self.HOP_SIZE,
                                                start_index=self.INPUT_LENGTH - self.OUTPUT_LENGTH)

    def train_model(self, keras_model, epochs=50, batch_size=32, val_freq=3, need_sample_weight=False):
        """
        Provides sample weights to parent method.
        """

        if need_sample_weight:
            # prepare sample weights
            # sample_weight = np.ones(self.x_train.shape, dtype=self.x_train.dtype)
            # sample_weight[:, 0:self.INPUT_LENGTH // 2, :] = 0.
            sample_weight = np.ones(self.x_train.shape[0:2], dtype=self.x_train.dtype)
            sample_weight[:, 0:self.INPUT_LENGTH // 2] = 0.
        else:
            sample_weight = None

        return super(KerasBufferizedNNHandler, self).train_model(keras_model,
                                                                epochs=epochs,
                                                                batch_size=batch_size,
                                                                val_freq=val_freq,
                                                                sample_weight=sample_weight
                                                                )

    def evaluate_model(self, keras_model, batch_size=None, start_end_idxs=None, need_plot=False):
        """
        Compute the average loss function score on the evaluation dataset.
        Plot the IO of the model if needed.
        """

        # Evaluation set
        # For RNN split the input data into chunks of length IN_SEQ_LENGTH and HOP_SIZE
        self.x_eval = bufferize_array(self.x_eval, self.INPUT_LENGTH, hop_size=self.HOP_SIZE)
        # For RNN split the groundtruth data into chunks of length OUT_SEQ_LENGTH and HOP_SIZE
        self.y_eval = bufferize_array(self.y_eval, self.OUTPUT_LENGTH, hop_size=self.HOP_SIZE,
                                            start_index=self.INPUT_LENGTH - self.OUTPUT_LENGTH)
        return super(KerasBufferizedNNHandler, self).evaluate_model(keras_model, batch_size=batch_size,
                                                                    start_end_idxs=start_end_idxs,
                                                                    need_plot=need_plot)


    def predict(self, keras_model, x_inputs, batch_size=None):
        """
        Override predict() parent method to unbufferize output
        """
        bufferized_predictions = keras_model.predict(x_inputs, batch_size=batch_size)
        return unbufferize_array(bufferized_predictions, self.HOP_SIZE)
