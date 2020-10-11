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

# Custom imports
from dataset_utils import get_dict_from_pkl, stack_array_from_dict_lastaxis
from array_utils import bufferize_array


class ModelTrainer():
    """
    A class that will load pickle dictionary files containing train and test data to perform model training.
    """
    def __init__(   self,
                    pkl_load_dir, input_keys, groundtruth_keys,
                    epochs=100, batch_size=32,
                    ):
        """
        """
        # define pickle directories (where the pickle files are located)
        self.pkl_load_dir = pkl_load_dir
        # Define train & test datasets pickle filepaths 
        self.pkl_train_filepath = os.path.join(os.path.join(self.pkl_load_dir, "train.pkl"))
        self.pkl_test_filepath = os.path.join(os.path.join(self.pkl_load_dir, "test.pkl"))
        # Load pickle data
        self.load_pkl_data()

        # Define signal keys (i.e. key to load in the pickle dict for both input and groundtruth data)
        self.input_keys = input_keys
        self.groundtruth_keys = groundtruth_keys

        # Training hyper-parameters
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

        return

    def load_pkl_data(self):
        """
        Load pickle dictionaries for train and test sets
        """
        self.train_dict = get_dict_from_pkl(self.pkl_train_filepath)
        self.test_dict = get_dict_from_pkl(self.pkl_test_filepath)

    def prepare_datasets(self):
        """
        For both train and test sets: Stack data into a numpy array acoording to input and groundtruth keys.
        """
        # Train set
        # Stack inputs (i.e. features) on last axis
        self.x_train = stack_array_from_dict_lastaxis(self.train_dict, self.input_keys)
        self.y_train = stack_array_from_dict_lastaxis(self.train_dict, self.groundtruth_keys)

        # Test set
        # Stack inputs (i.e. features) on last axis
        self.x_test = stack_array_from_dict_lastaxis(self.test_dict, self.input_keys)
        self.y_test = stack_array_from_dict_lastaxis(self.test_dict, self.groundtruth_keys)

    # Abstract
    def train_model(self):
        return

    # Abstract
    def evaluate_model(self):
        return



class KerasModelTrainer(ModelTrainer):
    """
    Model trainer child class using Keras API 
    """
    def __init__(   self,
                    pkl_load_dir, input_keys, groundtruth_keys,
                    keras_optim, keras_loss,
                    **kwargs,
                    ):
        """
        Init parent class and add optimizer & loss
        """
        super(KerasModelTrainer, self).__init__(pkl_load_dir, input_keys, groundtruth_keys, **kwargs)
        self.optimizer = keras_optim
        self.loss = keras_loss

    def train_model(self, model):
        """
        Compile provided model with selected optimizer andstart training
        """

        model.compile(loss=self.loss, optimizer=self.optimizer)
        training_state = model.fit(self.x_train, self.y_train, shuffle=True,
                                    epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                                    validation_data=(self.x_test, self.y_test),
                                    validation_freq=3,
                                    )

class KerasRNNTrainer(KerasModelTrainer):
    """
    Keras Model trainer child class.
    Dedicated for RNN models which require to bufferize datasets before training and testing.

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
        super(KerasRNNTrainer, self).__init__(pkl_load_dir, input_keys, groundtruth_keys,
                                                keras_optim, keras_loss,
                                                **kwargs)
        # RNN hyper-parameters
        self.INPUT_LENGTH = input_length
        self.HOP_SIZE = hop_size
        self.OUTPUT_LENGTH = output_length

    def prepare_datasets(self):
        """
        For both train and test sets: Stack data into a numpy array acoording to input and groundtruth keys.
        """
        super(KerasRNNTrainer, self).prepare_datasets()

        # Train set
        # For RNN split the input data into chunks of length IN_SEQ_LENGTH and HOP_SIZE
        self.x_train = bufferize_array(self.x_train, self.INPUT_LENGTH, hop_size=self.HOP_SIZE)
        # For RNN split the groundtruth data into chunks of length OUT_SEQ_LENGTH and HOP_SIZE
        self.y_train = bufferize_array(self.y_train, self.OUTPUT_LENGTH, hop_size=self.HOP_SIZE,
                                            start_index=self.INPUT_LENGTH - self.OUTPUT_LENGTH)

        # Test set
        # For RNN split the input data into chunks of length IN_SEQ_LENGTH and HOP_SIZE
        self.x_test = bufferize_array(self.x_test, self.INPUT_LENGTH, hop_size=self.HOP_SIZE)
        # For RNN split the groundtruth data into chunks of length OUT_SEQ_LENGTH and HOP_SIZE
        self.y_test = bufferize_array(self.y_test, self.OUTPUT_LENGTH, hop_size=self.HOP_SIZE,
                                            start_index=self.INPUT_LENGTH - self.OUTPUT_LENGTH)



