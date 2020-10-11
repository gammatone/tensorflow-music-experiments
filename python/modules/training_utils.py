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
from array_utils import bufferize_array, unbufferize_array
from plot_utils import plot_by_key


class ModelTrainer():
    """
    A class that will load pickle dictionary files containing train and validation data to perform model training.
    """
    def __init__(   self,
                    pkl_load_dir, input_keys, groundtruth_keys,
                    train_filename="train.pkl", val_filename="validation.pkl", eval_filename="evaluation.pkl",
                    epochs=100, batch_size=32,
                    ):
        """
        """
        # define pickle directories (where the pickle files are located)
        self.pkl_load_dir = pkl_load_dir
        # Define train & val datasets pickle filepaths 
        self.pkl_train_filepath = os.path.join(os.path.join(self.pkl_load_dir, train_filename))
        self.pkl_val_filepath = os.path.join(os.path.join(self.pkl_load_dir, val_filename))
        self.pkl_eval_filepath = os.path.join(os.path.join(self.pkl_load_dir, eval_filename))
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
        Load pickle dictionaries for train and validation sets
        """
        self.train_dict = get_dict_from_pkl(self.pkl_train_filepath)
        self.val_dict = get_dict_from_pkl(self.pkl_val_filepath)
        self.eval_dict = get_dict_from_pkl(self.pkl_eval_filepath)

    def prepare_datasets(self):
        """
        For both train and validation sets: Stack data into a numpy array acoording to input and groundtruth keys.
        """
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

    def train_model(self, keras_model):
        """
        Compile provided keras_model with selected optimizer andstart training
        """

        keras_model.compile(loss=self.loss, optimizer=self.optimizer)
        training_state = keras_model.fit(self.x_train, self.y_train, shuffle=True,
                                    epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                                    validation_data=(self.x_val, self.y_val),
                                    validation_freq=3,
                                    )

    def evaluate_model(self, keras_model, need_plot=False):
        """
        Compute the average loss function score on the evaluation dataset.
        Plot the IO of the model if needed
        """
        if need_plot:
            predicted_output = self.predict(keras_model, self.x_eval)
            array_dict = {}
            # Append signals from evaluation dict
            for key in self.eval_dict.keys():
                array_dict[key] = self.eval_dict[key][:]
            # Append predictions in dict
            array_dict["predictions"] = predicted_output
            plot_by_key(array_dict, array_dict.keys(), title="Model evaluation")
        return keras_model.evaluate(self.x_eval, self.y_eval)

    def predict(self, keras_model, x_inputs):
        """
        """
        return keras_model.predict(x_inputs)



class KerasBufferizedNNTrainer(KerasModelTrainer):
    """
    Keras Model trainer child class.
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
        super(KerasBufferizedNNTrainer, self).__init__(pkl_load_dir, input_keys, groundtruth_keys,
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
        super(KerasBufferizedNNTrainer, self).prepare_datasets()

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


        # Evaluation set
        # For RNN split the input data into chunks of length IN_SEQ_LENGTH and HOP_SIZE
        self.x_eval = bufferize_array(self.x_eval, self.INPUT_LENGTH, hop_size=self.HOP_SIZE)
        # For RNN split the groundtruth data into chunks of length OUT_SEQ_LENGTH and HOP_SIZE
        self.y_eval = bufferize_array(self.y_eval, self.OUTPUT_LENGTH, hop_size=self.HOP_SIZE,
                                            start_index=self.INPUT_LENGTH - self.OUTPUT_LENGTH)


    def predict(self, keras_model, x_inputs):
        """
        Override predict() parent method to unbufferize output
        """
        bufferized_predictions = keras_model.predict(x_inputs)
        return unbufferize_array(bufferized_predictions, self.HOP_SIZE)



