#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-09 10:42:28
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
filter8_RNN.py
Try to emulate Joranalogue's Filter8 eurorack module using RNN model with Keras API.
The aim is to reproduce the cutoff and resonance effects of the filter applied on any input sound signal.
To do so the model is trained with some recorded data and labels described above:
    - Training dataset (temporal signals recorded for 25s at 44100Hz):
        * Input = white analog noise from Zlob Modular's Entropy module
        * Groundtruth = Input filtered by filter8
        * Labels = Cutoff + resonance params modulated by 2 asynchronous triangle LFOs (0.22Hz and 1.04Hz) from VCV rack
    - Testing dataset (temporal signals recorded for 17s at 44100Hz):
        * Input = Sawtooth wave from After Later Audio's Knit (uPlaits) at 260Hz (C4)
        * Groundtruth = Idem
        * Labels = Idem
The use of RNN type is motivated by its temporal behaviour which enables to make a real-time implementation later.
"""

import os, sys

import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model

file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

# Append python modules directory into python path
sys.path.append(os.path.join(root_dir, "python/modules/"))

# Custom imports
from dataset_utils import create_pkl_audio_dataset
from plot_utils import plot_by_key
from NN_utils import optimizer_call, vanilla_LSTM
from dsp_utils import librosa_write_wav
from training_utils import KerasBufferizedNNHandler


# Comment this line if you have CUDA installed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)


def get_RNN_model(in_length, out_length, n_features, stateful=False, batch_size=None):
    model = vanilla_LSTM(   in_length, out_length, n_features=n_features,
                            stateful=stateful,
                            batch_size=batch_size,
                            )
    return model

def training_routine(pkl_dir, input_signal_keys, output_signal_keys,
                    optimizer=optimizer_call(lr=1e-3), loss_metric="mse",
                    audio_out_dir=None, model_root_save_dir=None,
                    pkl_filenames=("train.pkl", "validation.pkl", "evaluation.pkl")
                    ):
    """
    Training function:
        - Load train & validation dataset
        - Create many-to-many RNN (e.g. IO_size=2048; hop_size=512)
        - Train RNN in stateless mode (i.e. truncated backprop for each chunks of length IO_size)
        - Evaluate trained model
        - Create audio file corresponding to the infered data from evaluation dataset
        - Save model and results (in .h5 and .pkl files)
    """
    # Define model handler for training
    my_model_trainer = KerasBufferizedNNHandler(pkl_dir, input_signal_keys, output_signal_keys,
                                                optimizer, loss_metric,
                                                IO_SEQ_LENGTH, HOP_SIZE, IO_SEQ_LENGTH,
                                                need_training=True,
                                                train_filename=pkl_filenames[0],
                                                val_filename=pkl_filenames[1],
                                                eval_filename=pkl_filenames[2],
                                                )

    if NEED_PLOT:
        plot_by_key(my_model_trainer.train_dict, my_model_trainer.train_dict.keys(), title="Extract from train dataset",
                    start_idx=int(5e4), end_idx=int(5.1e4))
        plot_by_key(my_model_trainer.val_dict, my_model_trainer.val_dict.keys(), title="Extract from validation dataset",
                    start_idx=int(2e4), end_idx=int(2.1e4))
    my_model_trainer.prepare_datasets()
    # Define STATELESS RNN model
    model = get_RNN_model(in_length=IO_SEQ_LENGTH, out_length=IO_SEQ_LENGTH, n_features=N_FEATURES,
                            stateful=False,
                            )
    model.summary()

    ## MODEL TRAINING ##
    training_state = my_model_trainer.train_model(model, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                    val_freq=VALIDATION_FREQ)
    # Define a dict for training losses
    training_losses = {}
    training_losses["loss"] = np.array(training_state.history["loss"])
    training_losses["val_loss"] = np.repeat(np.array(training_state.history["val_loss"]), VALIDATION_FREQ)
    plot_by_key(training_losses, 
            ["loss", "val_loss",],
            title="Losses value during training",
            )

    ## MODEL EVALUATION ##
    # Evaluate trained model (i.e. many-to-many samples model)  
    print("Evaluating trained model")
    eval_score = my_model_trainer.evaluate_model(model, need_plot=NEED_PLOT)
    print("Evaluation score = {}".format(eval_score))

    print("Inference on trained model")
    # Predict output of evaluation dataset inputs from trained model
    eval_predictions = my_model_trainer.predict(model, my_model_trainer.x_eval)
    print("Saving inference output to audio file")

    ## AUDIO FILE WRITING ##
    if audio_out_dir is not None:
        # Save to audio file
        if not os.path.exists(audio_out_dir):
            os.makedirs(audio_out_dir)
        savefilepath = os.path.join(audio_out_dir, "infered_data_trained_model.wav")
        # Use numpy asfortranarray() function to ensure Fortran contiguity on the array
        librosa_write_wav(np.asfortranarray(eval_predictions), savefilepath, sr=SAMPLE_RATE)

    ## MODEL SAVING ##
    model_save_dir = None
    if model_root_save_dir is not None:
        # Save model and results into a sub directory specified by current data/time
        import datetime
        model_save_dir = os.path.join(model_root_save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(model_save_dir)
        # Save useful constants from training as pickle
        result_dict = {}
        result_dict["sample_rate"] = SAMPLE_RATE
        result_dict["epochs"] = EPOCHS
        result_dict["batch_size"] = BATCH_SIZE
        result_dict["hop_size"] = HOP_SIZE
        result_dict["loss_metric"] = loss_metric
        result_dict["optimizer"] = optimizer
        result_dict["val_freq"] = VALIDATION_FREQ
        result_dict["training_losses"] = training_losses
        with open(os.path.join(model_save_dir,"training_results.pkl"), "wb") as f:
            pickle.dump(result_dict, f)
        # Save model in the sub directory
        model.save(os.path.join(model_save_dir,'trained_model.h5'),)

    return model_save_dir

def testing_routine(savedmodel_dir, pkl_dir, input_signal_keys, output_signal_keys,
                    audio_out_dir=None, test_pkl_filename="evaluation.pkl",
                    ):
    """
    Testing function:
        - Load a pretrained many-to-many RNN Keras model
        - Create one-to-one stateful RNN (i.e. the real-time implementation model)
        with same graph than pretrained model
        - Copy weight from the pretrained model
        - Train RNN in stateless mode (i.e. truncated backprop for each chunks of length IO_size)
        - Evaluate stateful model
        - Create audio file corresponding to the infered data from a test dataset
    """
    ## PRETRAINED MODEL LOADING ##
    # Load pretrained stateless model
    for file in os.listdir(savedmodel_dir):
        if file == "trained_model.h5":
            pretrained_model = load_model(os.path.join(savedmodel_dir, file),)
            break
    # Load training infos from saved dictionary
    with open(os.path.join(savedmodel_dir, "training_results.pkl"), "rb") as f:
        training_results_dict = pickle.load(f)
    # Deduce optimizer and loss metric to be able to compile real-time model later
    optimizer = training_results_dict["optimizer"]
    loss_metric = training_results_dict["loss_metric"]

    ## REAL-TIME MODEL CREATION ##
    # Define model handler for testing one-to-one stateful model:
    # means only evaluation dataset (no train and validation datasets)
    # means io_seq_length=1; hop_size=1; 
    my_rt_model_tester = KerasBufferizedNNHandler(pkl_dir, input_signal_keys, output_signal_keys,
                                                optimizer, loss_metric,
                                                1, 1, 1,
                                                need_training=False,
                                                eval_filename=test_pkl_filename,
                                                )
    my_rt_model_tester.prepare_datasets()

    print("\nOne-to-one RNN model (i.e. the one used in real-time) creation according to trained model config")
    rt_model = get_RNN_model(in_length=1, out_length=1, n_features=N_FEATURES,
                            stateful=True,
                            batch_size=1,
                            )
    rt_model.summary()
    # Copy weights from pretrained model
    training_weights = pretrained_model.get_weights()
    rt_model.set_weights(training_weights)

    ## REAL-TIME MODEL EVALUATION ##
    print("Evaluating real-time model")
    eval_duration_s = 5.
    eval_score = my_rt_model_tester.evaluate_model(rt_model, batch_size=1,
                                                    start_end_idxs=(0, int(eval_duration_s * SAMPLE_RATE)),
                                                    need_plot=NEED_PLOT)
    print("Evaluation score = {}".format(eval_score))
    print("Inference on real-time model")
    # Predict output of evaluation dataset inputs from stateful model
    eval_predictions = my_rt_model_tester.predict(rt_model, my_rt_model_tester.x_eval, batch_size=1)
    print("Saving inference output to audio file")

    ## AUDIO FILE WRITING ##
    # Save to audio file
    if not os.path.exists(audio_out_dir):
        os.makedirs(audio_out_dir)
    savefilepath = os.path.join(audio_out_dir, "infered_data_realtime_model.wav")
    # Use numpy asfortranarray() function to ensure Fortran contiguity on the array
    librosa_write_wav(np.asfortranarray(eval_predictions), savefilepath, sr=SAMPLE_RATE)

    return

def main():
    """
    Main routine
    """
    global NEED_PLOT
    global SAMPLE_RATE
    global IO_SEQ_LENGTH, HOP_SIZE, N_FEATURES
    global EPOCHS, BATCH_SIZE, VALIDATION_FREQ
    # define plot flags
    NEED_PLOT = False

    # define audio parameters
    SAMPLE_RATE = 44100

    # Training hyper parameters
    IO_SEQ_LENGTH = 4096
    HOP_SIZE = 1024
    N_FEATURES = 3
    BATCH_SIZE = 32
    EPOCHS = 500
    VALIDATION_FREQ = 3

    # Define signal keys
    input_signal_keys = ["signal_in", "cutoff", "resonance"]
    output_signal_keys = ["signal_filtered"]
    signal_keys = input_signal_keys + output_signal_keys

    # define pickle directories (where the pickle files are located)
    pkl_dir = os.path.join(root_dir, "data/pickle/filter8_rec")
    pkl_train_filename = "train.pkl"
    pkl_val_filename = "validation.pkl"
    pkl_eval_filename = "evaluation.pkl"
    my_pkl_filenames = (pkl_train_filename, pkl_val_filename, pkl_eval_filename)
    # define audio directories (where the audio files are located)
    audio_dir = os.path.join(root_dir, "data/audio/filter8_rec")
    audio_train_dir = os.path.join(audio_dir, "white_noise")
    audio_val_dir = os.path.join(audio_dir, "saw_wave")
    audio_eval_dir = os.path.join(audio_dir, "saw_wave")
    # Define an audio output directory
    audio_out_dir = os.path.join(audio_dir, "inference")
    my_audio_dirs = (audio_train_dir, audio_val_dir, audio_eval_dir)
    # Define where to save trained model
    model_root_save_dir = os.path.join(root_dir, "data/keras_models/RNN")

    # Save to pickle if dataset does not exist yet
    # First make sure pickle directories exist
    for i in range(len(my_pkl_filenames)):
        if not os.path.exists(my_pkl_filenames[i]):
            # Create pickle dictionary
            create_pkl_audio_dataset(my_audio_dirs[i], pkl_dir, my_pkl_filenames[i], keynames=signal_keys, sr=SAMPLE_RATE)

    # Train, evaluate and save RNN stateless model
    savedmodel_dir = training_routine(pkl_dir, input_signal_keys, output_signal_keys,
                                    optimizer=optimizer_call(lr=1e-3), loss_metric="mse",
                                    audio_out_dir=audio_out_dir, model_root_save_dir=model_root_save_dir,
                                    )



    testing_routine(savedmodel_dir, pkl_dir,
                    input_signal_keys, output_signal_keys,
                    audio_out_dir=audio_out_dir, test_pkl_filename="evaluation.pkl",
                    )


if __name__ == "__main__":
    main()
