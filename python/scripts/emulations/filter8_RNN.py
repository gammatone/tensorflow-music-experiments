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

import tensorflow as tf


file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

# Append python modules directory into python path
sys.path.append(os.path.join(root_dir, "python/modules/"))

# Custom imports
from dataset_utils import create_pkl_audio_dataset
from plot_utils import plot_by_key
from NN_utils import optimizer_call, create_RNN_OutDense, create_RNN_InDense
from dsp_utils import librosa_write_wav
from training_utils import KerasBufferizedNNHandler


# Comment this line if you have CUDA installed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)


def main():
    """
    Main routine
    """
    # define plot flags
    need_plot = False

    # define audio parameters
    sr = 44100

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

    # Save to pickle if dataset does not exist yet
    # First make sure pickle directories exist
    for i in range(len(my_pkl_filenames)):
        if not os.path.exists(my_pkl_filenames[i]):
            # Create pickle dictionary
            create_pkl_audio_dataset(my_audio_dirs[i], pkl_dir, my_pkl_filenames[i], keynames=signal_keys, sr=sr)

    ## DATA FORMATING FOR RNN ##
    # Training hyper parameters
    IN_SEQ_LENGTH = 4096
    HOP_SIZE = 512
    N_FEATURES = 3
    # (Careful: for evaluation, out length should be higher than hop size to recover all the data)
    OUT_SEQ_LENGTH = 4096
    BATCH_SIZE = 32
    EPOCHS = 500

    # Define model trainer
    my_model_trainer = KerasBufferizedNNHandler(pkl_dir, input_signal_keys, output_signal_keys,
                                                optimizer_call(lr=1e-3), "mse",
                                                IN_SEQ_LENGTH, HOP_SIZE, OUT_SEQ_LENGTH,
                                                epochs=EPOCHS, batch_size=BATCH_SIZE)

    if need_plot:
        plot_by_key(my_model_trainer.train_dict, my_model_trainer.train_dict.keys(), title="Extract from train dataset",
                    start_idx=int(5e4), end_idx=int(5.1e4))
        plot_by_key(my_model_trainer.val_dict, my_model_trainer.val_dict.keys(), title="Extract from validation dataset",
                    start_idx=int(2e4), end_idx=int(2.1e4))
    my_model_trainer.prepare_datasets()
    # Define STATELESS RNN model
    model = create_RNN_InDense(IN_SEQ_LENGTH, OUT_SEQ_LENGTH, n_features=N_FEATURES, stateful=False)
    model.summary()
    # Start model training
    my_model_trainer.train_model(model)

    ## MODEL TESTING ##
    # Evaluate trained model (i.e. many-to-many samples model)  
    print("Evaluating trained model")
    eval_score = my_model_trainer.evaluate_model(model, need_plot=False)
    print("Evaluation score = {}".format(eval_score))

    # Redefine a STATEFUL new model to get a one-to-one sample RNN model
    print("\nOne-to-one sample model creation from trained model (i.e. the one used in real-time)")

    # Define model trainer
    my_stateful_model_handler = KerasBufferizedNNHandler(pkl_dir, input_signal_keys, output_signal_keys,
                                                optimizer_call(lr=1e-3), "mse",
                                                1, 1, 1,
                                                )
    my_stateful_model_handler.prepare_datasets()
    stateful_model = create_RNN_InDense(1, 1, n_features=N_FEATURES, stateful=True, batch_size=1)
    stateful_model.summary()
    # Copy weights from trained model
    training_weights = model.get_weights()
    stateful_model.set_weights(training_weights)
    # Do evaluation
    stateful_model.compile(optimizer_call(lr=1e-3), "mse",)
    print("Evaluating real-time model")
    eval_duration_s = 5.
    eval_score = my_stateful_model_handler.evaluate_model(stateful_model, batch_size=1,
                                                    start_end_idxs=(0, int(eval_duration_s * sr)),
                                                    need_plot=need_plot)
    print("Evaluation score = {}".format(eval_score))


    print("Saving trained model inference to audio file")
    # Predict output of evaluation dataset inputs from trained model
    eval_predictions = my_model_trainer.predict(model, my_model_trainer.x_eval, batch_size=1)
    # Save to audio file
    if not os.path.exists(audio_out_dir):
        os.makedirs(audio_out_dir)
    savefilepath = os.path.join(audio_out_dir, "infered_data_trained_model.wav")
    librosa_write_wav(eval_predictions, savefilepath, sr=sr)


    print("Saving real-time model inference to audio file")
    # Predict output of evaluation dataset inputs from stateful model
    eval_predictions = my_stateful_model_handler.predict(stateful_model, my_stateful_model_handler.x_eval, batch_size=1)
    # Save to audio file
    if not os.path.exists(audio_out_dir):
        os.makedirs(audio_out_dir)
    savefilepath = os.path.join(audio_out_dir, "infered_data_realtime_model.wav")
    librosa_write_wav(eval_predictions, savefilepath, sr=sr)

    return


if __name__ == "__main__":
    main()
