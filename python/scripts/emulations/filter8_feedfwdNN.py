#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-11 21:17:33
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
filter8_feedfwdNN.py
Try to emulate Joranalogue's Filter8 eurorack module using bufferized feed-forward NN model with Keras API.
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
The use of bufferized data feed to NN is motivated by a real-time implementation.
"""

import os, sys



file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

# Append python modules directory into python path
sys.path.append(os.path.join(root_dir, "python/modules/"))

# Custom imports
from dataset_utils import create_pkl_audio_dataset
from plot_utils import plot_by_key
from NN_utils import create_feedfwdNN, optimizer_call
from dsp_utils import librosa_write_wav
from training_utils import KerasBufferizedNNTrainer


# Comment this line if you have CUDA installed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    """
    Main routine
    """
    # define plot flags
    need_plot = True

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

    ## DATA FORMATING FOR NN ##
    # Training hyper parameters
    IO_BUFF_SIZE = 2048
    N_FEATURES = 3
    BATCH_SIZE = 16
    EPOCHS = 100

    # Define model trainer
    my_model_trainer = KerasBufferizedNNTrainer(pkl_dir, input_signal_keys, output_signal_keys,
                                                optimizer_call(lr=1e-3), "mse",
                                                IO_BUFF_SIZE, IO_BUFF_SIZE, IO_BUFF_SIZE,
                                                epochs=EPOCHS, batch_size=BATCH_SIZE)

    if need_plot:
        plot_by_key(my_model_trainer.train_dict, my_model_trainer.train_dict.keys(), title="Extract from train dataset",
                    start_idx=int(5e4), end_idx=int(5.1e4))
        plot_by_key(my_model_trainer.val_dict, my_model_trainer.val_dict.keys(), title="Extract from validation dataset",
                    start_idx=int(2e4), end_idx=int(2.1e4))
    my_model_trainer.prepare_datasets()
    # Define NN model
    model = create_feedfwdNN(IO_BUFF_SIZE, IO_BUFF_SIZE, n_features=N_FEATURES)
    # Start model training
    my_model_trainer.train_model(model)

    ## MODEL TESTING ##
    # Do evaluation
    eval_score = my_model_trainer.evaluate_model(model, need_plot=need_plot)
    print("Evaluation score = {}".format(eval_score))

    # Predict output of evaluation dataset inputs
    eval_predictions = my_model_trainer.predict(model, my_model_trainer.x_eval)
    # Save to audio file
    if not os.path.exists(audio_out_dir):
        os.makedirs(audio_out_dir)
    savefilepath = os.path.join(audio_out_dir, "infered_data.wav")
    librosa_write_wav(eval_predictions, savefilepath, sr=sr)

    return


if __name__ == "__main__":
    main()

