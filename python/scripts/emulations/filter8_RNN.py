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



file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

# Append python modules directory into python path
sys.path.append(os.path.join(root_dir, "python/modules/"))

# Custom imports
from dataset_utils import create_pkl_audio_dataset, get_dict_from_pkl
from plot_utils import plot_by_key

def main():
    """
    Main routine
    """
    # define plot flags
    need_plot = True

    # define audio parameters
    sr = 44100

    # Define signal keys
    signal_keys = ["signal_in", "signal_filtered", "cutoff", "resonance"]

    # define pickle directories (where the pickle files are located)
    pkl_dir = os.path.join(root_dir, "data/pickle/filter8_rec")
    pkl_train_filename = "train.pkl"
    pkl_test_filename = "test.pkl"
    my_pkl_filenames = (pkl_train_filename, pkl_test_filename)
    # define audio directories (where the audio files are located)
    audio_dir = os.path.join(root_dir, "data/audio/filter8_rec")
    audio_train_dir = os.path.join(audio_dir, "white_noise")
    audio_test_dir = os.path.join(audio_dir, "saw_wave")
    my_audio_dirs = (audio_train_dir, audio_test_dir)

    # Save to pickle if dataset does not exist yet
    # First make sure pickle directories exist
    for i in range(len(my_pkl_filenames)):
        if not os.path.exists(my_pkl_filenames[i]):
            # Create pickle dictionary
            create_pkl_audio_dataset(my_audio_dirs[i], pkl_dir, my_pkl_filenames[i], keynames=signal_keys, sr=sr)

    # Load pickle files
    train_dict = get_dict_from_pkl(os.path.join(pkl_dir, pkl_train_filename))
    test_dict = get_dict_from_pkl(os.path.join(pkl_dir, pkl_test_filename))

    if need_plot:
        plot_by_key(train_dict, train_dict.keys(), title="Extract from train dataset",
                    start_idx=int(5e4), end_idx=int(5.1e4))
        plot_by_key(test_dict, test_dict.keys(), title="Extract from test dataset",
                    start_idx=int(2e4), end_idx=int(2.1e4))

    return


if __name__ == "__main__":
    main()
