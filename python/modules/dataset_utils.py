#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-09 12:35:57
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
dataset_utils.py
Module gathering useful functions for dataset handling.
"""

import os

import pickle
import numpy as np

# Custom imports
from dsp_utils import librosa_load_wav
from array_utils import pad_array


def create_pkl_audio_dataset(load_dir, dest_dir, dest_filename, keynames=[], sr=44100):
    """
    Load all the .wav files specified by keynames located in load_dir.
    Store all the arrays in a dictionary and save it in .pkl file in dest dir.
    """
    pkl_dict = {}
    for file in os.listdir(load_dir):
        if file.endswith(".wav"):
            for keyname in keynames:
                if file.startswith(keyname):
                    array, _ = librosa_load_wav(os.path.join(load_dir, file), sr=sr)
                    break
            pkl_dict[keyname] = array
    # Make sure dest_dir exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Save dict
    save_dict_to_pkl(pkl_dict, os.path.join(dest_dir, dest_filename))

def get_dict_from_pkl(pkl_path):
    """
    Load array dictionary from pickle file
    """
    with open(pkl_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def save_dict_to_pkl(pkl_dict, pkl_path):
    # Save dict
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_dict, f)

def induce_IO_delay(pkl_path, input_signal_keys, output_signal_keys, num_samples=0):
    """
    Induce delay between output signals and input signals in dict.
    i.e. input signals are zero-padded at the end whereas output signals at the beginning of the array
    """
    data_dict = get_dict_from_pkl(pkl_path)
    for signal_key in input_signal_keys:
        data_dict[signal_key] = pad_array(data_dict[signal_key], num_samples=num_samples, at_the_end=True)
    for signal_key in output_signal_keys:
        data_dict[signal_key] = pad_array(data_dict[signal_key], num_samples=num_samples, at_the_end=False)
    save_dict_to_pkl(data_dict, pkl_path)

def stack_array_from_dict_lastaxis(data_dict, keys):
    """
    Stack array present in dictionary in a numpy array.
    NB arrays have to have same shape and same type
    """
    # Prepare new shape
    stacked_array = data_dict[keys[0]][:]
    stacked_array = np.expand_dims(stacked_array, axis=-1)
    for i in range(1, len(keys)):
        stacked_array = np.concatenate([stacked_array, np.expand_dims(data_dict[keys[i]], axis=-1)], axis=-1)
    return stacked_array[:]
