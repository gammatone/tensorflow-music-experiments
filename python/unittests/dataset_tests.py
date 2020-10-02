#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-02 22:23:00
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$

"""
dataset_tests.py
A set of unit tests for dataset manipulation
"""
import os, sys
import shutil

import unittest
import pickle
import numpy as np

file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

# Append python modules directory into python path
sys.path.append(os.path.join(root_dir, "python/modules/"))

# Custom imports
from dsp_utils import librosa_load_wav

# Define data directory
data_dir = os.path.join(root_dir, "data/unittests/dataset_tests")


class TestAudioDatasets(unittest.TestCase):

    def test_load_wav_file(self):
        """
        Load a test .wav file
        Check array length and samplerate
        """
        wav_filename = "test.wav"
        wanted_sr = 44100
        array, sr = librosa_load_wav(os.path.join(data_dir, wav_filename), sr=wanted_sr)
        self.assertEqual(array.shape, (44927,))
        self.assertEqual(sr, (wanted_sr))

    def test_save_load_wav_to_pickle(self):
        """
        Try to save and load .wav file samples converted into numpy array dictionary into a .pkl file 
        """
        wav_filename = "test.wav"
        pkl_filename = "test.pkl"
        my_dict = {}
        my_key = "test_wav"

        # Create tmp folder
        tmp_dir = os.path.join(data_dir, "tmp")
        os.makedirs(tmp_dir)

        pkl_path = os.path.join(tmp_dir, pkl_filename)

        array, _ = librosa_load_wav(os.path.join(data_dir, wav_filename), sr=44100)
        my_dict[my_key] = array

        # Save dict
        with open(pkl_path, 'wb') as f:
            pickle.dump(my_dict, f)        
        self.assertTrue(os.path.exists(pkl_path))

        # load dict
        with open(pkl_path, 'rb') as f:
            loaded_dict = pickle.load(f) 

        self.assertEqual(next(iter(loaded_dict)), my_key)  

        # Delete tmp folder and content
        shutil.rmtree(tmp_dir)
        self.assertFalse(os.path.exists(pkl_path))
        self.assertFalse(os.path.exists(tmp_dir))

    def test_bufferize_array(self):
        from array_utils import bufferize_array
        test_array = np.array([1,2,3,4,5,6,7,8,9])
        chunk_array = bufferize_array(test_array, chunk_size=4, hop_size=2)
        self.assertTrue(np.array_equal(chunk_array, np.array([[1,2,3,4], [3,4,5,6], [5,6,7,8]])))
        chunk_array = bufferize_array(test_array, chunk_size=3, hop_size=2)
        self.assertTrue(np.array_equal(chunk_array, np.array([[1,2,3], [3,4,5], [5,6,7], [7,8,9]])))
        chunk_array = bufferize_array(test_array[:-2], chunk_size=3, hop_size=2)
        self.assertTrue(np.array_equal(chunk_array, np.array([[1,2,3], [3,4,5], [5,6,7]])))

suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioDatasets)
unittest.TextTestRunner(verbosity=2).run(suite)
