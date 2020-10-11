#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-02 22:39:31
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
dsp_utils.py
Module gathering useful DSP functions for audio analysis and processings
"""

import os

import librosa

def librosa_load_wav(filepath, sr=22050):
    """
    Load wav file (using librosa) into numpy array

    Returns
        raw_array (np_array)
        sr (int):               samplerate value
    """
    return librosa.load(filepath, sr=sr)

def librosa_write_wav(np_array, savepath, sr=22050):
	"""
	Write wav file (using librosa) into numpy array
	"""
	librosa.output.write_wav(savepath, np_array, sr=sr)
