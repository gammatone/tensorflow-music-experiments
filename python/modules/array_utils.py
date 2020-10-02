#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-02 23:19:51
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
array_utils.py
Module gathering useful functions for numpy array manipulation
"""

import os
import numpy as np

def bufferize_array(in_array, chunk_size, hop_size):
    """
    Split a 1D array into a 2D array made of chunks from the sliced input array.
    Useful for recurrent models.
    Arguments
        in_array (1D np array):     input array
        chunk_size (int):           chunk size
        hop_size (int):             hop size

    Returns
        chunks_array (2D np array): bufferized array

    Example
        bufferize_array([1, 2, 3, 4, 5, 6], 3, 2)
        returns
        [   [1, 2, 3],
            [3, 4, 5]
                        ]
        (notice that last element is dropped)
    """
    chunks_nb = (in_array.shape[0] - chunk_size) // hop_size + 1
    chunks_array = np.zeros((chunks_nb, chunk_size), dtype=in_array.dtype)
    for i in range(chunks_nb):
        chunks_array[i] = in_array[i*hop_size: i*hop_size + chunk_size]
    return chunks_array
