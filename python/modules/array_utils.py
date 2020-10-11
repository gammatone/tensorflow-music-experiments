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

def bufferize_array(in_array, chunk_size, hop_size, start_index=0, end_index=None):
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
    if end_index is None:
        end_index = in_array.shape[0]
    chunks_nb = (end_index - start_index + hop_size - chunk_size) // hop_size
    chunks_array = np.zeros((chunks_nb, chunk_size) + in_array.shape[1:], dtype=in_array.dtype)
    for i in range(chunks_nb):
        chunks_array[i] = in_array[i*hop_size + start_index: i*hop_size + start_index + chunk_size]
    return chunks_array

def unbufferize_array(chunked_array, hop_size):
    """
    Do the reverse operation of bufferize_array() with the right provided hop_size.
    Overlapping samples (due to hop_size) are overwritten by next chunk samples.
    Assuming the chunk dimension is on the first axis.
    Arguments
        chunked_array (2D np array):bufferized array
        hop_size (int):             corresponding hop size

    Returns
        out_array (1D np array): unbufferized array

    Examples
        unbufferize_array([[1, 2, 3, 4],[3, 4, 5, 6]], 2) returns [1, 2, 3, 4, 5, 6]
        whereas unbufferize_array([[1, 2, 3, 4],[3, 4, 5, 6], 3) returns [1, 2, 3, 3, 4, 5, 6]
    """
    chunk_nb = chunked_array.shape[0]
    chunk_size = chunked_array.shape[1]
    array_length = hop_size * (chunk_nb - 1) + chunk_size
    out_array = np.zeros(((array_length,) + chunked_array.shape[2:]), dtype=chunked_array.dtype)
    chunk_count = 0
    for i in range(chunk_nb):
        out_array[i*hop_size:i*hop_size+chunk_size] = chunked_array[i]
    return out_array
