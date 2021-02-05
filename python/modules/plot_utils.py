#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-09 10:32:35
# @Author  : gammatone (adrien.bardet@live.fr)
# @Link    : https://github.com/gammatone/tensorflow-music-experiments
# @Version : $Id$
"""
plot_utils.py
Module gathering useful plot functions using matplotlib.
"""

import os

import matplotlib.pyplot as plt

def plot_by_key(arrays_dict, keys_str, title="", start_idx=0, end_idx=None, xaxis_str=None, yaxis_str=None):
    """
    Plot temporal data from a numpy array dictionary.
    """
    fig = plt.figure()

    for key_str in keys_str:
        if end_idx is not None:
            plt.plot(arrays_dict[key_str][start_idx:end_idx])
        else:
            plt.plot(arrays_dict[key_str][start_idx:])

    plt.title(title)
    plt.legend(keys_str)
    if xaxis_str is not None:
        plt.xlabel(xaxis_str)
    if yaxis_str is not None:
        plt.ylabel(yaxis_str)
    plt.grid()
    plt.show()

    plt.close(fig)

    return