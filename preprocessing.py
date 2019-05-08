#coding:utf-8
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
import h5py

from settings import *

def load_npz(npzpath):
    data = np.load(npzpath)
    arr = data["data"]
    return arr

def make_dataset(arr, cut_len):
    arr_left = cut_dataset(arr, cut_len)
    arr_right = cut_dataset(arr, cut_len, left=False)
    arr_left_rev = reverse_data(arr_left)
    arr_right_rev = reverse_data(arr_right)
    data_arr = np.vstack((arr_left, arr_right, arr_left_rev, arr_right_rev))
    return data_arr

def cut_dataset(arr, cut_len, left=True):
    if left:
        return arr[:, :cut_len]
    else:
        return arr[:, -cut_len:]

def reverse_data(arr):
    return arr[::-1]

def over_sampling(arr, cut_len, increase_rate):
    num_data, lc_len = data.shape
    diff = lc_len - cut_len
    window_len = diff // increase_rate
    arr_list = []
    for i in range(increase_rate):
        start = i * window_len
        end = lc_len - (increase_rate - i + 1) * window_len
        cut_arr = data[:, start : end]
    increased_arr = np.vstack(tuple(arr_list))
    return increased_arr

def normalize(arr):
    mid = np.median(arr)
    std = np.std(arr)
    arr = (arr - mid) / std
    return arr
