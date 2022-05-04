#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
import nibabel as nib


def MaxMinNorm(data, type='MR'):
    if type == 'MR':
        data_max = np.amax(data)
        data_min = np.amin(data)
    else:
        # type == 'CT'
        data_max = 2000.0
        data_min = -1000.0
        data[data > data_max] = data_max
        data[data < data_min] = data_min
    print("Before:", data_max, data_min)
    data -= data_min
    data /= (data_max-data_min)
    print("After:", np.amax(data), np.amin(data))
    return data


def NacNorm(data):
    return data/1500
