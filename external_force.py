import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from bilateral_filters import *


def get_ext_force(ext_filename):
    ext_df = pd.read_excel(ext_filename)
    ext_force = ext_df[0]
    ext_time = ext_df[1]
    return ext_force


def upsample_ext_force(ext_filename, target_length, fs=2000):
    '''
    upsample the origin external force data to the target length which is the  length of the motion data and emg data
    captured by the Noraxon devices
    :param ext_filename: filename of the file that contains the force data
    :param target_length: length of the array after upsample
    :param fs:
    :return: upsampled and filtered external force data
    '''
    ext_force = get_ext_force(ext_filename)
    time_length = target_length / fs

    ext_time = np.linspace(0, time_length, len(ext_force))

    ext_time_new = np.linspace(0, time_length, target_length)

    '''
    B spline
    '''
    ups_tck = interpolate.splrep(ext_time, ext_force)

    ext_force_bspline = interpolate.splev(ext_time_new, ups_tck)

    lpf_up_ext_force = filter_force(ext_force_bspline)
    for i in range(len(lpf_up_ext_force)):
        if lpf_up_ext_force[i] < 0:
            lpf_up_ext_force[i] = 0
    # print(len(ext_force_bspline))
    for i in range(len(lpf_up_ext_force)):
        lpf_up_ext_force[i] = lpf_up_ext_force[i] / 500
    # plt.plot(ext_time, ext_force, 'r', label='origin')
    # plt.plot(ext_time_new, ext_force_bspline, 'g', label='B-spline')
    # plt.plot(ext_time_new, lpf_up_ext_force, 'y', label='filtered B-spline')
    # plt.legend()
    # plt.show()
    # print(len(ext_time))
    return lpf_up_ext_force


# ext_force_file = './IsometricData/LFC/MaxForce_04.xlsx'
# ext_df = pd.read_excel(ext_force_file)
# upsample_ext_force(ext_force_file, 3300, fs=2000)



