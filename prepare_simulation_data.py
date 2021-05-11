import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from bilateral_filters import *
from Inverse_dynamics import *
import os
from biomechanical_algorithms import *
from trigger_motion_data import *
from external_force import *


def get_pickle(PATH):
    files = os.listdir(PATH)
    for item in files:
        if item[-3:] == 'txt':
            return PATH + item


testNums = [1, 2, 3, 4, 5]

for num in testNums:
    print('num is ', num)
    SubPATH = './IsometricData/FCL/'
    pickle_path = SubPATH + 'test' + str(num) + '/'
    Muscle_PATH = SubPATH + 'MUSCLE/'

    output_filename = SubPATH + 'test' + str(num) + '/' + 'recons.trc'
    origin_pickle = get_pickle(pickle_path)
    mot_dataFrame = read_pickle(origin_pickle)

    trig_bi, trig_ti, trig_time, trig_elbow_flexion, trig_supination, trig_elbow_acce = trigger_motion_data(mot_dataFrame)
    # plt.plot(trig_time, trig_elbow_flexion, c='r', label='Elbow Flexion')
    # plt.plot(trig_time, trig_supination, c='g', label='Wrist Supination')
    # plt.plot(trig_time, norm_bi, c='b', label='Biceps Activation')
    # plt.plot(trig_time[1:], elbow_velo, c='y', label='Elbow Acceleration')
    # plt.legend()
    # plt.show()
    ext_force_file = SubPATH + 'test' + str(num) + '/' + 'MaxForce.xlsx'
    target_len = len(trig_elbow_flexion)
    ext_force = upsample_ext_force(ext_force_file, target_len, fs=2000)
    # norm_bi, norm_ti = normalization(trig_bi), normalization(trig_ti)
    generate_reconst_data(trig_time, trig_elbow_flexion, trig_supination, trig_elbow_acce, trig_bi, trig_ti, Muscle_PATH, ext_force, output_filename)





