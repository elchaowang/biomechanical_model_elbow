import numpy as np
import pandas as pd
from pandas import DataFrame
import os
from bilateral_filters import *


def cal_deriv(x, y):  # x, y的类型均为列表
    diff_x = []  # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []  # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []  # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []  # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率

    for i in deriv:  # 打印结果，方便检查，调用时也可注释掉
        print(i)

    return deriv  # 返回存储一阶导数结果的列表


def trigger_motion_data(motion_df):
    '''
    :param motion_df: origin IMU and EMG data
    :return: trig data : biceps, triceps EMG, time, elbow flexion and wrist supination
    '''
    # motion_df = pd.read_excel(motion_file, skiprows=range(0, 3))
    print(motion_df.keys())
    bi_emg = list(motion_df['Noraxon Ultium.BIClong (uV)'])
    ti_emg = list(motion_df['Noraxon Ultium.TRIlong (uV)'])
    # bi_emg = list(motion_df['Noraxon Ultium-Noraxon Ultium.BIClong (uV)'])
    # ti_emg = list(motion_df['Noraxon Ultium-Noraxon Ultium.TRIlong (uV)'])

    bi_act, ti_act = emg_2_activation(bi_emg), emg_2_activation(ti_emg)
    time_seq = list(motion_df['time'])
    # elbow_flexion = list(motion_df['RT 肘关节屈曲 (deg)'])
    elbow_flexion = list(motion_df['LT Elbow Flexion (deg)'])
    # elbow_flexion = list(motion_df['Noraxon MyoMotion-Joints-Elbow LT-Flexion (deg)'])
    # trig = list(motion_df['Noraxon Ultium-Noraxon Ultium.Sync (On)'])
    trig = list(motion_df['Noraxon Ultium.Sync (On)'])  # LFC
    # trig = list(motion_df['Noraxon Ultium.同步 (开)'])  # LRL

    # elbow_supination = list(motion_df['Noraxon MyoMotion-Joints-Wrist LT-Supination (deg)'])
    elbow_supination = list(motion_df['LT Wrist Supination (deg)'])

    start_ind, end_ind = 0, 0
    for i, sig in enumerate(trig):
        if trig[i] == 1 and trig[i + 6] == 1:
            start_ind = i
        elif trig[i] == 1 and trig[i + 6] != 1:
            end_ind = i
    emg_shift = -460
    trig_bi, trig_ti, trig_time = bi_act[start_ind + emg_shift:end_ind + emg_shift], ti_act[start_ind + emg_shift:end_ind + emg_shift], time_seq[start_ind:end_ind]
    trig_elbow_flexion = elbow_flexion[start_ind:end_ind]
    trig_supination = elbow_supination[start_ind:end_ind]
    for i in range(len(trig_elbow_flexion)):
        trig_elbow_flexion[i] = trig_elbow_flexion[i] * np.pi / 180
        # trig_supination[i] = trig_supination[i] * np.pi / 180
    elbow_acce = get_joint_acceleration(time_seq, elbow_flexion)
    trig_elbow_acce = elbow_acce[start_ind: end_ind]

    print(len(trig_bi), len(trig_time))
    return trig_bi, trig_ti, trig_time, trig_elbow_flexion, trig_supination, trig_elbow_acce


def get_joint_acceleration(trig_time, trig_elbow_flexion):
    for i, angle in enumerate(trig_elbow_flexion):
        trig_elbow_flexion[i] = trig_elbow_flexion[i] * np.pi / 180
    low_cut = 150
    b, a = signal.butter(2, 2 * low_cut / 2000, 'lowpass')
    lpf_elbow_angle = signal.filtfilt(b, a, trig_elbow_flexion)
    elbow_velo = cal_deriv(trig_time, lpf_elbow_angle)
    lpf_elbow_velo = signal.filtfilt(b, a, elbow_velo)
    elbow_acce = cal_deriv(trig_time, lpf_elbow_velo)
    low_cut = 10
    b, a = signal.butter(2, 2 * low_cut / 2000, 'lowpass')
    lpf_elbow_acce = signal.filtfilt(b, a, elbow_acce)

    return lpf_elbow_acce



# motion_file = './IsometricData/LFC_IMU/2021-04-20-11-44_spasticity isometric configuration-3.xlsx'
#
# trigger_motion_data(motion_file)




