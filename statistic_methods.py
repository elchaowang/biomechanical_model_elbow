def mean(x):
  return sum(x) / len(x)


# error between mean and each item of a list
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


# methods: dot product sum_of_squares
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


# variance
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)


import math


#  standard deviation
def standard_deviation(x):
    return math.sqrt(variance(x))


def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)


# calculating correlation
def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0


import pandas as pd
from bilateral_filters import prediction_emg_filter, normalization
from biomechanical_algorithms import emg_muscle_activation_interpretion


df = pd.read_excel('./IsometricData/LRL/test1/recons.xlsx', skiprows=range(0, 1))
bi_emg = df['BIClong-EMG']
ti_emg = df['TRIlong-EMG']
bi_emg = normalization(bi_emg)
bi_emg = emg_muscle_activation_interpretion(bi_emg, A=-2.5)

bi_pre_cs1_df = pd.read_excel('./IsometricData/LRL/test1/test1result_with_Fpe_cost1.xlsx')
bi_pre_cs1 = list(bi_pre_cs1_df['BIClong'])
bi_pre_cs1 = prediction_emg_filter(bi_pre_cs1)
bi_pre_cs1 = normalization(bi_pre_cs1)

bi_pre_cs2_df = pd.read_excel('./IsometricData/LRL/test1/test1result_with_Fpe_cost2.xlsx')
bi_pre_cs2 = list(bi_pre_cs2_df['BIClong'])
bi_pre_cs2 = prediction_emg_filter(bi_pre_cs2)
bi_pre_cs2 = normalization(bi_pre_cs2)


