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
import numpy as np
from bilateral_filters import prediction_emg_filter, normalization
from biomechanical_algorithms import emg_muscle_activation_interpretion
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

# df = pd.read_excel('./IsometricData/YY_L/test3/recons.xlsx')
# bi_emg = df['BIClong-EMG']
# ti_emg = df['TRIlong-EMG']
# bi_emg = normalization(bi_emg)
# bi_emg = emg_muscle_activation_interpretion(bi_emg, A=-2.5)

# bi_pre_cs1_df = pd.read_excel('./IsometricData/YY_L/test3/test3result_with_Fpe.xlsx')
# bi_pre_cs1 = list(bi_pre_cs1_df['BIClong'])
# bi_pre_cs1 = prediction_emg_filter(bi_pre_cs1)
# bi_pre_cs1 = normalization(bi_pre_cs1)
# print(len(bi_pre_cs1), len(bi_emg))
# result_ = stats.pearsonr(bi_pre_cs1, bi_emg)
# print(result_)
# print(f"均方误差(MSE)：{mean_squared_error(bi_pre_cs1, bi_emg)}")
# print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(bi_pre_cs1, bi_emg))}")
# print(f"测试集R^2：{r2_score(bi_pre_cs1, bi_emg)}")

# bi_pre_cs2_df = pd.read_excel('./IsometricData/LRL/test1/test1result_with_Fpe_cube_cost_02.xlsx')
# bi_pre_cs2_df = pd.read_excel('./IsometricData/YY_L/test3/test3result_with_Fpe_cost_01_modified_muscledata.xlsx')
# bi_pre_cs2 = list(bi_pre_cs2_df['BIClong'])
# bi_pre_cs2 = prediction_emg_filter(bi_pre_cs2)
# bi_pre_cs2 = normalization(bi_pre_cs2)
# print(len(bi_pre_cs2), len(bi_emg))
# result_ = stats.pearsonr(bi_pre_cs2, bi_emg)
# print(result_)
# print(f"均方误差(MSE)：{mean_squared_error(bi_pre_cs2, bi_emg)}")
# print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(bi_pre_cs2, bi_emg))}")
# print(f"测试集R^2：{r2_score(bi_pre_cs2, bi_emg)}")
#
#
# plt.plot(bi_emg, label='BIC Measured')
# plt.plot(bi_pre_cs1, label='BIC Predicted without cost 1')
# plt.plot(bi_pre_cs2, label='BIC Predicted with cost 1 modified muscle')
# plt.legend(loc='upper left')
# plt.show()

rmse = list()
r_2 = list()

for i in [1, 2, 4, 5]:
    ori_file = './IsometricData/FCR/test' + str(i) + '/recons.xlsx'
    pre_file = './IsometricData/FCR/test%d/test%dresult_with_Fpe_cost_01_modified_muscledata.xlsx' % (i, i)
    df = pd.read_excel(ori_file)
    bi_emg = df['BIClong-EMG']
    ti_emg = df['TRIlong-EMG']
    bi_emg = normalization(bi_emg)
    bi_emg = emg_muscle_activation_interpretion(bi_emg, A=-2.5)

    bi_pre_cs2_df = pd.read_excel(pre_file)
    bi_pre_cs2 = list(bi_pre_cs2_df['BIClong'])
    bi_pre_cs2 = prediction_emg_filter(bi_pre_cs2)
    bi_pre_cs2 = normalization(bi_pre_cs2)
    print(len(bi_pre_cs2), len(bi_emg))
    # result_ = stats.pearsonr(bi_pre_cs2, bi_emg)
    # print(result_)
    tmp_mse = mean_squared_error(bi_pre_cs2, bi_emg)
    tmp_rmse = np.sqrt(mean_squared_error(bi_pre_cs2, bi_emg))
    tmp_r_2 = r2_score(bi_pre_cs2, bi_emg)
    rmse.append(tmp_rmse)
    r_2.append(tmp_r_2)
    print(f"均方误差(MSE)：{tmp_mse}")
    print(f"根均方误差(RMSE)：{tmp_rmse}")
    print(f"测试集R^2：{tmp_r_2}")

print('RMSE mean: %f,  RMSE std: %f' % (np.mean(rmse), np.std(rmse)))
print('R-squared mean: %f,  R-squared std: %f' % (np.mean(r_2), np.std(r_2)))



