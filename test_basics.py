import pandas as pd
from bilateral_filters import *
import matplotlib.pyplot as plt
from biomechanical_algorithms import *
import scipy.stats as stats
from statistic_methods import *
# import scipy.signal.c


# df = pd.read_excel('./IsometricData/YY_IMU/2021-04-21-16-13_SpasAssess_01-3_recons.xlsx', skiprows=range(0, 1))
df = pd.read_excel('./IsometricData/LFC/test5/recons.xlsx', skiprows=range(0, 1))
bi_emg = df['BIClong-EMG']
ti_emg = df['TRIlong-EMG']

# bi_pre_df = pd.read_excel('YYprediction_muscle_activation_test3-without_Fpe.xlsx')
# bi_pre_df = pd.read_excel('./IsometricData/LRL/test1/test1result_with_Fpe_BIC_15.xlsx')
bi_pre_df = pd.read_excel('./IsometricData/LFC/test5/test5result_with_Fpe.xlsx')
# ti_pre = list(bi_pre_df['TRIlong'])
bi_pre = list(bi_pre_df['BIClong'])
# ba_pre = list(bi_pre_df['BRA'])
# bd_pre = list(bi_pre_df['BRD'])
# pt_pre = list(bi_pre_df['PRO'])

bi_pre = prediction_emg_filter(bi_pre)
# bi_pre_without_Fpe = prediction_emg_filter(bi_pre_without_Fpe)

bi_pre = normalization(bi_pre)
bi_emg = normalization(bi_emg)

print(len(bi_pre), len(bi_emg))
print('biceps prediction length: %d' % len(bi_pre))
result_ = stats.pearsonr(bi_pre, bi_emg)
print(result_)
plt.subplot(211)
plt.plot(bi_emg, label='BIC Measured')
plt.plot(bi_pre, label='BIC Predicted')


bi_emg = emg_muscle_activation_interpretion(bi_emg, A=-2.5)
plt.subplot(212)
plt.plot(bi_emg, label='BIC Measured')
plt.plot(bi_pre, label='BIC Predicted')
result_ = stats.pearsonr(bi_pre, bi_emg)

# print('Pearson : ', result_)
corr = correlation(bi_emg, bi_pre)
print('cross-correlation: ', corr)
plt.legend()
plt.show()

