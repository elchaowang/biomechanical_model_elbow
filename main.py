import pandas as pd
from bilateral_filters import *
import matplotlib.pyplot as plt
from biomechanical_algorithms import *
import scipy.stats as stats


# df = pd.read_excel('./IsometricData/YY_IMU/2021-04-21-16-13_SpasAssess_01-3_recons.xlsx', skiprows=range(0, 1))
def get_pearsons(SUBPATH, num, withFlag, plotFlag=True):
    res_file = 'result_with_Fpe_06'
    res_file_WITHOUT = 'result_without_Fpe_06'
    recons_file = SUBPATH + str(num) + '/recons.xlsx'
    if withFlag:
        pre_file = SUBPATH + str(num) + '/test' + str(num) + res_file + '.xlsx'
        fig_name = SUBPATH[:-4] + 'ResultData' + '/test' + str(num) + res_file + '.jpg'
    else:
        pre_file = SUBPATH + str(num) + '/test' + str(num) + res_file_WITHOUT + '.xlsx'
        fig_name = SUBPATH[:-4] + 'ResultData' + '/test' + str(num) + res_file_WITHOUT + '.jpg'

    print('Recons File: %s \t Predict File: %s' % (recons_file, pre_file))
    df = pd.read_excel(recons_file, skiprows=range(0, 1))
    bi_emg = df['BIClong-EMG']
    # bi_emg = emg_muscle_activation_interpretion(bi_emg)
    # ti_emg = df['TRIlong-EMG']
    bi_pre_df = pd.read_excel(pre_file)
    # ti_pre = list(bi_pre_df['TRIlong'])
    bi_pre = list(bi_pre_df['BIClong'])
    bi_pre = prediction_emg_filter(bi_pre)
    bi_pre = normalization(bi_pre)

    bi_emg = normalization(bi_emg)
    bi_emg = emg_muscle_activation_interpretion(bi_emg, A=-2.5)
    # print(len(bi_pre), len(bi_emg))
    # print('biceps prediction length: %d' % len(bi_pre))
    if plotFlag:
        plt.plot(bi_emg, label='BIC Measured')
        plt.plot(bi_pre, label='BIC Predicted')
        plt.legend(loc='upper left')
        plt.savefig(fig_name)
        plt.show()
    result_ = stats.pearsonr(bi_pre, bi_emg)
    # print(result_)
    return result_


SUB_PATH = './IsometricData/YY/test'
testNums = [1, 2, 3, 4, 5]
for num in testNums:
    # res = get_pearsons(SUB_PATH, num, withFlag=True, plotFlag=True)
    withoutRes = get_pearsons(SUB_PATH, num, withFlag=False, plotFlag=True)
    # print(res)
    print(withoutRes, '\n')


# SUB_PATHS = ['./IsometricData/LFC/test', './IsometricData/LRL/test', './IsometricData/YY/test']
# testNums = [1, 2, 3, 4, 5]
# result_pearson_with = {'./IsometricData/LFC/test': list(), './IsometricData/LRL/test': list(), './IsometricData/YY/test': list()}
# # result_pearson_without = {'./IsometricData/LFC/test': list(), './IsometricData/LRL/test': list(), './IsometricData/YY/test': list()}
#
# for sub_i, sub_path in enumerate(SUB_PATHS):
#     for num in testNums:
#         (tmp_p, P) = get_pearsons(sub_path, num, withFlag=True, plotFlag=False)
#         # (t, P) = get_pearsons(sub_path, num, withFlag=True, plotFlag=False)
#         # withoutRes = get_pearsons(SUB_PATH, num, withFlag=False)
#         # print(res.)
#         result_pearson_with[sub_path].append(tmp_p)
#         # print(tmp_p, '\n')
#
#
# x = [1, 2, 3]
# y_mean = list()
# y_std_err = list()
#
# for sub in result_pearson_with.keys():
#     print(result_pearson_with[sub])
#     y_mean.append(np.mean(result_pearson_with[sub]))
#     y_std_err.append(np.std(result_pearson_with[sub]))



# a = [0.9687361014377889, 0.9466329674463418, 0.9738666547082966, 0.961909300931842, 0.974740584943934]
# b = [0.9874730409445551, 0.9731241428290527, 0.9879543410220732, 0.972329316108529, 0.9822548293982755]
# c = [0.956174079508681, 0.94048414817128, 0.9602824706089572, 0.9558402213966792, 0.9633885759051616]
# results = [a, b, c]
#
# for result in results:
#     # print(result_pearson_with[sub])
#     y_mean.append(np.mean(result))
#     y_std_err.append(np.std(result))
#
#
# # plt.bar(x, y_mean, yerr=y_std_err, tick_label=['Sub.A', 'Sub.B', 'Sub.C'])
# plt.errorbar(x, y_mean, yerr=y_std_err, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)
# plt.plot(bi_emg, label='BIC Measured')
# plt.plot(bi_pre, label='BIC Predicted')
# plt.legend()
plt.show()