import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def read_pickle(filename):
    myfile = open(filename, 'rb')
    data = pickle.load(myfile)
    myfile.close()
    return data


def write_pickle(filename, data):
    myfile = open(filename, 'wb')
    pickle.dump(data, myfile)
    myfile.close()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def emg_muscle_activation_interpretion(emg, A=-1):
    a_t = list()
    for i in range(len(emg)):
        tmp_a_t = (np.exp(A * emg[i]) - 1) / (np.exp(A) - 1)
        a_t.append(tmp_a_t)
    return a_t


def get_corre_filename(imu_ps_angle):
    angle_range = np.arange(181)
    index = (np.abs(imu_ps_angle - angle_range)).argmin()
    return index


def get_mot_dict(filename):
    df_ = {'time': list(), 'elbow_flexion': list(), 'BIClong': list(), 'BRA': list(), 'BRD': list(), 'PT': list(), 'TRIlong': list()}
    Flag = True
    with open(filename) as mot:
        for i, line in enumerate(mot):
            if i > 5:
                line = line.replace(' ', '')
                line = line.replace('\n', '')
                ls = line.split("\t")
                if i == 6:
                    for key in ['BIClong', 'BRA', 'BRD', 'PT', 'TRIlong']:
                        if key not in ls:
                            Flag = False
                if i > 6 and Flag:
                    df_['time'].append(int(float(ls[0])))
                    df_['elbow_flexion'].append((float(ls[1])))
                    df_['BIClong'].append((float(ls[2])))
                    df_['BRA'].append((float(ls[3])))
                    df_['BRD'].append((float(ls[4])))
                    df_['PT'].append((float(ls[5])))
                    df_['TRIlong'].append((float(ls[6])))
        df_ = pd.DataFrame(df_)
        return df_


def find_MAs_and_norm_lm(joint_flexion, wrist_sup, SubPath):
    MAs = {'TRIlong': list(), 'BIClong': list(), 'BRA': list(), 'BRD': list(), 'PRO': list()}
    LMs = {'TRIlong': list(), 'BIClong': list(), 'BRA': list(), 'BRD': list(), 'PRO': list()}
    LM_funcs = get_norm_muscle_fiber_length(SubPath)
    MA_funcs = get_MAs_funcs(SubPath)
    for i in range(len(joint_flexion)):
        cur_ps = get_corre_filename(wrist_sup[i])
        funcNum = int(180 - cur_ps)
        for muscle in ['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']:
            tmp_MA = MA_funcs[muscle][funcNum](joint_flexion[i])
            tmp_LM = LM_funcs[muscle][funcNum](joint_flexion[i])
            MAs[muscle].append(tmp_MA)
            LMs[muscle].append(tmp_LM)
    return MAs, LMs


def get_MAs_funcs(SubPath):
    MA_funcs = {'TRIlong': list(), 'BIClong': list(), 'BRA': list(), 'BRD': list(), 'PRO': list()}
    for i in range(181):
        filename = SubPath + 'MA_Sup_at_' + str(i) + '.mot'
        MA_df = get_mot_dict(filename)
        MA_BIC, MA_TRI, MA_BRA, MA_BRD, MA_PRO = list(MA_df['BIClong']), list(MA_df['TRIlong']), list(
            MA_df['BRA']), list(
            MA_df['BRD']), list(MA_df['PT'])
        joint_angle = list(MA_df['elbow_flexion'])
        for i in range(len(joint_angle)):
            joint_angle[i] = joint_angle[i] * np.pi / 180.0
        func_tri = np.polyfit(joint_angle, MA_TRI, 35)
        MA_funcs['TRIlong'].append(np.poly1d(func_tri))

        func_bic= np.polyfit(joint_angle, MA_BIC, 8)
        MA_funcs['BIClong'].append(np.poly1d(func_bic))

        func_bra = np.polyfit(joint_angle, MA_BRA, 38)
        MA_funcs['BRA'].append(np.poly1d(func_bra))

        func_brd = np.polyfit(joint_angle, MA_BRD, 35)
        MA_funcs['BRD'].append(np.poly1d(func_brd))

        func_pro = np.polyfit(joint_angle, MA_PRO, 15)
        MA_funcs['PRO'].append(np.poly1d(func_pro))
    return MA_funcs


def get_norm_muscle_fiber_length(SubPath):
    norm_l_m_funcs = {'TRIlong': list(), 'BIClong': list(), 'BRA': list(), 'BRD': list(), 'PRO': list()}
    for i in range(181):
        filename = SubPath + 'Sup_at_' + str(i) + '.mot'
        norm_l_m_df = get_mot_dict(filename)
        norm_l_m_BIC, norm_l_m_TRI, norm_l_m_BRA, norm_l_m_BRD, norm_l_m_PRO = list(norm_l_m_df['BIClong']), list(
            norm_l_m_df['TRIlong']), list(norm_l_m_df['BRA']), list(norm_l_m_df['BRD']), list(norm_l_m_df['PT'])
        joint_angle = list(norm_l_m_df['elbow_flexion'])
        for i in range(len(joint_angle)):
            joint_angle[i] = joint_angle[i] * np.pi / 180.0

        func_tri = np.polyfit(joint_angle, norm_l_m_TRI, 5)
        norm_l_m_funcs['TRIlong'].append(np.poly1d(func_tri))

        func_bic = np.polyfit(joint_angle, norm_l_m_BIC, 5)
        norm_l_m_funcs['BIClong'].append(np.poly1d(func_bic))

        func_bra = np.polyfit(joint_angle, norm_l_m_BRA, 5)
        norm_l_m_funcs['BRA'].append(np.poly1d(func_bra))

        func_brd = np.polyfit(joint_angle, norm_l_m_BRD, 5)
        norm_l_m_funcs['BRD'].append(np.poly1d(func_brd))

        func_pro = np.polyfit(joint_angle, norm_l_m_PRO, 5)
        norm_l_m_funcs['PRO'].append(np.poly1d(func_pro))
    return norm_l_m_funcs


def get_lengthening_velo(time, LMs):
    LM_velos = {'TRIlong': list(), 'BIClong': list(), 'BRA': list(), 'BRD': list(), 'PRO': list()}
    for i in range(len(LMs['TRIlong'])):
        if i >= 1:
            for muscle in ['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']:
                tmp_l_velo = (LMs[muscle][i] - LMs[muscle][i - 1]) / (time[i] - time[i - 1])
                LM_velos[muscle].append(tmp_l_velo)
        else:
            for muscle in ['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']:
                LM_velos[muscle].append(0)
    return LM_velos


def generate_reconst_data(time, elbow_flex, wrist_sup, elbow_acce, bicep_emg, tricep_emg, SubPath, ext_force, out_filename):
    MAs, LMs = find_MAs_and_norm_lm(elbow_flex, wrist_sup, SubPath)
    LM_velos = get_lengthening_velo(time, LMs)
    tmp_data = [time, elbow_flex, elbow_acce, wrist_sup, MAs['TRIlong'], MAs['BIClong'], MAs['BRA'], MAs['BRD'], MAs['PRO'], LMs['TRIlong'],
                LMs['BIClong'], LMs['BRA'], LMs['BRD'], LMs['PRO'], LM_velos['TRIlong'], LM_velos['BIClong'],
                LM_velos['BRA'], LM_velos['BRD'], LM_velos['PRO'], ext_force, bicep_emg, tricep_emg]
    tmp_data = np.array(tmp_data)
    output_data = tmp_data.transpose()
    with open(out_filename, 'w') as out_f:
        out_f.write("Reconstructed Data File\n")
        out_f.write("Frame\ttime\telbow_flex\telbow_acce\twrist_sup\tMA-TRIlong\tMA-BIClong\tMA-BRA\tMA-BRD\tMA-PRO")
        out_f.write("\tLM-TRIlong\tLM-BIClong\tLM-BRA\tLM-BRD\tLM-PRO")
        out_f.write("\tLV-TRIlong\tLV-BIClong\tLV-BRA\tLV-BRD\tLV-PRO\text_force\tBIClong-EMG\tTRIlong-EMG\n")
        for i in range(len(elbow_flex)):
            out_f.write('%d' % (i + 1))
            for j in range(len(tmp_data)):
                out_f.write("\t%f" % output_data[i][j])
            out_f.write("\n")
    Frame = list()
    for i in range(len(time)):
        Frame.append(i)
    tmp_dict = {'Frame': Frame, 'time': time, 'elbow_flex': elbow_flex, 'elbow_acce': elbow_acce, 'wrist_sup': wrist_sup,
              'MA-TRIlong': MAs['TRIlong'], 'MA-BIClong': MAs['BIClong'], 'MA-BRA': MAs['BRA'], 'MA-BRD': MAs['BRD'],
              'MA-PRO': MAs['PRO'], 'LM-TRIlong': LMs['TRIlong'], 'LM-BIClong': LMs['BIClong'], 'LM-BRA': LMs['BRA'], 'LM-BRD': LMs['BRD'],
              'LM-PRO': LMs['PRO'], 'LV-TRIlong': LM_velos['TRIlong'], 'LV-BIClong': LM_velos['BIClong'],
              'LV-BRA': LM_velos['BRA'], 'LV-BRD': LM_velos['BRD'], 'LV-PRO': LM_velos['PRO'], 'ext_force': ext_force,
              'BIClong-EMG': bicep_emg, 'TRIlong-EMG': tricep_emg}
    tmp_df = pd.DataFrame(tmp_dict)
    tmp_df.to_excel(out_filename[:-3] + 'xlsx', index=False)

# df = read_pickle('2021-03-30-16-43_TestSynch-3.txt')
# elbow_flexion = list(df['Noraxon MyoMotion-Joints-Elbow RT-Flexion (deg)'])
# # MA_df = pd.read_excel('sup_at90_normalized_muscle_fiber_length_against_elbowflexion.xlsx', skiprows=range(0, 6))
# MA_df = pd.read_excel('sup_at90_muscle_moment_arm_against_elbowflexion.xlsx', skiprows=range(0, 6))
# MA_BIC = list(MA_df['PT'])
# joint_angle = list(MA_df['/jointset/elbow/elbow_flexion/value'])
#
# func_bic = np.polyfit(joint_angle, MA_BIC, 5)
# print('func_bic is :\n', func_bic)
# p1 = np.poly1d(func_bic)
# print('p1 is :\n', p1)
# # ls = [elbow_flexion[0], elbow_flexion[1], elbow_flexion[2], elbow_flexion[3], elbow_flexion[4]]
# yvals = p1(joint_angle)  # 拟合y值
# print('yvals is :\n', yvals)
#
# plot1 = plt.plot(joint_angle, MA_BIC, 's', label='original values')
# plot2 = plt.plot(joint_angle, yvals, 'r', label='original values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4) #指定legend的位置右下角

# myMA_filename = 'sup_at90_muscle_moment_arm_against_elbowflexion.xlsx'
# print(MA_df)
# for i in range(len(elbow_flexion)):
# find_MAs(elbow_flexion, myMA_filename)









