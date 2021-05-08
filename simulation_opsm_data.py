import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from bilateral_filters import *
from Inverse_dynamics import *
from biomechanical_algorithms import *
import time


#  YY 74.5 186   LFC 102 189
# sub = SUB(M_fh=2.244, l_r=0.13104, l_h=0.19, l_f=0.27)  #  LFC
# muscle_group = {'TRIlong': Muscle('TRIlong', PCSA=15.93, max_iso_force=771.8 * 1.6, LM_opti=0.14954786938536857),
#                 'BIClong': Muscle('BIClong', PCSA=5.28, max_iso_force=525.1 * 1.6, LM_opti=0.1274375141444374),
#                 'BRA': Muscle('BRA', 6.18, max_iso_force=1177.37 * 1.4, LM_opti=0.09412172039927078),
#                 'BRD': Muscle('BRD', 1.7, max_iso_force=276.0 * 1.4, LM_opti=0.180769053337085),
#                 'PRO': Muscle('PRO', 4.11, max_iso_force=557.2 * 1.4, LM_opti=0.0510547516322843)}


# sub = SUB(M_fh=1.639, l_r=0.13104, l_h=0.20, l_f=0.28)  #  YY
# muscle_group = {'TRIlong': Muscle('TRIlong', PCSA=15.93, max_iso_force=771.8 * 0.6, LM_opti=0.14843214266132004),
#                 'BIClong': Muscle('BIClong', PCSA=5.28, max_iso_force=525.1 * 0.6, LM_opti=0.12741665324312956),
#                 'BRA': Muscle('BRA', 6.18, max_iso_force=1177.37 * 0.6, LM_opti=0.09465961482578092),
#                 'BRD': Muscle('BRD', 1.7, max_iso_force=276.0 * 0.6, LM_opti=0.18844965719547496),
#                 'PRO': Muscle('PRO', 4.11, max_iso_force=557.2 * 0.6, LM_opti=0.05360455206381705)}


sub = SUB(M_fh=1.683, l_r=0.1287, l_h=0.196, l_f=0.275)  # LRL
muscle_group = {'TRIlong': Muscle('TRIlong', PCSA=15.93, max_iso_force=771.8 * 1, LM_opti=0.13051784157517385),
                'BIClong': Muscle('BIClong', PCSA=5.28, max_iso_force=525.1 * 1, LM_opti=0.11551281106132918),
                'BRA': Muscle('BRA', 6.18, max_iso_force=1177.37 * 1, LM_opti=0.08519797910711775),
                'BRD': Muscle('BRD', 1.7, max_iso_force=276.0 * 1, LM_opti=0.1804005529459397),
                'PRO': Muscle('PRO', 4.11, max_iso_force=557.2 * 1, LM_opti=0.051972967699792136)}


def start_simulation(PATH, testNum):
    '''
    :param recons_datafile: filename of reconstructed datafile which contains the elbow-flexion angle, elbow-flexion accelerations,
                            movement arms, normalized muscle fiber lengths, fiber lengthening velocities
    :return:
    '''
    recons_df = pd.read_excel(PATH + 'recons.xlsx')
    # print(recons_df)
    frame_num = recons_df['Frame']
    elbow_flex_ = recons_df['elbow_flex']
    elbow_flex_ac_ = recons_df['elbow_acce']
    ext_force = recons_df['ext_force']
    time_seq = recons_df['time']
    MAs = {'TRIlong': recons_df['MA-TRIlong'], 'BIClong': recons_df['MA-BIClong'], 'BRA': recons_df['MA-BRA'],
           'BRD': recons_df['MA-BRD'], 'PRO': recons_df['MA-PRO']}
    LMs = {'TRIlong': recons_df['LM-TRIlong'], 'BIClong': recons_df['LM-BIClong'], 'BRA': recons_df['LM-BRA'],
           'BRD': recons_df['LM-BRD'], 'PRO': recons_df['LM-PRO']}
    LM_velos = {'TRIlong': recons_df['LV-TRIlong'], 'BIClong': recons_df['LV-BIClong'], 'BRA': recons_df['LV-BRA'],
                'BRD': recons_df['LV-BRD'], 'PRO': recons_df['LV-PRO']}
    muscle_acts = {'TRI': list(), 'BIC': list(), 'BRA': list(), 'BRD': list(), 'PRO': list(), 'JointTorque': list()}
    time_ = list()
    init_act = [0, 0, 0, 0, 0, 0]

    for i in range(len(frame_num)):
        frame_MAs, frame_LMs, frame_LVs = {}, {}, {}
        for muscle in ['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']:
            frame_MAs[muscle] = MAs[muscle][i]
            frame_LMs[muscle] = LMs[muscle][i]
            frame_LVs[muscle] = LM_velos[muscle][i]
        tmp_frame = Frame(time_inst=time_seq[i], elbow_flexion=elbow_flex_[i], acce=elbow_flex_ac_[i], elbow_supination=80,
                          frame_MAs=frame_MAs, frame_LMs=frame_LMs, frame_LVs=frame_LVs, ext_force=ext_force[i])
        opti_activation, joint_torque = static_opti(tmp_frame, sub, muscle_group=muscle_group, frameNum=i, init_opti=init_act, cost=1)
        init_act = opti_activation

        muscle_acts['TRI'].append(opti_activation[0])
        muscle_acts['BIC'].append(opti_activation[1])
        muscle_acts['BRA'].append(opti_activation[2])
        muscle_acts['BRD'].append(opti_activation[3])
        muscle_acts['PRO'].append(opti_activation[4])
        muscle_acts['JointTorque'].append(joint_torque)
        time_.append(time_seq[i])

    prediction = {'time': time_, 'TRIlong': muscle_acts['TRI'], 'BIClong': muscle_acts['BIC'], 'BRA': muscle_acts['BRA'],
                  'BRD': muscle_acts['BRD'], 'PRO': muscle_acts['PRO'], 'Torque': muscle_acts['JointTorque']}
    # prediction = np.array(prediction)
    # prediction_tosave = prediction.transpose()
    prediction_tosave = pd.DataFrame(prediction)
    prediction_tosave.to_excel(PATH + 'test' + str(testNum) + 'result_without_Fpe_cost_01.xlsx')


PATH = './IsometricData/LRL/test'

testNums = [1, 2, 3, 4, 5]
for num in testNums:
    print('start simulation for test %d' % num)
    testPATH = PATH + str(num) + '/'
    start_simulation(testPATH, num)


