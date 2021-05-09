import numpy as np
from scipy import optimize
import time


g_acce = 9.8066


class Frame:
    def __init__(self, time_inst, elbow_flexion, acce, elbow_supination, frame_MAs, frame_LMs, frame_LVs, ext_force):
        '''
        :param time_inst:
        :param elbow_flexion:
        :param acce:
        :param elbow_supination:
        :param frame_MAs: {'TRIlong': recons_df['MA-TRIlong'][frameNum], 'BIClong': recons_df['MA-BIClong'][frameNum],
                           'BRA': recons_df['MA-BRA'][frameNum], 'BRD': recons_df['MA-BRD'][frameNum], 'PRO': recons_df['MA-PRO'][frameNum]}
        :param frame_LMs:{'TRIlong': recons_df['LM-TRIlong'][frameNum], 'BIClong': recons_df['LM-BIClong'][frameNum], 'BRA': recons_df['LM-BRA'][frameNum],
                          'BRD': recons_df['LM-BRD'][frameNum], 'PRO': recons_df['LM-PRO'][frameNum]}
        :param frame_LVs:{'TRIlong': recons_df['LV-TRIlong'][frameNum], 'BIClong': recons_df['LV-BIClong'][frameNum], 'BRA': recons_df['LV-BRA'][frameNum],
                'BRD': recons_df['LV-BRD'][frameNum], 'PRO': recons_df['LV-PRO'][frameNum]}
        '''
        self.elbow_supination = elbow_supination
        self.elbow_flexion = elbow_flexion
        self.time_inst = time_inst
        self.acce = acce
        self.frame_MAs = frame_MAs
        self.frame_LMs = frame_LMs
        self.frame_LVs = frame_LVs
        self.ext_force = ext_force


class SUB:
    def __init__(self, l_r, M_fh, l_f, l_h):
        self.l_r = l_r
        self.M_fh = M_fh
        self.l_f = l_f
        self.l_h = l_h


class Muscle:

    def __init__(self, name, PCSA, max_iso_force, LM_opti):
        self.name = name
        self.PCSA = PCSA
        self.max_iso_force = max_iso_force
        self.LM_opti = LM_opti
        self.force = 0
        self.F_ce = 0
        self.F_pe = 0
        self.LV_opti = 10 * self.LM_opti
        # self.init_MA_paras()

    def init_MA_paras(self):
        if self.name == 'BIC':
            self.paras = [8.4533, 36.6147, 2.4777, -19.432, 2.0571, 13.6502, 0, 0, -5.6172, 0, -2.0854, 0, 0, 0, 0, 0]
        elif self.name == 'TRI':
            self.paras = [-24.5454, -8.8691, 9.3509, -1.7518, 0]
        elif self.name == 'BRA':
            self.paras = [16.1991, -16.1463, 24.5512, -6.3335, 0]
        elif self.name == 'BRD':
            self.paras = [15.2564, -11.8355, 2.8129, -5.7781, 44.8143, 0, 2.9032, 0, 0, -13.4956, 0, -0.3940, 0, 0, 0,
                          0]
        elif self.name == 'PRO':
            self.paras = [11.0405, -1.0079, 0.3933, -10.4824, -12.1639, -0.4369, 36.9174, 3.5232, -10.4223, 21.2604,
                          -37.2444, 10.2666, -11.0060, 14.5974, -3.9919, 1.7526, -2.0089, 0.5460]
        else:
            print('Muscle: ', self.name, 'is invalid')

    def get_moment_arm(self, joint_angle):
        paras = self.paras
        MA = 0
        if len(paras) == 5:
            MA = paras[0] + paras[1] * joint_angle[0] + paras[2] * (joint_angle[0] ** 2) + paras[3] * (
                        joint_angle[0] ** 3) + paras[4] * (joint_angle[0] ** 4)
        elif len(paras) == 16:
            q1, q2 = joint_angle[0], joint_angle[1]
            MA = paras[0] + paras[1] * q1 + paras[2] * q2 + paras[3] * q1 * q2 + paras[4] * (q1 ** 2) + paras[5] * (
                        q2 ** 2) + paras[6] * (q1 ** 2) * q2 + paras[7] * q1 * (q2 ** 2) + paras[8] * (q1 ** 2) * (
                             q2 ** 2) + paras[9] * (q1 ** 3) + paras[10] * (q2 ** 3) + paras[11] * (q1 ** 3) * q2 + \
                 paras[12] * q1 * (q2 ** 3) + paras[13] * (q1 ** 3) * (q2 ** 2) + paras[14] * (q1 ** 2) * (q2 ** 3) + \
                 paras[15] * (q1 ** 3) * (q2 ** 3)
        elif len(paras) == 18:
            q1, q2 = joint_angle[0], joint_angle[1]
            MA = paras[0] + paras[1] * q1 + paras[2] * q2 + paras[3] * q1 * q2 + paras[4] * (q1 ** 2) + paras[5] * (
                        q2 ** 2) + paras[6] * (q1 ** 2) * q2 + paras[7] * q1 * (q2 ** 2) + paras[8] * (q1 ** 2) * (
                             q2 ** 2) + paras[9] * (q1 ** 3) + paras[10] * (q1 ** 3) * q2 + paras[11] * (q1 ** 3) * (
                             q2 ** 2) + paras[12] * (q1 ** 4) + paras[13] * (q1 ** 4) * q2 + paras[14] * (q1 ** 4) * (
                             q2 ** 2) + paras[15] * (q1 ** 5) + paras[16] * (q1 ** 5) * q2 + paras[17] * (q1 ** 5) * (
                             q2 ** 2)
        return MA

    def get_F_pe(self, LM):
        norm_LM = LM / self.LM_opti
        if LM > self.LM_opti:
            F_pe = np.exp(10*(norm_LM - 1)) * self.max_iso_force / np.exp(5)
        else:
            F_pe = 0

        return F_pe

    def get_F_ce(self, LM, LV, activation):
        norm_LM = LM / (self.LM_opti * (0.15 * (1 - activation) + 1))
        norm_LV = LV / self.LV_opti

        f_l_a = np.exp(-(norm_LM - 1)**2 / 0.45)

        if norm_LV > 0:
            f_v_ = (2.34 * norm_LV + 0.039) / (1.3 * norm_LV + 0.039)
        else:
            f_v_ = 0.3 * (norm_LV + 1) / (0.3 - norm_LV)

        F_ce = activation * self.max_iso_force * f_v_ * f_l_a
        return F_ce


def motion_equation(frame, sub_info):
    # torque_m = 0
    torque_g = sub_info.M_fh * sub_info.l_r * np.sin(frame.elbow_flexion) * g_acce
    torque_acce = sub_info.M_fh * sub_info.l_r * sub_info.l_r * frame.acce
    torque_ex = frame.ext_force * (sub_info.l_f * np.cos(frame.elbow_flexion - np.pi / 2) + sub_info.l_h * 0.5)
    # torque_m = torque_g + torque_ex

    return torque_g, torque_ex, torque_acce


# @jit(float64[:](int64, Muscle, float64[:]), nopython=True, parallel=True)
def calculate_Fce_for_all(frame, muscle_group, activation):
    F_ce = {}
    for i, muscle in enumerate(['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']):
        # F_pe[muscle] = muscle_group[muscle].get_F_pe(frame.frame_LMs[muscle])
        F_ce[muscle] = muscle_group[muscle].get_F_ce(frame.frame_LMs[muscle], frame.frame_LVs[muscle], activation[i])

    return F_ce


def calculate_Fpe_for_all(frame, muscle_group):
    F_pe = {}
    for i, muscle in enumerate(['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']):
        F_pe[muscle] = muscle_group[muscle].get_F_pe(frame.frame_LMs[muscle])
    return F_pe


def get_solution_range(init_opt):
    low_bound, high_bound, resolu = list(), list(), list()
    range_ = 0.5
    for i, val in enumerate(init_opt):
        tmp_low = val - range_
        tmp_high = val + range_
        if tmp_low < 0:
            tmp_low = 0

        if tmp_high > 1.3:
            tmp_high = 1.3
        low_bound.append(tmp_low)
        high_bound.append(tmp_high)
        resolu.append(40)
    return low_bound, high_bound, resolu


def static_opti(frame, sub_info, muscle_group, frameNum, init_opti, cost=1):
    torque_g, torque_ex, torque_acce = motion_equation(frame, sub_info)
    joint_torque = torque_g + torque_ex
    # joint_angle = [np.pi - frame.elbow_flexion, frame.elbow_supination]
    low_bound, high_bound, resolu = get_solution_range(init_opti)
    F_pe = calculate_Fpe_for_all(frame, muscle_group)
    print(
        '\rframeNumber: %d\tBIC F_pe: %.3f\tTRI F_pe: %.4f\tElbow flexion: %.3f\tGravity torque: %.6f\tExternal torque: %.4f\tAcceleration torque: %.4f' % (
        frameNum, F_pe['BIClong'], F_pe['TRIlong'], frame.elbow_flexion, torque_g, torque_ex, torque_acce), flush=True, end='')
    # print(
    #     '\rframeNumber: %d\tBIC F_pe: %.3f\tTRI F_pe: %.4f\tElbow flexion: %.3f\tGravity torque: %.6f\tExternal torque: %.4f\tAcceleration torque: %.4f' % (
    #         frameNum, F_pe['BIClong'], F_pe['TRIlong'], frame.elbow_flexion, torque_g, torque_ex, torque_acce))

    def cost_f1(act):
        '''
        :param force: list object, length is 5, force_ce for all seletect muscle
        :return:
        '''
        Fce = calculate_Fce_for_all(frame, muscle_group, act)
        cost = 0
        for i, muscle in enumerate(['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']):
            cost += ((Fce[muscle] + F_pe[muscle]) / muscle_group[muscle].PCSA)**2
            # cost += (Fce[muscle] / muscle_group[muscle].PCSA) ** 2
        return cost

    def cost_f2(act):
        '''
        :param force: list object, length is 5, force_ce for all seletect muscle
        :return:
        '''
        Fce = calculate_Fce_for_all(frame, muscle_group, act)
        cost = 0
        for i, muscle in enumerate(['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']):
            cost = ((Fce[muscle] + F_pe[muscle]) / muscle_group[muscle].max_iso_force) ** 3 + (
                        (Fce[muscle] + F_pe[muscle]) / muscle_group[muscle].PCSA) ** 2

            # cost = (Fce[muscle] / muscle_group[muscle].max_iso_force) ** 2 + (
            #         Fce[muscle] / muscle_group[muscle].PCSA) ** 2
        return cost

    def subject_f(act):
        Fce = calculate_Fce_for_all(frame, muscle_group, act)
        st = 0
        for i, muscle in enumerate(['TRIlong', 'BIClong', 'BRA', 'BRD', 'PRO']):
            st += ((Fce[muscle] + F_pe[muscle]) * frame.frame_MAs[muscle])
            # st += (Fce[muscle] * frame.frame_MAs[muscle])
        st = st - joint_torque
        return st

    constrain = (
        {'type': 'eq', 'fun': subject_f},
        {'type': 'ineq', 'fun': lambda act: high_bound[0] - act[0]},
        {'type': 'ineq', 'fun': lambda act: high_bound[1] - act[1]},
        {'type': 'ineq', 'fun': lambda act: high_bound[2] - act[2]},
        {'type': 'ineq', 'fun': lambda act: high_bound[3] - act[3]},
        {'type': 'ineq', 'fun': lambda act: high_bound[4] - act[4]},
        {'type': 'ineq', 'fun': lambda act: act[0] - low_bound[0]},
        {'type': 'ineq', 'fun': lambda act: act[1] - low_bound[1]},
        {'type': 'ineq', 'fun': lambda act: act[2] - low_bound[2]},
        {'type': 'ineq', 'fun': lambda act: act[3] - low_bound[3]},
        {'type': 'ineq', 'fun': lambda act: act[4] - low_bound[4]}
    )
    if cost == 1:
        '''
        pre optimize
        '''
        x_start = optimize.brute(cost_f1, (
            slice(low_bound[0], high_bound[0], resolu[0]), slice(low_bound[1], high_bound[1], resolu[1]),
            slice(low_bound[2], high_bound[2], resolu[2]),
            slice(low_bound[3], high_bound[3], resolu[3]), slice(low_bound[4], high_bound[4], resolu[4])), finish=None)
        '''Optimization'''
        result_pre = optimize.minimize(cost_f1, x_start, method='SLSQP', constraints=constrain,
                                       options={'maxiter': 1e4})
    elif cost == 2:
        x_start = optimize.brute(cost_f2, (
            slice(low_bound[0], high_bound[0], resolu[0]), slice(low_bound[1], high_bound[1], resolu[1]),
            slice(low_bound[2], high_bound[2], resolu[2]),
            slice(low_bound[3], high_bound[3], resolu[3]), slice(low_bound[4], high_bound[4], resolu[4])), finish=None)
        result_pre = optimize.minimize(cost_f2, x_start, method='SLSQP', constraints=constrain,
                                       options={'maxiter': 1e4})

    opti_act = result_pre.x
    return opti_act, joint_torque



