import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from bilateral_filters import *
from Inverse_dynamics import *
import math


def write_pickle(filename, data):
    myfile = open(filename, 'wb')
    pickle.dump(data, myfile)
    myfile.close()


def read_pickle(filename):
    myfile = open(filename, 'rb')
    data = pickle.load(myfile)
    myfile.close()
    return data


def writemyTRC(filename, data):
    with open(filename, 'w') as file_obj:
        DataRate = 2000
        CameraRate = 2000
        NumFrames = len(data) - 1
        NumMarkers = 6
        OrigDataRate = 2000
        OrigDataStartFrame = 1
        OrigNumFrames = len(data) - 1
        Labels = ['r_elbow', 'r_shoulder', 'r_ulnar', 'r_jugular', 'C7', 'T10']
        file_obj.write("PathFileType\t4\t(X/Y/Z)\toutput.trc\n")
        file_obj.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        file_obj.write("%d\t%d\t%d\t%d\tmm\t%d\t%d\t%d\n" % (DataRate, CameraRate, NumFrames, NumMarkers, OrigDataRate,
                                                         OrigDataStartFrame, OrigNumFrames))
        # Write labels
        file_obj.write("Frame#\tTime\t")
        for i, label in enumerate(Labels):
            if i != 0:
                file_obj.write("\t\t\t")
            file_obj.write("%s" % (label))
        file_obj.write("\n")
        file_obj.write("\t")
        for i in range(len(Labels * 3)):
            file_obj.write("\t%c%d" % (chr(ord('X') + (i % 3)), math.ceil((i + 1) / 3)))
        file_obj.write("\n\n")

        # write data
        print('len of the first column is: %d' % NumFrames)
        for i in range(NumFrames):
            file_obj.write("%d\t%f" % (i + 1, data[i][0]))
            for j in range(NumMarkers * 3):
                file_obj.write("\t%f" % data[i][j + 1])
            file_obj.write("\n")


# filename =
# df = pd.read_excel(filename, skiprows=range(0, 3))
# write_pickle('2021-03-30-16-43_TestSynch-3.txt', df)
df = read_pickle('./IsometricData/YY_IMU/2021-04-21-16-13_SpasAssess_01-3.txt')

# print(df.keys())


time_seq = list(df['time'])

elbow_x = list(df['Noraxon MyoMotion-Trajectories-Elbow RT-x (mm)'])
elbow_y = list(df['Noraxon MyoMotion-Trajectories-Elbow RT-y (mm)'])
elbow_z = list(df['Noraxon MyoMotion-Trajectories-Elbow RT-z (mm)'])

shoulder_x = list(df['Noraxon MyoMotion-Trajectories-Shoulder RT-x (mm)'])
shoulder_y = list(df['Noraxon MyoMotion-Trajectories-Shoulder RT-y (mm)'])
shoulder_z = list(df['Noraxon MyoMotion-Trajectories-Shoulder RT-z (mm)'])

Ulnar_x = list(df['Noraxon MyoMotion-Trajectories-Ulnar styloid process RT-x (mm)'])
Ulnar_y = list(df['Noraxon MyoMotion-Trajectories-Ulnar styloid process RT-y (mm)'])
Ulnar_z = list(df['Noraxon MyoMotion-Trajectories-Ulnar styloid process RT-z (mm)'])

Jugular_x = list(df['Noraxon MyoMotion-Trajectories-Jugular notch-x (mm)'])
Jugular_y = list(df['Noraxon MyoMotion-Trajectories-Jugular notch-y (mm)'])
Jugular_z = list(df['Noraxon MyoMotion-Trajectories-Jugular notch-z (mm)'])

C7_x = list(df['Noraxon MyoMotion-Trajectories-7th cervical vertebrae-x (mm)'])
C7_y = list(df['Noraxon MyoMotion-Trajectories-7th cervical vertebrae-y (mm)'])
C7_z = list(df['Noraxon MyoMotion-Trajectories-7th cervical vertebrae-z (mm)'])

T10_x = list(df['Noraxon MyoMotion-Trajectories-10th thoracic vertebrae-x (mm)'])
T10_y = list(df['Noraxon MyoMotion-Trajectories-10th thoracic vertebrae-y (mm)'])
T10_z = list(df['Noraxon MyoMotion-Trajectories-10th thoracic vertebrae-z (mm)'])

ps = [time_seq, elbow_x, elbow_y, elbow_z, shoulder_x, shoulder_y, shoulder_z, Ulnar_x, Ulnar_y, Ulnar_z, Jugular_x,
      Jugular_y, Jugular_z, C7_x, C7_y, C7_z, T10_x, T10_y, T10_z]
ps = np.array(ps)
print(ps.shape)
# psdf = ps.transpose()
# print(psdf.shape)
# psdf = pd.DataFrame(psdf)
# psdf.to_excel('extracted_origin.xlsx')

alpha = -np.pi / 2

rotate_metrix_z = np.array([[ np.cos(alpha), np.sin(alpha), 0],
                            [-np.sin(alpha), np.cos(alpha), 0],
                            [             0,             0, 1]])

rotate_metrix_x = np.array([[  1,              0,              0],
                            [  0,  np.cos(alpha), -np.sin(alpha)],
                            [  0,  np.sin(alpha),  np.cos(alpha)]])

for i in range(4):
    for j in range(len(ps[i])):
        tmp_ps = np.array([ps[i * 3 + 1][j], ps[i * 3 + 2][j], ps[i * 3 + 3][j]])
        rot_ps = np.dot(rotate_metrix_x, tmp_ps)
        ps[i * 3 + 1][j], ps[i * 3 + 2][j], ps[i * 3 + 3][j] = rot_ps[0], rot_ps[1], rot_ps[2]

rot_df = ps.transpose()
out_filename = 'joint_trajectory.trc'
writemyTRC(out_filename, rot_df)
print(rot_df.shape, type(rot_df))
# rot_df = pd.DataFrame(rot_df)
# rot_df.to_excel('rot_ps.xlsx')


from mpl_toolkits.mplot3d import Axes3D


fig0 = plt.figure()
ax0 = plt.axes(projection='3d')
# ax1.scatter3D(elbow_x, elbow_y, elbow_z, cmap='r')  # 绘制散点图
# ax1.scatter3D(shoulder_x, shoulder_y, shoulder_z, cmap='g')  # 绘制散点图
# ax1.scatter3D(Ulnar_x, Ulnar_y, Ulnar_z, cmap='b')  # 绘制散点图
# ax1.scatter3D(Jugular_x, Jugular_y, Jugular_z, cmap='y')
ax0.plot3D(elbow_x, elbow_y, elbow_z, c='r')  # 绘制散点图
ax0.plot3D(shoulder_x, shoulder_y, shoulder_z, c='g')  # 绘制散点图
ax0.plot3D(Ulnar_x, Ulnar_y, Ulnar_z, c='b')  # 绘制散点图
ax0.plot3D(Jugular_x, Jugular_y, Jugular_z, c='y')
plt.show()


fig = plt.figure()
ax1 = plt.axes(projection='3d')
# for i in range(4):
#     ax1.plot3D(ps[i], ps[i + 1], ps[i + 2])  #
ax1.plot3D(ps[1], ps[2], ps[3], c='r')  #
ax1.plot3D(ps[4], ps[5], ps[6], c='g')  #
ax1.plot3D(ps[7], ps[8], ps[9], c='b')  #
ax1.plot3D(ps[10], ps[11], ps[12], c='y')
# ax1.plot3D(x, y, z, 'gray')    # 绘制空间曲线
plt.show()

