"""
Author: thuaj@connect.ust.hk
Date: 2024-10-23 12:40:25
LastEditTime: 2024-10-23 13:17:05
Description: calculate the projection matrix for kitti dataset
Copyright (c) 2024 by thuaj@connect.ust.hk, All Rights Reserved.
"""

import cv2
import fire
import numpy as np
import pandas as pd

"""
usage example python3 panoptic_mapping_utils/src/kitti_dataset/lidar_camera_projection_matrix.py ~/Data/KITTI/
"""


def calculation_projection_matrix(calib_file):
    calibration = pd.read_csv(calib_file, delimiter=" ", header=None, index_col=0)

    # x = Pi * Tr * X -----> this formular can already achieve , we do not need to decompose the projection matrix P
    P2 = np.array(calibration.loc["P2:"]).reshape((3, 4))
    Tr = np.array(calibration.loc["Tr:"]).reshape((3, 4))  # T_C0_L

    Rr, tr = Tr[:3, :3], Tr[:3, 3].reshape(3, 1)
    # qr = R.from_matrix(Rr).as_quat()

    # decomposing projection matrix
    # ref
    # k0,r0,t0,_,_,_,_= cv2.decomposeProjectionMatrix(P0)
    # t0 = (t0/t0[3])[:3].reshape(3,1) #for homogeneous coordinates

    # Pcam=[P0,P1,P2,P3]
    Pcam = [P2]
    for id, P in enumerate(Pcam):
        id = 2
        print(f"-------------{calib_file}---------------")
        k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)  # get the position of target camera under ref T_C0_Cx
        t = (t / t[3])[:3].reshape(3, 1)  # for homogeneous coordinates

        # T_c0_cx = [r, t]
        # T_cx_c0 =  inv(T_c0_cx) = [r.T, -r.T@t]
        # T_c0_L = [Rr, tr]
        # T_cx_L = T_cx_c0 * T_c0_L = [r.T, -r.T@t] [Rr, tr] = [r.T@Rr ,  r.T@tr - r.T@t]

        ext_r = r.transpose() @ Rr
        # print(r.shape, t[:3].shape, tr.shape)
        ext_t = r.transpose() @ (tr.reshape(3, 1) - t[:3].reshape(3, 1))

        print(f"rotation matrix_ext R_C{id}_L: \n", ext_r.flatten())
        print(f"translation vector_ext t_C{id}_L:\n", ext_t.flatten().round(10))
        print("")

        # P2@Tr@X = k@[r.transpose(), -r.transpose()@t]@[Rr,tr]@[x, 1]


def main(dataset_root_path):
    for idx in range(10):
        # seq is two digit number like 00 01 02 03..
        seq = f"{idx:02d}"
        calib_file = dataset_root_path + seq + "/calib.txt"
        calculation_projection_matrix(calib_file)


if __name__ == "__main__":

    fire.Fire(main)

# 00
# rotation matrix_ext R_C2_L:
#  [ 4.27680239e-04 -9.99967248e-01 -8.08449168e-03 -7.21062651e-03
#   8.08119847e-03 -9.99941316e-01  9.99973865e-01  4.85948581e-04
#  -7.20693369e-03]
# translation vector_ext t_C2_L:
#  [ 0.04795398 -0.05517103 -0.2884171 ]
# 01
# rotation matrix_ext R_C2_L:
#  [ 4.27680239e-04 -9.99967248e-01 -8.08449168e-03 -7.21062651e-03
#   8.08119847e-03 -9.99941316e-01  9.99973865e-01  4.85948581e-04
#  -7.20693369e-03]
# translation vector_ext t_C2_L:
#  [ 0.04795398 -0.05517103 -0.2884171 ]
# 02
# rotation matrix_ext R_C2_L:
#  [ 4.27680239e-04 -9.99967248e-01 -8.08449168e-03 -7.21062651e-03
#   8.08119847e-03 -9.99941316e-01  9.99973865e-01  4.85948581e-04
#  -7.20693369e-03]
# translation vector_ext t_C2_L:
#  [ 0.04795398 -0.05517103 -0.2884171 ]
# 03
# rotation matrix_ext R_C2_L:
#  [ 2.34773698e-04 -9.99944155e-01 -1.05634778e-02  1.04494074e-02
#   1.05653536e-02 -9.99889574e-01  9.99945389e-01  1.24365378e-04
#   1.04513030e-02]
# translation vector_ext t_C2_L:
#  [ 0.05705245 -0.07546672 -0.26938691]
#  04
# rotation matrix_ext R_C2_L:
#  [-0.00185774 -0.99996595 -0.00803998 -0.00648147  0.00805186 -0.99994661
#   0.99997731 -0.00180553 -0.0064962 ]
# translation vector_ext t_C2_L:
#  [ 0.05624655 -0.07481402 -0.32779358]
# 05
# rotation matrix_ext R_C2_L:
#  [-0.00185774 -0.99996595 -0.00803998 -0.00648147  0.00805186 -0.99994661
#   0.99997731 -0.00180553 -0.0064962 ]
# translation vector_ext t_C2_L:
#  [ 0.05624655 -0.07481402 -0.32779358]
# 06
# rotation matrix_ext R_C2_L:
#  [-0.00185774 -0.99996595 -0.00803998 -0.00648147  0.00805186 -0.99994661
#   0.99997731 -0.00180553 -0.0064962 ]
# translation vector_ext t_C2_L:
#  [ 0.05624655 -0.07481402 -0.32779358]
# 07
# rotation matrix_ext R_C2_L:
#  [-0.00185774 -0.99996595 -0.00803998 -0.00648147  0.00805186 -0.99994661
#   0.99997731 -0.00180553 -0.0064962 ]
# translation vector_ext t_C2_L:
#  [ 0.05624655 -0.07481402 -0.32779358]
# 08
# rotation matrix_ext R_C2_L:
#  [-0.00185774 -0.99996595 -0.00803998 -0.00648147  0.00805186 -0.99994661
#   0.99997731 -0.00180553 -0.0064962 ]
# translation vector_ext t_C2_L:
#  [ 0.05624655 -0.07481402 -0.32779358]
# 09
# rotation matrix_ext R_C2_L:
#  [-0.00185774 -0.99996595 -0.00803998 -0.00648147  0.00805186 -0.99994661
#   0.99997731 -0.00180553 -0.0064962 ]
# translation vector_ext t_C2_L:
#  [ 0.05624655 -0.07481402 -0.32779358]
# 10
# rotation matrix_ext R_C2_L:
#  [-0.00185774 -0.99996595 -0.00803998 -0.00648147  0.00805186 -0.99994661
#   0.99997731 -0.00180553 -0.0064962 ]
# translation vector_ext t_C2_L:
#  [ 0.05624655 -0.07481402 -0.32779358]
