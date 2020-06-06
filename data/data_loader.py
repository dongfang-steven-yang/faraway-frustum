import cv2
import numpy as np
import os


class DatasetLoader:
    def __init__(self, data_type, data_path):
        self.data_type = data_type
        self.data_path = data_path

    def read_raw_data(self, sample_num: str):
        # data paths
        path_sample_velodyne = os.path.join(self.data_path, 'training', 'velodyne', '%s.bin' % sample_num)
        path_sample_img = os.path.join(self.data_path, 'training', 'image_2', '%s.png' % sample_num)
        path_sample_cal = os.path.join(self.data_path, 'training', 'calib', '%s.txt' % sample_num)

        # ------ read pointcloud ------
        points_3d_lidar = np.fromfile(path_sample_velodyne, dtype=np.float32, count=-1).reshape([-1, 4])
        points_3d_lidar = points_3d_lidar[points_3d_lidar[:, 0] > 0]  # only use points in front of the vehicle

        # ------ read image ------
        img = cv2.imread(path_sample_img)

        # ------ read calibration info ------
        Ms_cal = []
        with open(path_sample_cal) as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip('\n').split(' ')
                Ms_cal.append(l)
        # P0-P3 for cameras, see the sensor setup page: http://www.cvlibs.net/datasets/kitti/setup.php
        P0 = np.array(Ms_cal[0][1:]).reshape(3, 4).astype(float)
        P1 = np.array(Ms_cal[1][1:]).reshape(3, 4).astype(float)
        P2 = np.array(Ms_cal[2][1:]).reshape(3, 4).astype(float)
        P3 = np.array(Ms_cal[3][1:]).reshape(3, 4).astype(float)
        R0_rect = np.array(Ms_cal[4][1:]).reshape(3, 3).astype(float)
        Tr_velo_to_cam = np.array(Ms_cal[5][1:]).reshape(3, 4).astype(float)

        # change to homogeneous coordinates
        R0 = np.zeros((4, 4))
        R0[:3, :3] = R0_rect
        R0[3, 3] = 1
        R0_rect = R0

        Tr = np.zeros((4, 4))
        Tr[:3, :] = Tr_velo_to_cam
        Tr[3, 3] = 1
        Tr_velo_to_cam = Tr

        # to-be-used calibration info
        cal_info = (P2, R0_rect, Tr_velo_to_cam)

        return img, points_3d_lidar, cal_info