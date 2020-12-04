import cv2
import numpy as np
import os
import tensorflow as tf
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if path.startswith('http'):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)


class DatasetLoader:
    def __init__(self, data_type, data_path):
        self.data_type = data_type
        self.data_path = data_path

    def read_raw_data(self, sample_num, split_set):
        print('-- %s sample %s loaded.' % (split_set, sample_num))
        # data paths
        path_sample_velodyne = os.path.join(self.data_path, split_set, 'velodyne', '%s.bin' % sample_num)
        path_sample_img = os.path.join(self.data_path, split_set, 'image_2', '%s.png' % sample_num)
        path_sample_cal = os.path.join(self.data_path, split_set, 'calib', '%s.txt' % sample_num)
        if split_set == 'training':
            path_sample_gt = os.path.join(self.data_path, split_set, 'label_2', '%s.txt' % sample_num)
        else:
            pass
        # ------ read pointcloud ------
        # points_3d_lidar = np.fromfile(open(path_sample_velodyne, 'r'), dtype=np.float32, count=-1).reshape([-1, 4])
        points_3d_lidar = np.fromfile(path_sample_velodyne, dtype=np.float32, count=-1).reshape([-1, 4])
        points_3d_lidar = points_3d_lidar[points_3d_lidar[:, 0] > 0]  # only use points in front of the vehicle

        # ------ read image ------
        img = cv2.imread(path_sample_img)
        # img = load_image_into_numpy_array(path_sample_img)

        # ------ read calibration info ------
        infos_cal = []
        with open(path_sample_cal) as f:
            lines = f.readlines()
            for line in lines:
                infos_cal.append(line.strip('\n').split(' '))

        # ------ read ground truth ------
        if split_set == 'training':
            gt_labels = []
            with open(path_sample_gt) as f:
                lines = f.readlines()
                for line in lines:
                    gt_labels.append(line.strip('\n').split(' '))
        else:
            gt_labels = None

        # P0-P3 for cameras, see the sensor setup page: http://www.cvlibs.net/datasets/kitti/setup.php
        P0 = np.array(infos_cal[0][1:]).reshape(3, 4).astype(float)
        P1 = np.array(infos_cal[1][1:]).reshape(3, 4).astype(float)
        P2 = np.array(infos_cal[2][1:]).reshape(3, 4).astype(float)
        P3 = np.array(infos_cal[3][1:]).reshape(3, 4).astype(float)
        R0_rect = np.array(infos_cal[4][1:]).reshape(3, 3).astype(float)
        Tr_velo_to_cam = np.array(infos_cal[5][1:]).reshape(3, 4).astype(float)

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

        return img, points_3d_lidar, cal_info, gt_labels
