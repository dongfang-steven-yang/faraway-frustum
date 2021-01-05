import pickle
import numpy as np
import matplotlib.pyplot as plt


def compute_3d_box(x, y, z, h, w, l, yaw):
    """
    :return: corners of 3d boxes
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d += np.vstack([x, y, z])
    return corners_3d


def test():
    # see the coordinates convention: http://www.cvlibs.net/datasets/kitti/setup.php
    # all the coordinates follow the cam 0's coordinates in the above link

    # load data
    # format:
    # - dim 0: num. of frustums
    # - dim 1: num. of points
    # - dim 2: x, y, z, intensity, useless
    data = pickle.load(open('data/samples.p', 'rb'))

    # load labels
    # format:
    # - dim 0: num. of frustums
    # - dim 1: x, y, z, height, width, length, rotation_y

    labels = pickle.load(open('data/labels.p', 'rb'))
    assert len(data) == len(labels)


    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    # test plot in BEV
    for i in range(10):
        points = data[i]
        label = labels[i]
        corners = compute_3d_box(label[0], label[1], label[2], label[3], label[4], label[5], label[6])
        plt.axis('equal')
        plt.plot(points[:, 0], points[:, 2], '.b', alpha=0.3)
        plt.plot(label[0], label[2], 'xr')
        plt.plot(corners[0], corners[2], 'g')
        plt.show()


if __name__ == '__main__':
    test()


