import numpy as np


def transform(points_3d_lidar, cal_info):
    """ projecting lidar points onto image coordinates of camera 2 """
    P2, R0_rect, Tr_velo_to_cam = cal_info
    points_3d_intensity = np.copy(points_3d_lidar[:, 3])
    points_3d_lidar[:, 3] = 1
    points_3d_cam0 = Tr_velo_to_cam.dot(points_3d_lidar.T).T
    points_3d_cam0[:, 3] = points_3d_intensity
    points_2d_img = P2.dot(R0_rect.dot(points_3d_cam0.T))
    points_2d_img = points_2d_img / points_2d_img[2]
    points_2d_img = points_2d_img.T
    return points_2d_img[:, :2], points_3d_cam0


def frustum_project(points_2d_img, points_3d_cam0, boxes, masks=None):
    # check if using mask
    useing_mask = masks is not None
    if useing_mask and len(masks) > 0:
        h, w = masks[0].shape  # get image size (mask size is the same as image size)

    # add index column for the labels
    points_2d_img = np.insert(points_2d_img, 2, values=-1, axis=1)
    points_3d_cam0 = np.insert(points_3d_cam0, 4, values=-1, axis=1)

    # assign instance labels
    if len(points_2d_img) != 0:
        # find association
        for i in range(len(boxes)):
            for j in range(len(points_2d_img)):
                if useing_mask:
                    r = int(round(points_2d_img[j, 1]))
                    c = int(round(points_2d_img[j, 0]))
                    if 0 <= r < h and 0 <= c < w:
                        if masks[i, r, c]:
                            points_2d_img[j, 2] = i
                else:  # use bounding box
                    if boxes[i, 0] <= points_2d_img[j, 0] <= boxes[i, 2] and boxes[i, 1] <= points_2d_img[j, 1] <= boxes[i, 3]:
                        points_2d_img[j, 2] = i
                    # except:
                    #     if boxes[0] <= points_2d_img[0] <= boxes[2] and boxes[1] <= points_2d_img[1] <= boxes[3]:
                    #         points_2d_img[j, 2] = i
        # copy the labels to 3d points
        points_3d_cam0[:, 4] = points_2d_img[:, 2]

    # generate clusters
    clusters_cam0 = []
    for i in range(len(boxes)):
        cluster = points_3d_cam0[points_3d_cam0[:, 4] == i]
        clusters_cam0.append(cluster)

    return clusters_cam0, points_2d_img, points_3d_cam0
