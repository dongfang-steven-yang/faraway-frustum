import numpy as np
import os
from object_detection.utils import label_map_util


PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# class mapping
labels_coco_to_kitti = {
    'person': 'Pedestrian',
    'bicycle': 'Cyclist',
    'car': 'Car',
    'truck': 'Truck',
    'bus': 'Van'
}


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
                    if boxes[i, 0] <= points_2d_img[j, 1] <= boxes[i, 2] and boxes[i, 1] <= points_2d_img[j, 0] <= boxes[i, 3]:
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


def save_txt_results(path_output, name_sample, positions_bev, boxes, labels, scores):
    # kitti result
    os.makedirs(path_output, exist_ok=True)
    with open('%s/%s.txt' % (path_output, name_sample), 'w') as f:
        f.truncate()
        for i in range(len(labels)):
            if positions_bev[i] is None:  # only has 2D detection result but not 3D result
                pass
            else:
                if labels[i] in category_index.keys():
                    class_name = category_index[labels[i]]['name']
                else:
                    continue
                # Kitti type:
                # 'Car', 'Van', 'Truck' 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
                if class_name in labels_coco_to_kitti.keys():
                    #    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                    #                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                    #                      'Misc' or 'DontCare'
                    f.write(labels_coco_to_kitti[class_name] + ' ')
                    #    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                    #                      truncated refers to the object leaving image boundaries
                    f.write('%.2f ' % 0)
                    #    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                    #                      0 = fully visible, 1 = partly occluded
                    #                      2 = largely occluded, 3 = unknown
                    f.write('%d ' % 0)
                    #    1    alpha        Observation angle of object, ranging [-pi..pi]
                    #  todo: check what is observation angle
                    f.write('%.2f ' % 0)
                    #    4    bbox         2D bounding box of object in the image (0-based index):
                    #                      contains left, top, right, bottom pixel coordinates
                    f.write('%.2f ' % boxes[i][1])
                    f.write('%.2f ' % boxes[i][0])
                    f.write('%.2f ' % boxes[i][3])
                    f.write('%.2f ' % boxes[i][2])
                    #    3    dimensions   3D object dimensions: height, width, length (in meters)
                    f.write('%.2f ' % 1)
                    f.write('%.2f ' % 0.7)
                    f.write('%.2f ' % 0.7)
                    #    3    location     3D object location x,y,z in camera coordinates (in meters)
                    f.write('%.2f ' % positions_bev[i][0])
                    f.write('%.2f ' % 0.50)
                    f.write('%.2f ' % positions_bev[i][2])
                    #    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
                    f.write('%.2f ' % 0)
                    f.write('%.4f' % scores[i] + '\n')
                    #    1    score        Only for results: Float, indicating confidence in
                    #                      detection, needed for p/r curves, higher is better.
                else:  # DontCare
                    f.write('DontCare ')
                    f.write('-1 ')
                    f.write('-1 ')
                    f.write('-10 ')
                    f.write('%.2f ' % boxes[i][0])
                    f.write('%.2f ' % boxes[i][1])
                    f.write('%.2f ' % boxes[i][2])
                    f.write('%.2f ' % boxes[i][3])
                    f.write('-1 ')
                    f.write('-1 ')
                    f.write('-1 ')
                    f.write('-1000 ')
                    f.write('-1000 ')
                    f.write('-1000 ')
                    f.write('-10 ')
                    # todo: verify how to report score of `DontCare`
                    f.write('-1\n')

    return None