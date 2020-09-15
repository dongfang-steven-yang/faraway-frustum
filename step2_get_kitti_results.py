import os
import argparse
import numpy as np
import pickle
import math
import sys
from tensorflow import keras
import tensorflow as tf


from data.data_loader import DatasetLoader
from utils.data_operations import transform, frustum_project, save_kitti_txts
from utils.categories import labels_coco_to_kitti, classes_dict_coco
from utils.pc_histogram import pointcloud_clustering

np.set_printoptions(precision=4, suppress=True)


"""
This script requires tensorflow > 2.0.
"""

# global parameters for frustum-NN
width_meters = 40
height_meters = 70
resoultion = 0.1  # meter per pixel

width = int(width_meters * (1 / resoultion))
height = int(height_meters * (1 / resoultion))


def convert_frustum_to_grid(data):
    # initiliaze grid. widhtxheight, resolution is in decimeters
    input_data = np.zeros([len(data), width, height, 1], dtype=np.int8)

    for counter, data_point in enumerate(data):
        # initiliaze grid. 1200x1200, resolution is in decimeters

        data_x = data_point[:, 0]
        data_z = data_point[:, 2]

        data_grid_2D, edgesX, edgesZ = np.histogram2d(
            data_x, data_z, bins=[width, height],
            range=[[int(-width_meters/2), int(width_meters/2)], [0, int(height_meters)]]
        )

        data_grid_2D = np.reshape(data_grid_2D, (width, height, 1))
        input_data[counter, :, :, :] = data_grid_2D


        # target_data[counter] = [labels[counter][0]+75, labels[counter][2]]
        # we shift the X data by 75 meters to positive.
        # This is neccessary, because we only want positive numbers (or only negative) for the output of NN (because of RELU)
        # This can be corrected after training. During test, simply substract 75 from the X dimension of the output of NN.
    return input_data


def run_detection(model, path_output, path_2d, data_loader, sample_list):
    for sample_name in sample_list:
        # if int(sample_name) < 7255:
        #     continue
        print('Generating result (.txt) for %s sample %s ...' % (data_loader.data_type, sample_name))

        # 1. read raw data
        img, points_3d_lidar, cal_info, gt_info = data_loader.read_raw_data(sample_num=sample_name)

        # 2. call yolo on `img` to get 2d boxes
        path_file = os.path.join(path_2d, '%s.p' % sample_name)
        masks_img, boxes_img, labels_img, scores_img = pickle.load(open(path_file, 'rb'))

        # 3 .transform 3d points into 2d points
        points_2d_img, points_3d_cam0 = transform(points_3d_lidar, cal_info)

        # 4. ground removal (skip)

        # 5. frustum projection
        clusters_cam0, _, _ = frustum_project(
            points_2d_img=points_2d_img,
            points_3d_cam0=points_3d_cam0,
            boxes=boxes_img,
            masks=masks_img
        )

        # 6. calculate bird view positions using neural network

        if model is not None:  # use network
            test_data = convert_frustum_to_grid(data=clusters_cam0)
            assert len(test_data) == len(boxes_img)

            if len(test_data) > 0:
                test_data_tf = tf.cast(test_data, tf.float32)
                bev_predicted = model.predict(test_data_tf, batch_size=len(test_data))
                bev_predicted = bev_predicted - [75, 0]  # shift back
                positions_3d = np.insert(bev_predicted, 1, 0.875, axis=1)
            else:  # if len() == 0:  # no frustums
                positions_3d = np.array([])
        else:  # use clustering instead
            positions_3d = pointcloud_clustering(clusters_cam0)

        # 7. calculate 3d positions of each objects (skip)

        # 8. calculate score and save txt file
        save_kitti_txts(
            path_output=path_output,
            name_sample=sample_name,
            classes=classes_dict_coco,
            positions_3d=positions_3d,
            boxes=boxes_img,
            labels=labels_img,
            scores=scores_img
        )


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Detecting Road-Users via Frustum-based Methods')
    argparser.add_argument('--data_split', default='data/split/kitti', help='path to data split info.')
    # argparser.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_kitti', default='/media/steven/Data/datasets_cv_autonomous_driving/KITTI/', help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_result', default='/home/steven/Projects/faraway-frustum-data', help='select 2D detector (mask_rcnn, yolo_v3)')
    argparser.add_argument('--model', default='clustering', help='select 2D detector (clustering, NN)')

    args = argparser.parse_args()

    # data loader
    data_loader = DatasetLoader(data_type='kitti', data_path=args.path_kitti)
    # sample list
    with open(os.path.join(args.data_split, 'eval.txt'), 'r') as f:
        data_list = f.read().split('\n')

    # model
    if args.model == 'network':
        model = keras.models.load_model('detectors/saved_model_resnet50_1')
    elif args.model == 'clustering':
        model = None
    else:
        raise Exception('Invalid model')

    # run detection
    path_2d = os.path.join(args.path_result, 'kitti_2d_mask_rcnn')
    path_result = os.path.join(args.path_result, 'mask_plus_histogram_all_cat')
    run_detection(model=model, path_output=path_result, path_2d=path_2d, data_loader=data_loader, sample_list=data_list)


if __name__ == '__main__':
    main()
