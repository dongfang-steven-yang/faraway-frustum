import os
import argparse
import numpy as np
import pickle
import math
import sys
import pandas as pd
from tensorflow import keras
import tensorflow as tf


from data.data_loader import DatasetLoader
from utils.data_operations import transform, frustum_project, save_kitti_txts
from utils.categories import labels_coco_to_kitti, classes_dict_coco
from utils.pc_histogram import pointcloud_clustering

np.set_printoptions(precision=4, suppress=True)


"""
Instruction for running this script:
Update: 2020-12-04

This script is to generate final results for the faraway object detection. It uses the intermediate results from 
previous step.

Environment configuration:
- This script requires tensorflow > 2.0.

Before running, select the experiment configuration in the `main()` function:
```
    model_type = 'car'
    use_mask = True
    use_gt = False
    split = 'testing'
    testing_example = False
```

When running, specify the following paths as parameters:
--path_kitti: path to the Kitti dataset 
--path_result: path to save the output result. This same path must be the same as the one specified in the previous 
step (step1_save_2d_results.py)

You also need to specify the paths to pedestrian/car NN models.  
--path_car_nn_model: path to the trained car model (folder name: `car_mobile_net`)
--path_ped_nn_model: path to the trained pedestrian model (folder name: `ped_mobile_net`)
Check the README for the download link.

"""

# global parameters for frustum-NN
width_meters = 40
height_meters = 70
resoultion = 0.1  # meter per pixel

width = int(width_meters * (1 / resoultion))
height = int(height_meters * (1 / resoultion))

bb_shift = 75


def preprocess_data(pointcloud_frustum_data, bb_label_data, centroids=[], width_meters=15, height_meters=20, resolution=0.1, bb_shift=75):

    # Preprocesses data for neural network training.

    # First, convert raw frustum pointcloud (PC) data -> A 2D bird's eye view frustum PC image with a fixed width and height.
    # Origin of this new frustum PC image is clustering centroids. Follow these steps:
    # 1 - find the histogram (cluster) centroids (X and Z) in the raw frustum point cloud data.
    # 2 - convert the frustum raw pointcloud coordinates into a new coordinate system whose origin is histogram centroids
    # 3 - Now, make a grid image via 2D histograming (different from step 1!). The value in each cell in the grid is
    #     equal to the number of 3D points that falls into that cell. The range of the 2D histogram is defined by
    #     -width_meters/2 to width_meters/2 and 0 to height_meters, number of bins depend on resolution.
    #     If you face memory issues for preallocating NN weights, try to increase resolution (it will make your image smaller)

    # Then, preprocess the label data. We want to do 2 things here:
    # 1- Make sure everything is positive (the final output of the NN will be positive only). This is very straight
    #    forward: Just add a big positive number (e.g 75) to the bounding box center coordinates. We do this via bb_shift
    # 2- Change the coordinate system of the labels to a new coordinate system whose origin is histogram
    #    centroids of input data (yes, the same step as step 1 for input data processing)

    # initiliaze Bird's Eye View (BEV) grid. WidthxHeight, resolution is defined explicitly below.
    # Take only x and z coordinates of the pointcloud.

    data = pointcloud_frustum_data
    labels = bb_label_data
    #centroids = pointcloud_clustering(data)

    width_meters_2 = width_meters
    height_meters_2 = height_meters
    resolution_2 = resolution  # meter per pixel

    width_2 = int(width_meters_2 * (1 / resolution_2))
    height_2 = int(height_meters_2 * (1 / resolution_2))

    # initiliaze grid. widhtxheight, resolution is in decimeters
    input_data = np.zeros([len(data), width_2, height_2, 1], dtype=np.int8)

    if bb_label_data is None:
        target_data = None
    else:
        target_data = np.zeros([len(labels), 7])

    for counter, (data_point, centroid) in enumerate(zip(data, centroids)):

        #if len(data_point)==0:
        #continue
        data_x = data_point[:, 0]
        data_z = data_point[:, 2]

        centroid_x = centroid[0]
        # print(counter)
        centroid_z = centroid[2]

        data_x_new_coordinate = data_x - centroid_x
        data_z_new_coordinate = data_z - centroid_z

        # Here we get the centroid of the pointcloud frustum

        data_grid_2D_new, edgesX_new, edgesZ_new = np.histogram2d(data_x_new_coordinate, data_z_new_coordinate,
                                                                  bins=[width_2, height_2],
                                                                  range=[[int((-width_meters_2 / 2) - centroid_x),
                                                                          int((width_meters_2 / 2) - centroid_x)],
                                                                         [(0 - centroid_z),
                                                                          int(height_meters_2 - centroid_z)]])


        # fig = plt.figure()
        # plt.imshow(data_grid_2D_new)

        data_grid_2D_new = np.reshape(data_grid_2D_new, (width_2, height_2, 1))
        input_data[counter, :, :, :] = data_grid_2D_new

        if bb_label_data is not None:
            # target_data[counter] = [labels[counter][0]+75, labels[counter][2]]
            target_data[counter] = [labels[counter][0] + bb_shift - centroid_x, labels[counter][1], labels[counter][2] +
                                    bb_shift - centroid_z, labels[counter][3], labels[counter][4], labels[counter][5], labels[counter][6]]
        # we shift the X and Z data by bb_shift meters to positive.
        # This is neccessary, because we only want positive numbers (or only negative) for the output of NN (because of RELU)
        # This can be corrected after training. During test, simply substract bb_shift from the X and Z dimension of the output of NN.

    return input_data, target_data


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


def read_gt_2d(name_sample, path_kitti, split):
    # read ground truth
    path_txt = os.path.join(path_kitti, split, 'label_2', '%s.txt' % name_sample)
    gt = pd.read_csv(path_txt, header=None, sep=' ')
    gt.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
    boxes_img=[]
    masks_img= None
    labels_img=[]
    scores_img=[]
    # ground truth boxes
    for i in range(len(gt)):
        d = gt.loc[i]
        if d['type'] == 'Misc' or d['type'] =='DontCare' :
            continue
        else:
            boxes_img.append([d['bbox_left'], d['bbox_top'], d['bbox_right'], d['bbox_bottom']])
            scores_img.append(0.99)
            if d['type'] == 'Pedestrian':
                labels_img.append(1)
            elif d['type'] == 'Car' or d['type'] == 'Van' or d['type'] == 'Truck' :
                labels_img.append(3)
            elif d['type'] == 'Cyclist':
                labels_img.append(2)
            else:
                labels_img.append(0)
    boxes_img=np.array(boxes_img)
    labels_img=np.array(labels_img)
    scores_img=np.array(scores_img)
    # masks_img=np.array(masks_img)
    return masks_img,boxes_img,labels_img,scores_img


def run_detection(model, model_type, path_output, path_2d, data_loader, sample_list, split_set, use_mask, use_gt):
    for sample_name in sample_list:
        # if int(sample_name) < 7255:
        #     continue
        print('Generating result (.txt) for %s sample %s ...' % (data_loader.data_type, sample_name))

        # 1. read raw data
        img, points_3d_lidar, cal_info, gt_info = data_loader.read_raw_data(sample_num=sample_name,
                                                                            split_set=split_set)

        # 2. call yolo on `img` to get 2d boxes
        if use_gt:
            print('-- use 2d box from ground truth')
            masks_img, boxes_img, labels_img, scores_img = read_gt_2d(name_sample=sample_name,
                                                                      path_kitti=data_loader.data_path,
                                                                      split=split_set)
        else:
            print('-- use 2d result from mask r-cnn')
            path_file = os.path.join(path_2d, '%s.p' % sample_name)
            masks_img, boxes_img, labels_img, scores_img = pickle.load(open(path_file, 'rb'))

        # 3 .transform 3d points into 2d points
        points_2d_img, points_3d_cam0 = transform(points_3d_lidar, cal_info)

        # 4. ground removal (skip)

        # 5. frustum projection
        if not use_mask:
            print('-- using box')
            masks = None
        else:
            print('-- using mask')
            masks = masks_img

        clusters_cam0, _, _ = frustum_project(
            points_2d_img=points_2d_img,
            points_3d_cam0=points_3d_cam0,
            boxes=boxes_img,
            masks=masks
        )

        # filter out frustum with zero point
        boxes_img_new = []
        labels_img_new = []
        scores_img_new = []
        clusters_cam0_new = []
        for i, cluster in enumerate(clusters_cam0):
            if len(cluster) == 0:
                continue
            clusters_cam0_new.append(cluster)
            boxes_img_new.append(boxes_img[i])
            labels_img_new.append(labels_img[i])
            scores_img_new.append(scores_img[i])

        # 6. calculate bird view positions using neural network

        if model is not None:  # use network
            test_centroid = pointcloud_clustering(clusters_cam0_new)
            test_data, test_target = preprocess_data(clusters_cam0_new, None, test_centroid)

            assert len(test_data) == len(boxes_img_new)

            if len(test_data) > 0:

                test_data_tf = tf.cast(test_data, tf.float32)
                predicted_3d_boxes = model.predict(test_data_tf, batch_size=len(test_data))

                for i, label in enumerate(predicted_3d_boxes):
                    label[0] = label[0] - bb_shift + test_centroid[i][0]
                    label[2] = label[2] - bb_shift + test_centroid[i][2]
                    if model_type == 'car':
                        label[6] = label[6] - 3.14

            else:  # if len() == 0:  # no frustums
                predicted_3d_boxes = np.array([])

            # test_data = convert_frustum_to_grid(data=clusters_cam0)
            # assert len(test_data) == len(boxes_img)
            #
            # if len(test_data) > 0:
            #     test_data_tf = tf.cast(test_data, tf.float32)
            #     bev_predicted = model.predict(test_data_tf, batch_size=len(test_data))
            #     bev_predicted = bev_predicted - [75, 0]  # shift back
            #     positions_3d = np.insert(bev_predicted, 1, 0.875, axis=1)
            # else:  # if len() == 0:  # no frustums
            #     positions_3d = np.array([])
        else:  # use clustering instead
            positions_3d = pointcloud_clustering(clusters_cam0)
            predicted_3d_boxes = np.hstack([np.array(positions_3d), np.full((len(positions_3d), 4), None)]).astype(np.float32)

        # 7. calculate 3d positions of each objects (skip)

        # 8. calculate score and save txt file
        save_kitti_txts(
            path_output=path_output,
            name_sample=sample_name,
            classes=classes_dict_coco,
            boxes_3d=predicted_3d_boxes,
            boxes_2d=boxes_img_new,
            labels=labels_img_new,
            scores=scores_img_new
        )


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Detecting Road-Users via Frustum-based Methods')
    argparser.add_argument('--path_kitti', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_result', required=True, help='select 2D detector (mask_rcnn, yolo_v3)')
    argparser.add_argument('--path_car_nn_model', required=True, help='path to the car NN model checkpoint')
    argparser.add_argument('--path_ped_nn_model', required=True, help='path to the pedestrian NN model checkpoint')
    args = argparser.parse_args()

    # path variables
    # cp_car_model = '/home/steven/Downloads/car_mobile_net'
    # cp_ped_model = '/home/steven/Downloads/saved_model_hist_norm_1'
    cp_car_model = args.path_car_nn_model
    cp_ped_model = args.path_ped_nn_model

    # configurations - TODO: change this to generate results of different combinations
    model_type = 'car'
    use_mask = True
    use_gt = False
    split = 'testing'
    testing_example = False

    # data loader
    data_loader = DatasetLoader(data_type='kitti', data_path=args.path_kitti)

    # sample list
    if split == 'training':  # Kitti training set
        path_split_txt = '/home/steven/Projects/faraway-frustum/data/split/kitti/training.txt'
        path_2d_m_rcnn_saved_result = os.path.join(args.path_result, 'kitti_2d_mask_rcnn training')
    elif split == 'testing':
        path_split_txt = '/home/steven/Projects/faraway-frustum/data/split/kitti/testing.txt'
        path_2d_m_rcnn_saved_result = os.path.join(args.path_result, 'kitti_2d_mask_rcnn testing')
    else:
        raise Exception('invalid data split')
    with open(path_split_txt, 'r') as f:
        data_list = f.read().split('\n')

    # model
    if model_type == 'ped' or model_type == 'car':
        if model_type == 'car':
            path_nn_model = cp_car_model
        else:
            path_nn_model = cp_ped_model
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        model = keras.models.load_model(path_nn_model)
    else:
        print('No NN model for 3D box estimation, using clustering only.')
        model = None

    # output folder name
    if use_mask:
        folder_prefix = 'm-rcnn-mask'
    else:
        if use_gt:
            folder_prefix = 'gt-box'
        else:
            folder_prefix = 'm-rcnn-box'
    if model is None:
        folder_body = 'hist-clustering'
    else:
        folder_body = 'nn-clustering %s' % model_type

    path_result = os.path.join(args.path_result, '%s %s %s-set' % (folder_prefix, folder_body, split))

    if testing_example:
        if model_type == 'car':
            data_list = ['003666', '002071', '002099', '001846']
        else: # pedestrian
            data_list = ['003533', '002751', '002331', '001095']
        path_result = path_result + ' example'


    # run detection
    run_detection(
        model=model,
        model_type=model_type,
        path_output=path_result,
        path_2d=path_2d_m_rcnn_saved_result,
        data_loader=data_loader,
        sample_list=data_list,
        split_set=split,
        use_mask=use_mask,
        use_gt=use_gt
    )


if __name__ == '__main__':
    main()
