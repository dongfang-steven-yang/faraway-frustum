import os
import argparse
import numpy as np
import pickle
from data.data_loader import DatasetLoader
from utils.data_operations import transform, frustum_project
from utils.categories import labels_coco_to_kitti

np.set_printoptions(precision=4, suppress=True)


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Detecting Road-Users via Frustum-based Methods')
    argparser.add_argument('--data_type', default='kitti', help='select data type (e.g. kitti).')
    argparser.add_argument('--data_split', default='/home/steven/Projects/faraway-frustum/data/split', help='path to data split info.')
    # argparser.add_argument('--path_kitti', required=True, help='path to the data dir. See README for detail.')
    # argparser.add_argument('--path_output', required=True, help='select 2D detector (mask_rcnn, yolo_v3)')
    argparser.add_argument('--path_kitti', default='/media/steven/Backup Plus/Datasets/datasets_cv_autonomous_driving/KITTI', help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_output', default='/home/steven/Projects/faraway-frustum-data/', help='select 2D detector (mask_rcnn, yolo_v3)')
    args = argparser.parse_args()



    # data loader
    data_loader = DatasetLoader(data_type=args.data_type, data_path=args.path_kitti)

    # data list
    with open(os.path.join(args.data_split, 'kitti', 'training.txt'), 'r') as f:
        data_list_kitti_training = f.read().split('\n')
    with open(os.path.join(args.data_split, 'split01', 'raw', 'val.txt'), 'r') as f:
        data_list_val = f.read().split('\n')

    # # check if the split of FP is different from the split of PCdet
    # with open(os.path.join(args.data_split, 'frustum_pointnet', 'val.txt'), 'r') as f:
    #     frustum_list = f.read().split('\n')
    # for name in data_list_val:
    #     if name not in frustum_list:
    #         print('not match')
    # # passed ! they are the same.

    # run detection
    # thr_faraway = 60
    # data_type = 'data_split01_val'

    category = 'Car'
    # category = 'Pedestrian'

    # partition = 'train'
    partition = 'val'

    folder_name = 'data_split01_%s_%s' % (partition, category)


    path_frustum_data = os.path.join(args.path_output, '%s/x' % folder_name)
    path_frustum_labels = os.path.join(args.path_output, '%s/y' % folder_name)
    path_frustum = os.path.join(args.path_output, folder_name)
    os.makedirs(path_frustum_data, exist_ok=True)
    os.makedirs(path_frustum_labels, exist_ok=True)

    i_data = 0
    pointclouds, labels_3d = [], []
    for sample_name in data_list_kitti_training:
        if partition == 'train':
            if sample_name in data_list_val:  # skip val split
                continue
        elif partition == 'val':
            if sample_name not in data_list_val:  # skip train split
                continue
        else:
            raise Exception('Invalid Partition Selection.')

        print('Working on sample %s ...' % sample_name)

        # read raw data
        img, points_3d_lidar, cal_info, gt_info = data_loader.read_raw_data(sample_num=sample_name)

        # transform 3d points into 2d points
        points_2d_img, points_3d_cam0 = transform(points_3d_lidar, cal_info)

        # get ground truth labels
        boxes_2d_img = []
        boxes_3d_cam0 = []
        for gt in gt_info:
            if gt[0] == category:
                # 2d boxes
                x_min = gt[4]
                y_min = gt[5]
                x_max = gt[6]
                y_max = gt[7]
                box = [x_min, y_min, x_max, y_max]
                # 3d boxes
                height, width, length = gt[8], gt[9], gt[10]
                x, y, z = gt[11], gt[12], gt[13]
                rotation_y = gt[14]
                # if float(z) < thr_faraway:
                #     continue
                label = (x, y, z, height, width, length, rotation_y)
                boxes_2d_img.append(box)
                boxes_3d_cam0.append(label)
        boxes_2d_img = np.array(boxes_2d_img).astype(float)
        boxes_3d_cam0 = np.array(boxes_3d_cam0).astype(float)

        # frustum projection
        clusters_cam0, _, _ = frustum_project(
            points_2d_img=points_2d_img,
            points_3d_cam0=points_3d_cam0,
            boxes=boxes_2d_img,
            masks=None
        )

        assert len(clusters_cam0) == len(boxes_2d_img) == len(boxes_3d_cam0)

        # save data
        for i, pointcloud in enumerate(clusters_cam0):
            pointclouds.append(pointcloud)
            labels_3d.append(boxes_3d_cam0[i])
            pickle.dump(pointcloud, open(os.path.join(path_frustum_data, '%06d.p' % i_data), 'wb'))
            pickle.dump(boxes_3d_cam0[i], open(os.path.join(path_frustum_labels, '%06d.p' % i_data), 'wb'))
            i_data += 1

    print('Total num. of samples: %d' % len(pointclouds))

    assert len(pointclouds) == len(labels_3d)
    pickle.dump(pointclouds, open(os.path.join(path_frustum, 'samples.p'), 'wb'))
    pickle.dump(labels_3d, open(os.path.join(path_frustum, 'labels.p'), 'wb'))


if __name__ == '__main__':
    main()

