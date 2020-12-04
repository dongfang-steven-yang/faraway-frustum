import os
import argparse
import numpy as np
import pickle
import tensorflow as tf
from detectors.mask_rcnn_interface import MaskRCNNInterface
from data.data_loader import DatasetLoader

np.set_printoptions(precision=4, suppress=True)

"""
Instruction for running this script:
Update: 2020-12-04

This script is to generate intermediate results of mask-rcnn from 2D images.

Environment configuration:
1. tensorflow == 1.15 is required

Make sure the weights for mask-RCNN model (mask_rcnn_coco.h5) is included in the local directory: detectors/mask_rcnn. 
Check the README for the download link of model weights.

When running, specify the following paths as parameters:
--path_kitti: path to the Kitti dataset 
--path_result: path to save the output result. This same path is required for the next step (step2_get_kitti_results.py)

"""


class Detector:
    def __init__(self, model, data_loader, sample_list, split):
        self.model = model
        self.data_loader = data_loader
        self.sample_list = sample_list
        self.split = split

    def run_detection(self, path_output):
        for sample_name in self.sample_list:
            print('Generating result (.txt) for %s sample %s ...' % (self.data_loader.data_type, sample_name))

            # 1. read raw data
            img, points_3d_lidar, cal_info, gt_info = self.data_loader.read_raw_data(sample_num=sample_name,
                                                                                     split_set=self.split)

            # 2. call yolo on `img` to get 2d boxes
            masks_img, boxes_img, labels_img, scores_img = self.model.detect(img)

            # 3. save 2d detection result as pickle files
            path_file = os.path.join(path_output, '%s.p' % sample_name)
            pickle.dump((masks_img, boxes_img, labels_img, scores_img), open(path_file, 'wb'))


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Detecting Road-Users via Frustum-based Methods')
    # argparser.add_argument('--data_split', default='data/split/kitti', help='path to data split info.')
    argparser.add_argument('--path_kitti', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_result', required=True, help='select 2D detector (mask_rcnn, yolo_v3)')

    args = argparser.parse_args()
    model = MaskRCNNInterface()

    # config
    # split = 'training'
    split = 'testing'

    # data loader
    data_loader = DatasetLoader(data_type='kitti', data_path=args.path_kitti)

    # sample list
    if split == 'training':  # Kitti training set
        path_split_txt = '/home/steven/Projects/faraway-frustum/data/split/kitti/training.txt'
    elif split == 'testing':
        path_split_txt = '/home/steven/Projects/faraway-frustum/data/split/kitti/testing.txt'
    else:
        raise Exception('invalid data split')
    with open(path_split_txt, 'r') as f:
        data_list = f.read().split('\n')

    # detecter
    detector = Detector(model=model, data_loader=data_loader, sample_list=data_list, split=split)

    # run detection
    path_output = os.path.join(args.path_result, 'kitti_2d_mask_rcnn %s' % split)
    os.makedirs(path_output, exist_ok=True)
    detector.run_detection(path_output=path_output)


if __name__ == '__main__':
    main()
