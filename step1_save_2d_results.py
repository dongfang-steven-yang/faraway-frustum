import os
import argparse
import numpy as np
import pickle
from detectors.mask_rcnn_interface import MaskRCNNInterface
from data.data_loader import DatasetLoader

np.set_printoptions(precision=4, suppress=True)


class Detector:
    def __init__(self, model, data_loader, sample_list):
        self.model = model
        self.data_loader = data_loader
        self.sample_list = sample_list

    def run_detection(self, path_output):
        for sample_name in self.sample_list:
            print('Generating result (.txt) for %s sample %s ...' % (self.data_loader.data_type, sample_name))

            # 1. read raw data
            img, points_3d_lidar, cal_info, gt_info = self.data_loader.read_raw_data(sample_num=sample_name)

            # 2. call yolo on `img` to get 2d boxes
            masks_img, boxes_img, labels_img, scores_img = self.model.detect(img)

            # 3. save 2d detection result as pickle files
            path_file = os.path.join(path_output, '%s.p' % sample_name)
            pickle.dump((masks_img, boxes_img, labels_img, scores_img), open(path_file, 'wb'))


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Detecting Road-Users via Frustum-based Methods')
    argparser.add_argument('--data_split', default='data/split/kitti', help='path to data split info.')
    # argparser.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_kitti', default='/media/steven/Data/datasets_cv_autonomous_driving/KITTI/', help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_result', default='/home/steven/Projects/faraway-frustum-data', help='select 2D detector (mask_rcnn, yolo_v3)')

    args = argparser.parse_args()
    model = MaskRCNNInterface()

    # data loader
    data_loader = DatasetLoader(data_type='kitti', data_path=args.path_kitti)

    # detecter
    with open(os.path.join(args.data_split, 'eval.txt'), 'r') as f:
        data_list = f.read().split('\n')
    detector = Detector(model=model, data_loader=data_loader, sample_list=data_list)

    # run detection
    path_output = os.path.join(args.path_result, 'kitti_2d_mask_rcnn')
    os.makedirs(path_output, exist_ok=True)
    detector.run_detection(path_output=path_output)


if __name__ == '__main__':
    main()
