import os
import argparse
import numpy as np
import pickle
from data.data_loader import DatasetLoader

np.set_printoptions(precision=4, suppress=True)


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Detecting Road-Users via Frustum-based Methods')
    argparser.add_argument('--data_type', default='kitti', help='select data type (e.g. kitti).')
    argparser.add_argument('--data_split', default='data/split', help='path to data split info.')
    # argparser.add_argument('--path_kitti', required=True, help='path to the data dir. See README for detail.')
    # argparser.add_argument('--path_output', required=True, help='select 2D detector (mask_rcnn, yolo_v3)')
    argparser.add_argument('--path_kitti', default='/media/steven/Data/datasets_cv_autonomous_driving/KITTI/', help='path to the data dir. See README for detail.')
    argparser.add_argument('--path_output', default='/home/steven/Projects/faraway-frustum-data/', help='select 2D detector (mask_rcnn, yolo_v3)')
    args = argparser.parse_args()

    # data list
    with open(os.path.join(args.data_split, 'kitti', 'training.txt'), 'r') as f:
        data_list_kitti_training = f.read().split('\n')
    with open(os.path.join(args.data_split, 'split01', 'raw', 'val.txt'), 'r') as f:
        data_list_val = f.read().split('\n')
    out_put_folder = 'split01'

    list_train, list_val = [], []
    list_train_ped_50, list_val_ped_50 = [], []
    list_train_ped_60, list_val_ped_60 = [], []

    list_train_car_75, list_val_car_75 = [], []
    list_train_car_60, list_val_car_60 = [], []

    list_train_cyc_50, list_val_cyc_50 = [], []
    list_train_cyc_60, list_val_cyc_60 = [], []


    for sample_name in data_list_kitti_training:

        print('Working on sample %s ...' % sample_name)

        # read gt labels
        path_sample_gt = os.path.join(args.path_kitti, 'training', 'label_2', '%s.txt' % sample_name)
        gt_labels = []
        with open(path_sample_gt) as f:
            lines = f.readlines()
            for line in lines:
                gt_labels.append(line.strip('\n').split(' '))

        # get ground truth labels
        for gt in gt_labels:

            if gt[0] == 'Pedestrian':
                # 3d positions
                x, y, z = gt[11], gt[12], gt[13]
                rotation_y = gt[14]
                if sample_name in data_list_val:
                    if sample_name not in list_val:
                        list_val.append(sample_name)
                    if float(z) >= 50:
                        if sample_name not in list_val_ped_50:
                            list_val_ped_50.append(sample_name)
                    if float(z) >= 60:
                        if sample_name not in list_val_ped_60:
                            list_val_ped_60.append(sample_name)
                else:
                    if sample_name not in list_train:
                        list_train.append(sample_name)
                    if float(z) >= 50:
                        if sample_name not in list_train_ped_50:
                            list_train_ped_50.append(sample_name)
                    if float(z) >= 60:
                        if sample_name not in list_train_ped_60:
                            list_train_ped_60.append(sample_name)
            if gt[0] == 'Car':
                x, y, z = gt[11], gt[12], gt[13]
                if sample_name in data_list_val:
                    if float(z) >= 75:
                        if sample_name not in list_val_car_75:
                            list_val_car_75.append(sample_name)
                    if float(z) >= 60:
                        if sample_name not in list_val_car_60:
                            list_val_car_60.append(sample_name)
                else:
                    if float(z) >= 75:
                        if sample_name not in list_train_car_75:
                            list_train_car_75.append(sample_name)
                    if float(z) >= 60:
                        if sample_name not in list_train_car_60:
                            list_train_car_60.append(sample_name)

            if gt[0] == 'Cyclist':
                x, y, z = gt[11], gt[12], gt[13]
                if sample_name in data_list_val:
                    if float(z) >= 50:
                        if sample_name not in list_val_cyc_50:
                            list_val_cyc_50.append(sample_name)
                    if float(z) >= 60:
                        if sample_name not in list_val_cyc_60:
                            list_val_cyc_60.append(sample_name)
                else:
                    if float(z) >= 50:
                        if sample_name not in list_train_cyc_50:
                            list_train_cyc_50.append(sample_name)
                    if float(z) >= 60:
                        if sample_name not in list_train_cyc_60:
                            list_train_cyc_60.append(sample_name)

    os.makedirs(os.path.join(args.data_split, out_put_folder), exist_ok=True)

    with open(os.path.join(args.data_split, out_put_folder, 'val.txt'), 'w') as f:
        for i, name in enumerate(list_val):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'train.txt'), 'w') as f:
        for i, name in enumerate(list_train):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    # -------------------- ped
    with open(os.path.join(args.data_split, out_put_folder, 'val_ped_50.txt'), 'w') as f:
        for i, name in enumerate(list_val_ped_50):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'train_ped_50.txt'), 'w') as f:
        for i, name in enumerate(list_train_ped_50):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'val_ped_60.txt'), 'w') as f:
        for i, name in enumerate(list_val_ped_60):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'train_ped_60.txt'), 'w') as f:
        for i, name in enumerate(list_train_ped_60):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    # ------------------ car
    with open(os.path.join(args.data_split, out_put_folder, 'val_car_75.txt'), 'w') as f:
        for i, name in enumerate(list_val_car_75):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'train_car_75.txt'), 'w') as f:
        for i, name in enumerate(list_train_car_75):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'val_car_60.txt'), 'w') as f:
        for i, name in enumerate(list_val_car_60):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'train_car_60.txt'), 'w') as f:
        for i, name in enumerate(list_train_car_60):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    # ------------------ cyclist
    with open(os.path.join(args.data_split, out_put_folder, 'val_cyc_50.txt'), 'w') as f:
        for i, name in enumerate(list_val_cyc_50):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'train_cyc_50.txt'), 'w') as f:
        for i, name in enumerate(list_train_cyc_50):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'val_cyc_60.txt'), 'w') as f:
        for i, name in enumerate(list_val_cyc_60):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    with open(os.path.join(args.data_split, out_put_folder, 'train_cyc_60.txt'), 'w') as f:
        for i, name in enumerate(list_train_cyc_60):
            if i == 0:
                f.write('%s' % name)
            else:
                f.write('\n%s' % name)

    print('Completed.')


if __name__ == '__main__':
    main()

