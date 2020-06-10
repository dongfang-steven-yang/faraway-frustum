import argparse
import os
import math
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import random
import itertools
import colorsys
import time

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def plot_curve():
    pass
    return None

def cal_score(txt_a, txt_b, out_path):
    # todo: calculating and saving scores

    plot_curve()

    return True

def cal_IOU(txt_a, txt_b, out_path):
    # todo: calculating and saving IOU
    num_sample=0
    IOU=0.0
    record=time.gmtime()
    record_name = '_' + str(record.tm_mon)+'_'+str(record.tm_mday)+'_'+str(record.tm_hour) +'_'+str(record.tm_min)
    files = os.listdir(txt_b)
    for file in files:
        if file[-4:] == '.txt':
            sample_name = file[:-4]
            print('Computing IOU for sample %s ...' % sample_name)
            # read ground truth
            gt = pd.read_csv(str(Path(txt_a, '%s.txt' % sample_name)), header=None, sep=' ')
            gt.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                          'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']

            # read detected result
            if os.path.getsize(str(Path(txt_b, '%s.txt' % sample_name))) > 0:
                dr = pd.read_csv(str(Path(txt_b, '%s.txt' % sample_name)), header=None, sep=' ')
                dr.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                              'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z',
                              'rot_y', 'score']
            # detected
            BEV_dt_boxes = []
            BEV_dt_center = []
            for i in range(len(dr)):
                d = dr.iloc[i]
                # modification from BEV
                # todo : remove the warning
                # d['height'] = 1.0
                # d['pos_y'] = 1.0
                # d['rot_y'] = 0.01
                # if d['type'] in TYPES_kitti_important:
                if d['type'] in 'Pedestrian':
                    dt_center_point = (d['pos_x'], d['pos_z'])
                    dt_left_bot = (d['pos_x'] - (d['length'] / 2), d['pos_z'] - (d['width'] / 2))
                    dt_left_top = (d['pos_x'] - (d['length'] / 2), d['pos_z'] + (d['width'] / 2))
                    dt_right_top = (d['pos_x'] + (d['length'] / 2), d['pos_z'] + (d['width'] / 2))
                    dt_right_bot = (d['pos_x'] + (d['length'] / 2), d['pos_z'] - (d['width'] / 2))
                    New_dt_left_bot = rotate(dt_center_point, dt_left_bot, -d['rot_y'])
                    New_dt_left_top = rotate(dt_center_point, dt_left_top, -d['rot_y'])
                    New_dt_right_top = rotate(dt_center_point, dt_right_top, -d['rot_y'])
                    New_dt_right_bot = rotate(dt_center_point, dt_right_bot, -d['rot_y'])
                    BEV_dt_boxes.append([New_dt_left_bot, New_dt_left_top, New_dt_right_top, New_dt_right_bot])
                    BEV_dt_center.append(dt_center_point)

            # groud truth
            BEV_gt_boxes = []
            BEV_gt_center = []
            for i in range(len(gt)):
                d = gt.loc[i]
                # if d['type'] in TYPES_kitti_important:
                if d['type'] == 'Pedestrian':
                    gt_center_point = (d['pos_x'], d['pos_z'])
                    gt_left_bot = (d['pos_x'] - (d['length'] / 2), d['pos_z'] - (d['width'] / 2))
                    gt_left_top = (d['pos_x'] - (d['length'] / 2), d['pos_z'] + (d['width'] / 2))
                    gt_right_top = (d['pos_x'] + (d['length'] / 2), d['pos_z'] + (d['width'] / 2))
                    gt_right_bot = (d['pos_x'] + (d['length'] / 2), d['pos_z'] - (d['width'] / 2))
                    New_gt_left_bot = rotate(gt_center_point, gt_left_bot, -d['rot_y'])
                    New_gt_left_top = rotate(gt_center_point, gt_left_top, -d['rot_y'])
                    New_gt_right_top = rotate(gt_center_point, gt_right_top, -d['rot_y'])
                    New_gt_right_bot = rotate(gt_center_point, gt_right_bot, -d['rot_y'])
                    BEV_gt_boxes.append([New_gt_left_bot, New_gt_left_top, New_gt_right_top, New_gt_right_bot])
                    BEV_gt_center.append(gt_center_point)

            for i in range(len(BEV_gt_boxes)):
                p1 = Polygon(BEV_gt_boxes[i])
                num_sample += 1
                high_score = 0
                for j in range(len(BEV_dt_boxes)):
                    p2 = Polygon(BEV_dt_boxes[j])
                    inter = p2.intersection(p1).area
                    union = p1.area + p2.area - inter
                    score = inter / union
                    if score > high_score:
                        high_score = score
                IOU = IOU + high_score

    IOU = IOU / num_sample
    with open(f'{out_path}/IOU{record_name}.txt', 'w') as resulttxt:
        resulttxt.truncate()
        resulttxt.write('Number of samples: ' + str(num_sample) + '\n')
        resulttxt.write('Average IOU: ' + str(IOU))
    return True


def main():

    # parsing arguments
    argparser = argparse.ArgumentParser(description='Calculate Scores')
    argparser.add_argument('--txt_alpha', required=True, help='path to source txt.')
    argparser.add_argument('--txt_beta', required=True, help='path to target txt.')
    argparser.add_argument('--out_path', required=True, help='path to output')
    argparser.add_argument('--use_subset', default=False, help='path to output')
    args = argparser.parse_args()
    # please use txt_a as ground truth path.
    if args.use_subset:
        max_points = [5, 10, 15, 20]  # maximum number of lidar points in each object.
        min_dists = [30.0, 40.0, 50.0, 60.0]  # minimum distance of objects in the subset.
        for max_point in max_points:
            for min_dist in min_dists:
                print('==> Generating subset:')
                print(' - Maximum number of points: %d' % max_point)
                print(' - Minimum distance of objects: %.2f' % min_dist)
        # process
                path_subset = os.path.join(args.txt_alpha, 'label_2_p_%d_d_%.2f' % (max_point, min_dist), 'pd' )

                if not os.path.exists(
                        os.path.join(args.out_path, 'result_IOU', 'label_2_p_%d_d_%.2f' % (max_point, min_dist))):
                    os.makedirs(
                        os.path.join(args.out_path, 'result_IOU', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)))
                if not os.path.exists(
                        os.path.join(args.out_path, 'result_mAP', 'label_2_p_%d_d_%.2f' % (max_point, min_dist))):
                    os.makedirs(
                        os.path.join(args.out_path, 'result_mAP', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)))

                done1 = cal_IOU(path_subset, args.txt_beta, os.path.join(args.out_path, 'result_IOU', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)))
                done2 = cal_score(path_subset, args.txt_beta, os.path.join(args.out_path, 'result_mAP', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)))
                if done1:
                    print(f" Average IOU of Subset with max point: {max_point} min distance: {min_dist} - Done!")
                if done2:
                    print(f" MAP score of Subset with max point: {max_point} min distance: {min_dist} - Done!")
    else:
        if not os.path.exists(
                os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti')):
            os.makedirs(
                os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti'))
        if not os.path.exists(
                os.path.join(args.out_path, 'result_mAP', 'label_2_Kitti')):
            os.makedirs(
                os.path.join(args.out_path, 'result_mAP', 'label_2_Kitti'))
        done1 = cal_IOU(args.txt_alpha, args.txt_beta,os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti'))
        done2 = cal_score(args.txt_alpha, args.txt_beta, os.path.join(args.out_path, 'result_mAP', 'label_2_Kitti'))

    if done1:
        print(" Average IOU of KITTI dataset  - Done!")
    if done2:
        print(" MAP score of KITTI dataset  - Done!")


if __name__ == '__main__':
    main()