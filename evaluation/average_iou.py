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

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

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


def cal_IOU(txt_a, txt_b, out_path,current_class=1):
    num_sample=0
    IOU=0.0
    record=time.gmtime()
    record_name = '_' + str(record.tm_mon)+'_'+str(record.tm_mday)+'_'+str(record.tm_hour) +'_'+str(record.tm_min)
    files = os.listdir(txt_b)
    for file in files:
        if file[-4:] == '.txt':
            sample_name = file[:-4]
            # print('Computing IOU for sample %s ...' % sample_name)
            # read ground truth
            gt = pd.read_csv(str(Path(txt_a, '%s.txt' % sample_name)), header=None, sep=' ')
            gt.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                          'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']

            # read detected result
            if os.path.getsize(str(Path(txt_b, '%s.txt' % sample_name))) > 0:
                try:
                    dr = pd.read_csv(str(Path(txt_b, '%s.txt' % sample_name)), header=None, sep=' ')
                    dr.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top','bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z','rot_y','score']
                except:
                    dr = pd.read_csv(str(Path(txt_b, '%s.txt' % sample_name)), header=None, sep=' ')
                    dr.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top','bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z','rot_y']

            # detected
            BEV_dt_boxes = []
            BEV_dt_center = []
            for i in range(len(dr)):
                d = dr.iloc[i]
                if current_class==1:
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
                elif current_class==0:
                    if d['type'] in 'Car':
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
                if current_class == 1:
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
                elif current_class == 0:
                    if d['type'] == 'Car':
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
    print(f'Number of samples : {num_sample}  Average IOU : {IOU}')
    with open(f'{out_path}/IOU{record_name}.txt', 'w') as resulttxt:
        resulttxt.truncate()
        resulttxt.write('Number of samples: ' + str(num_sample) + '\n')
        resulttxt.write('Average IoU: ' + str(IOU))
    print(f'IOU result file was saved at {out_path}/IOU{record_name}.txt')
    # return IOU
    return True


def main():
    '''
    You can revise 'current class' to evaluate scores for different classes
    current class ------ 0:car 1:pedestrian
    note: cal_IOU() only supports BEV detection now
    '''
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Calculate Scores')
    argparser.add_argument('--txt_alpha', required=True, help='path to source txt. label')
    argparser.add_argument('--txt_beta', required=True, help='path to target txt.')
    argparser.add_argument('--txt_split', required=True, help='path to label split txt.')
    argparser.add_argument('--out_path', required=True, help='path to output')
    argparser.add_argument('--use_subset', default=False, help='path to output')
    args = argparser.parse_args()

    # please use txt_alpha as path for labels, and txt_beta as path for detection results

    print('==> Calculating:')
    if not os.path.exists(
            os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti')):
        os.makedirs(
            os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti'))
    done= cal_IOU(args.txt_alpha, args.txt_beta, os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti'), current_class=1)
    if done:
        print(" BEV Average IoU Calculation  - Done!")

if __name__ == '__main__':
    main()