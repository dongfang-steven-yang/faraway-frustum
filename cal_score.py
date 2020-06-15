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
import utils.kitti_common as kitti
from utils.eval import get_official_eval_result, get_coco_eval_result

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

def cal_score(label_path,result_path,label_split_file,out_path,current_class=0,score_thresh=-1):
    # todo: calculating and saving scores
    record = time.gmtime()
    record_name = '_' + str(record.tm_mon) + '_' + str(record.tm_mday) + '_' + str(record.tm_hour) + '_' + str(
        record.tm_min)
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    result=''
    if current_class == -1:
        for c_class in [0,1,2]:
            result_current = get_official_eval_result(gt_annos, dt_annos, c_class)
            result = result + result_current
    else:
        result = get_official_eval_result(gt_annos, dt_annos, current_class)
    print(result)
    with open(f'{out_path}/mAP{record_name}.txt', 'w') as resulttxt:
        resulttxt.truncate()
        resulttxt.write(label_path + '\n' )
        resulttxt.write(result)
    print(f'mAP result file was saved at {out_path}/mAP{record_name}.txt')
    plot_curve()
    return True

def cal_IOU(txt_a, txt_b, out_path,current_class=1):
    # todo: calculating and saving IOU
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
        resulttxt.write('Average IOU: ' + str(IOU))
    print(f'IOU result file was saved at {out_path}/IOU{record_name}.txt')
    # return IOU
    return True


def main():
    '''
    You can revise 'current class' to evaluate scores for different classes
    current class ------ -1:all class 0:car 1:pedestrian 2:cyclist
    note: cal_IOU() only supports pedestrian now (current class=1)
    '''
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Calculate Scores')
    argparser.add_argument('--txt_alpha', required=True, help='path to source txt.')
    argparser.add_argument('--txt_beta', required=True, help='path to target txt.')
    argparser.add_argument('--txt_split', required=True, help='path to label split txt.')
    argparser.add_argument('--out_path', required=True, help='path to output')
    argparser.add_argument('--use_subset', default=False, help='path to output')
    args = argparser.parse_args()
    # please use txt_a as ground truth path.
    if args.use_subset=='True':
        IOUs=[]
        max_points = [5,10,15,20,30,50,100,200,400,1500,20000]  # maximum number of lidar points in each object.
        min_dists = [0.0, 10.0, 20.0,30.0,40.0,50.0,60.0]  # minimum distance of objects in the subset.
        for max_point in max_points:
            for min_dist in min_dists:
                print('==> Calculating subset:')
                print(' - Maximum number of points: %d' % max_point)
                print(' - Minimum distance of objects: %.2f' % min_dist)

                path_subset = os.path.join(args.txt_alpha, 'label_2_p_%d_d_%.2f' % (max_point, min_dist), 'pd' )

                if not os.path.exists(
                        os.path.join(args.out_path, 'result_IOU', 'label_2_p_%d_d_%.2f' % (max_point, min_dist))):
                    os.makedirs(
                        os.path.join(args.out_path, 'result_IOU', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)))
                if not os.path.exists(
                        os.path.join(args.out_path, 'result_mAP', 'label_2_p_%d_d_%.2f' % (max_point, min_dist))):
                    os.makedirs(
                        os.path.join(args.out_path, 'result_mAP', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)))

                done1=False
                done2=False
                done1 = cal_IOU(path_subset, args.txt_beta, os.path.join(args.out_path, 'result_IOU', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)),current_class=1)
                # done2 = cal_score(path_subset,args.txt_beta,args.txt_split, os.path.join(args.out_path, 'result_mAP', 'label_2_p_%d_d_%.2f' % (max_point, min_dist)),current_class=1)
                # IOUs.append(done1)
                if done1:
                    print(f" Average IOU of Subset with max point: {max_point} min distance: {min_dist} - Done!")
                if done2:
                    print(f" MAP score of Subset with max point: {max_point} min distance: {min_dist} - Done!")
        # print(IOUs)
    else:
        print('==> Calculating dataset:')
        if not os.path.exists(
                os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti')):
            os.makedirs(
                os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti'))
        if not os.path.exists(
                os.path.join(args.out_path, 'result_mAP', 'label_2_Kitti')):
            os.makedirs(
                os.path.join(args.out_path, 'result_mAP', 'label_2_Kitti'))

        done1 = False
        done2 = False
        # done1 = cal_IOU(args.txt_alpha, args.txt_beta,os.path.join(args.out_path, 'result_IOU', 'label_2_Kitti'),current_class=1)
        done2 = cal_score(args.txt_alpha, args.txt_beta, args.txt_split, os.path.join(args.out_path, 'result_mAP', 'label_2_Kitti'),current_class=-1)

    if done1:
        print(" Average IOU of KITTI dataset  - Done!")
    if done2:
        print(" MAP score of KITTI dataset  - Done!")


if __name__ == '__main__':
    main()