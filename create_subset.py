import argparse
import os
from data.data_loader import DatasetLoader
from utils.data_operations import transform, frustum_project
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import cmath as math
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

def get_distance_and_number_each(name_sample,points_3d, cal_info, gt_info,path):
    points_2d_img, points_3d_cam0 = transform(points_3d,cal_info)
    result = np.zeros([len(gt_info),3])
    for i in range(len(gt_info)):
        if gt_info[i][0] == 'Car' or gt_info[i][0] == 'Van' or gt_info[i][0]=='Truck':
            result[i][2]= 0
        elif gt_info[i][0] == 'Pedestrian':
            result[i][2] = 1
        elif gt_info[i][0] == 'Cyclist' :
            result[i][2] = 2
        else:
            result[i][0] = -1
            result[i][1] = -1
            result[i][2] = 3
    for j in range(len(gt_info)):
        number=0
        if result[j][2] !=3:
            rotation =-float(gt_info[j][14])
            Rmatrix = [[math.cos(-rotation), -math.sin(-rotation)], [math.sin(-rotation), math.cos(-rotation)]]
            # Rmatrix = np.asarray(Rmatrix)
            dimensions = [float(gt_info[j][8]),float(gt_info[j][9]),float(gt_info[j][10])]
            location = [float(gt_info[j][11]),float(gt_info[j][12]),float(gt_info[j][13])]
            distance = location[2]
            result[j][0] = distance
            BEV_max=max(dimensions[1],dimensions[2])
            cluster = []
            for jj in range(len(points_3d_cam0)):
                # points_3d_rotation = np.zeros([1, 2])
                if points_3d_cam0[jj][0] >= location[0]-(BEV_max/2) and points_3d_cam0[jj][0] <= location[0]+(BEV_max/2) and points_3d_cam0[jj][2] >= location[2]-(BEV_max/2) and points_3d_cam0[jj][2] <= location[2]+(BEV_max/2):
                    points_3d_translate_x=points_3d_cam0[jj][0]-location[0]
                    points_3d_translate_z=points_3d_cam0[jj][2]-location[2]
                    points_3d_rotation_x=Rmatrix[0][0]*points_3d_translate_x + Rmatrix[0][1]*points_3d_translate_z
                    points_3d_rotation_z=Rmatrix[1][0]*points_3d_translate_x + Rmatrix[1][1]*points_3d_translate_z
                    if abs(points_3d_rotation_x)<dimensions[2]/2 and abs(points_3d_rotation_z)<dimensions[1]/2:
                        number = number + 1
                        cluster.append([points_3d_cam0[jj][0],points_3d_cam0[jj][2]])
            result[j][1] = number
    np.savetxt(path + "/%s.txt" % (name_sample), result, fmt='%f',
                   delimiter=' ')
    return 0
def extract_subset_each(path_gt, path_txt,path_gt_small,name_sample,min_distance,max_points,focus):
    if focus == "pd":
        class_focus = 1.0

    gt = pd.read_csv(str(Path(path_gt, '%s.txt' % name_sample)), header=None, sep=' ')
    gt.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']

    his_txt = pd.read_csv(str(Path(path_txt, '%s.txt' % name_sample)), header=None, sep=' ')
    rowNum = his_txt.shape[0]
    his_txt.columns = ['distance', 'points', 'classes']
    isExists = os.path.exists('%s/%s/' % (path_gt_small, str(focus)))
    if not isExists:
        os.mkdir('%s/%s/' % (path_gt_small, str(focus)))
    with open('%s/%s/%s.txt' % (
            path_gt_small, str(focus), name_sample), 'w') as resulttxt:
        resulttxt.truncate()
        for i in range(rowNum):
            distance_his = his_txt.iloc[i,0]
            points_his = his_txt.iloc[i,1]
            classes_his = his_txt.iloc[i,2]
            if distance_his > min_distance and points_his < max_points and classes_his == class_focus :
                    resulttxt.write(str(gt.iloc[i,0]) + ' ')
                    resulttxt.write(str(gt.iloc[i,1]) + ' ')
                    resulttxt.write(str(gt.iloc[i,2]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 3]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 4]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 5]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 6]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 7]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 8]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 9]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 10]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 11]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 12]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 13]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 14]) + '\n')

            elif classes_his == class_focus:
                    resulttxt.write(str('DontCare') + ' ')
                    resulttxt.write(str(-1.0) + ' ')
                    resulttxt.write(str(-1) + ' ')
                    resulttxt.write(str(-10.0) + ' ')
                    resulttxt.write(str(gt.iloc[i, 4]) + ' ')  # boxes
                    resulttxt.write(str(gt.iloc[i, 5]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 6]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 7]) + ' ')
                    resulttxt.write(str(-1.0) + ' ')
                    resulttxt.write(str(-1.0) + ' ')
                    resulttxt.write(str(-1.0) + ' ')
                    resulttxt.write(str(-1000.0) + ' ')
                    resulttxt.write(str(-1000.0) + ' ')
                    resulttxt.write(str(-1000.0) + ' ')
                    resulttxt.write(str(-10.0) + '\n')

            elif classes_his != class_focus:
                    resulttxt.write(str(gt.iloc[i,0]) + ' ')
                    resulttxt.write(str(gt.iloc[i,1]) + ' ')
                    resulttxt.write(str(gt.iloc[i,2]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 3]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 4]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 5]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 6]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 7]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 8]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 9]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 10]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 11]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 12]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 13]) + ' ')
                    resulttxt.write(str(gt.iloc[i, 14]) + '\n')


def creating_subset(data_loader, data_list, max_point, min_dist, path_out_histogram,path_out_subset):
    # todo: creating subset, new txt should save to `path_out` dir.
    if not os.path.exists(path_out_histogram):
        print('Histogram txt files do not exist!')
        print('Start to create histogram txt files ... ')
        os.makedirs(path_out_histogram)
        for sample_name in data_list:
            try:
                if int(sample_name)<7481:
                    img, points_3d_lidar, cal_info, gt_info = data_loader.read_raw_data(sample_num=sample_name)
                    get_distance_and_number_each(sample_name, points_3d_lidar, cal_info, gt_info, path=path_out_histogram)
            except:
                print('all samples were processed!')
        print('Histogram txt files were created!')
        print('Start to create subset label files ... ')
    else:
        print('Histogram txt files exist!')
        print('Start to create subset label files ... ')
    for sample_name in data_list:
        try:
            if int(sample_name) < 7481:
                print('==> running sample (Extract subset)' + sample_name)
                extract_subset_each(path_gt=os.path.join(data_loader.data_path, 'training','label_2'), path_txt=path_out_histogram,
                                    path_gt_small=path_out_subset, name_sample=sample_name, min_distance=min_dist,
                                    max_points=max_point, focus='pd')
        except:
            print('all samples were processed!')
    return True

def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Generating subset data of Kitti format.')
    argparser.add_argument('--data_type', default='kitti', help='select data type (e.g. kitti).')
    argparser.add_argument('--data_split', default='data/kitti/split', help='path to data split info.')
    argparser.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--out_txt_path', required=True, help='path to the output txt folder.')
    args = argparser.parse_args()
    print('- The subsets will be saved at: %s' % args.out_txt_path)

    # data loader
    data_loader = DatasetLoader(data_type=args.data_type, data_path=args.data_path)
    with open(os.path.join(args.data_split, 'training.txt'), 'r') as f:
        data_list = f.read().split('\n')

    max_points = [15]  # maximum number of lidar points in each object.
    min_dists = [0.0, 10.0, 20.0]  # minimum distance of objects in the subset.

    for max_point in max_points:
        for min_dist in min_dists:
            print('==> Generating subset:')
            print(' - Maximum number of points: %d' % max_point)
            print(' - Minimum distance of objects: %.2f' % min_dist)
            if not os.path.exists(os.path.join(args.out_txt_path, 'result_subset','label_2_p_%d_d_%.2f' % (max_point, min_dist))):
                os.makedirs(os.path.join(args.out_txt_path, 'result_subset','label_2_p_%d_d_%.2f' % (max_point, min_dist)))
            path_out_histogram = os.path.join(args.out_txt_path, 'result_histogram')
            path_out_subset = os.path.join(args.out_txt_path, 'result_subset','label_2_p_%d_d_%.2f' % (max_point, min_dist))
            # process
            done = creating_subset(data_loader, data_list, max_point, min_dist, path_out_histogram,path_out_subset)
            if done:
                print(f" - Done! subset with max point : {max_point} min distance : {min_dist} was created ")



if __name__ == '__main__':
    main()