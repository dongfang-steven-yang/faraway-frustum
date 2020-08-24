import argparse
import os
from data.data_loader import DatasetLoader
from utils.data_operations import transform, frustum_project
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pathlib import Path
import cv2
import random
import itertools
import colorsys
import os
import math
# from shapely.geometry import Polygon
TYPES_kitti_important = {'Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist'}


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        # cv2.putText(img, label, ((c1[0]+c2[0])/2, (c1[1]+c2[1])/2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1]- 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def get_objs_from_sample(folder_txt, sample_name):
    boxes_2d = []
    boxes_3d = []
    labels = []
    scores = []
    txt_file = os.path.join(folder_txt, sample_name + '.txt')
    # todo: read the txt files and get the objects info
    dr = pd.read_csv(txt_file, header=None, sep=' ')
    with_score = True if dr.shape[1] == 16 else False
    if with_score:
        dr.columns = ['type', 'truncated', 'occluded', 'alpha',
                      'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                      'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score']
    else:
        dr.columns = ['type', 'truncated', 'occluded', 'alpha',
                      'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                      'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']

    for i in range(len(dr)):
        d = dr.loc[i]
        if d['type'] in TYPES_kitti_important:
            box_2d = [d['bbox_left'], d['bbox_top'], d['bbox_right'], d['bbox_bottom']]
            box_3d = [d['height'], d['width'], d['length'], d['pos_x'], d['pos_y'], d['pos_z'], d['rot_y']]
            boxes_2d.append(box_2d)
            boxes_3d.append(box_3d)
            labels.append(d['type'])
            if with_score:
                scores.append(d['score'])
            else:
                scores.append(1.0)
    boxes_2d = np.array(boxes_2d)
    boxes_3d = np.array(boxes_3d)
    labels = np.array(labels)
    scores = np.array(scores)
    return boxes_2d, boxes_3d, labels, scores


def plot_2d(boxes_2d_a, labels_a, scores_a, boxes_2d_b, labels_b, scores_b, img):
    assert len(boxes_2d_a) == len(labels_a) == len(scores_a)
    assert len(boxes_2d_b) == len(labels_b) == len(scores_b)
    # todo: conduct operation on images -- ploting boxes, labels, and scores.
    # return the same img object
    for i in range(len(boxes_2d_a)):
        plot_one_box(img, boxes_2d_a[i],label=labels_a[i], color=[0, 0, 204])

    for i in range(len(boxes_2d_b)):
        if scores_b[i] > 0.5:
            plot_one_box(img, boxes_2d_b[i], label=labels_b[i], color=[255, 102, 51])

    return img


def plot_bev(boxes_3d_a, labels_a, scores_a, boxes_3d_b, labels_b, scores_b, points_3d_labelled, clusters):
    assert len(boxes_3d_a) == len(labels_a) == len(scores_a)
    assert len(boxes_3d_b) == len(labels_b) == len(scores_b)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    # todo: add content into `ax` -- plotting the results
    # return the fig object
    # detected
    for i in range(len(boxes_3d_b)):
        if boxes_3d_b[i] is None:
            pass
        else:
            ax.plot(clusters[i][:, 0].T, clusters[i][:, 2].T, '.', alpha=0.2)
    for i in range(len(boxes_3d_b)):
        # modification from BEV
        # todo : remove the warning
        # height= 1.0
        # pos_y= 1.0
        # rot_y = 0.01
        # boxes_3d.append([d['height'], d['width'], d['length'], d['pos_x'], d['pos_y'], d['pos_z'], d['rot_y']])
        if labels_b[i] in TYPES_kitti_important and scores_b[i] > 0.7:
            corners_3d_cam2 = compute_3d_box_cam2(
                boxes_3d_b[i][0], boxes_3d_b[i][1], boxes_3d_b[i][2], boxes_3d_b[i][3],
                boxes_3d_b[i][4], boxes_3d_b[i][5], boxes_3d_b[i][6]
            )
            ax.plot(boxes_3d_b[i][3], boxes_3d_b[i][5], 'xb', markersize=12)  # center
            ax.plot(corners_3d_cam2[0, :5], corners_3d_cam2[2, :5], 'b')
            ax.text(boxes_3d_b[i][3] + 0.4, boxes_3d_b[i][5] - 0.4, labels_b[i] + ', {:.2f}%'.format(float(scores_b[i]) * 100),
                    color='b')

    for i in range(len(boxes_3d_a)):
        # modification from BEV
        # todo : remove the warning
        # boxes_3d.append([d['height'], d['width'], d['length'], d['pos_x'], d['pos_y'], d['pos_z'], d['rot_y']])
        if labels_a[i] in TYPES_kitti_important:
            corners_3d_cam2 = compute_3d_box_cam2(boxes_3d_a[i][0], boxes_3d_a[i][1], boxes_3d_a[i][2], boxes_3d_a[i][3], boxes_3d_a[i][4], boxes_3d_a[i][5], boxes_3d_a[i][6])
            ax.plot(boxes_3d_a[i][3], boxes_3d_a[i][5], 'xr', markersize=12)  # center
            ax.plot(corners_3d_cam2[0, :5], corners_3d_cam2[2, :5], '--r')
            ax.text(boxes_3d_a[i][3] + 0.4, boxes_3d_a[i][5] - 0.4, labels_a[i] + ', {:.2f}%'.format(float(scores_a[i]) * 100),
                    color='r')
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_title('BEV of Camera 0 Coordinates')
    ax.grid()
    return fig


def visualize(data_loader, txt_a, txt_b, out_path):
    # the sample list will always be selected based on the samples with `txt_b`.
    # if use ground truth label, please use txt_a as ground truth path.
    files = sorted(os.listdir(txt_b))
    for file in files:
        if file[-4:] == '.txt':
            sample_name = file[:-4]
            print('Visualizing sample %s ...' % sample_name)

            # get object boxes and scores
            boxes_img_a, boxes_3d_a, labels_a, scores_a = get_objs_from_sample(txt_a, sample_name)
            boxes_img_b, boxes_3d_b, labels_b, scores_b = get_objs_from_sample(txt_b, sample_name)

            # read raw data
            img_tensor, points_3d_lidar, cal_info, gt_info = data_loader.read_raw_data(sample_name)
            # transform 3d points into 2d points
            points_2d_img, points_3d_cam0 = transform(points_3d_lidar, cal_info)
            # get clusters that correspond to target txts (results)
            clusters_cam0, points_2d_img, points_3d_cam0 = frustum_project(
                points_2d_img=points_2d_img,
                points_3d_cam0=points_3d_cam0,
                boxes=boxes_img_b,
                masks=None
            )
            img = img_tensor[0]

            # 2d vis
            img_plotted = plot_2d(boxes_img_a, labels_a, scores_a, boxes_img_b, labels_b, scores_b, img)
            if not os.path.exists(os.path.join(out_path, 'result_2D')):
                os.makedirs(os.path.join(out_path, 'result_2D'))
            cv2.imwrite(os.path.join(out_path, 'result_2D', '%s_img_cam2.png' % sample_name), img_plotted)

            # bev vis
            fig_bev = plot_bev(boxes_3d_a, labels_a, scores_a, boxes_3d_b, labels_b, scores_b, points_3d_cam0, clusters_cam0)
            if not os.path.exists(os.path.join(out_path, 'result_BEV')):
                os.makedirs(os.path.join(out_path, 'result_BEV'))
            fig_bev.savefig(os.path.join(out_path,'result_BEV','%s_bev.png' % sample_name))
            ax = fig_bev.axes[0]  # zoomed in and save again
            ax.set_xlim(-20, 20)
            ax.set_ylim(40, 80)
            fig_bev.savefig(os.path.join(out_path, 'result_BEV','%s_bev_zoomed_in.png' % sample_name))
            plt.close(fig_bev)

    return True


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Visualizing results.')
    argparser.add_argument('--data_type', default='kitti', help='select data type (e.g. kitti).')
    argparser.add_argument('--data_split', default='data/kitti/split', help='path to data split info.')
    argparser.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    argparser.add_argument('--txt_alpha', required=True, help='path to source txt.')
    argparser.add_argument('--txt_beta', required=True, help='path to target txt.')
    argparser.add_argument('--out_path', required=True, help='path to output')
    args = argparser.parse_args()

    print('- Source txt (ground truth): %s' % args.txt_alpha)
    print('- Target txt (results): %s' % args.txt_beta)
    print('visualization results will be saved at: %s' % args.out_path)

    # data loader
    data_loader = DatasetLoader(data_type=args.data_type, data_path=args.data_path)

    # run visualization
    # if use ground truth label, please use txt_a as ground truth path.
    done = visualize(data_loader=data_loader, txt_a=args.txt_alpha, txt_b=args.txt_beta, out_path=args.out_path)
    if done:
        print("Visualization Done. ")


if __name__ == '__main__':
    main()