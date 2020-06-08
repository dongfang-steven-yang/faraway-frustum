import argparse
import os
from data.data_loader import DatasetLoader
from utils.data_operations import transform, frustum_project
import matplotlib.pyplot as plt
import cv2


def get_objs_from_sample(folder_txt, sample_name):
    txt_file = os.path.join(folder_txt, sample_name + '.txt')
    # todo: read the txt files and get the objects info

    return boxes_2d, boxes_3d, labels, scores


def plot_2d(boxes_2d_a, labels_a, scores_a, boxes_2d_b, labels_b, scores_b, img):
    assert len(boxes_2d_a) == len(labels_a) == len(scores_a)
    assert len(boxes_2d_b) == len(labels_b) == len(scores_b)
    # todo: conduct operation on images -- ploting boxes, labels, and scores.
    # return the same img object

    return img


def plot_bev(boxes_3d_a, labels_a, scores_a, boxes_3d_b, labels_b, scores_b, points_3d_labelled, clusters):
    assert len(boxes_3d_a) == len(labels_a) == len(scores_a)
    assert len(boxes_3d_b) == len(labels_b) == len(scores_b)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    # todo: add content into `ax` -- plotting the results
    # return the fig object

    return fig


def visualize(data_loader, txt_a, txt_b, out_path):
    # the sample list will always be selected based on the samples with `txt_a`.
    files = os.listdir(txt_a)
    for file in files:
        if file[-4:] == '.txt':
            sample_name = file[:-4]
            print('Visualizing sample %s ...' % sample_name)

            # get object boxes and scores
            boxes_img_a, boxes_3d_a, labels_a, scores_a = get_objs_from_sample(txt_a, sample_name)
            boxes_img_b, boxes_3d_b, labels_b, scores_b = get_objs_from_sample(txt_b, sample_name)

            # read raw data
            img, points_3d_lidar, cal_info = data_loader.read_raw_data(sample_name)
            # transform 3d points into 2d points
            points_2d_img, points_3d_cam0 = transform(points_3d_lidar, cal_info)
            # get clusters that correspond to target txts (results)
            clusters_cam0, points_2d_img, points_3d_cam0 = frustum_project(
                points_2d_img=points_2d_img,
                points_3d_cam0=points_3d_cam0,
                boxes=boxes_img_b,
                masks=None
            )

            # 2d vis
            img_plotted = plot_2d(boxes_img_a, labels_a, scores_a, boxes_img_b, labels_b, scores_b, img)
            cv2.imwrite(os.path.join(out_path, '%s_img_cam2.png' % sample_name), img_plotted)

            # bev vis
            fig_bev = plot_bev(boxes_3d_a, labels_a, scores_a, boxes_3d_b, labels_b, scores_b, points_3d_cam0, clusters_cam0)
            fig_bev.savefig(os.path.join(out_path, '%s_bev.png' % sample_name))
            ax = fig_bev.axes[0]  # zoomed in and save again
            ax.set_xlim(-20, 20)
            ax.set_ylim(40, 80)
            fig_bev.savefig(os.path.join(out_path, '%s_bev_zoomed_in.png' % sample_name))
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
    done = visualize(data_loader=data_loader, txt_a=args.txt_alpha, txt_b=args.txt_beta, out_path=args.out_path)
    if done:
        print("Visualization Done. ")


if __name__ == '__main__':
    main()