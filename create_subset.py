import argparse
import os
from data.data_loader import DatasetLoader


def creating_subset(data_loader, data_list, path_out):
    # todo: creating subset, new txt should save to `path_out` dir.
    for sample_name in data_list:
        img, points_3d_lidar, cal_info = data_loader.read_raw_data(sample_num=sample_name)

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

    max_points = [15, 20]  # maximum number of lidar points in each object.
    min_dists = [30.0, 40.0]  # minimum distance of objects in the subset.

    for max_point in max_points:
        for min_dist in min_dists:
            print('==> Generating subset:')
            print(' - Maximum number of points: %d' % max_point)
            print(' - Minimum distance of objects: %.2f' % min_dist)
            path_out = os.path.join(args.out_txt_path, 'label_2_p_%d_d_%.2f' % (args.max_points, args.min_dist))

            # process
            done = creating_subset(data_loader, data_list, path_out)
            if done:
                print(" - Done!")


if __name__ == '__main__':
    main()