import os
import argparse
import numpy as np
import math
from sklearn import mixture
from scipy.signal import savgol_filter
from datetime import datetime
from data.data_loader import DatasetLoader
from utils.data_operations import transform, frustum_project, save_txt_results
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops


import tensorflow as tf
import tensorflow_hub as hub


tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_virtual_device_configuration(
#     physical_devices[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
# )

np.set_printoptions(precision=4, suppress=True)


ALL_MODELS = {
    'CenterNet HourGlass104 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1',
    'CenterNet HourGlass104 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1',
    'CenterNet HourGlass104 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1',
    'CenterNet HourGlass104 Keypoints 1024x1024' : 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1',
    'CenterNet Resnet50 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1',
    'CenterNet Resnet50 V1 FPN Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1',
    'CenterNet Resnet101 V1 FPN 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1',
    'CenterNet Resnet50 V2 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1',
    'CenterNet Resnet50 V2 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1',
    'EfficientDet D0 512x512' : 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
    'EfficientDet D1 640x640' : 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
    'EfficientDet D2 768x768' : 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
    'EfficientDet D3 896x896' : 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
    'EfficientDet D4 1024x1024' : 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
    'EfficientDet D5 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
    'EfficientDet D6 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
    'EfficientDet D7 1536x1536' : 'https://tfhub.dev/tensorflow/efficientdet/d7/1',
    'SSD MobileNet v2 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
    'SSD MobileNet V1 FPN 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1',
    'SSD MobileNet V2 FPNLite 320x320' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1',
    'SSD MobileNet V2 FPNLite 640x640' : 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1',
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1',
    'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)' : 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1',
    'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1',
    'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)' : 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1',
    'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1',
    'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)' : 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1',
    'Faster R-CNN ResNet50 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',
    'Faster R-CNN ResNet50 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1',
    'Faster R-CNN ResNet50 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1',
    'Faster R-CNN ResNet101 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1',
    'Faster R-CNN ResNet101 V1 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1',
    'Faster R-CNN ResNet101 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1',
    'Faster R-CNN ResNet152 V1 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1',
    'Faster R-CNN ResNet152 V1 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1',
    'Faster R-CNN ResNet152 V1 800x1333' : 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1',
    'Faster R-CNN Inception ResNet V2 640x640' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1',
    'Faster R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1',
    'Mask R-CNN Inception ResNet V2 1024x1024' : 'https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1'
}


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
            results = self.model(img)
            result = {key: value.numpy() for key, value in results.items()}

            if 'detection_masks' in result:
                # we need to convert np.arrays to tensors
                detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
                detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])

                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes,
                    img.shape[1], img.shape[2])
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
                result['detection_masks_reframed'] = detection_masks_reframed.numpy()

                masks_img = result['detection_masks_reframed']
            else:
                masks_img = None

            boxes_img = result['detection_boxes'][0] * [[img.shape[1], img.shape[2], img.shape[1], img.shape[2]]]
            labels_img = (result['detection_classes'][0] + 0).astype(int)
            scores_img = result['detection_scores'][0]

            # 3 .transform 3d points into 2d points
            points_2d_img, points_3d_cam0 = transform(points_3d_lidar, cal_info)

            # 4. ground removal (skip)

            # 5. frustum projection
            clusters_cam0, _, _ = frustum_project(
                points_2d_img=points_2d_img,
                points_3d_cam0=points_3d_cam0,
                boxes=boxes_img,
                masks=masks_img
            )

            # 6. calculate bird view positions of each objects
            positions_bev = self.cal_bev_pos(clusters_cam0)

            # 7. calculate 3d positions of each objects (skip)

            # 8. calculate score and save txt file
            save_txt_results(
                path_output=path_output,
                name_sample=sample_name,
                positions_bev=positions_bev,
                boxes=boxes_img,
                labels=labels_img,
                scores=scores_img
            )

    @staticmethod
    def cal_bev_pos(clusters, method='histogram'):
        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')

        positions = []
        # for i in range(len(boxes_img)):
        for cluster in clusters:
            if len(cluster) == 0:
                positions.append(None)
            else:
                if method == 'average':
                    pos = [cluster[:, 0].mean(), cluster[:, 1].mean(), cluster[:, 2].mean()]
                elif method == 'mix_gaussian':
                    if len(cluster[:, [0, 2]]) < 3:
                        pos = [cluster[:, 0].mean(), cluster[:, 1].mean(), cluster[:, 2].mean()]
                    else:
                        clf.fit(cluster[:, [0, 2]])
                        k = np.argmax(np.argsort(clf.covariances_[:, 0, 0]) + np.argsort(clf.covariances_[:, 1, 1]))
                        pos = [clf.means_[k, 0], None, clf.means_[k, 1]]
                elif method == 'histogram':
                    pos = []
                    for j in range(3):
                        hist = np.histogram(cluster[:, j])
                        k = np.argmax(hist[0])
                        pos.append((hist[1][k] + hist[1][k + 1]) / 2)
                else:
                    raise Exception('Invalid definition of method.')
                positions.append(tuple(pos))

        return positions

    @staticmethod
    def remove_ground(points3D):
        pi = math.pi
        # points3D=points3D[points3D[:,2]>0,:]
        points3D = np.insert(points3D, 5, values=0, axis=1)
        tanpoints3D = np.true_divide(points3D[:, 2], points3D[:, 0])

        points3D = np.insert(points3D, 6, tanpoints3D, axis=1)
        distance = np.multiply(points3D[:, 0], points3D[:, 0]) + np.multiply(points3D[:, 2], points3D[:, 2])
        distance = distance ** 0.5
        points3D = np.insert(points3D, 7, distance, axis=1)  # distance
        points3D = np.insert(points3D, 8, 0, axis=1)  # angel for z and distance
        points3D = np.insert(points3D, 9, 0, axis=1)  # ray number
        raysize = 5000
        rayspace = []
        for i in range(raysize):
            size = len(rayspace)
            current = points3D[points3D[:, 6] <= math.tan(pi / raysize * (i + 1)), :]
            current = current[current[:, 6] >= math.tan(pi / raysize * (i)), :]
            if len(current) != 0:
                current = current[current[:, 7].argsort()]
                current[:, 9] = size + 1
            else:
                continue
            # rayspace.append(current)
            rayspace.append(current)

        size1 = len(rayspace)
        newray = np.array(rayspace[0][:][:])
        for i in range(1, size1):
            newray = np.append(newray, np.array(rayspace[i][:][:]), axis=0)
        newpoint = np.zeros((1, 10))
        for i in range(size1):
            size = len(rayspace[i])
            current1 = newray[newray[:, 9] == i + 1, :]
            current1[0, 5] = 0
            size = len(current1)
            for j in range(1, size):
                current1[j, 8] = abs(
                    math.atan((current1[j, 1] - current1[j - 1, 1]) / (current1[j, 7] - current1[j - 1, 7]))) / pi * 180
                if j == size - 1 and j > 5:
                    current1[:, 8] = savgol_filter(current1[:, 8], 7, 5)
            for j in range(1, size):
                if current1[j, 8] > 9:
                    current1[j, 5] = 1
            newpoint = np.append(newpoint, current1, axis=0)
            # newpoint.append(current)
        # size1=len(rayspace)
        # newray=np.array(rayspace[0][:][:])
        # for i in range(1,size1):
        #     newray=np.append(newray,np.array(rayspace[i][:][:]),axis=0)
        #
        # newpoint1 = []
        # for i in range(size1):
        #     newpoint1.append(rayspace[i][:])
        # newpoint1=np.asarray(newpoint1)
        points_3D_ground_label = newpoint[0:-1, :]

        # plot result
        removeground = points_3D_ground_label
        # plot removed ground
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111)
        # ax.plot(removeground[removeground[:,5]==0,0].T, removeground[removeground[:,5]==0,2].T,'.', alpha=0.2)
        # ax.set_xlim(-50, 50)
        # ax.set_ylim(0, 100)
        # ax.set_xlabel('x axis')
        # ax.set_ylabel('z axis')
        # ax.set_title('BEV of ground')
        # fig.savefig(PATH_OUT_IMG + '/%s_bevground.png' % name_sample)
        # plt.close(fig)
        # ax.plot(removeground[removeground[:,5]==1,0].T, removeground[removeground[:,5]==1,2].T, '.', alpha=0.2)
        # ax.set_xlim(-50, 50)
        # ax.set_ylim(0, 100)
        # ax.set_xlabel('x axis')
        # ax.set_ylabel('z axis')
        # ax.set_title('BEV of groundremove')
        # fig.savefig(PATH_OUT_IMG + '/%s_bevgroundremove.png' % name_sample)
        # plt.close(fig)

        return points_3D_ground_label

    @staticmethod
    def save_as_txts(path_output, name_sample, classes, positions_bev, boxes, labels, scores):
        # kitti result
        os.makedirs(path_output, exist_ok=True)
        with open('%s/%s.txt' % (path_output, name_sample), 'w') as f:
            f.truncate()
            for i in range(len(labels)):
                if positions_bev[i] is None:
                    pass
                else:
                    # Kitti type:
                    # 'Car', 'Van', 'Truck' 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
                    if classes[labels[i]] in labels_coco_to_kitti.keys():
                        #    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                        #                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        #                      'Misc' or 'DontCare'
                        f.write(labels_coco_to_kitti[classes[labels[i]]] + ' ')
                        #    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                        #                      truncated refers to the object leaving image boundaries
                        f.write(str(0.00) + ' ')
                        #    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                        #                      0 = fully visible, 1 = partly occluded
                        #                      2 = largely occluded, 3 = unknown
                        f.write(str(0) + ' ')
                        #    1    alpha        Observation angle of object, ranging [-pi..pi]
                        #  todo: check what is observation angle
                        f.write(str(0.00) + ' ')
                        #    4    bbox         2D bounding box of object in the image (0-based index):
                        #                      contains left, top, right, bottom pixel coordinates
                        f.write('%.2f ' % boxes[i][0])
                        f.write('%.2f ' % boxes[i][1])
                        f.write('%.2f ' % boxes[i][2])
                        f.write('%.2f ' % boxes[i][3])
                        #    3    dimensions   3D object dimensions: height, width, length (in meters)
                        f.write(str(-1.00) + ' ')
                        f.write(str(0.70) + ' ')
                        f.write(str(0.70) + ' ')
                        #    3    location     3D object location x,y,z in camera coordinates (in meters)
                        f.write('%.2f ' % positions_bev[i][0])
                        f.write('%.2f ' % -1000.00)
                        f.write('%.2f ' % positions_bev[i][2])
                        #    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
                        f.write(str(0.00) + ' ')
                        f.write('%.4f' % scores[i] + '\n')
                        #    1    score        Only for results: Float, indicating confidence in
                        #                      detection, needed for p/r curves, higher is better.
                    else:  # DontCare
                        f.write('DontCare ')
                        f.write('-1 ')
                        f.write('-1 ')
                        f.write('-10 ')
                        f.write('%.2f ' % boxes[i][0])
                        f.write('%.2f ' % boxes[i][1])
                        f.write('%.2f ' % boxes[i][2])
                        f.write('%.2f ' % boxes[i][3])
                        f.write('-1 ')
                        f.write('-1 ')
                        f.write('-1 ')
                        f.write('-1000 ')
                        f.write('-1000 ')
                        f.write('-1000 ')
                        f.write('-10')
                        # todo: verify how to report score of `DontCare`
                        f.write('-1\n')

        return None


def main():
    # parsing arguments
    argparser = argparse.ArgumentParser(description='Faraway-frustum 3d object detection')
    argparser.add_argument('--data_type', default='kitti', help='select data type (e.g. kitti).')
    argparser.add_argument('--data_split', default='data/kitti/split', help='path to data split info.')
    argparser.add_argument('--data_path', required=True, help='path to the data dir. See README for detail.')
    # argparser.add_argument('--detector', default='mask_rcnn', help='select 2D detector (mask_rcnn, yolo_v3)')
    argparser.add_argument('--detector', default='faster_rcnn', help='select 2D detector (mask_rcnn, yolo_v3)')

    args = argparser.parse_args()

    # model
    # @title Model Selection { display-mode: "form", run: "auto" }
    if args.detector == 'mask_rcnn':
        model_display_name = 'Mask R-CNN Inception ResNet V2 1024x1024'  # @param ['CenterNet HourGlass104 512x512','CenterNet HourGlass104 Keypoints 512x512','CenterNet HourGlass104 1024x1024','CenterNet HourGlass104 Keypoints 1024x1024','CenterNet Resnet50 V1 FPN 512x512','CenterNet Resnet50 V1 FPN Keypoints 512x512','CenterNet Resnet101 V1 FPN 512x512','CenterNet Resnet50 V2 512x512','CenterNet Resnet50 V2 Keypoints 512x512','EfficientDet D0 512x512','EfficientDet D1 640x640','EfficientDet D2 768x768','EfficientDet D3 896x896','EfficientDet D4 1024x1024','EfficientDet D5 1280x1280','EfficientDet D6 1280x1280','EfficientDet D7 1536x1536','SSD MobileNet v2 320x320','SSD MobileNet V1 FPN 640x640','SSD MobileNet V2 FPNLite 320x320','SSD MobileNet V2 FPNLite 640x640','SSD ResNet50 V1 FPN 640x640 (RetinaNet50)','SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)','SSD ResNet101 V1 FPN 640x640 (RetinaNet101)','SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)','SSD ResNet152 V1 FPN 640x640 (RetinaNet152)','SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)','Faster R-CNN ResNet50 V1 640x640','Faster R-CNN ResNet50 V1 1024x1024','Faster R-CNN ResNet50 V1 800x1333','Faster R-CNN ResNet101 V1 640x640','Faster R-CNN ResNet101 V1 1024x1024','Faster R-CNN ResNet101 V1 800x1333','Faster R-CNN ResNet152 V1 640x640','Faster R-CNN ResNet152 V1 1024x1024','Faster R-CNN ResNet152 V1 800x1333','Faster R-CNN Inception ResNet V2 640x640','Faster R-CNN Inception ResNet V2 1024x1024','Mask R-CNN Inception ResNet V2 1024x1024']
    elif args.detector == 'faster_rcnn':
        model_display_name = 'Faster R-CNN ResNet101 V1 800x1333'
    else:
        raise Exception('Invalid Detector %s' % args.detector)
    model_handle = ALL_MODELS[model_display_name]

    print('Selected model:' + model_display_name)
    print('Model Handle at TensorFlow Hub: {}'.format(model_handle))
    print('loading model...')
    model = hub.load(model_handle)
    print('model loaded!')


    # data loader
    data_loader = DatasetLoader(data_type=args.data_type, data_path=args.data_path)

    # detecter
    with open(os.path.join(args.data_split, 'eval.txt'), 'r') as f:
        data_list = f.read().split('\n')
    detector = Detector(model=model, data_loader=data_loader, sample_list=data_list)

    # run detection
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_result = os.path.join('results', '%s_%s_%s' % (cur_time, args.detector, args.data_type), 'txts')
    detector.run_detection(path_output=path_result)


if __name__ == '__main__':
    main()
