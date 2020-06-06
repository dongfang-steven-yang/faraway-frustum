import tensorflow as tf
import time
import cv2
import numpy as np

from models.pre_trained.yolo_v3.utils.misc_utils import parse_anchors, read_class_names
from models.pre_trained.yolo_v3.utils.nms_utils import gpu_nms
from models.pre_trained.yolo_v3.utils.plot_utils import get_color_table
from models.pre_trained.yolo_v3.utils.data_aug import letterbox_resize
from models.pre_trained.yolo_v3.model import yolov3

# CuDNN error
# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
PATH_YOLOv3_DATA = 'models/pre_trained/yolo_v3'


class YOLOv3Interface:
    def __init__(self):
        self.name = 'yolo_v3'
        # create tf session
        self.tf_session = tf.Session(config=config)

        # config model
        path_class_name = PATH_YOLOv3_DATA + '/data/coco.names'
        path_anchor = PATH_YOLOv3_DATA + '/data/yolo_anchors.txt'
        path_restore = PATH_YOLOv3_DATA + '/data/darknet_weights/yolov3.ckpt'

        self.img_size_yolo3 = (416, 416)
        self.classes = read_class_names(path_class_name)
        self.num_class = len(self.classes)
        self.color_table = get_color_table(self.num_class)
        self.anchors = parse_anchors(path_anchor)

        # initialize model
        self.img_empty = tf.placeholder(tf.float32, [1, self.img_size_yolo3[0], self.img_size_yolo3[1], 3], name='input_data')
        self.yolo_model = yolov3(self.num_class, self.anchors)

        with tf.variable_scope('yolov3'):
            pred_feature_maps = self.yolo_model.forward(self.img_empty, False)
        pred_boxes, pred_confs, pred_probs = self.yolo_model.predict(pred_feature_maps)
        pred_scores = pred_confs * pred_probs
        self.boxes, self.scores, self.labels = gpu_nms(
            pred_boxes, pred_scores, self.num_class,
            max_boxes=200,
            score_thresh=0.3,
            nms_thresh=0.45
        )
        saver = tf.train.Saver()
        saver.restore(self.tf_session, path_restore)

        print("Yolo-v3 2D detection initialized.")

    def detect(self, img):
        t1 = time.time()

        img_resized, resize_ratio, dw, dh = letterbox_resize(img, self.img_size_yolo3[0], self.img_size_yolo3[1])
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = np.asarray(img_resized, np.float32)
        img_resized = img_resized[np.newaxis, :] / 255.
        boxes_, scores_, labels_ = self.tf_session.run(
            [self.boxes, self.scores, self.labels],
            feed_dict={self.img_empty: img_resized}
        )
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        t2 = time.time()

        print('Yolo-v3 inference completed, time elapsed = %.4f' % (t2 - t1))

        return None, boxes_, labels_, scores_
