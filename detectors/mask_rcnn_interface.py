import time
import os
import sys
import random
import numpy as np
from detectors.mask_rcnn.samples.coco import coco  # Import COCO config
from detectors.mask_rcnn.mrcnn import utils
import detectors.mask_rcnn.mrcnn.model as modellib

# Root directory of the project
ROOT_DIR = os.path.abspath("detectors/mask_rcnn")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# # Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNInterface:
    def __init__(self):
        self.name = 'mask_rcnn'
        self.classes = dict(zip(range(len(class_names)), class_names))
        self.num_class = len(self.classes)
        self.color_table = get_color_table(self.num_class)

        # todo @ initialize the model
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def detect(self, img):
        t1 = time.time()
        # todo @ execute detection
        results = self.model.detect([img], verbose=1)

        # Visualize results
        r = results[0]
        cl_id_o = np.asarray(r['class_ids'])
        cl_rois_o = np.asarray(r['rois'])
        cl_masks_o = np.asarray(r['masks'])
        cl_scores_o = np.asarray(r['scores'])

        cl_rois_o = cl_rois_o[:, [1, 0, 3, 2]]
        cl_masks_o = np.moveaxis(cl_masks_o, 2, 0)

        t2 = time.time()
        print('Mask RCNN inference completed, time elapsed = %.4f' % (t2 - t1))

        return cl_masks_o, cl_rois_o, cl_id_o, cl_scores_o