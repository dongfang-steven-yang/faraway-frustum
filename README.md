# faraway-frustum
Faraway-Frustum: Dealing with LiDAR Sparsity for 3D Object Detection using Fusion


## Environment

The program is tested with `python 3.7` with `tensorflow 1.15`. 
You also need to install the packages listed in `requirements.txt`.

## Getting Started

1. Download the pre-trained Mask-RCNN model 
([link](https://drive.google.com/file/d/1QsfRE5NV6a9aCs6SG0LEklmOS9at2EFK/view?usp=sharing)) 
and put it into folder `detectors/mask_rcnn/`.

2. Prepare Kitti Dataset: download Kitti Dataset and arrange it as follows:
    ```shell script
    ├── testing
    │   ├── calib
    │   ├── image_2
    │   └── velodyne
    └── training
        ├── calib
        ├── image_2
        ├── label_2
        └── velodyne
    ```
   
3. Run stage one of 2D detection nd save results: execute the script `step1_save_2d_results.py` to obtain 
the 2D detection result (including boxes, masks, labels, scores). It will be saved as pickle file.
You need to specify the path to Kitti dataset `--path_kitti` and the path to store the 2D detection results 
`--path_result`.

4. Run stage two of frustum-projection and 3D box estimation: execute the script `step2_get_kitti_results.py` 
to obtain the final results of Kitti txt format. It will read the pickle files obtained in previous step and 
generating final results in the same directory. Again, you need to specify the path to Kitti dataset `--path_kitti` 
and the path to store the 2D detection results `--path_result`. They should be the same as in the previous 
step.
