# faraway-frustum
Faraway-Frustum: Dealing with LiDAR Sparsity for 3D Object Detection using Fusion

Paper in ArXiv: https://arxiv.org/abs/2011.01404

Submitted to ICAR 2021.

## Environment
There two scripts for running this program. Each script requires a different version of `tensorflow`. Check the python script files for version detail. 

We recommend to use `anaconda` to manage the tensorflow environment. You can use the following commands to configure your environment:
```shell
conda create -n {your environment name} tensorflow-gpu={a specific version} python=3.7
```
Then `anaconda` will solve the dependencies automatically for you. (Make sure you have successfully installed the NVIDIA driver.)

You also need to install the additional standard python packages listed in `requirements.txt`.

## Getting Started

Note: there are additional instructions inside the python scripts of step 1 and step 2. Do check them. 

1. Download the pre-trained Mask-RCNN model 
([link](https://drive.google.com/file/d/1QsfRE5NV6a9aCs6SG0LEklmOS9at2EFK/view?usp=sharing)) 
and put it into folder `detectors/mask_rcnn/`.

2. Prepare Kitti Dataset: download [Kitti Dataset](http://www.cvlibs.net/datasets/kitti/) and arrange it as follows.
   In Kitti, `training` has 7481 samples and `testing` has 7518 samples.
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

4. Download the trained NN models for pedestrian/car position detection/refinement in the frustum pointcloud.
Here is the link to download the models: [NN models - Google Drive](https://drive.google.com/file/d/1_BdfX87hUUXLlytWfNxYeOD-qpmGvZmc/view?usp=sharing)

5. Run stage two of frustum-projection and 3D box estimation: execute the script `step2_get_kitti_results.py` 
to obtain the final results of Kitti txt format. It will read the pickle files obtained in previous step and 
generating final results in the same directory. Again, you need to specify the path to Kitti dataset `--path_kitti` 
and the path to store the 2D detection results `--path_result`. They should be the same as in the previous 
step. You also need to specify the path to trained NN models. See additional instruction in the `step2_get_kitti_results.py`.


## Contact

- Dongfang Yang: yang.3455@osu.edu
- Haolin Zhang: zhang.10749@osu.edu
- Ekim Yurtsever: yurtsever.2@osu.edu