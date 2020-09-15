# faraway-frustum
A frustum-based method for detecting far-away objects in autonomous driving.

Title option: A Frustum-based Sensor Fusion Approach for Faraway Pedestrian Detection


## Getting Started

### Environment Config

We recommend to use [Anaconda](https://docs.anaconda.com/anaconda/install/) to create a virtual environment for running this program.

The program is tested with `python 3.7` with `tensorflow 1.15`. In the terminal, create a new `conda` environment with the correct `python` version and `tensorflow` version:
```shell script
conda create -n faraway-frustum python=3.7 tensorflow-gpu=1.15
```
The `conda` will automatically figure out what are the dependencies of the above specific `tensorflow` version.

Activate the newly created conda environment:
```shell script
conda activate faraway-frustum
```

Clone this repository:
```shell script
git clone https://github.com/dongfang-steven-yang/faraway-frustum.git
```
Enter the project folder:
```shell script
cd faraway-frustum
```

Install necessary basic packages in the conda environment `faraway-frustum`:
```shell script
pip install -r requirements.txt
```
Or you can manually install the dependencies listed in `requirements.txt`

### Downloading Pre-trained Models

Download the following models, unzip and put them into `models/pre_trained/` folder:
- Mask-RCNN model: [Google Drive](https://drive.google.com/file/d/10UrdoYgqBQhNGHLFFLUHTXnw0Qv-TYvE/view?usp=sharing)
- Yolo-v3 model: [Google Drive](https://drive.google.com/file/d/1rAgNGKjeXoSjNfHXqiuvoie5KckbBSAN/view?usp=sharing)

### Dataset Directory Structure
For Kitti:
```shell script
.
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

### Running Scripts

In total there are 4 scripts:
- `detect.py`: running object detection.
- `visualize.py`: visualizing the results by plotting.
- `create_subset.py`: creating subsets for far-away objects
- `cal_score.py`: calculating finals scores for specific subsets.

Check the `argparser` in each script to learn about how to give parameters.

## TODO List

- [x] Detection pipeline
- [x] Visualize
- [x] Creating subset pipeline
- [x] Evaluation pipeline (score calculation and comparison)
- [ ] Update readme
- [ ] Ground removal algorithms (future work)
