
# KITTI
## 📌 Download the dataset
- Download kitti odometry (grayscale, color, velodyne, calibration, ground truth poses, development kit) from https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- Download SemanticKITTI label (labels) from http://www.semantic-kitti.org/dataset.html

source data organization:
```
├── source folder
│   ├── data_odometry_color
│   ├── data_odometry_labels
│   ├── data_odometry_poses
│   ├── data_odometry_velodyne
|   ├── data_odometry_calib
```
target data organization:
```
├── target folder (defaults as xxx/dataset/sequences)
│   ├── 00
│   │   ├── velodyne
│   │   |   ├── 000000.bin
│   │   |   ├── ....bin
│   |   ├── labels
│   │   |   ├── 000000.label
│   │   |   ├── ....label
│   │   ├── image2
│   │   |   ├── 000000.png
│   │   |   ├── ....png
│   │   ├── image3 （default we do not use this but also store it in this folder）
│   │   |   ├── 000000.png
│   │   |   ├── ....png
│   ├── 01
│   ├── ...
│   ├── poses.txt
│   ├── calib.txt
│   ├── timestamps.txt
|   ├── detectron_labels.csv
```
---
## 📌 Data process
```
# >> set up detectron2 environment
conda create -n dhp python=3.9 -y
conda activate dhp
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install natsort fire opencv-python pandas scipy open3d
# >> run the python script
python3 preprocess_kitti.py $source_folder $target_folder
# >> get lidar camera extrinsic calibration
python3 get_lidar_camera_extrinsic.py $data_root_folder
```
Your have finished the data process, and you can back to the main README.md to continue the next step.

The following is the optional step to check and visualize the data.
## ▶️ Check ROS topic player

modify the launch file to set the correct
<span style="background:WhiteSmoke">data path</span> and <span style="background:WhiteSmoke">max_frames</span> in the launch file
```
cd <workspace_dir>
catkin build panoptic_mapping_utils
source ./devel/setup.bash
# >>> using the segmentation results of detectron2 for experiments
roslaunch panoptic_mapping_utils play_kitti_image_label.launch
rviz -d $(rospack find panoptic_mapping_utils)/config/kitti_player_check.rviz
# >>> using the ground truth labels(from kitti label panoptic labels) for generate ground truth map
roslaunch panoptic_mapping_utils play_kitti_lidar_label.launch
rviz -d $(rospack find panoptic_mapping_utils)/config/kitti_player_check_lidar.rviz
```
## ▶️ Visualize the segmentation results of detectron2
```
# vis detectron segmentation
python3 vis_detectron_segmentation.py
```
