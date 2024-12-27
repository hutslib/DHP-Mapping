# DHP-Mapping
**DHP-Mapping: A Dense Panoptic Mapping System with Hierarchical World Representation and Label Optimization Techniques**

[![arXiv](https://img.shields.io/badge/arXiv-2403.16880-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2403.16880)
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/F1NCSWK26I8)
[![githubpage](https://img.shields.io/badge/Website-DHPMapping-blue)](https://hutslib.github.io/DHP-Mapping/)
[![poster](https://img.shields.io/badge/IROS2024|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://hkustconnect-my.sharepoint.com/:b:/g/personal/thuaj_connect_ust_hk/ESZvkPJNLNhJgKkzg-YgRg0BE7kvqr6TG9x7gPKziITIGQ?e=JkErLB)
---
## 🔔 News

🎊 We plan to integrate [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), [Yolo-World](https://github.com/AILab-CVC/YOLO-World), and [OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2) into this mapping pipeline, for **online open-world semantic-mapping**.
Additionally, we will support [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) to advance developments in embodied-AI. Stay tuned for our upcoming feature releases and don’t forget to give us a star!

🔥 **27/12/2024** Release experiment configs and launch files.

🔥 **26/11/2024** Release main algorithms!

🤗 **15/10/2024** Presented at [iros2024-abudhabi](https://iros2024-abudhabi.org/)

🚀 **30/06/2024** Accepted by IROS2024!

📜 **26/03/2024** arXiv version [paper](https://arxiv.org/abs/2403.16880)

## 🎈 Getting Started

### Step1: Dataset and Pre-process
**SemanticKITTI**: Prepare the dataset to desired format, following [dataset](panoptic_mapping_utils/src/kitti_dataset/README.md).

**flat**: Download the dataset from the [ASL Datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=panoptic_mapping).

### Step2 Workspace setup
```bash
sudo apt-get install python3-pip python3-wstool
mkdir -p DHP_ws/src
cd DHP_ws/src
git clone git@github.com:hutslib/DHP-mapping.git
wstool init . ./DHP-mapping/panoptic_mapping_ssh.rosinstall    # SSH
cd ..
catkin init
catkin config --extend /opt/ros/noetic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
catkin config --merge-devel
catkin build panoptic_mapping_utils
source ../devel/setup.zsh
```

### Step3 Modify config and launch file to run an experiment
The launch files are located in:
[panoptic_mapping_ros/launch/iros_exp](panoptic_mapping_ros/launch/iros_exp)

The config files are located in:
[panoptic_mapping_ros/config/mapper/iros_exp](panoptic_mapping_ros/config/mapper/iros_exp)

### Key Modifications:
- Update `base_path` and `config` in the launch file.
- Note: For long sequences in SemanticKITTI, adjust `max_frames` in the launch file to limit the number of frames based on your device's memory.
- Update the following in the config file:
  - `save_map_path_when_finished`
  - `label_info_print_path`
  - `save_mesh_folder_path`
  - `submap_info_path`
  - `Tr`
  - `P2`
  - `labels: file_name`
  - `visualization: colormap_print_path`
- Note: The `Tr` and `P2` parameters should be modified based on the extrinsics calculated in the [data preprocess step](panoptic_mapping_utils/src/kitti_dataset/README.md#-📌-data-process).

```bash
# run the experiment
roslaunch panoptic_mapping_ros iros_exp_kitti_detectron_07.launch
```
---
### To Do

The code is under cleaning and will be released gradually.

- [x] Initial repo & paper
- [x] Dataset process & player & visualization
- [x] Main algorithms code
- [x] Config and launch files
- [ ] Visualization and other tools
- [ ] Improve docs
- [ ] Feature imporve: integrate open-world segmentation/detection algorithms
- [ ] Feature improve: supporting Habitat-Lab

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@misc{hu2024dhpmapping,
  title={DHP-Mapping: A Dense Panoptic Mapping System with Hierarchical World Representation and Label Optimization Techniques},
  author={Tianshuai Hu and Jianhao Jiao and Yucheng Xu and Hongji Liu and Sheng Wang and Ming Liu},
  year={2024},
  eprint={2403.16880},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}

```
---
This implementation is based on codes from several repositories. Thanks to these authors who kindly open-sourcing their work to the community.
💕 Thanks to [panoptic_mapping](https://github.com/ethz-asl/panoptic_mapping), [Voxfield Panmap](https://github.com/VIS4ROB-lab/voxfield-panmap), [Semantic 3D mapping](https://github.com/shichaoy/semantic_3d_mapping)
