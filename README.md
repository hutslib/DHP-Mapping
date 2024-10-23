# DHP-Mapping
**DHP-Mapping: A Dense Panoptic Mapping System with Hierarchical World Representation and Label Optimization Techniques**

[![arXiv](https://img.shields.io/badge/arXiv-2403.16880-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2403.16880)
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/F1NCSWK26I8)
[![githubpage](https://img.shields.io/badge/Website-DHPMapping-blue)](https://hutslib.github.io/DHP-Mapping/)
---
![](assets/docs/main.png)

## 🔔 Update
**26/03/2024** arXiv version [paper](https://arxiv.org/abs/2403.16880)

**30/06/2024** accepted by IROS2024!

**15/10/2024** present at [iros2024-abudhabi](https://iros2024-abudhabi.org/)

### To Do

The code is under cleaning and will be released gradually.

- [x] initial repo & paper
- [x] dataset process & player & visualization
- [ ] main algorithms code
- [ ] visualization and other tools
- [ ] improve docs
## 🎈 Getting Start
### Step1: Dataset and pre-process
#### SemanticKITTI
- prepare the dataset to desire format, following [dataset](panoptic_mapping_utils/src/kitti_dataset/README.md).
#### flat
- Download the dataset from the [ASL Datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=panoptic_mapping).
### Step2 Workspace setup
```sudo apt-get install python3-pip python3-wstool
mkdir -p DHP_ws/src
cd DHP_ws/src
git clone git@github.com:hutslib/DHP-mapping.git
wstool init . ./DHP_Mapping/panoptic_mapping_ssh.rosinstall    # SSH
cd ./DHP_Mapping
catkin init
catkin config --extend /opt/ros/noetic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
catkin config --merge-devel
catkin build panoptic_mapping_utils
source ../devel/setup.zsh
```
##
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
