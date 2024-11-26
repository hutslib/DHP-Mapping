/*
 * @Author: thuaj@connect.ust.hk
 * @Description: DHP-Mapping data structure to manage the final input data (as a 3d points).
 * Copyright (c) 2022 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#ifndef PANOPTIC_MAPPING_COMMON_POINTS3D_H_
#define PANOPTIC_MAPPING_COMMON_POINTS3D_H_
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/common/input_data.h"
#include "panoptic_mapping/labels/label_handler_base.h"

namespace panoptic_mapping {
class Points3d {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 0;
    std::vector<FloatingPoint> Tr;
    std::vector<FloatingPoint> P2;
    std::string sensor_setup = "rgbd";
    FloatingPoint min_ray_length_m = 0.1f;
    FloatingPoint max_ray_length_m = 15.f;
    std::string color_ply_save_folder;
    bool save_color_pcd = false;
    Config() { setConfigName("Points3d"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  explicit Points3d(const Config& config);
  virtual ~Points3d() = default;

  // Access.
  const Config& getConfig() const { return config_; }
  void setExtrinsicParametersAndPrint();

  // this function is for kitti dataset
  void convert2Points3d(const LidarPointCloud& ptcloud,
                        const cv::Mat& color_img,
                        const std::shared_ptr<Camera>& cam_ptr,
                        const cv::Mat& label_img,
                        const KittiLabels kitti_labels, Pointcloud* points,
                        Colors* colors, Labels* labels, cv::Mat* depth_image);

  void convert2Points3d(const cv::Mat& depth_img, const cv::Mat& color_img,
                        const std::shared_ptr<Camera>& cam_ptr,
                        const cv::Mat& label_img, Pointcloud* points,
                        Colors* colors, Labels* labels);
  bool create_directory_if_not_exists(const std::string& folder_path);

 private:
  const Config config_;

  Eigen::Matrix<float, 3, 4> Tr_;  // extrinsic form velo to cam ref
  Eigen::Matrix<float, 3, 4> P2_;  // projection matrix from cam ref to cam 2

  int frame_;
  Eigen::Matrix<FloatingPoint, 3, 3> k_;
  Eigen::Matrix<FloatingPoint, 3, 3> ext_r_;
  Eigen::Matrix<FloatingPoint, 3, 1> ext_t_;
};
};      // namespace panoptic_mapping
#endif  // PANOPTIC_MAPPING_COMMON_POINTS3D_H_
