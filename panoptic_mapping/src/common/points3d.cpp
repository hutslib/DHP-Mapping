/*
 * @Author: thuaj@connect.ust.hk
 * @Date: 2022-12-13 16:57:22
 * @LastEditTime: 2024-11-26 19:15:43
 * @Description: lidar class dataloader
 * Copyright (c) 2022 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#include "panoptic_mapping/common/points3d.h"

#include <limits>
#include <memory>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <voxblox/core/color.h>
#include <opencv2/opencv.hpp>

#include "panoptic_mapping/labels/semantic_kitti_label.h"

namespace panoptic_mapping {

void Points3d::Config::checkParams() const {
  // TODO(thuaj): check quaterion and translation is valid
}

void Points3d::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("sensor_setup", &sensor_setup);
  setupParam("min_ray_length_m", &min_ray_length_m, "point.norm() [m]");
  setupParam("max_ray_length_m", &max_ray_length_m, "point.norm() [m]");
  setupParam("Tr", &Tr);
  setupParam("P2", &P2);
  setupParam("color_ply_save_folder", &color_ply_save_folder);
  setupParam("save_color_pcd", &save_color_pcd);
}

void Points3d::setExtrinsicParametersAndPrint() {

  if (config_.sensor_setup == "kitti_lidar_camera" ||
      config_.sensor_setup == "kitti_lidar") {
    Tr_(0, 0) = config_.Tr.at(0);
    Tr_(0, 1) = config_.Tr.at(1);
    Tr_(0, 2) = config_.Tr.at(2);
    Tr_(0, 3) = config_.Tr.at(3);
    Tr_(1, 0) = config_.Tr.at(4);
    Tr_(1, 1) = config_.Tr.at(5);
    Tr_(1, 2) = config_.Tr.at(6);
    Tr_(1, 3) = config_.Tr.at(7);
    Tr_(2, 0) = config_.Tr.at(8);
    Tr_(2, 1) = config_.Tr.at(9);
    Tr_(2, 2) = config_.Tr.at(10);
    Tr_(2, 3) = config_.Tr.at(11);
    P2_(0, 0) = config_.P2.at(0);
    P2_(0, 1) = config_.P2.at(1);
    P2_(0, 2) = config_.P2.at(2);
    P2_(0, 3) = config_.P2.at(3);
    P2_(1, 0) = config_.P2.at(4);
    P2_(1, 1) = config_.P2.at(5);
    P2_(1, 2) = config_.P2.at(6);
    P2_(1, 3) = config_.P2.at(7);
    P2_(2, 0) = config_.P2.at(8);
    P2_(2, 1) = config_.P2.at(9);
    P2_(2, 2) = config_.P2.at(10);
    P2_(2, 3) = config_.P2.at(11);
  }

  // decompose
  // 3x4 projection matrix
  // Convert Eigen matrix to OpenCV matrix
  cv::Mat cvProjectionMatrix;
  cv::eigen2cv(P2_, cvProjectionMatrix);

  // Decompose the projection matrix
  cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY,
      rotMatrixZ, eulerAngles;
  cv::decomposeProjectionMatrix(cvProjectionMatrix, cameraMatrix, rotMatrix,
                                transVect, rotMatrixX, rotMatrixY, rotMatrixZ,
                                eulerAngles);

  // Convert OpenCV matrices to Eigen matrices
  Eigen::Matrix<FloatingPoint, 3, 3> r;
  Eigen::Matrix<FloatingPoint, 4, 1> t_origin;

  cv::cv2eigen(cameraMatrix, k_);
  cv::cv2eigen(rotMatrix, r);
  cv::cv2eigen(transVect, t_origin);

  Eigen::Matrix<FloatingPoint, 3, 1> t;
  t << (t_origin[0] / t_origin[3]), (t_origin[1] / t_origin[3]),
      (t_origin[2] / t_origin[3]);

  Eigen::Matrix<FloatingPoint, 3, 3> Rr;
  Eigen::Matrix<FloatingPoint, 3, 1> tr;
  Rr = Tr_.block<3, 3>(0, 0);
  tr = Tr_.block<3, 1>(0, 3);
  ext_r_ = r.transpose() * Rr;
  ext_t_ = r.transpose() * (tr - t);

}

Points3d::Points3d(const Config& config) : config_(config.checkValid()) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
  // config_.setIntrinsicParametersAndPrint();
  setExtrinsicParametersAndPrint();
  frame_ = 0;
}

template <typename T>
T getLinearInterpolation(const FloatingPoint x, const FloatingPoint y,
                         const T vlt, const T vrt, const T vlb, const T vrb) {
  FloatingPoint x0 = floor(x);
  FloatingPoint y0 = floor(y);
  // clang-format off
T interpolate = static_cast<T>(
        (x0 + 1 - x) * (y0 + 1 - y) * vlt +
        (x - x0) * (y0 + 1 - y) * vrt +
        (x0 + 1 - x) * (y - y0) * vlb +
        (x - x0) * (y - y0) * vrb);
  // clang-format on
  return interpolate;
}

// this function is for kitti dataset
void Points3d::convert2Points3d(const LidarPointCloud& ptcloud,
                                const cv::Mat& color_img,
                                const std::shared_ptr<Camera>& cam_ptr,
                                const cv::Mat& label_img,
                                const KittiLabels kitti_labels,
                                Pointcloud* points, Colors* colors,
                                Labels* labels, cv::Mat* depth_image) {
  points->clear();
  colors->clear();
  labels->clear();
  points->reserve(ptcloud.size());
  colors->reserve(ptcloud.size());
  labels->reserve(ptcloud.size());

  if (config_.sensor_setup == "kitti_lidar_camera") {
    LOG_IF(INFO, config_.verbosity >= 1)
        << "we generate the points3d using kitti lidar camera";
    int index = 0;
    ColorLidarPointCloud ptcloud_rgb;
    for (const auto& pt : ptcloud) {
      KittiLabel cur_label = kitti_labels[index];
      index++;
      Point point(pt.x, pt.y, pt.z);  // points in lidar coordinate

      const FloatingPoint ray_distance = point.norm();
      if (ray_distance > config_.min_ray_length_m &&
          ray_distance < config_.max_ray_length_m) {

        Point pointCam = ext_r_ * point + ext_t_;
        if (pointCam.z() <= 0.0f) {
          continue;
        }
        FloatingPoint u, v;
        if (!cam_ptr->projectKittiPointToImagePlane(pointCam, &u, &v, k_)) {
          continue;
        }

        cv::Vec3b bgr = getLinearInterpolation(
            u, v, color_img.at<cv::Vec3b>(floor(v), floor(u)),
            color_img.at<cv::Vec3b>(floor(v), floor(u) + 1),
            color_img.at<cv::Vec3b>(floor(v) + 1, floor(u)),
            color_img.at<cv::Vec3b>(floor(v) + 1, floor(u) + 1));
        // r g b
        Color color(bgr[2], bgr[1], bgr[0]);

        // NOTE(thuaj): we take this part from voxfile-panmap
        // only for semantic kitti dataset
        // Filter those outlier and dynamic objects
        // sem_label <=1 means unlabeled or outlier
        // sem_label > 250 means moving (dynamic) objects
        // TODO(py): in practice, these moving objects should also be considered
        if (cur_label.sem_label <= 1) {
          // outliers
          continue;
        }
        // filter_moving_object car has two sem_label moving and static we
        // filter the moving one
        if (cur_label.sem_label > 250) {
          continue;
        }
        // get label
        // TODO(thuaj): get the semantic and instance number and probabilistic
        Label label(label_img.at<int>(v, u), label_img.at<int>(v, u), 0.9, 0);
        if (label.ins_label_ > 10000) {
          std::cout << "failed" << std::endl;
          std::cout << "label" << label.ins_label_ << std::endl;
          std::cout << "v , u : " << v << " , " << u << std::endl;
        }
        // generate Points3d (in the lidar coordinate)
        points->push_back(point);
        colors->push_back(color);
        labels->push_back(label);

        if (config_.save_color_pcd) {
          // visualization debug colored points3d
          ColorLidarPoint pt_rgb;
          pt_rgb.x = point(0, 0);
          pt_rgb.y = point(1, 0);
          pt_rgb.z = point(2, 0);
          pt_rgb.r = color.r;
          pt_rgb.g = color.g;
          pt_rgb.b = color.b;

          ptcloud_rgb.push_back(pt_rgb);
        }
      }
    }
    if (config_.save_color_pcd) {
      pcl::PCDWriter writer;
      std::string save_colored_folder =
          config_.color_ply_save_folder + "/colorlidar";
      create_directory_if_not_exists(save_colored_folder);
      std::string save_colored_file =
          save_colored_folder + "/color_" + std::to_string(frame_) + ".pcd";
      writer.write<ColorLidarPoint>(save_colored_file, ptcloud_rgb, false);
      frame_++;
    }
  } else if (config_.sensor_setup == "kitti_lidar") {
    LOG_IF(INFO, config_.verbosity >= 1)
        << "we generate the points3d using kitti lidar (for gt generate)";
    int index = 0;
    for (const auto& pt : ptcloud) {
      KittiLabel cur_label = kitti_labels[index];

      index++;
      Point point(pt.x, pt.y, pt.z);  // points in lidar coordinate
      const FloatingPoint ray_distance = point.norm();
      if (ray_distance > config_.min_ray_length_m &&
          ray_distance < config_.max_ray_length_m) {
        Point pointCam = ext_r_ * point + ext_t_;

        if (pointCam.z() <= 0.0f) {
          continue;
        }

        FloatingPoint u, v;
        if (!cam_ptr->projectKittiPointToImagePlane(pointCam, &u, &v, k_)) {
          continue;
        }

        // interpolate and get color
        // TODO(thuaj): replace the interpolation part using the interpolate
        // class
        cv::Vec3b bgr = getLinearInterpolation(
            u, v, color_img.at<cv::Vec3b>(floor(v), floor(u)),
            color_img.at<cv::Vec3b>(floor(v), floor(u) + 1),
            color_img.at<cv::Vec3b>(floor(v) + 1, floor(u)),
            color_img.at<cv::Vec3b>(floor(v) + 1, floor(u) + 1));
        // r g b
        Color color(bgr[2], bgr[1], bgr[0]);

        // NOTE(thuaj): we take this part from voxfile-panmap
        // only for semantic kitti dataset
        // Filter those outlier and dynamic objects
        // sem_label <=1 means unlabeled or outlier
        // sem_label > 250 means moving (dynamic) objects
        // TODO(py): in practice, these moving objects should also be considered
        if (cur_label.sem_label <= 1) {
          // outliers
          continue;
        }
        // filter_moving_object car has two sem_label moving and static we
        // filter the moving one

        if (cur_label.sem_label > 250) {
          continue;
        }
        // directly get the label form semantickitti (for gt generation)
        Label label(cur_label.sem_label, cur_label.ins_label,
                    cur_label.id_label, 0.9, 0);
        // generate Points3d (in the lidar coordinate)
        points->push_back(point);
        colors->push_back(color);
        labels->push_back(label);
      }
    }
  }
}

// rgbd
void Points3d::convert2Points3d(const cv::Mat& depth_img,
                                const cv::Mat& color_img,
                                const std::shared_ptr<Camera>& cam_ptr,
                                const cv::Mat& label_img, Pointcloud* points,
                                Colors* colors, Labels* labels) {
  points->clear();
  colors->clear();
  labels->clear();
  points->reserve(depth_img.total());
  colors->reserve(depth_img.total());
  labels->reserve(depth_img.total());


  // Compute the 3D pointcloud from a depth image.
  const float cam_fx = cam_ptr->getConfig().fx;
  const float cam_fy = cam_ptr->getConfig().fy;
  const float cam_vx = cam_ptr->getConfig().vx;
  const float cam_vy = cam_ptr->getConfig().vy;
  const float fx_inv = 1.f / cam_fx;
  const float fy_inv = 1.f / cam_fy;

  for (int v = 0; v < depth_img.rows; v++) {
    for (int u = 0; u < depth_img.cols; u++) {
      cv::Vec3f vertex;  // x, y, z
      vertex[2] = depth_img.at<float>(v, u);
      vertex[0] = (static_cast<float>(u) - cam_vx) * vertex[2] * fx_inv;
      vertex[1] = (static_cast<float>(v) - cam_vy) * vertex[2] * fy_inv;
      Point point(vertex[0], vertex[1], vertex[2]);

      const FloatingPoint ray_distance = point.norm();
      if (ray_distance > config_.min_ray_length_m &&
          ray_distance < config_.max_ray_length_m) {
        // only process valid points
        cv::Vec3b bgr = color_img.at<cv::Vec3b>(v, u);
        // r g b
        Color color(bgr[2], bgr[1], bgr[0]);
        // get label
        Label label(label_img.at<int>(v, u), label_img.at<int>(v, u), 0.9, 0);

        // generate Points3d (in the lidar coordinate)
        points->push_back(point);
        colors->push_back(color);
        labels->push_back(label);

      }
    }
  }
}

bool Points3d::create_directory_if_not_exists(const std::string& folder_path) {
  // Check if the folder exists
  if (!std::filesystem::exists(folder_path)) {
    // If it doesn't exist, try to create it recursively
    if (std::filesystem::create_directories(folder_path)) {
      std::cout << "Folder created: " << folder_path << std::endl;
      return true;
    } else {
      std::cerr << "Error creating folder: " << folder_path << std::endl;
      return false;
    }
  } else {
    // If it exists, return true
    std::cout << "Folder already exists: " << folder_path << std::endl;
    return true;
  }
}

}  // namespace panoptic_mapping
