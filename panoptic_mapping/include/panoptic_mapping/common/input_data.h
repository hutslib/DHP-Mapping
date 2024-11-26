/*
 * @Author: thuaj@connect.ust.hk
 * @Description: DHP-Mapping input_data.
 * Copyright (c) 2024 by thuaj@connect.ust.hk, All Rights Reserved.
 */

#ifndef PANOPTIC_MAPPING_COMMON_INPUT_DATA_H_
#define PANOPTIC_MAPPING_COMMON_INPUT_DATA_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core/mat.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/common/segment.h"
namespace panoptic_mapping {
class InputDataUser;

/**
 * All custom types of input data are defined here.
 */

// Labels supplied by the detectron2 network.
struct DetectronLabel {
  int id = 0;
  bool is_thing = true;
  int category_id = 0;
  int instance_id = 0;
  float score = 0.f;
};
struct KittiLabel {
  KittiLabel() : sem_label(0), ins_label(0) {}
  KittiLabel(short int _sem_label, short int _ins_label)
      : sem_label(_sem_label), ins_label(_ins_label) {}
  KittiLabel(uint32_t label) {
    full_label = label;
    sem_label = label & 0xFFFF;
    ins_label = label >> 16;
    id_label = sem_label * kKITTIMaxIntstance + ins_label;
  }

  int id_label;
  uint32_t full_label;
  short int sem_label;  // int16_t
  short int ins_label;  // int16_t
  // std::string name;
};

typedef std::unordered_map<int, DetectronLabel> DetectronLabels;  // <id-label>
typedef AlignedVector<KittiLabel> KittiLabels;                    // <id-label>
/**
 * A class that wraps all input data to be processed into a common structure.
 * Optional fields are also included here but the data is only set if required.
 */
class InputData {
 public:
  // Lists all types of possible inputs.
  enum class InputType {
    kRawImage,
    kLidarPoints,
    kDepthImage,
    kColorImage,
    kSegmentationImage,
    kDetectronLabels,
    kKittiLabels,  // kitti gt panoptic labels
    kVertexMap,
    kValidityImage,
    kUncertaintyImage,
    kPoints,   // Point3d structure 3d Points
    kColors,   // Point3d structure 3d Points corresponding color
    kLabels,   // Point3d structure 3d Points corresponding label
    kSegments  // Segments to be integration
  };

  static std::string inputTypeToString(InputType type) {
    switch (type) {
      case InputType::kRawImage:
        return "Raw Image";
      case InputType::kLidarPoints:
        return "Lidar Points";
      case InputType::kDepthImage:
        return "Depth Image";
      case InputType::kColorImage:
        return "Color Image";
      case InputType::kSegmentationImage:
        return "Segmentation Image";
      case InputType::kDetectronLabels:
        return "Detectron Labels";
      case InputType::kKittiLabels:
        return "Kitti Labels";
      case InputType::kUncertaintyImage:
        return "Uncertainty Image";
      case InputType::kVertexMap:
        return "Vertex Map";
      case InputType::kValidityImage:
        return "Validity Image";
      case InputType::kPoints:
        return "Points";
      case InputType::kColors:
        return "Colors";
      case InputType::kLabels:
        return "Labels";
      case InputType::kSegments:
        return "Segments";
      default:
        return "Unknown Input";
    }
  }

  using InputTypes = std::unordered_set<InputType>;

  // Construction.
  InputData() = default;
  virtual ~InputData() = default;

  /* Data input */
  void setT_M_C(const Transformation& T_M_C) { T_M_C_ = T_M_C; }
  void setTimeStamp(double timestamp) { timestamp_ = timestamp; }
  void setFrameName(const std::string& frame_name) {
    sensor_frame_name_ = frame_name;
  }
  void setRawImage(const cv::Mat& raw_image) {
    raw_image_ = raw_image;
    contained_inputs_.insert(InputType::kRawImage);
  }
  void setLidarPoints(const pcl::PointCloud<pcl::PointXYZ>& lidar_points) {
    lidar_points_ = lidar_points;
    contained_inputs_.insert(InputType::kLidarPoints);
  }
  void setDepthImage(const cv::Mat& depth_image) {
    depth_image_ = depth_image;
    contained_inputs_.insert(InputType::kDepthImage);
  }
  void setColorImage(const cv::Mat& color_image) {
    color_image_ = color_image;
    contained_inputs_.insert(InputType::kColorImage);
  }
  void setIdImage(const cv::Mat& id_image) {
    id_image_ = id_image;
    contained_inputs_.insert(InputType::kSegmentationImage);
  }
  void setDetectronLabels(const DetectronLabels& labels) {
    detectron_labels_ = labels;
    contained_inputs_.insert(InputType::kDetectronLabels);
  }
  void setKittiLabels(const KittiLabels& labels) {
    kitti_labels_ = labels;
    contained_inputs_.insert(InputType::kKittiLabels);
  }
  void setVertexMap(const cv::Mat& vertex_map) {
    vertex_map_ = vertex_map;
    contained_inputs_.insert(InputType::kVertexMap);
  }
  void setValidityImage(const cv::Mat& validity_image) {
    validity_image_ = validity_image;
    contained_inputs_.insert(InputType::kValidityImage);
  }

  void setUncertaintyImage(const cv::Mat& uncertainty_image) {
    uncertainty_image_ = uncertainty_image;
    contained_inputs_.insert(InputType::kUncertaintyImage);
  }

  void setPoints(const Pointcloud& points) {
    points_ = points;
    contained_inputs_.insert(InputType::kPoints);
  }

  void setColors(const Colors& colors) {
    colors_ = colors;
    contained_inputs_.insert(InputType::kColors);
  }

  void setLabels(const Labels& labels) {
    labels_ = labels;
    contained_inputs_.insert(InputType::kLabels);
  }

  void setSegments(const Segments& segments) {
    segments_ = segments;
    contained_inputs_.insert(InputType::kSegments);
  }

  void setSubmapIDList(const std::vector<int>& submap_id_list) {
    submap_ids_ = submap_id_list;
  }

  // Access.
  // Access to constant data.
  const Transformation& T_M_C() const { return T_M_C_; }
  const std::string& sensorFrameName() const { return sensor_frame_name_; }
  double timestamp() const { return timestamp_; }
  const cv::Mat& rawImage() const { return raw_image_; }
  const pcl::PointCloud<pcl::PointXYZ>& lidarPoints() const {
    return lidar_points_;
  }
  const cv::Mat& depthImage() const { return depth_image_; }
  const cv::Mat& colorImage() const { return color_image_; }
  const DetectronLabels& detectronLabels() const { return detectron_labels_; }
  const cv::Mat& vertexMap() const { return vertex_map_; }
  const cv::Mat& idImage() const { return id_image_; }
  const KittiLabels& kittiLabels() const { return kitti_labels_; }
  const cv::Mat& validityImage() const { return validity_image_; }
  const cv::Mat& uncertaintyImage() const { return uncertainty_image_; }
  const Pointcloud& points() const { return points_; }
  const Colors& colors() const { return colors_; }
  const Labels& labels() const { return labels_; }
  const Segments& segments() const { return segments_; }
  const std::vector<int>& submapIDList() const { return submap_ids_; }

  // Access to modifyable data.
  cv::Mat* idImagePtr() { return &id_image_; }
  cv::Mat* validityImagePtr() { return &validity_image_; }
  Labels* labelsPtr() { return &labels_; }
  Colors* colorsPtr() { return &colors_; }
  Segments* segmentsPtr() {return &segments_;}
  Pointcloud* PointcloudPtr() { return &points_; }

  // Tools.
  bool has(InputType input_type) const {
    return contained_inputs_.find(input_type) != contained_inputs_.end();
  }

  void list() const {
    std::cout << "contained_inputs:" << std::endl;
    for (auto type : contained_inputs_) {
      std::cout << InputData::inputTypeToString(type) << std::endl;
    }
  }

 private:
  friend InputDataUser;
  friend class InputSynchronizer;

  // Permanent data.
  Transformation T_M_C_;  // Transform from Camera/Sensor (C) to Mission (M).
  std::string sensor_frame_name_;  // Sensor frame name, taken from depth data.
  double timestamp_ = 0.0;         // Timestamp of the inputs.

  // Common Input data.
  cv::Mat raw_image_;  // RGB (CV_8U). raw (with distortion)
  pcl::PointCloud<pcl::PointXYZ> lidar_points_;  // lidar points cloud
  cv::Mat depth_image_;        // Float depth image (CV_32FC1).
  cv::Mat color_image_;        // BGR (CV_8U). raw -> undistorted
  cv::Mat id_image_;           // Mutable assigned ids as ints (CV_32SC1).
  cv::Mat uncertainty_image_;  // Float image containing uncertainty information
                               // (CV_32FC1)

  // Common derived data.
  cv::Mat vertex_map_;      // XYZ points (CV32FC3), can be compute via camera.
  cv::Mat validity_image_;  // 0-1 image for valid pixels (CV_8UC1).

  // Optional Input data.
  DetectronLabels detectron_labels_;
  KittiLabels kitti_labels_;
  // Points3d structure
  Pointcloud points_;
  Colors colors_;
  Labels labels_;

  // after spatial matching process segments to be integrated
  Segments segments_;
  // store the submap ids in the current input segments_;
  std::vector<int> submap_ids_;

  // Content tracking.
  InputData::InputTypes contained_inputs_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_INPUT_DATA_H_
