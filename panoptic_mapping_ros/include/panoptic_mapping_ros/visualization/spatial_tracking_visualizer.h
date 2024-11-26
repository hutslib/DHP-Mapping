/*
 * @Author: thuaj@connect.ust.hk
 * @Description: DHP-Mapping visualizer module.
 * Copyright (c) 2023 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#ifndef PANOPTIC_MAPPING_ROS_VISUALIZATION_SPATIAL_TRACKING_VISUALIZER_H_
#define PANOPTIC_MAPPING_ROS_VISUALIZATION_SPATIAL_TRACKING_VISUALIZER_H_

#include <string>
#include <unordered_map>

#include <panoptic_mapping/3rd_party/config_utilities.hpp>
#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/tracking/id_tracker_base.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/node_handle.h>

namespace panoptic_mapping {

class SpatialTrackingVisualizer {
 public:
  // config
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 1;
    bool visualize_tracking = true;
    std::string ros_namespace;

    Config() { setConfigName("SpatialTrackingVisualizer"); }

   protected:
    void setupParamsAndPrinting() override;
    void fromRosParam() override;
  };

  // Constructors.
  explicit SpatialTrackingVisualizer(const Config& config);
  virtual ~SpatialTrackingVisualizer() = default;

  // Setup.
  void registerIDTracker(IDTrackerBase* tracker);

  // Publish visualization requests.
  void publishImage(const cv::Mat& image, const std::string& name);
  void publishptCloud(const pcl::PointCloud<pcl::PointXYZRGB>& ptcloud_rbg,
                      const std::string& name);

 private:
  const Config config_;

  // Publishers.
  ros::NodeHandle nh_;
  std::unordered_map<std::string, ros::Publisher> publishers_;
  ros::Publisher rawcolor_pub_;
  ros::Publisher inputid_pub_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_ROS_VISUALIZATION_SPATIAL_TRACKING_VISUALIZER_H_
