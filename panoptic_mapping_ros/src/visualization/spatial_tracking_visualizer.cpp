#include "panoptic_mapping_ros/visualization/spatial_tracking_visualizer.h"

#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>

namespace panoptic_mapping {

void SpatialTrackingVisualizer::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("visualize_tracking", &visualize_tracking);
}

void SpatialTrackingVisualizer::Config::fromRosParam() {
  ros_namespace = rosParamNameSpace();
}

SpatialTrackingVisualizer::SpatialTrackingVisualizer(const Config& config)
    : config_(config.checkValid()) {
  // Print config after setting up the modes.
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  // Setup nodehandle.
  nh_ = ros::NodeHandle(config_.ros_namespace);
  rawcolor_pub_ = nh_.advertise<sensor_msgs::Image>("rawcolor", 100);
  inputid_pub_ = nh_.advertise<sensor_msgs::Image>("inputid", 100);
  publishers_.emplace("rawcolor", rawcolor_pub_);
  publishers_.emplace("inputid", inputid_pub_);
}

void SpatialTrackingVisualizer::registerIDTracker(IDTrackerBase* tracker) {
  if (config_.visualize_tracking) {
    CHECK_NOTNULL(tracker);
    tracker->setVisualizationCallback(
        [this](const cv::Mat& image, const std::string& name) {
          publishImage(image, name);
        });
    tracker->setptCloudVisualizationCallback(
        [this](const pcl::PointCloud<pcl::PointXYZRGB>& ptcloud_rbg,
               const std::string& name) { publishptCloud(ptcloud_rbg, name); });
  }
}

void SpatialTrackingVisualizer::publishImage(const cv::Mat& image,
                                             const std::string& name) {
  std::cout << "will publish rgb image" << std::endl;
  auto it = publishers_.find(name);
  if (it == publishers_.end()) {
    // Advertise a new topic if there is no publisher for the given name.
    it = publishers_.emplace(name, nh_.advertise<sensor_msgs::Image>(name, 1))
             .first;
  }

  // Publish the image, expected as BGR8.
  std_msgs::Header header;
  header.stamp = ros::Time::now();
  it->second.publish(
      cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, image)
          .toImageMsg());
}

void SpatialTrackingVisualizer::publishptCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>& ptcloud_rbg,
    const std::string& name) {
  auto it = publishers_.find(name);
  if (it == publishers_.end()) {
    // Advertise a new topic if there is no publisher for the given name.
    it = publishers_
             .emplace(name, nh_.advertise<sensor_msgs::PointCloud2>(name, 100))
             .first;
  }

  sensor_msgs::PointCloud2 ros_cloud;
  pcl::toROSMsg(ptcloud_rbg, ros_cloud);
  ros_cloud.header.frame_id = "world";
  ros_cloud.header.stamp = ros::Time::now();

  it->second.publish(ros_cloud);
}

}  // namespace panoptic_mapping
