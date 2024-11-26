#ifndef PANOPTIC_MAPPING_ROS_CONVERSIONS_CONVERSIONS_H_
#define PANOPTIC_MAPPING_ROS_CONVERSIONS_CONVERSIONS_H_

#include <panoptic_mapping/common/input_data.h>
#include <panoptic_mapping_msgs/DetectronLabel.h>
#include <panoptic_mapping_msgs/DetectronLabels.h>
#include <panoptic_mapping_msgs/KittiLabel.h>
#include <panoptic_mapping_msgs/KittiLabels.h>
namespace panoptic_mapping {

DetectronLabel detectronLabelFromMsg(
    const panoptic_mapping_msgs::DetectronLabel& msg);

DetectronLabels detectronLabelsFromMsg(
    const panoptic_mapping_msgs::DetectronLabels& msg);

KittiLabel kittiLabelFromMsg(const panoptic_mapping_msgs::KittiLabel& msg);

KittiLabels kittiLabelsFromMsg(const panoptic_mapping_msgs::KittiLabels& msg);

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_ROS_CONVERSIONS_CONVERSIONS_H_
