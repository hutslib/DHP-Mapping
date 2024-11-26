#include "panoptic_mapping_ros/conversions/conversions.h"

namespace panoptic_mapping {

DetectronLabel detectronLabelFromMsg(
    const panoptic_mapping_msgs::DetectronLabel& msg) {
  DetectronLabel result;
  result.id = msg.id;
  result.instance_id = msg.instance_id;
  result.is_thing = msg.is_thing;
  result.category_id = msg.category_id;
  result.score = msg.score;
  return result;
}

DetectronLabels detectronLabelsFromMsg(
    const panoptic_mapping_msgs::DetectronLabels& msg) {
  DetectronLabels result;
  for (const panoptic_mapping_msgs::DetectronLabel& label : msg.labels) {
    result[label.id] = detectronLabelFromMsg(label);
  }
  return result;
}

KittiLabel kittiLabelFromMsg(const panoptic_mapping_msgs::KittiLabel& msg) {
  KittiLabel result(msg.full_id);
  return result;
}

KittiLabels kittiLabelsFromMsg(const panoptic_mapping_msgs::KittiLabels& msg) {
  KittiLabels result;
  for (const panoptic_mapping_msgs::KittiLabel& label : msg.labels) {
    result.push_back(kittiLabelFromMsg(label));
  }
  return result;
}

}  // namespace panoptic_mapping
