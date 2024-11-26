/*
 * @Author: thuaj@connect.ust.hk
 * @Description: DHP-Mapping segment class.
 * Copyright (c) 2023 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#ifndef PANOPTIC_MAPPING_COMMON_SEGMENT_H_
#define PANOPTIC_MAPPING_COMMON_SEGMENT_H_

// #include <pcl/point_cloud.h>
#include <vector>
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/map/submap.h"

namespace panoptic_mapping {

class Segment {
 public:
  Segment(int submap_id, int semantic_id, int segmentation_id, Submap* submap_ptr,
          Transformation T_G_C);
  void Pushback(Point pt, Color color, Label label);
  Submap* getSubmapPtr() { return submap_ptr_; }
  Transformation& getT() { return T_G_C_; }
  Pointcloud& points() { return points_; }
  Colors& colors() { return colors_; }
  Labels& labels() { return labels_; }
  int& getSubmapID() { return submap_id_; }
  int& getClassID() { return semantic_id_; }
  int& getSegmentationID() { return segmentation_id_; }

 private:
  Pointcloud points_;
  Colors colors_;
  Labels labels_;
  int submap_id_;
  int semantic_id_;
  int segmentation_id_; //input id
  Submap* submap_ptr_;
  Transformation T_G_C_;
};
typedef AlignedVector<Segment> Segments;
}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_SEGMENT_H_