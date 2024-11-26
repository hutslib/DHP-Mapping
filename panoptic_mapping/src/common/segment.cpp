#include "panoptic_mapping/common/segment.h"

namespace panoptic_mapping {

Segment::Segment(int submap_id, int semantic_id, int segmentation_id, Submap* submap_ptr,
                 Transformation T_G_C)
    : submap_id_(submap_id),
      semantic_id_(semantic_id),
      segmentation_id_(segmentation_id),
      submap_ptr_(submap_ptr),
      T_G_C_(T_G_C){};

void Segment::Pushback(Point pt, Color color, Label label) {
  points_.push_back(pt);
  colors_.push_back(color);
  labels_.push_back(label);
}

}  // namespace panoptic_mapping
