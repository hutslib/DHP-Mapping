#ifndef PANOPTIC_MAPPING_LABELS_LABEL_ENTRY_H_
#define PANOPTIC_MAPPING_LABELS_LABEL_ENTRY_H_

#include <map>
#include <string>
#include <unordered_map>

#include <voxblox/core/color.h>

#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/tools/text_colors.h"
namespace panoptic_mapping {

/**
 * @brief Labels are read from external sources to interpret the semantic
 meaning of the segmentation inputs. Depending on the source of panoptic input
 data, different labels might be needed. The LabelEntry is the base struct
 containing minimal information that needs to be provided to panoptic mapping.
 The default values represent an unknown label.
 */
struct LabelEntry {
  // Required fields.
  // The ID of the input data referring to this label.
  int segmentation_id = -1;

  // The ID of the semantic class.
  int class_id = -1;

  // The panoptic category of the label.
  PanopticLabel label = PanopticLabel::kUnknown;

  // Print the contents of the label.
  virtual std::string toString() const {
    std::stringstream ss;
    ss << orangeText << "SegmentationID: " << endColor << segmentation_id
       << orangeText << ", ClassID: " << endColor << class_id << orangeText
       << ", PanopticID: " << endColor << panopticLabelToString(label)
       << orangeText << ", Color: " << endColor << static_cast<int>(color.r)
       << " " << static_cast<int>(color.g) << " " << static_cast<int>(color.b)
       << orangeText << ", Name: " << endColor << name << orangeText
       << ", Size: " << endColor << size;
    return ss.str();
  }

  // Optional fields.
  // Human readable or interpretable class name.
  std::string name = "UninitializedName";

  // Associated size of (S,M,L) of the object. Used by specific submap
  // allocators.
  std::string size = "Unknown";

  // Color to be used to display this label.
  Color color = Color(80, 80, 80);
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_LABELS_LABEL_ENTRY_H_
