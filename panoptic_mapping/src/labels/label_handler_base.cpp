#include "panoptic_mapping/labels/label_handler_base.h"

#include <string>
#include <fstream>

namespace panoptic_mapping {

bool LabelHandlerBase::segmentationIdExists(int segmentation_id) const {
  return labels_.find(segmentation_id) != labels_.end();
}

bool LabelHandlerBase::classIdExists(int class_id) const {
  for (const auto& label : labels_) {
    if (label.second->class_id == class_id) {
      return true;
    }
  }
  return false;
}

int LabelHandlerBase::getClassID(int segmentation_id) const {
  return labels_.at(segmentation_id)->class_id;
}

bool LabelHandlerBase::isBackgroundClass(int segmentation_id) const {
  return labels_.at(segmentation_id)->label == PanopticLabel::kBackground;
}

bool LabelHandlerBase::isInstanceClass(int segmentation_id) const {
  return labels_.at(segmentation_id)->label == PanopticLabel::kInstance;
}

bool LabelHandlerBase::isUnknownClass(int segmentation_id) const {
  return labels_.at(segmentation_id)->label == PanopticLabel::kUnknown;
}

bool LabelHandlerBase::isSpaceClass(int segmentation_id) const {
  return labels_.at(segmentation_id)->label == PanopticLabel::kFreeSpace;
}

PanopticLabel LabelHandlerBase::getPanopticLabel(int segmentation_id) const {
  return labels_.at(segmentation_id)->label;
}

const voxblox::Color& LabelHandlerBase::getColor(int segmentation_id) const {
  return labels_.at(segmentation_id)->color;
}

const voxblox::Color& LabelHandlerBase::getColorbyClass(int class_id) const {
  for (const auto& label : labels_) {
    if (label.second->class_id == class_id) {
      return label.second->color;
    }
  }
}

const std::string& LabelHandlerBase::getName(int segmentation_id) const {
  return labels_.at(segmentation_id)->name;
}

const LabelEntry& LabelHandlerBase::getLabelEntry(int segmentation_id) const {
  return *labels_.at(segmentation_id);
}

const LabelEntry& LabelHandlerBase::getLabelEntrybyClass(int class_id) const {
  for (const auto& label : labels_) {
    if (label.second->class_id == class_id) {
      return *label.second;
    }
  }
}

bool LabelHandlerBase::getLabelEntryIfExists(int segmentation_id,
                                             LabelEntry* label_entry) const {
  auto it = labels_.find(segmentation_id);
  if (it != labels_.end()) {
    CHECK_NOTNULL(label_entry);
    *label_entry = *it->second;
    return true;
  }
  return false;
}

size_t LabelHandlerBase::numberOfLabels() const { return labels_.size(); }

void LabelHandlerBase::printLabels(const std::string& filename) const {
  for (const auto& pair : labels_) {
    std::cout << "label : " << pair.first
              << "with label info: " << pair.second->toString() << std::endl;
  }
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    std::cerr << "Error: Failed to open file " << filename << std::endl;
    return;
  }

  for (const auto& pair : labels_) {
    outfile << "label : " << pair.first << std::endl;
    outfile << "with Label Info: " << pair.second->toString() << std::endl;
  }

  outfile.close();
}

}  // namespace panoptic_mapping
