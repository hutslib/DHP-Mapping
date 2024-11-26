#include "panoptic_mapping/map/classification/variable_count.h"

#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace panoptic_mapping {

ClassVoxelType VariableCountVoxel::getVoxelType() const {
  return ClassVoxelType::kVariableCount;
}

bool VariableCountVoxel::isObserverd() const { return !counts.empty(); }

bool VariableCountVoxel::belongsToSubmap() const {
  // In doubt we count the voxel as belonging. This also applies for unobserved
  // voxels.
  return current_index == 0;
}

float VariableCountVoxel::getBelongingProbability() const {
  return getProbability(0);
}

void VariableCountVoxel::getProbabilityCRFList(const std::set<int> id_list,
                                               VectorXf_1col* probability) {
  int i = 0;
  for (auto element : id_list) {
    float p_i = getProbability(element);
    (*probability)(i) = p_i;
    i++;
  }
}

void VariableCountVoxel::getSemanticCRFList(const std::set<int> class_list,
                                            VectorXf_1col* probability) {
  int i = 0;
  for (auto element : class_list) {
    float p_i = getSemanticProbability(element);
    (*probability)(i) = p_i;
    i++;
  }
}

void VariableCountVoxel::getSemanticCRFList(int class_size,
                                            VectorXf_1col* probability,
                                            bool use_detectron) {
  if (use_detectron) {
    for (int i = 2; i < class_size + 2; ++i) {
      float p_i = getSemanticProbability(i);
      (*probability)(i - 2) = p_i;
    }
  } else {
    for (int i = 0; i < class_size; ++i) {
      float p_i = getSemanticProbability(i);
      (*probability)(i) = p_i;
    }
  }
}

int VariableCountVoxel::getBelongingID() const { return current_index; }

float VariableCountVoxel::getProbability(const int id) const {
  if (counts.empty()) {
    return 0.f;
  }
  auto it = counts.find(id);
  if (it == counts.end()) {
    return 0.f;
  }
  return static_cast<float>(it->second) / static_cast<float>(total_count);
}

float VariableCountVoxel::getSemanticProbability(const int id) const {
  if (semantic_records.empty()) {
    return 0.f;
  }
  auto it = semantic_records.find(id);
  if (it == semantic_records.end()) {
    return 0.f;
  }
  return static_cast<float>(it->second) /
         static_cast<float>(total_semantic_count);
}

void VariableCountVoxel::incrementCount(const int id, const float weight) {
  counts[id] += weight;
  const FloatingPoint new_count = counts[id];
  if (new_count > current_count) {
    if (id != 0) {
      std::cout << "id 0 count: " << counts[0];
      std::cout << " new id : " << id << " new_count " << new_count
                << " origin index: " << current_index << " origin_count"
                << current_count << std::endl;
      exit(0);
    }
    current_index = id;
    current_count = new_count;
  }

  // since we will add the count twice for itself belong count id =0 ++ and id = submapid ++
  if (id != 0) {
    total_count += weight;
  }

  if (total_count > 65000) {
    LOG(ERROR) << "too large count";
  }
}

void VariableCountVoxel::incrementSemanticCount(const int id,
                                                const float weight) {

  if (id == -1) {
    LOG(WARNING) << "set semantic label -1";
  }
  semantic_records[id] += weight;
  const FloatingPoint new_semantic_count = semantic_records[id];
  if (new_semantic_count > current_semantic_count) {
    semantic_class = id;
    current_semantic_count = new_semantic_count;
  }
  total_semantic_count += weight;

}

bool VariableCountVoxel::mergeVoxel(const ClassVoxel& other) {
  // Check type compatibility.
  auto voxel = dynamic_cast<const VariableCountVoxel*>(&other);
  if (!voxel) {
    LOG(WARNING) << "Can not merge voxels that are not of same type "
                    "(VariableCountVoxel).";
    return false;
  }
  // No averaging is performed here. This inflates the number of total counts
  // but keeps the accuracy higher.
  for (const auto& id_count_pair : voxel->counts) {
    if (id_count_pair.first == 0) {
      // skip the id 0 since 0 is reserved for each submap itself
      continue;
    }
    // for debug output
    bool verbosity = false;
    if (verbosity) {
      std::cout << "will merge class voxel id: " << id_count_pair.first
                << " count: " << id_count_pair.second << std::endl;
    }
    total_count += id_count_pair.second;
    counts[id_count_pair.first] += id_count_pair.second;
    if (counts[id_count_pair.first] > current_count) {
      current_count = counts[id_count_pair.first];
      current_index = id_count_pair.first;
      // LOG(WARNING) << "when merge change index to: " << current_index
      // <<std::endl;
    }
    // NOTE(thuaj): we need to check count do not to large
    // TODO(thuaj): need further implement to check floating value
    if (current_count > 65000) {
      LOG(WARNING) << "aft merge too large current count";
    }
  }

  for (const auto& id_semantic_pair : voxel->semantic_records) {
    if (id_semantic_pair.first == -1) {
      LOG(ERROR) << "we find error semantic records";
      continue;
    }
    // for debug output
    bool verbosity = false;
    if (verbosity) {
      std::cout << "will merge class voxel id: " << id_semantic_pair.first
                << " count: " << id_semantic_pair.second << std::endl;
    }
    total_semantic_count += id_semantic_pair.second;
    semantic_records[id_semantic_pair.first] += id_semantic_pair.second;
    if (semantic_records[id_semantic_pair.first] > current_semantic_count) {
      current_semantic_count = semantic_records[id_semantic_pair.first];
      semantic_class = id_semantic_pair.first;
      // LOG(WARNING) << "when merge change semantic class to: " <<
      // current_index << std::endl;
    }

    // NOTE(thuaj): we need to check count do not to large
    // TODO(thuaj): need further implement to check floating value
    if (current_semantic_count > 65000) {
      LOG(WARNING) << "aft merge too large current count";
    }
  }
  return true;
}

std::string VariableCountVoxel::printCount() const {
  std::stringstream ss;
  for (const auto& pair : counts) {
    ss << "submap instance id: " << pair.first << ", count: " << pair.second
       << std::endl;
  }
  ss << "current_index: " << current_index << std::endl;
  ss << "current_count: " << std::setprecision(20) << current_count
     << std::endl;
  ss << "total_count: " << std::setprecision(20) << total_count << std::endl;
  ss << "belonging id: " << getBelongingID() << std::endl;
  ss << "belong to this submap: " << belongsToSubmap() << std::endl;

  return ss.str();
}

int VariableCountVoxel::getClassId() const { return semantic_class; }

std::string VariableCountVoxel::printSemantic() const {
  std::stringstream ss;
  for (const auto& pair : semantic_records) {
    ss << " submap semantic class id: " << pair.first
       << ", count: " << pair.second << std::endl;
  }
  ss << "semantic_class: " << semantic_class << std::endl;
  ss << "current_semantic_count: " << std::setprecision(20)
     << current_semantic_count << std::endl;
  ss << "total_semantic_count: " << std::setprecision(20)
     << total_semantic_count << std::endl;
  ss << "belonging class id: " << getClassId() << std::endl;

  return ss.str();
}

std::vector<uint32_t> VariableCountVoxel::serializeVoxelToInt() const {
  // To save memory, here we just assume that the IDs stored in the map are in
  // int_16 range (-32k:32k).
  std::vector<uint32_t> result(counts.size() + 1);
  result[0] = counts.size();

  // Store all counts as id-value pair.
  size_t index = 0;
  for (const auto& id_count_pair : counts) {
    index++;
    if (id_count_pair.first < std::numeric_limits<int16_t>::lowest() ||
        id_count_pair.first > std::numeric_limits<int16_t>::max()) {
      LOG(WARNING) << "ID: '" << id_count_pair.first
                   << "' is out of Int16 range and will be ignored.";
      continue;
    }
    result[index] = int32FromTwoInt16(
        static_cast<uint16_t>(id_count_pair.first), id_count_pair.second);
  }
  return result;
}

bool VariableCountVoxel::deseriliazeVoxelFromInt(
    const std::vector<uint32_t>& data, size_t* data_index) {
  if (*data_index >= data.size()) {
    LOG(WARNING)
        << "Can not deserialize voxel from integer data: Out of range (index: "
        << *data_index << ", data: " << data.size() << ")";
    return false;
  }

  // Check number of counts to load.
  const size_t length = data[*data_index] + 1;
  if (*data_index + length > data.size()) {
    LOG(WARNING) << "Can not deserialize voxel from integer data: Not enough "
                    "data (index: "
                 << *data_index << "-" << (*data_index + length)
                 << ", data: " << data.size() << ")";
    return false;
  }

  // Get the data.
  counts.clear();
  total_count = 0;
  current_index = -1;
  current_count = 0;
  for (size_t i = 1; i < length; ++i) {
    std::pair<uint16_t, uint16_t> datum =
        twoInt16FromInt32(data[*data_index + i]);
    counts[static_cast<int16_t>(datum.first)] = datum.second;
    total_count += datum.second;
    if (datum.second > current_count) {
      current_count = datum.second;
      current_index = datum.first;
    }
  }
  *data_index += length;
  return true;
}

config_utilities::Factory::RegistrationRos<ClassLayer, VariableCountLayer,
                                           float, int>
    VariableCountLayer::registration_("variable_count");

VariableCountLayer::VariableCountLayer(const Config& config,
                                       const float voxel_size,
                                       const int voxels_per_side)
    : config_(config.checkValid()),
      ClassLayerImpl(voxel_size, voxels_per_side) {}

ClassVoxelType VariableCountLayer::getVoxelType() const {
  return ClassVoxelType::kVariableCount;
}

std::unique_ptr<ClassLayer> VariableCountLayer::clone() const {
  return std::make_unique<VariableCountLayer>(*this);
}

std::unique_ptr<ClassLayer> VariableCountLayer::loadFromStream(
    const SubmapProto& submap_proto, std::istream* /* proto_file_ptr */,
    uint64_t* /* tmp_byte_offset_ptr */) {
  // Nothing special needed to configure for binary counts.
  return std::make_unique<VariableCountLayer>(VariableCountLayer::Config(),
                                              submap_proto.voxel_size(),
                                              submap_proto.voxels_per_side());
}

}  // namespace panoptic_mapping
