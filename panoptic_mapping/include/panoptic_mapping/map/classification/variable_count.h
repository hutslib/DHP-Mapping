#ifndef PANOPTIC_MAPPING_MAP_CLASSIFICATION_VARIABLE_COUNT_H_
#define PANOPTIC_MAPPING_MAP_CLASSIFICATION_VARIABLE_COUNT_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/Submap.pb.h"
#include "panoptic_mapping/map/classification/class_layer_impl.h"
#include "panoptic_mapping/map/classification/class_voxel.h"

namespace panoptic_mapping {

/**
 * @brief Keep track of arbitrary number of IDs in an unordered map. ID 0 is
 * generally used to store the belonging submap and shifting other IDs by 1.
 */
struct VariableCountVoxel : public ClassVoxel {
 public:
  // Implement interfaces.
  ClassVoxelType getVoxelType() const override;
  bool isObserverd() const override;
  bool belongsToSubmap() const override;
  float getBelongingProbability() const override;
  int getBelongingID() const override;
  float getProbability(const int id) const override;
  float getSemanticProbability(const int id) const;
  void incrementCount(const int id, const float weight = 1.f) override;
  void incrementSemanticCount(const int id, const float weight = 1.f) override;
  bool mergeVoxel(const ClassVoxel& other) override;
  std::vector<uint32_t> serializeVoxelToInt() const override;
  bool deseriliazeVoxelFromInt(const std::vector<uint32_t>& data,
                               size_t* data_index) override;
  void getProbabilityCRFList(const std::set<int> id_list,
                             VectorXf_1col* probability) override;
  void getSemanticCRFList(int class_size, VectorXf_1col* probability,
                          bool use_detectron) override;
  void getSemanticCRFList(const std::set<int> class_list,
                          VectorXf_1col* probability) override;
  float getTotalCount() override { return total_count; }
  float getSemanticTotalCount() override { return total_semantic_count; }
  float getCurrentCount() override { return current_count; }
  float getSemanticCurrentCount() override { return current_semantic_count; }
  void setAfterMerge(float set_new_count) override {
    current_index = 0;
    counts[0] = set_new_count;
  }
  void setCount(int id, float set_new_count) override {
    counts[id] = set_new_count;
  }

  void setSemanticCount(int id, float set_new_count) override {
    semantic_records[id] = set_new_count;
  }

  void setAfterCrf(int new_current_id, float new_current_count,
                   float new_total_count) override {
    current_index = new_current_id;
    current_count = new_current_count;
    total_count = new_total_count;
    if (new_current_id == 0) {
      counts[0] = current_count;
    }
  }
  void setSemanticAfterCrf(int new_semantic_class,
                           float new_current_semantic_count,
                           float new_total_semantic_count) override {
    semantic_class = new_semantic_class;
    current_semantic_count = new_current_semantic_count;
    total_semantic_count = new_total_semantic_count;
  }

  void clearCount() override {
    counts.clear();
    semantic_records.clear();
    current_index = 0;
    semantic_class = -1;
    current_count = 0;
    total_count = 0;
    current_semantic_count = 0;
    total_semantic_count = 0;
  }
  int getClassId() const override;
  std::vector<int> getSemanticIDVec() override {
    std::vector<int> semantic_id_vec;
    for (auto const& record : semantic_records) {
      semantic_id_vec.push_back(record.first);
    }
    return semantic_id_vec;
  }
  std::string printCount() const override;
  std::string printSemantic() const override;
  // Data.
  std::unordered_map<int, FloatingPoint> counts;
  std::unordered_map<int, FloatingPoint> semantic_records;
  int current_index = 0;
  int semantic_class = -1;  // -1 for not set
  FloatingPoint current_count = 0;
  FloatingPoint total_count = 0;
  FloatingPoint current_semantic_count = 0;
  FloatingPoint total_semantic_count = 0;
};

class VariableCountLayer : public ClassLayerImpl<VariableCountVoxel> {
 public:
  struct Config : public config_utilities::Config<Config> {
    Config() { setConfigName("VariableCountLayer"); }

   protected:
    void fromRosParam() override {}
    void printFields() const override {}
  };

  VariableCountLayer(const Config& config, const float voxel_size,
                     const int voxels_per_side);

  ClassVoxelType getVoxelType() const override;
  std::unique_ptr<ClassLayer> clone() const override;
  static std::unique_ptr<ClassLayer> loadFromStream(
      const SubmapProto& submap_proto, std::istream* /* proto_file_ptr */,
      uint64_t* /* tmp_byte_offset_ptr */);

 protected:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      ClassLayer, VariableCountLayer, float, int>
      registration_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_MAP_CLASSIFICATION_VARIABLE_COUNT_H_
