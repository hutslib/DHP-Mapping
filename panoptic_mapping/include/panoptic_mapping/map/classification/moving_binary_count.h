#ifndef PANOPTIC_MAPPING_MAP_CLASSIFICATION_MOVING_BINARY_COUNT_H_
#define PANOPTIC_MAPPING_MAP_CLASSIFICATION_MOVING_BINARY_COUNT_H_

#include <memory>
#include <vector>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/Submap.pb.h"
#include "panoptic_mapping/map/classification/class_layer_impl.h"
#include "panoptic_mapping/map/classification/class_voxel.h"

namespace panoptic_mapping {

/**
 * @brief Binary classification by simple counting, where ID 0 indicates the
 * voxel belongs. Uses a reduced datatype to save memory, where older counts are
 * de-weighted to prevent overflow.
 */
struct MovingBinaryCountVoxel : public ClassVoxel {
 public:
  // Implement interfaces.
  ClassVoxelType getVoxelType() const override;
  bool isObserverd() const override;
  bool belongsToSubmap() const override;
  float getBelongingProbability() const override;
  int getBelongingID() const override;
  float getProbability(const int id) const override;
  void incrementCount(const int id, const float weight = 1.f) override;
  void incrementSemanticCount(const int id,
                              const float weight = 1.f) override{};
  bool mergeVoxel(const ClassVoxel& other) override;
  std::vector<uint32_t> serializeVoxelToInt() const override;
  bool deseriliazeVoxelFromInt(const std::vector<uint32_t>& data,
                               size_t* data_index) override;
  void getProbabilityCRFList(const std::set<int> id_list,
                             VectorXf_1col* probability) override{};
  void getSemanticCRFList(int class_size, VectorXf_1col* probability,
                          bool use_detectron) override{};
  void getSemanticCRFList(const std::set<int> class_list,
                          VectorXf_1col* probability) override {}
  float getTotalCount() override { return 0; }
  float getSemanticTotalCount() override {}
  void setAfterMerge(float set_new_count) override{};
  float getCurrentCount() override { return 0; }
  float getSemanticCurrentCount() override {}
  void setCount(int id, float set_new_count) override {}
  void setSemanticCount(int id, float set_new_count) override {}
  void setAfterCrf(int new_current_id, float new_current_count,
                   float new_total_count) override {}
  void setSemanticAfterCrf(int new_semantic_class,
                           float new_current_semantic_count,
                           float new_total_semantic_count) override {}
  // void setCurrentCount(ClassificationCount new_current_count) override {}
  // void setNewID(int id, ClassificationCount new_current_count,
  //               ClassificationCount new_total_count) override {}
  void clearCount() override {}
  int getClassId() const override{};
  std::vector<int> getSemanticIDVec() override {
    std::vector<int> a;
    return a;
  }
  std::string printCount() const override {}
  std::string printSemantic() const override {}
  // Data.
  uint8_t belongs_count = 0u;
  uint8_t foreign_count = 0u;
};

class MovingBinaryCountLayer : public ClassLayerImpl<MovingBinaryCountVoxel> {
 public:
  struct Config : public config_utilities::Config<Config> {
    Config() { setConfigName("MovingBinaryCountLayer"); }

   protected:
    void fromRosParam() override {}
    void printFields() const override {}
  };

  MovingBinaryCountLayer(const Config& config, const float voxel_size,
                         const int voxels_per_side);

  // Overwrite these method since we only need half a word per voxel.
  bool saveBlockToStream(BlockIndex block_index,
                         std::fstream* outfile_ptr) const override;
  bool addBlockFromProto(const voxblox::BlockProto& block_proto) override;

  ClassVoxelType getVoxelType() const override;
  std::unique_ptr<ClassLayer> clone() const override;
  static std::unique_ptr<ClassLayer> loadFromStream(
      const SubmapProto& submap_proto, std::istream* /* proto_file_ptr */,
      uint64_t* /* tmp_byte_offset_ptr */);

 protected:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      ClassLayer, MovingBinaryCountLayer, float, int>
      registration_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_MAP_CLASSIFICATION_MOVING_BINARY_COUNT_H_
