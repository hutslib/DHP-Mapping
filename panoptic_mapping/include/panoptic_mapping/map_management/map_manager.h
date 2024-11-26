/*
 * @Author: thuaj@connect.ust.hk
 * @Description: DHP-Mapping map manage module
 * (inter-submap label management and multi-variable crf)
 * Copyright (c) 2024 by thuaj@connect.ust.hk, All Rights Reserved.
 */

#ifndef PANOPTIC_MAPPING_MAP_MANAGEMENT_MAP_MANAGER_H_
#define PANOPTIC_MAPPING_MAP_MANAGEMENT_MAP_MANAGER_H_

#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <voxblox/utils/color_maps.h>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/map/submap.h"
#include "panoptic_mapping/map/submap_collection.h"
#include "panoptic_mapping/map_management/activity_manager.h"
#include "panoptic_mapping/map_management/layer_manipulator.h"
#include "panoptic_mapping/map_management/map_manager_base.h"
#include "panoptic_mapping/map_management/tsdf_registrator.h"

#include "densecrf.h"

namespace panoptic_mapping {

/**
 * @brief High level class that wraps all map management actions and tools.
 */
class MapManager : public MapManagerBase {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    // Perform actions every n ticks (frames), set 0 to turn off.
    int prune_active_blocks_frequency = 0;
    int change_detection_frequency = 0;
    int activity_management_frequency = 0;
    int voxel_class_management_frequency = 0;
    bool excute_crf = true;
    int crf_frequency = 0;

    // If true, submaps that are deactivated are checked for alignment with
    // inactive maps and merged together if a match is found.
    bool merge_deactivated_submaps_if_possible = false;

    // If true, the class layer will be integrated into the TSDF layer and
    // discraded afterwards when submaps are deactivated. This saves memory at
    // the loss of classification information.
    bool apply_class_layer_when_deactivating_submaps = false;

    // Member configs.
    TsdfRegistrator::Config tsdf_registrator_config;
    ActivityManager::Config activity_manager_config;
    LayerManipulator::Config layer_manipulator_config;

    // crf related config
    int semantic_size = 0;
    int crf_iterations = 5;
    bool use_high_order = false;
    float smooth_xy_stddev = 3;
    float smooth_z_stddev = 3;
    float smooth_weight = 8;
    float appear_xy_stddev = 160;
    float appear_z_stddev = 40;
    float appear_rgb_stddev = 4;
    float appear_weight = 10;
    bool exit_after_management = false;

    bool use_detectron = false;
    bool use_kitti = false;

    bool crfmultithread = true;
    int threads = std::thread::hardware_concurrency();

    std::string submap_info_path;

    int submap_color_discretization;

    Config() { setConfigName("MapManager"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  explicit MapManager(const Config& config);
  virtual ~MapManager() = default;

  // Perform all actions when with specified timings.
  void tick(SubmapCollection* submaps) override;
  void finishMapping(SubmapCollection* submaps) override;

  // Perform specific tasks.
  void pruneActiveBlocks(SubmapCollection* submaps);
  void manageSubmapActivity(SubmapCollection* submaps);
  void manageVoxelClass(SubmapCollection* submaps);
  void performChangeDetection(SubmapCollection* submaps);

  // Tools.
  bool mergeSubmapIfPossible(SubmapCollection* submaps, int submap_id,
                             int* merged_id = nullptr);
  void mergeSubmapVoxelInfo(std::set<int>& frozen_submaps_id,
                            SubmapCollection* submaps);

  void setBelongingsMultiThread(Submap* frozen_submap,
                                SubmapCollection* submaps);

  void processBlock(Submap* frozen_submap, SubmapCollection* submaps,
                    size_t start_block, size_t end_block,
                    const voxblox::BlockIndexList& block_indices,
                    std::atomic<int>& change_count);

  void processVoxels(size_t start_voxel, size_t end_voxel,
                     Submap* frozen_submap, TsdfBlock::Ptr tsdf_block,
                     ClassBlock::Ptr class_block, SubmapCollection* submaps,
                     std::atomic<int>& change_count);


  void processCrfMultiThread(std::set<int>& frozen_submaps_id,
                             SubmapCollection* submaps);


  void setCrfProcessBlock(Submap* frozen_submap, SubmapCollection* submaps,
                          const voxblox::BlockIndexList& block_indices,
                          const int start_block, const int end_block,
                          std::vector<GlobalIndex>& voxel_indices_in_order,
                          const std::set<int>& frozen_submaps_id,
                          const std::set<int>& frozen_submaps_semantic_id);

  void setCrfProcessVoxel(Submap* frozen_submap, TsdfBlock::Ptr tsdf_block,
                          ClassBlock::Ptr class_block, const int start_voxel,
                          const int end_voxel,
                          std::vector<GlobalIndex>& voxel_indices_in_order,
                          const std::set<int>& frozen_submaps_id,
                          const std::set<int>& frozen_submaps_semantic_id,
                          BlockIndex& block_index);

  void assignCrfProEachGlobalIndex(
      const std::vector<GlobalIndex>& global_voxel_index_vector,
      int frozen_submap_id, const std::set<int>& frozen_submaps_id,
      const std::set<int>& frozen_submaps_semantic_id, Submap* frozen_submap,
      const Eigen::MatrixXf& instance_crf_output_prob,
      const Eigen::MatrixXf& semantic_crf_output_prob,
      const Eigen::VectorXi& instance_results,
      const Eigen::VectorXi& semantic_results, int start_index, int end_index);
  void computeCorners(Point corners[8], Point& origin,
                                     FloatingPoint block_size) ;


 protected:
  std::string pruneBlocks(Submap* submap) const;

 private:
  static config_utilities::Factory::RegistrationRos<MapManagerBase, MapManager>
      registration_;
  // Members.
  const Config config_;

  std::shared_ptr<ActivityManager> activity_manager_;
  std::shared_ptr<TsdfRegistrator> tsdf_registrator_;
  std::shared_ptr<LayerManipulator> layer_manipulator_;

  std::unordered_set<int> last_inview_submaps_;
  std::set<int> accumulate_frozen_submap_;
  std::set<int> accumulate_deactivated_submap_;

  voxblox::ExponentialOffsetIdColorMap id_color_map_;

  // crf multi thread mutex
  std::mutex crf_cur_voxel_idx_mutex_;
  std::mutex crf_mat_mutex_;
  std::mutex crf_voxel_num_actual_mutex_;
  std::mutex crf_voxel_indices_in_order_mutex_;
  std::mutex crf_instance_change_voxel_num_mutex_;
  std::mutex crf_instance_label_voxel_change_record_mutex_;
  std::mutex crf_semantic_label_voxel_change_record_mutex_;
  std::mutex crf_semantic_change_voxel_num_mutex_;
  // crf variable and matrix
  MatrixXf_row semantic_unary_mat_;
  MatrixXf_row instance_unary_mat_;
  MatrixXf_row rgb_mat_;
  MatrixXf_row pose_mat_;
  int cur_voxel_idx_ = 0;
  std::map<int, std::vector<GlobalIndex>> voxel_index_map_;
  int crf_voxel_num_actual_ = 0;
  int crf_instance_num_;  // N: instance label size
  int crf_semantic_num_;  // M: semantic label size
  // assign results
  int voxel_instance_label_changed_num_ = 0;
  int voxel_semantic_label_changed_num_ = 0;
  int instance_change_voxel_num_ = 0;
  int semantic_change_voxel_num_ = 0;
  int cur_map_pro_index_count_ = 0;
  // crf const variable (we only read it so d not need mutex for them)

  std::map<int, std::map<int, int>>
      instance_label_voxel_change_record_;  // <origin,<new, count>>
  std::map<int, std::map<int, int>>
      semantic_label_voxel_change_record_;  // <origin,<new, count>>
  // TODO(thuaj): wthether need this ??
  int submap_index_ = 0;
  // Action tick counters.
  //
  int crf_frame_counter_ = 0;
  bool crf_at_the_end_ = false;
  class Ticker {
   public:
    Ticker(unsigned int max_ticks,
           std::function<void(SubmapCollection* submaps)> action)
        : max_ticks_(max_ticks), action_(std::move(action)) {}
    void tick(SubmapCollection* submaps);

   private:
    unsigned int current_tick_ = 0;
    const unsigned int max_ticks_;
    const std::function<void(SubmapCollection* submaps)> action_;
  };
  std::vector<Ticker> tickers_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_MAP_MANAGEMENT_MAP_MANAGER_H_
