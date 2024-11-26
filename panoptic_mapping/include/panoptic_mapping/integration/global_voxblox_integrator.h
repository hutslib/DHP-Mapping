/*
 * @Author: thuaj@connect.ust.hk
 * @Date: 2023-03-30 20:47:32
 * @LastEditTime: 2024-11-26 21:24:13
 * @Description: DHP-Mapping integration module.
 * Copyright (c) 2023 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#ifndef PANOPTIC_MAPPING_INTEGRATION_GLOBAL_VOXBLOX_INTEGRATOR_H_
#define PANOPTIC_MAPPING_INTEGRATION_GLOBAL_VOXBLOX_INTEGRATOR_H_

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/common.h"
#include <voxblox/integrator/integrator_utils.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/utils/color_maps.h>

#include "panoptic_mapping/common/globals.h"
#include "panoptic_mapping/map/classification/class_voxel.h"
#include "panoptic_mapping/map/classification/uncertainty.h"
#include "panoptic_mapping/map/classification/variable_count.h"
#include "panoptic_mapping/tools/text_colors.h"

namespace panoptic_mapping {

/**
 * @brief Integrator that integrates all data into a global submap.Combine this
 * module with the raytracing tsdf integrator.
 */

class GlobalVoxbloxIntegrator : public voxblox::MergedTsdfIntegrator {
 public:
  struct Config : public config_utilities::Config<Config> {
    voxblox::MergedTsdfIntegrator::Config tsdf_config;
    FloatingPoint min_ray_length_m = 0.1f;
    FloatingPoint max_ray_length_m = 15.f;
    int verbosity = 4;
    // Distance-based log-normal distribution of label confidence weights.
    bool enable_confidence_weight_dropoff = false;
    float lognormal_weight_mean = 0.0f;
    float lognormal_weight_sigma = 1.8f;
    float lognormal_weight_offset = 0.7f;
    bool enable_anti_grazing = false;
    int integrator_threads = std::thread::hardware_concurrency();
    float truncation_distance = -2;
    float label_truncation_distance = -2;
    bool skip_update_label_faraway_surface = true;
    bool voxel_carving_enabled = false;
    bool use_detectron = false;
    float instance_weight = 1.0f;
    float background_weight = 0.5f;
    int submap_color_discretization;
    Config() { setConfigName("GlobalVoxbloxIntegrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  GlobalVoxbloxIntegrator(const Config& tsdf_config, Submap* map);
  ~GlobalVoxbloxIntegrator() = default;
  void processIntegration(const Transformation& T_G_C,
                          const Pointcloud& points_C, const Colors& colors,
                          const Labels& labels, const bool freespace_points);

  ClassVoxel* allocateStorageAndGetClassVoxelPtr(
      const GlobalIndex& global_voxel_idx);
  void integrateRays(
      const Transformation& T_G_C, const Pointcloud& points_C,
      const Colors& colors, const Labels& labels, bool enable_anti_grazing,
      bool clearing_ray,
      const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type&
          voxel_map,
      const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type&
          clear_map);
  void integrateVoxels(
      const Transformation& T_G_C, const Pointcloud& points_C,
      const Colors& colors, const Labels& labels, bool enable_anti_grazing,
      bool clearing_ray,
      const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type&
          voxel_map,
      const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type&
          clear_map,
      size_t thread_idx);
  void integrateVoxel(
      const Transformation& T_G_C, const Pointcloud& points_C,
      const Colors& colors, const Labels& labels, bool enable_anti_grazing,
      bool clearing_ray,
      const std::pair<GlobalIndex, AlignedVector<size_t>>& kv,
      const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type&
          voxel_map);

 private:
  const Config config_;
  Submap* map_;
  std::mutex temp_class_block_mutex_;
  voxblox::ExponentialOffsetIdColorMap id_color_map_;
  int unknow_class_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATION_GLOBAL_VOXBLOX_INTEGRATOR_H_
