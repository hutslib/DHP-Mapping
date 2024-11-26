/*
 * @Author: thuaj@connect.ust.hk
 * @Date: 2023-03-30 20:47:32
 * @LastEditTime: 2024-03-08 15:45:56
 * @Description: DHP-Mapping Raycasting integrator module.
 * Copyright (c) 2023 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#ifndef PANOPTIC_MAPPING_INTEGRATION_RAYTRACING_TSDF_INTEGRATOR_H_
#define PANOPTIC_MAPPING_INTEGRATION_RAYTRACING_TSDF_INTEGRATOR_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/common.h"
#include <voxblox/integrator/integrator_utils.h>
#include <voxblox/integrator/tsdf_integrator.h>

#include "panoptic_mapping/common/globals.h"
#include "panoptic_mapping/common/segment.h"
#include "panoptic_mapping/integration/global_voxblox_integrator.h"
#include "panoptic_mapping/integration/tsdf_integrator_base.h"
#include "panoptic_mapping/map/classification/uncertainty.h"
#include "panoptic_mapping/tools/text_colors.h"

namespace panoptic_mapping {

/**
 * @brief Integrator that integrates all data into a global submap.Combine this
 * module with the SpatialTsdfIDTracker.
 */
class RaytracingTsdfIntegrator : public TsdfIntegratorBase {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;
    // If true require a color image and update voxel colors.
    bool use_color = true;

    // If true require a segmentation image and integrate it into a class layer.
    bool use_segmentation = true;

    // If true require an uncertainty image and integrate it into an class layer
    // of type 'UncertaintyLayer'.
    bool use_uncertainty = false;

    // If true require an uncertainty image and integrate it into a score layer.
    bool use_score = false;

    // Decay rate in [0, 1] used to update uncertainty voxels. Only used if
    // 'use_uncertainty' is true.
    float uncertainty_decay_rate = 0.5f;

    float min_ray_length_m = 0.1f;

    float max_ray_length_m = 5.f;

    bool allow_clear = false;

    // voxblox config
    FloatingPoint voxel_size = 0.05;
    // bool enable_anti_grazing = false;

    int integrator_threads = std::thread::hardware_concurrency();
    // int integrator_threads = 8;

    std::string integration_order_mode = "mixed";

    bool use_detectron = false;

    bool use_kitti = false;

    GlobalVoxbloxIntegrator::Config
        global_voxblox_integrator_config;  // this is the config for voxblox
                                           // integrator

    Config() { setConfigName("RaytracingTsdfIntegrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  RaytracingTsdfIntegrator(const Config& config,
                           std::shared_ptr<Globals> globals);
  ~RaytracingTsdfIntegrator() = default;
  void processInput(SubmapCollection* submaps, InputData* input) override;
  void processSubmap(Segments* segments, Submap* submap);
  void GenerateInputPointCloud(Pointcloud& points, Colors& colors,
                               Labels& labels, std::string ply_path);
  bool create_directory_if_not_exists(const std::string& folder_path);

 private:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      TsdfIntegratorBase, RaytracingTsdfIntegrator, std::shared_ptr<Globals>>
      registration_;
  std::shared_ptr<voxblox::MergedTsdfIntegrator> voxblox_integrator_ptr_;
  std::unordered_map<int, std::shared_ptr<GlobalVoxbloxIntegrator>>
      submap_geometry_integrtor_ptr_;  // map_id, ptr_integrator
  int frame_;
  std::mutex segments_mutex_;
  std::mutex integrator_mutex_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATION_RAYTRACING_TSDF_INTEGRATOR_H_
