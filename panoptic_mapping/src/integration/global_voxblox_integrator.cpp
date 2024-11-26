#include "panoptic_mapping/integration/global_voxblox_integrator.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <voxblox/integrator/merge_integration.h>

#include "panoptic_mapping/common/index_getter.h"

namespace panoptic_mapping {

void GlobalVoxbloxIntegrator::Config::checkParams() const {

}

void GlobalVoxbloxIntegrator::Config::setupParamsAndPrinting() {
  setupParam("min_ray_length_m", &min_ray_length_m);
  setupParam("max_ray_length_m", &max_ray_length_m);
  setupParam("enable_anti_grazing", &enable_anti_grazing);
  setupParam("verbosity", &verbosity);
  setupParam("truncation_distance", &truncation_distance);
  setupParam("label_truncation_distance", &label_truncation_distance);
  setupParam("enable_confidence_weight_dropoff", &enable_confidence_weight_dropoff);
  setupParam("lognormal_weight_mean", &lognormal_weight_mean);
  setupParam("lognormal_weight_sigma", &lognormal_weight_sigma);
  setupParam("lognormal_weight_offset", &lognormal_weight_offset);
  setupParam("integrator_threads", &integrator_threads);
  setupParam("skip_update_label_faraway_surface",
             &skip_update_label_faraway_surface);
  setupParam("voxel_carving_enabled", &voxel_carving_enabled);
  setupParam("use_detectron", &use_detectron);
  setupParam("instance_weight", &instance_weight);
  setupParam("background_weight", &background_weight);
  setupParam("submap_color_discretization", &submap_color_discretization);
  tsdf_config.min_ray_length_m = min_ray_length_m;
  tsdf_config.max_ray_length_m = max_ray_length_m;
  tsdf_config.enable_anti_grazing = enable_anti_grazing;
  tsdf_config.default_truncation_distance = truncation_distance;
  tsdf_config.voxel_carving_enabled = voxel_carving_enabled;
}

GlobalVoxbloxIntegrator::GlobalVoxbloxIntegrator(const Config& config,
                                                 Submap* map)
    : config_(config.checkValid()),
      voxblox::MergedTsdfIntegrator(config.tsdf_config,
                                    map->getTsdfLayerPtr().get()),
      map_(map) {
  LOG_IF(INFO, config_.verbosity >= 4) << "\n" << config_.toString() << "\n";

  if (config_.use_detectron) {
    unknow_class_ = 0;
  } else {
    unknow_class_ = -1;
  }

  id_color_map_.setItemsPerRevolution(config_.submap_color_discretization);
}

void GlobalVoxbloxIntegrator::processIntegration(const Transformation& T_G_C,
                                                 const Pointcloud& points_C,
                                                 const Colors& colors,
                                                 const Labels& labels,
                                                 const bool freespace_points) {
  CHECK_EQ(points_C.size(), colors.size());
  CHECK_EQ(points_C.size(), labels.size());

  // Pre-compute a list of unique voxels to end on.
  // Create a hashmap: VOXEL INDEX -> index in original cloud.
  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type voxel_map;
  // This is a hash map (same as above) to all the indices that need to be cleared.
  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type clear_map;

  std::unique_ptr<voxblox::ThreadSafeIndex> index_getter(
      voxblox::ThreadSafeIndexFactory::get(
          config_.tsdf_config.integration_order_mode, points_C));

  MergedTsdfIntegrator::bundleRays(T_G_C, points_C, freespace_points,
                                   index_getter.get(), &voxel_map, &clear_map);


// For each submap integration, perform integratePointCloud.
// This will sequentially execute bundleRay (since all belong to the same instance class,
// there won't be bundling of different labels),
  integrateRays(T_G_C, points_C, colors, labels,
                config_.tsdf_config.enable_anti_grazing, false, voxel_map,
                clear_map);
}
ClassVoxel* GlobalVoxbloxIntegrator::allocateStorageAndGetClassVoxelPtr(
    const GlobalIndex& global_voxel_idx) {
  const BlockIndex block_idx = voxblox::getBlockIndexFromGlobalVoxelIndex(
      global_voxel_idx, voxels_per_side_inv_);

  std::lock_guard<std::mutex> lock(temp_class_block_mutex_);
  auto block_ptr = map_->getClassLayerPtr()->allocateBlockPtrByIndex(block_idx);
  block_ptr->setUpdatedAll();
  const VoxelIndex local_voxel_idx =
      voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_);

  return &(block_ptr->getVoxelByVoxelIndex(local_voxel_idx));
}

void GlobalVoxbloxIntegrator::integrateRays(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const Labels& labels, bool enable_anti_grazing,
    bool clearing_ray,
    const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map,
    const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type&
        clear_map) {
  if (config_.integrator_threads == 1) {
    constexpr size_t thread_idx = 0;
    integrateVoxels(T_G_C, points_C, colors, labels, enable_anti_grazing,
                    clearing_ray, voxel_map, clear_map, thread_idx);
  } else {
    std::list<std::thread> integration_threads;
    for (size_t i = 0; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back(
          &GlobalVoxbloxIntegrator::integrateVoxels, this, T_G_C, points_C,
          colors, labels, enable_anti_grazing, clearing_ray, voxel_map,
          clear_map, i);
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }

  updateLayerWithStoredBlocks();

}

void GlobalVoxbloxIntegrator::integrateVoxels(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const Labels& labels, bool enable_anti_grazing,
    bool clearing_ray,
    const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map,
    const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type& clear_map,
    size_t thread_idx) {
  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type::const_iterator it;
  size_t map_size;
  if (clearing_ray) {
    it = clear_map.begin();
    map_size = clear_map.size();
  } else {
    it = voxel_map.begin();
    map_size = voxel_map.size();
  }

  for (size_t i = 0; i < map_size; ++i) {
    if (((i + thread_idx + 1) % config_.integrator_threads) == 0) {
      integrateVoxel(T_G_C, points_C, colors, labels, enable_anti_grazing,
                     clearing_ray, *it, voxel_map);
    }
    ++it;
  }
}

void GlobalVoxbloxIntegrator::integrateVoxel(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const Colors& colors, const Labels& labels, bool enable_anti_grazing,
    bool clearing_ray, const std::pair<GlobalIndex, AlignedVector<size_t>>& kv,
    const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type&
        voxel_map) {
  if (kv.second.empty()) {
    return;
  }
  const Point& origin = T_G_C.getPosition();
  Color merged_color;
  Point merged_point_C = Point::Zero();
  FloatingPoint merged_weight = 0.0;
  std::vector<Label> candidate_sets;
  for (const size_t pt_idx : kv.second) {
    const Point& point_C = points_C[pt_idx];
    const Color& color = colors[pt_idx];
    const Label& candidate_L = labels[pt_idx];
    const float point_weight = getVoxelWeight(point_C);
    if (point_weight < voxblox::kEpsilon) {
      continue;
    }
    merged_point_C = (merged_point_C * merged_weight + point_C * point_weight) /
                     (merged_weight + point_weight);
    merged_color =
        Color::blendTwoColors(merged_color, merged_weight, color, point_weight);
    merged_weight += point_weight;

    candidate_sets.push_back(candidate_L);

    // only take first point when clearing
    if (clearing_ray) {
      break;
    }
  }

  const Point merged_point_G = T_G_C * merged_point_C;

  // we do not use voxel carving enable
  float truncation_distance;
  if (config_.truncation_distance < 0.f) {
    float vs = map_->getTsdfLayer().voxel_size();
    truncation_distance = config_.truncation_distance * (-vs);
    LOG_IF(INFO, config_.verbosity >= 5)
        << "when do ray cast the default truncation distance is : "
        << truncation_distance << std::endl;
  }

  voxblox::RayCaster ray_caster(origin, merged_point_G, clearing_ray,
                                config_.tsdf_config.voxel_carving_enabled,
                                config_.tsdf_config.max_ray_length_m,
                                voxel_size_inv_, truncation_distance);

  GlobalIndex global_voxel_idx;
  bool first_cast = true;
  GlobalIndex last_global_voxel_idx;
  // we enable the enti grazing
  while (ray_caster.nextRayIndex(&global_voxel_idx)) {
    bool update_label = true;
    if (enable_anti_grazing) {
      // Check if this one is already the the block hash map for this
      // insertion. Skip this to avoid grazing.
      if ((clearing_ray || global_voxel_idx != kv.first) &&
          voxel_map.find(global_voxel_idx) != voxel_map.end()) {
        continue;
      }
    }
    // we only update the label info once a ray hit this voxel
    if (first_cast) {
      first_cast = false;
    } else if (global_voxel_idx == last_global_voxel_idx) {
      update_label = false;
    }
    last_global_voxel_idx = global_voxel_idx;

    voxblox::Block<TsdfVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    TsdfVoxel* voxel =
        allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);
    ClassVoxel* class_voxel =
        allocateStorageAndGetClassVoxelPtr(global_voxel_idx);
    float sdf;
    bool updated_successful =
        updateTsdfVoxel(origin, merged_point_G, global_voxel_idx, merged_color,
                        merged_weight, voxel, &sdf);

    if (!updated_successful) {
      if (voxel->distance != 0 && voxel->weight != 0) {
        std::cout << "voxel distance: " << std::setprecision(20)
                  << voxel->distance
                  << " voxel weight: " << std::setprecision(20) << voxel->weight
                  << std::endl;
        std::cout << "is observed: " << class_voxel->isObserverd() << std::endl;
        std::cout << class_voxel->printCount();
        LOG(WARNING) << "update voxel unsuccessfully";
      }
      return;
    }
    // we only update the label layer near the surface
    float label_truncation_distance;
    if (config_.label_truncation_distance < 0.f) {
      float vs = map_->getTsdfLayer().voxel_size();
      label_truncation_distance = config_.label_truncation_distance * (-vs);
      LOG_IF(INFO, config_.verbosity >= 5)
          << "the label truncatuon distance is : " << label_truncation_distance
          << std::endl;
    }
    if (fabs(sdf) > label_truncation_distance &&
        config_.skip_update_label_faraway_surface) {
      LOG_IF(INFO, config_.verbosity >= 4)
          << "skip update label voxel far from surface with sdf: " << sdf
          << std::endl;
      return;
    }
    // point in one ray get same label
    // DCHECK(class_voxel != nullptr);
    // CHECK_NOTNULL(class_voxel);
    if (candidate_sets.size() == 0) {
      LOG(WARNING) << "candidate size is 0";
    }
    int label_ins = candidate_sets[0].ins_label_;
    int label_sem = candidate_sets[0].sem_label_;
    int panoptic_id = candidate_sets[0].panoptic_id_;
    float weight;
    if(panoptic_id == 1 ) {
      // for instance, we assign the weight to 1
      weight = config_.instance_weight;
    } else if(panoptic_id == 2 ) {
      // for background, we assign the weight to 0.5
      weight = config_.background_weight;
    }
    if (class_voxel) {
      if (label_sem != unknow_class_ && update_label && label_ins != 0) {
        std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));
        class_voxel->incrementCount(0,weight);
        class_voxel->incrementCount(label_ins,weight);
        class_voxel->incrementSemanticCount(label_sem,weight);
        // assign color based on the voxel semantic class info
        // color based on voxel class info
        const int class_id = class_voxel->getClassId();
        Color voxel_color = id_color_map_.colorLookup(class_id);
        voxel->vis_color.r = voxel_color.r;
        voxel->vis_color.b = voxel_color.b;
        voxel->vis_color.g = voxel_color.g;
        for (auto label_i : candidate_sets) {
          if (label_i.ins_label_ != label_ins ||
              label_i.sem_label_ != label_sem) {
            std::cout << "label_i.ins_label_: " << label_i.ins_label_
                      << std::endl;
            std::cout << "label_ins: " << label_ins << std::endl;
            std::cout << "label_i.sem_label_: " << label_i.sem_label_
                      << std::endl;
            std::cout << "label_sem: " << label_sem << std::endl;
            LOG(ERROR) << "inconsistent label";
            exit(0);
          }
        }
      } else if (!update_label) {
        LOG_IF(WARNING, config_.verbosity >= 5)
            << "multi time get into this voxel and skip update label info";
      }
    } else {
      LOG(ERROR) << "allocate class voxel failed" << std::endl;
    }

  }
}
} // namespace panoptic_mapping
