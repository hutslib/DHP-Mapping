#include "panoptic_mapping/map_management/tsdf_registrator.h"

#include <algorithm>
#include <future>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <voxblox/integrator/merge_integration.h>
#include <voxblox/mesh/mesh_integrator.h>

#include "panoptic_mapping/common/index_getter.h"
#include "panoptic_mapping/map/submap.h"
#include "panoptic_mapping/map/submap_collection.h"

namespace panoptic_mapping {

void TsdfRegistrator::Config::checkParams() const {
  checkParamNE(error_threshold, 0.f, "error_threshold");
  checkParamGE(min_voxel_weight, 0.f, "min_voxel_weight");
  checkParamGE(match_rejection_points, 0, "match_rejection_points");
  checkParamGE(match_rejection_percentage, 0.f, "match_rejection_percentage");
  checkParamLE(match_rejection_percentage, 1.f, "match_rejection_percentage");
  checkParamGE(match_acceptance_points, 0, "match_acceptance_points");
  checkParamGE(match_acceptance_percentage, 0.f, "match_acceptance_percentage");
  checkParamLE(match_acceptance_percentage, 1.f, "match_acceptance_percentage");
  if (normalize_by_voxel_weight) {
    checkParamGT(normalization_max_weight, 0.f, "normalization_max_weight");
  }
  checkParamGT(integration_threads, 0, "integration_threads");
}

void TsdfRegistrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("min_voxel_weight", &min_voxel_weight);
  setupParam("error_threshold", &error_threshold);
  setupParam("match_rejection_points", &match_rejection_points);
  setupParam("match_rejection_percentage", &match_rejection_percentage);
  setupParam("match_acceptance_points", &match_acceptance_points);
  setupParam("match_acceptance_percentage", &match_acceptance_percentage);
  setupParam("normalize_by_voxel_weight", &normalize_by_voxel_weight);
  setupParam("normalization_max_weight", &normalization_max_weight);
  setupParam("integration_threads", &integration_threads);
  setupParam("submap_color_discretization", &submap_color_discretization);
}

TsdfRegistrator::TsdfRegistrator(const Config& config)
    : config_(config.checkValid()) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
  id_color_map_.setItemsPerRevolution(config_.submap_color_discretization);
}

void TsdfRegistrator::checkSubmapCollectionForChange(
    SubmapCollection* submaps) const {
  auto t_start = std::chrono::high_resolution_clock::now();
  std::string info;

  // Check all inactive maps for alignment with the currently active ones.
  std::vector<int> id_list;
  for (const Submap& submap : *submaps) {
    if (!submap.isActive() && submap.getLabel() != PanopticLabel::kFreeSpace &&
        !submap.getIsoSurfacePoints().empty()) {
      id_list.emplace_back(submap.getID());
    }
  }

  // Perform change detection in parallel.
  SubmapIndexGetter index_getter(id_list);
  std::vector<std::future<std::string>> threads;
  for (int i = 0; i < config_.integration_threads; ++i) {
    threads.emplace_back(
        std::async(std::launch::async, [this, &index_getter, submaps]() {
          int index;
          std::string info;
          while (index_getter.getNextIndex(&index)) {
            info += this->checkSubmapForChange(*submaps,
                                               submaps->getSubmapPtr(index));
          }
          return info;
        }));
  }

  // Join all threads.
  for (auto& thread : threads) {
    info += thread.get();
  }
  auto t_end = std::chrono::high_resolution_clock::now();

  LOG_IF(INFO, config_.verbosity >= 2)
      << "Performed change detection in "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
             .count()
      << (config_.verbosity < 3 || info.empty() ? "ms." : "ms:" + info);
}

std::string TsdfRegistrator::checkSubmapForChange(
    const SubmapCollection& submaps, Submap* submap) const {
  // Check overlapping submaps for conflicts or matches.
  for (const Submap& other : submaps) {
    if (!other.isActive() ||
        !submap->getBoundingVolume().intersects(other.getBoundingVolume())) {
      continue;
    }

    // Note(schmluk): Exclude free space for thin structures. Although there's
    // potentially a nicer way of solving this.
    if (other.getLabel() == PanopticLabel::kFreeSpace &&
        submap->getConfig().voxel_size < other.getConfig().voxel_size * 0.5) {
      continue;
    }

    bool submaps_match;
    if (submapsConflict(*submap, other, &submaps_match)) {
      // No conflicts allowed.
      if (submap->getChangeState() != ChangeState::kAbsent) {
        submap->setChangeState(ChangeState::kAbsent);
      }
      std::stringstream info;
      info << "\nSubmap " << submap->getID() << " (" << submap->getName()
           << ") conflicts with submap " << other.getID() << " ("
           << other.getName() << ").";
      return info.str();
    } else if (submap->getClassID() == other.getClassID() && submaps_match) {
      // Semantically and geometrically match.
      submap->setChangeState(ChangeState::kPersistent);
    }
  }
  return "";
}

void TsdfRegistrator::computeCorners(Point corners[8], Point& origin,
                                     FloatingPoint block_size) {
  corners[0] = origin;
  corners[1] = origin + Point(block_size, 0, 0);
  corners[2] = origin + Point(0, block_size, 0);
  corners[3] = origin + Point(block_size, block_size, 0);
  corners[4] = origin + Point(0, 0, block_size);
  corners[5] = origin + Point(block_size, 0, block_size);
  corners[6] = origin + Point(0, block_size, block_size);
  corners[7] = origin + Point(block_size, block_size, block_size);
}

void TsdfRegistrator::processVoxels(
    const size_t start_voxel, const size_t end_voxel, TsdfBlock::Ptr tsdf_block,
    ClassBlock::Ptr class_block, const Transformation& T_O_R, Submap* other,
    int& merge_count, int& same_position_voxel_size,
    const float& rejection_distance, bool& was_updated) {
  for (size_t voxel_idx = start_voxel; voxel_idx < end_voxel; ++voxel_idx) {
    TsdfVoxel& reference_voxel = tsdf_block->getVoxelByLinearIndex(voxel_idx);
    ClassVoxel& reference_class_voxel =
        class_block->getVoxelByLinearIndex(voxel_idx);
    if (reference_voxel.weight >= 1e-6) {
      // get the center point transfered to other submap coordinate and get
      // the corresponding tsdf voxel and class voxel
      const Point voxel_center =
          T_O_R * tsdf_block->computeCoordinatesFromLinearIndex(voxel_idx);
      TsdfBlock::Ptr other_block_ptr =
          other->getTsdfLayerPtr()->getBlockPtrByCoordinates(voxel_center);
      ClassBlock::Ptr other_class_ptr =
          other->getClassLayerPtr()->getBlockPtrByCoordinates(voxel_center);
      if (other_block_ptr) {
        if (!other_class_ptr) {
          LOG(ERROR) << "other block do not has class layer!" << std::endl;
        } else {
          TsdfVoxel& other_voxel =
              other_block_ptr->getVoxelByCoordinates(voxel_center);
          ClassVoxel& other_class_voxel =
              other_class_ptr->getVoxelByCoordinates(voxel_center);
          if (other_voxel.weight > 1e-6) {
            LOG_IF(INFO, config_.verbosity >= 5) << "same position voxel";
            {
              std::unique_lock<std::mutex> lock(
                  same_position_voxel_size_mutex_);
              same_position_voxel_size++;
            }
            if (std::fabs(other_voxel.distance - reference_voxel.distance) <
                rejection_distance) {
              LOG_IF(INFO, config_.verbosity >= 5) << "merge voxel";
              {
                std::unique_lock<std::mutex> lock(was_updated_mutex_);
                was_updated = true;
              }
              if (config_.verbosity >= 5) {
                std::cout << "reference voxel class origin info: "
                          << reference_class_voxel.printCount() << " semantic "
                          << reference_class_voxel.printSemantic();
                std::cout << "other voxel class origin info: "
                          << other_class_voxel.printCount() << " semantic "
                          << other_class_voxel.printSemantic();
              }
              voxblox::mergeVoxelAIntoVoxelB(reference_voxel, &other_voxel);
              other_class_voxel.mergeVoxel(reference_class_voxel);
              // clear classlayer voxel value
              reference_class_voxel.clearCount();
              // clear tsdflayer voxel value
              reference_voxel = TsdfVoxel();
              const int class_id = other_class_voxel.getClassId();
              Color voxel_color = id_color_map_.colorLookup(class_id);
              other_voxel.vis_color.r = voxel_color.r;
              other_voxel.vis_color.b = voxel_color.b;
              other_voxel.vis_color.g = voxel_color.g;
              if (config_.verbosity >= 5) {
                std::cout << "after merge reference voxel class info: "
                          << reference_class_voxel.printCount() << " semantic "
                          << reference_class_voxel.printSemantic();
                std::cout << "after merge other voxel class info: "
                          << other_class_voxel.printCount() << " semantic "
                          << other_class_voxel.printSemantic();
              }
              {
                std::unique_lock<std::mutex> lock(merge_count_mutex_);
                merge_count++;
              }
              tsdf_block->setUpdatedAll();
              other_block_ptr->setUpdatedAll();
            }
          } else {
            // LOG(WARNING) << "check other voxel failed";
          }
        }
      } else {
        // LOG(WARNING) << "other do not has block with this position "
        //              << std::endl;
      }
    }
  }
}

void TsdfRegistrator::processBlocks(
    const size_t start_block, const size_t end_block, Submap* reference,
    Submap* other, const Transformation& T_O_R,
    const voxblox::BlockIndexList& reference_all_block_indices,
    int& merge_count, int& intersection_block_size,
    int& same_position_voxel_size, const float& rejection_distance) {
  for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
    const auto& block_index = reference_all_block_indices[block_idx];
    ClassBlock::Ptr class_block;
    TsdfBlock::Ptr tsdf_block;
    if (reference->hasClassLayer()) {
      if (reference->getClassLayer().hasBlock(block_index) &&
          reference->getTsdfLayer().hasBlock(block_index)) {
        class_block =
            reference->getClassLayerPtr()->getBlockPtrByIndex(block_index);
        tsdf_block =
            reference->getTsdfLayerPtr()->getBlockPtrByIndex(block_index);
      } else {
        LOG(ERROR) << "error to find the block within the reference submap";
      }
    } else {
      LOG(ERROR) << "reference submap do not has classlayer";
    }

    // track whether this submap block is updated
    bool was_updated = false;

    Point block_origin = tsdf_block->origin();
    FloatingPoint block_size = reference->getTsdfLayer().block_size();
    if (config_.verbosity >= 5) {
      std::cout << "block size is : " << block_size << std::endl;
    }
    // get the eight corner of the block
    voxblox::Point corners[8];
    computeCorners(&corners[0], block_origin, block_size);

    // Check if at least one corner is within the bounding volume of the other
    // submap.
    bool has_intersection = false;
    for (int i = 0; i < 8; ++i) {
      const voxblox::Point& corner = corners[i];
      if (other->getBoundingVolume().contains_M(corner)) {
        // If at least one corner is within the bounding volume, add the block
        // index to the list.
        has_intersection = true;
        {
          std::unique_lock<std::mutex> lock(intersection_block_size_mutex_);
          intersection_block_size++;
        }
        break;  // Exit the loop early since we've found an intersection.
      }
    }

    if (!has_intersection) {
      // if this block is not within the other submap skip to check the inside
      // voxels
      continue;
    }

    const int voxel_indices =
        std::pow(reference->getConfig().voxels_per_side, 3);

    std::vector<std::future<void>> voxel_threads;
    const size_t num_voxel_threads = std::thread::hardware_concurrency();
    const size_t voxels_per_thread =
        (voxel_indices + num_voxel_threads - 1) / num_voxel_threads;

    for (size_t voxel_thread_index = 0; voxel_thread_index < num_voxel_threads;
         ++voxel_thread_index) {
      size_t start_voxel = voxel_thread_index * voxels_per_thread;
      size_t end_voxel = std::min(start_voxel + voxels_per_thread,
                                  static_cast<size_t>(voxel_indices));
      voxel_threads.push_back(std::async(
          std::launch::async,
          [this, &tsdf_block, &class_block, &T_O_R, start_voxel, end_voxel,
           other, &merge_count, &same_position_voxel_size, &rejection_distance,
           &was_updated] {
            processVoxels(start_voxel, end_voxel, tsdf_block, class_block,
                          T_O_R, other, merge_count, same_position_voxel_size,
                          rejection_distance, was_updated);
          }));
    }

    // Wait for all voxel threads to finish
    for (auto& voxel_thread : voxel_threads) {
      voxel_thread.wait();
    }
  }
}

void TsdfRegistrator::samePositionMergeMultiThread(Submap* reference,
                                                   Submap* other) {
  LOG_IF(INFO, config_.verbosity >= 4) << "same Position Merge Multi Thread";
  int reference_change_count = 0;
  int other_change_count = 0;
  int merge_count = 0;
  int all_block_size = 0;
  int intersection_block_size = 0;
  int same_position_voxel_size = 0;
  Transformation T_O_R = other->getT_S_M() * reference->getT_M_S();
  voxblox::BlockIndexList reference_all_block_indices;
  const float rejection_distance =
      config_.error_threshold > 0.f
          ? config_.error_threshold
          : config_.error_threshold * -other->getTsdfLayer().voxel_size();
  reference->getTsdfLayer().getAllAllocatedBlocks(&reference_all_block_indices);
  all_block_size = reference_all_block_indices.size();
  std::vector<std::future<void>> block_threads;
  const size_t num_block_threads = std::thread::hardware_concurrency();
  const size_t block_size = reference_all_block_indices.size();
  const size_t blocks_per_thread =
      (block_size + num_block_threads - 1) / num_block_threads;

  for (size_t thread_index = 0; thread_index < num_block_threads;
       ++thread_index) {
    size_t start_block = thread_index * blocks_per_thread;
    size_t end_block = std::min(start_block + blocks_per_thread, block_size);
    block_threads.push_back(std::async(
        std::launch::async,
        [this, &reference, &other, &T_O_R, &reference_all_block_indices,
         start_block, end_block, &merge_count, &intersection_block_size,
         &same_position_voxel_size, &rejection_distance] {
          processBlocks(start_block, end_block, reference, other, T_O_R,
                        reference_all_block_indices, merge_count,
                        intersection_block_size, same_position_voxel_size,
                        rejection_distance);
        }));
  }

  // Wait for all block threads to finish
  for (auto& block_thread : block_threads) {
    block_thread.wait();
  }
  LOG_IF(INFO, config_.verbosity >= 4)
      << "reference submap : " << reference->getID()
      << "all block size: " << all_block_size
      << "other submap : " << other->getID() << " merge count: " << merge_count
      << " intersection block size: " << intersection_block_size
      << "same_position_voxel_size: " << same_position_voxel_size;
}

bool TsdfRegistrator::submapsConflict(const Submap& reference,
                                      const Submap& other,
                                      bool* submaps_match) const {
  // Reference is the finished submap (with Iso-surfce-points) that is
  // compared to the active submap other.
  Transformation T_O_R = other.getT_S_M() * reference.getT_M_S();
  const float rejection_count =
      config_.normalize_by_voxel_weight
          ? std::numeric_limits<float>::max()
          : std::max(static_cast<float>(config_.match_rejection_points),
                     config_.match_rejection_percentage *
                         reference.getIsoSurfacePoints().size());
  const float rejection_distance =
      config_.error_threshold > 0.f
          ? config_.error_threshold
          : config_.error_threshold * -other.getTsdfLayer().voxel_size();
  float conflicting_points = 0.f;
  float matched_points = 0.f;
  float total_weight = 0.f;
  voxblox::Interpolator<TsdfVoxel> interpolator(&(other.getTsdfLayer()));

  // Check for disagreement.
  float distance, weight;
  for (const auto& point : reference.getIsoSurfacePoints()) {
    if (getDistanceAndWeightAtPoint(&distance, &weight, point, T_O_R,
                                    interpolator)) {
      // Compute the weight to be used for counting.
      if (config_.normalize_by_voxel_weight) {
        weight = computeCombinedWeight(weight, point.weight);
        total_weight += weight;
      } else {
        weight = 1.f;
      }

      // Count.
      if (other.getLabel() == PanopticLabel::kFreeSpace) {
        if (distance >= rejection_distance) {
          conflicting_points += weight;
        }
      } else {
        // Check for class belonging.

        if (other.hasClassLayer()) {
          const ClassVoxel* class_voxel =
              other.getClassLayer().getVoxelPtrByCoordinates(point.position);
          if (class_voxel) {
            if (!class_voxel->belongsToSubmap()) {
              distance = other.getConfig().truncation_distance;
            }
          }
        }
        if (distance <= -rejection_distance) {
          conflicting_points += weight;
        } else if (distance <= rejection_distance) {
          matched_points += weight;
        }
      }

      if (conflicting_points > rejection_count) {
        // If the rejection count is known and reached submaps conflict.
        if (submaps_match) {
          *submaps_match = false;
        }
        return true;
      }
    }
  }
  // Evaluate the result.
  if (config_.normalize_by_voxel_weight) {
    const float rejection_weight =
        std::max(static_cast<float>(config_.match_rejection_points) /
                     reference.getIsoSurfacePoints().size(),
                 config_.match_rejection_percentage) *
        total_weight;
    if (conflicting_points > rejection_weight) {
      if (submaps_match) {
        *submaps_match = false;
      }
      return true;
    } else if (submaps_match) {
      const float acceptance_weight =
          std::max(static_cast<float>(config_.match_acceptance_points) /
                       reference.getIsoSurfacePoints().size(),
                   config_.match_acceptance_percentage) *
          total_weight;
      if (matched_points > acceptance_weight) {
        *submaps_match = true;
      } else {
        *submaps_match = false;
      }
    }
  } else {
    if (submaps_match) {
      const float acceptance_count =
          std::max(static_cast<float>(config_.match_acceptance_points),
                   config_.match_acceptance_percentage *
                       reference.getIsoSurfacePoints().size());
      *submaps_match = matched_points > acceptance_count;
    }
  }
  return false;
}  // namespace panoptic_mapping

bool TsdfRegistrator::getDistanceAndWeightAtPoint(
    float* distance, float* weight, const IsoSurfacePoint& point,
    const Transformation& T_P_S,
    const voxblox::Interpolator<TsdfVoxel>& interpolator) const {
  // Check minimum input point weight.
  if (point.weight < config_.min_voxel_weight) {
    return false;
  }

  // Try to interpolate the voxel in the  map.
  // NOTE(Schmluk): This also interpolates color etc, but otherwise the
  // interpolation lookup has to be done twice. Getting only what we want is
  // also in voxblox::interpolator but private, atm not performance critical.
  TsdfVoxel voxel;
  const Point position = T_P_S * point.position;
  if (!interpolator.getVoxel(position, &voxel, true)) {
    return false;
  }
  if (voxel.weight < config_.min_voxel_weight) {
    return false;
  }
  *distance = voxel.distance;
  *weight = voxel.weight;
  return true;
}

float TsdfRegistrator::computeCombinedWeight(float w1, float w2) const {
  if (w1 <= 0.f || w2 <= 0.f) {
    return 0.f;
  } else if (w1 >= config_.normalization_max_weight &&
             w2 >= config_.normalization_max_weight) {
    return 1.f;
  } else {
    return std::sqrt(std::min(w1 / config_.normalization_max_weight, 1.f) *
                     std::min(w2 / config_.normalization_max_weight, 1.f));
  }
}

void TsdfRegistrator::mergeMatchingSubmaps(SubmapCollection* submaps) {
  // Merge all submaps of identical Instance ID into one.
  // TODO(schmluk): This is a preliminary function for prototyping, update
  // this.
  submaps->updateInstanceToSubmapIDTable();
  int merged_maps = 0;
  for (const auto& instance_submaps : submaps->getInstanceToSubmapIDTable()) {
    const auto& ids = instance_submaps.second;
    Submap* target;
    for (auto it = ids.begin(); it != ids.end(); ++it) {
      if (it == ids.begin()) {
        target = submaps->getSubmapPtr(*it);
        continue;
      }
      // Merging.
      merged_maps++;
      voxblox::mergeLayerAintoLayerB(submaps->getSubmap(*it).getTsdfLayer(),
                                     target->getTsdfLayerPtr().get());
      submaps->removeSubmap(*it);
    }
    // Set the updated flags of the changed layer.
    voxblox::BlockIndexList block_list;
    target->getTsdfLayer().getAllAllocatedBlocks(&block_list);
    for (auto& index : block_list) {
      target->getTsdfLayerPtr()->getBlockByIndex(index).setUpdatedAll();
    }
  }
  LOG_IF(INFO, config_.verbosity >= 2)
      << "Merged " << merged_maps << " submaps.";
}

}  // namespace panoptic_mapping
