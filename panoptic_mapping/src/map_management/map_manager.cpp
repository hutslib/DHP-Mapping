#include "panoptic_mapping/map_management/map_manager.h"

#include <float.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<MapManagerBase, MapManager>
    MapManager::registration_("submaps");

void MapManager::Config::checkParams() const {
  checkParamConfig(activity_manager_config);
  checkParamConfig(tsdf_registrator_config);
  checkParamConfig(layer_manipulator_config);
}

void MapManager::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("prune_active_blocks_frequency", &prune_active_blocks_frequency);
  setupParam("activity_management_frequency", &activity_management_frequency);
  setupParam("voxel_class_management_frequency",
             &voxel_class_management_frequency);
  setupParam("change_detection_frequency", &change_detection_frequency);
  setupParam("excute_crf", &excute_crf);
  setupParam("crf_frequency", &crf_frequency);
  setupParam("merge_deactivated_submaps_if_possible",
             &merge_deactivated_submaps_if_possible);
  setupParam("apply_class_layer_when_deactivating_submaps",
             &apply_class_layer_when_deactivating_submaps);
  setupParam("activity_manager_config", &activity_manager_config,
             "activity_manager");
  setupParam("tsdf_registrator_config", &tsdf_registrator_config,
             "tsdf_registrator");
  setupParam("layer_manipulator_config", &layer_manipulator_config,
             "layer_manipulator");
  setupParam("semantic_size", &semantic_size);
  setupParam("crf_iterations", &crf_iterations);
  setupParam("use_high_order", &use_high_order);
  setupParam("smooth_xy_stddev", &smooth_xy_stddev);
  setupParam("smooth_z_stddev", &smooth_z_stddev);
  setupParam("smooth_weight", &smooth_weight);
  setupParam("appear_xy_stddev", &appear_xy_stddev);
  setupParam("appear_z_stddev", &appear_z_stddev);
  setupParam("appear_rgb_stddev", &appear_rgb_stddev);
  setupParam("appear_weight", &appear_weight);
  setupParam("exit_after_management", &exit_after_management);
  setupParam("use_detectron", &use_detectron);
  setupParam("use_kitti", &use_kitti);
  setupParam("crfmultithread", &crfmultithread);
  setupParam("threads", &threads);
  setupParam("submap_info_path", &submap_info_path);
  setupParam("submap_color_discretization", &submap_color_discretization);
}

MapManager::MapManager(const Config& config) : config_(config.checkValid()) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  // Setup members.
  activity_manager_ =
      std::make_shared<ActivityManager>(config_.activity_manager_config);
  tsdf_registrator_ =
      std::make_shared<TsdfRegistrator>(config_.tsdf_registrator_config);
  layer_manipulator_ =
      std::make_shared<LayerManipulator>(config_.layer_manipulator_config);

  // Add all requested tasks.
  if (config_.prune_active_blocks_frequency > 0) {
    tickers_.emplace_back(
        config_.prune_active_blocks_frequency,
        [this](SubmapCollection* submaps) { pruneActiveBlocks(submaps); });
  }
  if (config_.activity_management_frequency > 0) {
    tickers_.emplace_back(
        config_.activity_management_frequency,
        [this](SubmapCollection* submaps) { manageSubmapActivity(submaps); });
  }
  if (config_.voxel_class_management_frequency > 0) {
    tickers_.emplace_back(
        config_.voxel_class_management_frequency,
        [this](SubmapCollection* submaps) { manageVoxelClass(submaps); });
  }
  if (config_.change_detection_frequency > 0) {
    tickers_.emplace_back(
        config_.change_detection_frequency,
        [this](SubmapCollection* submaps) { performChangeDetection(submaps); });
  }
  id_color_map_.setItemsPerRevolution(config_.submap_color_discretization);
}

void MapManager::tick(SubmapCollection* submaps) {
  // Increment counts for all tickers, which execute the requested actions.
  for (Ticker& ticker : tickers_) {
    ticker.tick(submaps);
  }
}

void MapManager::pruneActiveBlocks(SubmapCollection* submaps) {
  // Process all active instance and background submaps.
  auto t1 = std::chrono::high_resolution_clock::now();
  Timer timer("map_management/prune_active_blocks");
  std::stringstream info;
  std::vector<int> submaps_to_remove;
  for (Submap& submap : *submaps) {
    if (submap.getLabel() == PanopticLabel::kFreeSpace || !submap.isActive()) {
      continue;
    }
    info << pruneBlocks(&submap);

    // If a submap does not contain data anymore it can be removed.
    if (submap.getTsdfLayer().getNumberOfAllocatedBlocks() == 0) {
      submaps_to_remove.emplace_back(submap.getID());
      if (config_.verbosity >= 4) {
        info << "Removed submap!";
      }
    }
  }

  // Remove submaps.
  for (int id : submaps_to_remove) {
    submaps->removeSubmap(id);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  timer.Stop();
  LOG_IF(INFO, config_.verbosity >= 2)
      << "Pruned active blocks in "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms." << info.str();
}

void MapManager::manageSubmapActivity(SubmapCollection* submaps) {
  CHECK_NOTNULL(submaps);
  LOG_IF(INFO, config_.verbosity >= 2) << "manage submap activity";
  std::set<int> active_submaps_id;  // store the currently active instance
  // we use the std::set to preserve the ordered of the element
  std::set<int> frozen_submaps_id;  // store the inactive instance and
                                    // the background submaps

  // find all active_submaps and inactive_submaps
  for (const Submap& submap : *submaps) {
    if (submap.getID() == 0) {
      continue;
    }
    if (submap.isActive() && submap.getLabel() == PanopticLabel::kInstance) {
      active_submaps_id.insert(submap.getID());
    } else {
      frozen_submaps_id.insert(submap.getID());
    }
  }

  // std::set<int> frozen_submaps_id_origin = frozen_submaps_id;

  for (auto id : frozen_submaps_id) {
    accumulate_frozen_submap_.insert(id);
  }

  if (config_.verbosity >= 4) {
    std::cout << "currently active submaps";
    for (const auto& submap : active_submaps_id) {
      std::cout << submap << " ";
    }
    std::cout << "\n";
    std::cout << "currently frozen submaps:";
    for (const auto& submap : frozen_submaps_id) {
      std::cout << submap << " ";
    }
    std::cout << "\n";
  }

  // Perform activity management.
  // check redetection and missdetection do activity like delete and inactive
  // submaps
  std::stringstream prune_info;
  for (Submap& submap : *submaps) {
    prune_info << pruneBlocks(&submap);
  }
  LOG_IF(INFO, config_.verbosity >= 2)
      << "Pruned blocks info: " << prune_info.str();
  activity_manager_->processSubmaps(submaps);

  // Get the newly de-actived submaps and merge them into the frozen submaps
  std::set<int> deactivated_submaps_id;
  for (Submap& submap : *submaps) {
    if (!submap.isActive() &&
        active_submaps_id.find(submap.getID()) != active_submaps_id.end()) {
      deactivated_submaps_id.insert(submap.getID());
      frozen_submaps_id.insert(submap.getID());
    }
  }

  if (config_.verbosity >= 4) {
    std::cout << " deactivated submaps ";
    for (const auto& submap : deactivated_submaps_id) {
      std::cout << submap << " ";
    }
    std::cout << "\n";
    std::cout << " frozen submaps:";
    for (const auto& submap : frozen_submaps_id) {
      std::cout << submap << " ";
    }
    std::cout << "\n";
  }

  for (auto id : deactivated_submaps_id) {
    accumulate_deactivated_submap_.insert(id);
  }
}

void MapManager::manageVoxelClass(SubmapCollection* submaps) {
  crf_frame_counter_ += 1;
  LOG_IF(INFO, config_.verbosity >= 2) << "manage voxel class";
  auto t0 = std::chrono::high_resolution_clock::now();
  std::set<int> pure_frozen_submap;
  std::set<int> all_inactived_submap;
  std::set_difference(
      accumulate_frozen_submap_.begin(), accumulate_frozen_submap_.end(),
      accumulate_deactivated_submap_.begin(),
      accumulate_deactivated_submap_.end(),
      std::inserter(pure_frozen_submap, pure_frozen_submap.begin()));
  std::set_union(
      accumulate_frozen_submap_.begin(), accumulate_frozen_submap_.end(),
      accumulate_deactivated_submap_.begin(),
      accumulate_deactivated_submap_.end(),
      std::inserter(all_inactived_submap, all_inactived_submap.begin()));
  if (config_.verbosity >= 2) {
    std::cout << "accumulate_frozen_submap_: " << std::endl;
    for (int map_id : accumulate_frozen_submap_) {
      std::cout << map_id << " ";
    }
    std::cout << "\n";
    std::cout << "pure_frozen_submap: " << std::endl;
    for (int map_id : pure_frozen_submap) {
      std::cout << map_id << " ";
    }
    std::cout << "\n";
    std::cout << "accumulate_deactivated_submap_: " << std::endl;
    for (int map_id : accumulate_deactivated_submap_) {
      std::cout << map_id << " ";
    }
    std::cout << "\n";
    std::cout << "all_inactived_submap: " << std::endl;
    for (int map_id : all_inactived_submap) {
      std::cout << map_id << " ";
    }
    std::cout << "\n";
  }

  // process
  // Try to merge the submaps.
  // first merge voxel info across all inactive submaps
  auto t1 = std::chrono::high_resolution_clock::now();
  std::set<int> all_inactived_submap_copy = all_inactived_submap;
  mergeSubmapVoxelInfo(all_inactived_submap_copy, submaps);
  // set the class voxel to it belonging submap
  std::stringstream info;
  std::vector<int> submaps_to_remove;
  for (const auto& inactive_id : all_inactived_submap) {
    LOG_IF(INFO, config_.verbosity >= 4)
        << "check submap " << inactive_id
        << " voxels and set to belonging voxels ";
    Submap* inactive_submap = submaps->getSubmapPtr(inactive_id);
    setBelongingsMultiThread(inactive_submap, submaps);
    if (inactive_submap->getTsdfLayer().getNumberOfAllocatedBlocks() == 0) {
      submaps_to_remove.emplace_back(inactive_submap->getID());
      if (config_.verbosity >= 4) {
        info << inactive_submap->getID() << " Removed submap! ";
      }
    }
  }
  // Remove unusefull submaps.
  for (int id : submaps_to_remove) {
    submaps->removeSubmap(id);
    all_inactived_submap.erase(id);
    pure_frozen_submap.erase(id);
    accumulate_deactivated_submap_.erase(id);
  }
  LOG_IF(INFO, config_.verbosity >= 2) << "Pruned frozen blocks " << info.str();


  auto t2 = std::chrono::high_resolution_clock::now();
  // get the submaps collection which need to do crf optimize
  // NOTE(thuaj): here we set accumulate_deactivated_submap_ + which
  // previously frozen submap has an intersection with the deactivated one
  std::set<int> crf_submaps_id = accumulate_deactivated_submap_;

  for (auto id : crf_submaps_id) {
    if (id == 0) {
      LOG(WARNING) << "crf try to process submap id 0";
      exit(0);
    }
  }

  // multi-thread
  std::mutex crf_submaps_id_mutex;
  std::vector<std::future<void>> futures;
  for (auto& this_submap_id : accumulate_deactivated_submap_) {
    futures.emplace_back(std::async(std::launch::async, [&]() {
      Submap* this_submap = submaps->getSubmapPtr(this_submap_id);
      for (auto& other_submap_id : pure_frozen_submap) {
        Submap* other_submap = submaps->getSubmapPtr(other_submap_id);
        if (this_submap->getBoundingVolume().intersects(
                other_submap->getBoundingVolume())) {
          std::lock_guard<std::mutex> lock(crf_submaps_id_mutex);
          if (config_.verbosity >= 4) {
            std::cout << "insert crf submap: " << other_submap_id << std::endl;
          }
          crf_submaps_id.insert(other_submap_id);
        }
      }
    }));
  }

  for (auto& future : futures) {
    future.wait();
  }

  if (config_.verbosity >= 2) {
    std::cout << "crf submap size: " << crf_submaps_id.size() << std::endl;
    for (const auto& submap_id : crf_submaps_id) {
      Submap* submap = submaps->getSubmapPtr(submap_id);
      std::cout << submap->toString() << std::endl;
    }
    std::cout << "\n";
  }

  if (crf_submaps_id.size() > 0 && config_.excute_crf) {
    // crf optimizer
    processCrfMultiThread(crf_submaps_id, submaps);
    // set the class voxel to it belonging submap
    std::stringstream crf_info;
    std::vector<int> submaps_to_remove_aft_crf;
    for (const auto& inactivated_id : all_inactived_submap) {
      LOG_IF(INFO, config_.verbosity >= 4)
          << "aft crf check submap " << inactivated_id
          << " voxels and set to belonging voxels ";
      Submap* inactivated_submap = submaps->getSubmapPtr(inactivated_id);
      setBelongingsMultiThread(inactivated_submap, submaps);
      if (inactivated_submap->getTsdfLayer().getNumberOfAllocatedBlocks() ==
          0) {
        submaps_to_remove_aft_crf.emplace_back(inactivated_submap->getID());
        if (config_.verbosity >= 4) {
          crf_info << "Removed submap!";
        }
      }
      // Remove submaps after crf
      for (int id : submaps_to_remove_aft_crf) {
        submaps->removeSubmap(id);
      }
      LOG_IF(INFO, config_.verbosity >= 2)
          << "Pruned frozen blocks " << crf_info.str();
    }
  }
  // at the manage end we clear the accumulate set
  accumulate_frozen_submap_.clear();
  accumulate_deactivated_submap_.clear();
}
template <typename Derived>
inline bool is_finite(const Eigen::MatrixBase<Derived>& x) {
  return ((x.array() == x.array())).all();
}

void MapManager::processCrfMultiThread(std::set<int>& frozen_submaps_id,
                                       SubmapCollection* submaps) {
  LOG_IF(INFO, config_.verbosity >= 1)
      << "map management process crf to regularization the instance and "
         "semantic";
  int updated_vertex_num = 0;
  crf_instance_num_ = frozen_submaps_id.size();  // N: instance label size
  crf_semantic_num_ = config_.semantic_size;     // M: semantic label size
  int crf_voxel_num_guess = 0;
  // process all frozen_submaps to set the unary and pair-wise energy
  // Create a vector of futures to hold the results of the async calls.
  std::vector<std::future<int>> calculate_guess_num_futures;

  // Iterate through each frozen_submap_id and launch a new async call for each
  // one.
  for (const auto& frozen_submap_id : frozen_submaps_id) {
    // Use async with launch::async to launch a new thread for each submap.
    calculate_guess_num_futures.emplace_back(
        std::async(std::launch::async, [frozen_submap_id, &submaps]() {
          Submap* frozen_submap = submaps->getSubmapPtr(frozen_submap_id);
          const int voxel_indices =
              std::pow(frozen_submap->getConfig().voxels_per_side, 3);
          voxblox::BlockIndexList all_block_indices;
          frozen_submap->getTsdfLayer().getAllAllocatedBlocks(
              &all_block_indices);
          int all_block_size = all_block_indices.size();
          return voxel_indices * all_block_size;
        }));
  }

  // Wait for all async calls to finish and accumulate the results.
  for (auto& future : calculate_guess_num_futures) {
    crf_voxel_num_guess += future.get();
  }

  std::set<int> frozen_submaps_semantic_id;
  // get the semantic id
  for (auto& frozen_submap_id : frozen_submaps_id) {
    if (config_.verbosity >= 2) {
      std::cout << "get semantic submap id: " << frozen_submap_id << std::endl;
    }
    Submap* frozen_submap = submaps->getSubmapPtr(frozen_submap_id);
    if (!frozen_submap) {
      LOG(ERROR) << "invalid crf submap id ";
    }
    const int voxel_indices =
        std::pow(frozen_submap->getConfig().voxels_per_side, 3);
    voxblox::BlockIndexList all_block_indices;
    frozen_submap->getTsdfLayer().getAllAllocatedBlocks(&all_block_indices);
    for (const auto& block_index : all_block_indices) {
      if (frozen_submap->hasClassLayer() &&
          frozen_submap->getClassLayer().hasBlock(block_index) &&
          frozen_submap->getTsdfLayer().hasBlock(block_index)) {
        ClassBlock::Ptr class_block =
            frozen_submap->getClassLayerPtr()->getBlockPtrByIndex(block_index);
        TsdfBlock::Ptr tsdf_block =
            frozen_submap->getTsdfLayerPtr()->getBlockPtrByIndex(block_index);
        // access each voxel in this block
        for (int voxel_index = 0; voxel_index < voxel_indices; ++voxel_index) {
          ClassVoxel& class_voxel =
              class_block->getVoxelByLinearIndex(voxel_index);
          TsdfVoxel& tsdf_voxel =
              tsdf_block->getVoxelByLinearIndex(voxel_index);
          if (tsdf_voxel.weight >= 1e-6 && class_voxel.isObserverd()) {
            std::vector<int> this_voxel_semantic_vec =
                class_voxel.getSemanticIDVec();
            frozen_submaps_semantic_id.insert(this_voxel_semantic_vec.begin(),
                                              this_voxel_semantic_vec.end());
          }
        }
      }
    }
  }
  crf_semantic_num_ = frozen_submaps_semantic_id.size();
  if (config_.verbosity >= 2) {
    std::cout << "we willl process the semantic size:  " << crf_semantic_num_
              << std::endl;
    std::cout << "semantic id :";
    for (const auto& semantic_id : frozen_submaps_semantic_id) {
      std::cout << semantic_id << " ";
    }
    std::cout << std::endl;
  }

  LOG_IF(INFO, config_.verbosity >= 4)
      << "crf voxel num guess: " << crf_voxel_num_guess
      << "crf instance num: " << crf_instance_num_
      << "crf semantic num: " << crf_semantic_num_;
  LOG_IF(INFO, config_.verbosity >= 4)
      << "we will create semantic unary mat: " << crf_semantic_num_ << " * "
      << crf_voxel_num_guess << "instance unary mat: " << crf_instance_num_
      << " * " << crf_voxel_num_guess << "rbg_mat: " << 3 << " * "
      << crf_voxel_num_guess << "pose_mat: " << 3 << " * "
      << crf_voxel_num_guess;
  semantic_unary_mat_.resize(crf_semantic_num_, crf_voxel_num_guess);
  instance_unary_mat_.resize(crf_instance_num_, crf_voxel_num_guess);
  rgb_mat_.resize(3, crf_voxel_num_guess);
  pose_mat_.resize(3, crf_voxel_num_guess);

  for (auto& frozen_submap_id : frozen_submaps_id) {
    if (config_.verbosity >= 3) {
      std::cout << "crf submap id: " << frozen_submap_id << std::endl;
    }
    Submap* frozen_submap = submaps->getSubmapPtr(frozen_submap_id);
    if (!frozen_submap) {
      LOG(ERROR) << "invalid crf submap id ";
    }
    std::vector<GlobalIndex> voxel_indices_in_order;
    const int voxel_indices =
        std::pow(frozen_submap->getConfig().voxels_per_side, 3);
    voxblox::BlockIndexList all_block_indices;
    frozen_submap->getTsdfLayer().getAllAllocatedBlocks(&all_block_indices);
    int all_block_size = all_block_indices.size();
    // Create a vector of futures to hold the results of each block thread.
    std::vector<std::future<void>> block_threads;
    // const int num_block_threads = std::thread::hardware_concurrency();
    const int num_block_threads = config_.threads;
    // Divide the blocks among the threads.
    const int blocks_per_thread =
        (all_block_size + num_block_threads - 1) / num_block_threads;

    for (int thread_index = 0; thread_index < num_block_threads;
         ++thread_index) {
      int start_block = thread_index * blocks_per_thread;
      int end_block = std::min(start_block + blocks_per_thread, all_block_size);
      block_threads.push_back(std::async(
          std::launch::async, [this, frozen_submap, submaps, &all_block_indices,
                               start_block, end_block, &voxel_indices_in_order,
                               frozen_submaps_id, frozen_submaps_semantic_id] {
            setCrfProcessBlock(frozen_submap, submaps, all_block_indices,
                               start_block, end_block, voxel_indices_in_order,
                               frozen_submaps_id, frozen_submaps_semantic_id);
          }));
    }

    // Wait for all block threads to finish.
    for (auto& block_thread : block_threads) {
      block_thread.wait();
    }

    voxel_index_map_[frozen_submap_id] = voxel_indices_in_order;
  }

  // create the densecrf for both the instance variable and the semantic
  // variable

  DenseCRF3D instance_crf_3d(crf_voxel_num_actual_, crf_instance_num_);
  DenseCRF3D semantic_crf_3d(crf_voxel_num_actual_, crf_semantic_num_);
  // instance unary
  instance_crf_3d.setUnaryEnergy(
      instance_unary_mat_.leftCols(crf_voxel_num_actual_));
  // semantic unary
  semantic_crf_3d.setUnaryEnergy(
      semantic_unary_mat_.leftCols(crf_voxel_num_actual_));

  // pari-wise paramater
  float smooth_xy_stddev = config_.smooth_xy_stddev;
  float smooth_z_stddev = config_.smooth_z_stddev;
  float smooth_weight = config_.smooth_weight;
  float appear_xy_stddev = config_.appear_xy_stddev;
  float appear_z_stddev = config_.appear_z_stddev;
  float appear_rgb_stddev = config_.appear_rgb_stddev;
  float appear_weight = config_.appear_weight;

  std::cout << smooth_xy_stddev << " " << smooth_z_stddev << " "
            << smooth_weight << " " << appear_xy_stddev << " "
            << appear_z_stddev << " " << appear_rgb_stddev << " "
            << appear_weight << std::endl;

  // instance pair-wise Gaussian and Bilateral
  instance_crf_3d.addPairwiseGaussian(smooth_xy_stddev, smooth_xy_stddev,
                                      smooth_z_stddev,
                                      pose_mat_.leftCols(crf_voxel_num_actual_),
                                      new PottsCompatibility(smooth_weight));
  instance_crf_3d.addPairwiseBilateral(
      appear_xy_stddev, appear_xy_stddev, appear_z_stddev, appear_rgb_stddev,
      appear_rgb_stddev, appear_rgb_stddev,
      pose_mat_.leftCols(crf_voxel_num_actual_),
      rgb_mat_.leftCols(crf_voxel_num_actual_),
      new PottsCompatibility(appear_weight));

  // semantic pair-wise Gaussian and Bilateral
  semantic_crf_3d.addPairwiseGaussian(smooth_xy_stddev, smooth_xy_stddev,
                                      smooth_z_stddev,
                                      pose_mat_.leftCols(crf_voxel_num_actual_),
                                      new PottsCompatibility(smooth_weight));
  semantic_crf_3d.addPairwiseBilateral(
      appear_xy_stddev, appear_xy_stddev, appear_z_stddev, appear_rgb_stddev,
      appear_rgb_stddev, appear_rgb_stddev,
      pose_mat_.leftCols(crf_voxel_num_actual_),
      rgb_mat_.leftCols(crf_voxel_num_actual_),
      new PottsCompatibility(appear_weight));

  // crf optimizer
  int crf_iterations = config_.crf_iterations;
  bool use_high_order = config_.use_high_order;
  instance_crf_3d.set_ho(use_high_order);
  semantic_crf_3d.set_ho(use_high_order);
  Eigen::MatrixXf instance_crf_output_prob;
  Eigen::MatrixXf semantic_crf_output_prob;
  Eigen::VectorXi instance_results;
  Eigen::VectorXi semantic_results;
  instance_crf_output_prob = instance_crf_3d.inference(crf_iterations);
  semantic_crf_output_prob = semantic_crf_3d.inference(crf_iterations);
  instance_results = instance_crf_3d.currentMap(instance_crf_output_prob);
  semantic_results = semantic_crf_3d.currentMap(semantic_crf_output_prob);
  // apply the crf optimize results
  if (config_.verbosity >= 4) {
    for (auto debug_element : voxel_index_map_) {
      int debug_submap_id = debug_element.first;
      int element_size = debug_element.second.size();
      std::cout << submaps->getSubmapPtr(debug_submap_id)->toString()
                << std::endl;
      std::cout << "element size: " << element_size << std::endl;
    }
  }

  // Parallelize the loops using std::async and std::futures.
  for (auto& element : voxel_index_map_) {
    int frozen_submap_id = element.first;
    if (config_.verbosity >= 2) {
      std::cout << "assign the result to submap: " << frozen_submap_id
                << std::endl;
      std::cout << "submap index: " << submap_index_ << std::endl;
    }
    std::vector<GlobalIndex> global_voxel_index_vector = element.second;
    Submap* frozen_submap = submaps->getSubmapPtr(frozen_submap_id);
    if (!frozen_submap) {
      LOG(WARNING) << "failed to access the submap with id: " << frozen_submap;
    }
    std::vector<std::future<void>> assign_crf_results_futures;
    // Divide the global voxel indices among multiple threads.
    const int all_index_size = global_voxel_index_vector.size();
    // const int num_index_threads = std::thread::hardware_concurrency();
    const int num_index_threads = config_.threads;
    const int indices_per_thread =
        (all_index_size + num_index_threads - 1) / num_index_threads;
    for (int thread_index = 0; thread_index < num_index_threads;
         ++thread_index) {
      int start_index = thread_index * indices_per_thread;
      int end_index =
          std::min(start_index + indices_per_thread, all_index_size);
      assign_crf_results_futures.push_back(std::async(
          std::launch::async,
          [this, frozen_submap_id, frozen_submaps_id,
           frozen_submaps_semantic_id, frozen_submap, &instance_crf_output_prob,
           &semantic_crf_output_prob, &instance_results, &semantic_results,
           &global_voxel_index_vector, start_index, end_index]() {
            assignCrfProEachGlobalIndex(
                global_voxel_index_vector, frozen_submap_id, frozen_submaps_id,
                frozen_submaps_semantic_id, frozen_submap,
                instance_crf_output_prob, semantic_crf_output_prob,
                instance_results, semantic_results, start_index, end_index);
          }));
    }

    // Wait for all the futures to finish.
    for (auto& future : assign_crf_results_futures) {
      future.wait();
    }
    submap_index_++;
    cur_map_pro_index_count_ += all_index_size;
  }

  std::cout << blueText << "====crf results====" << endColor << std::endl;
  if (config_.verbosity >= 4) {
    for (const auto outer_entry : instance_label_voxel_change_record_) {
      int origin = outer_entry.first;
      const auto& innermap = outer_entry.second;
      for (const auto& innerentry : innermap) {
        int new_value = innerentry.first;
        int count = innerentry.second;
        std::cout << "origin:" << submaps->getSubmapPtr(origin)->toString()
                  << std::endl;
        std::cout << "new" << submaps->getSubmapPtr(new_value)->toString()
                  << "count" << count << std::endl;
      }
    }
    for (const auto outer_entry : semantic_label_voxel_change_record_) {
      int origin = outer_entry.first;
      const auto& innermap = outer_entry.second;
      for (const auto& innerentry : innermap) {
        int new_value = innerentry.first;
        int count = innerentry.second;
        std::cout << "origin:" << origin << std::endl;
        std::cout << "new" << new_value << "count" << count << std::endl;
      }
    }
  }

  // cleaning
  semantic_unary_mat_.resize(0, 0);
  instance_unary_mat_.resize(0, 0);
  rgb_mat_.resize(0, 0);
  pose_mat_.resize(0, 0);
  cur_voxel_idx_ = 0;
  voxel_index_map_.clear();
  crf_voxel_num_actual_ = 0;
  crf_instance_num_ = 0;
  crf_semantic_num_ = 0;
  voxel_instance_label_changed_num_ = 0;
  voxel_semantic_label_changed_num_ = 0;
  instance_change_voxel_num_ = 0;
  semantic_change_voxel_num_ = 0;
  instance_label_voxel_change_record_.clear();
  semantic_label_voxel_change_record_.clear();
  submap_index_ = 0;
  cur_map_pro_index_count_ = 0;
}

void MapManager::setCrfProcessVoxel(
    Submap* frozen_submap, TsdfBlock::Ptr tsdf_block,
    ClassBlock::Ptr class_block, const int start_voxel, const int end_voxel,
    std::vector<GlobalIndex>& voxel_indices_in_order,
    const std::set<int>& frozen_submaps_id,
    const std::set<int>& frozen_submaps_semantic_id, BlockIndex& block_index) {
  for (int voxel_index = start_voxel; voxel_index < end_voxel; ++voxel_index) {
    ClassVoxel& class_voxel = class_block->getVoxelByLinearIndex(voxel_index);
    TsdfVoxel& tsdf_voxel = tsdf_block->getVoxelByLinearIndex(voxel_index);
    if (tsdf_voxel.weight >= 1e-6 && class_voxel.isObserverd()) {
      {
        std::unique_lock<std::mutex> lock_actual_num(
            crf_voxel_num_actual_mutex_);
        crf_voxel_num_actual_++;
      }
      // instance unary for voxel
      VectorXf_1col voxel_ins_p(crf_instance_num_);
      VectorXf_1col voxel_sem_p(crf_semantic_num_);
      class_voxel.getProbabilityCRFList(frozen_submaps_id, &voxel_ins_p);

      class_voxel.getSemanticCRFList(frozen_submaps_semantic_id, &voxel_sem_p);
      if (config_.verbosity >= 5) {
        std::cout << "voxel_ins_p: " << voxel_ins_p.transpose() << std::endl;
        std::cout << "voxel_sem_p: " << voxel_sem_p.transpose() << std::endl;
        std::cout << class_voxel.printCount();
        std::cout << class_voxel.printSemantic();
      }
      // check the probabilistic from instance
      if (!((voxel_ins_p.array() >= 0).all())) {
        std::cout << "tsdf voxel weight: " << std::setprecision(20)
                  << tsdf_voxel.weight << " distance: " << tsdf_voxel.distance
                  << std::endl;
        std::cout << class_voxel.printCount();
        std::cout << "voxel_ins_p has probabilstic <0 " << std::endl;
        std::cout << "voxel_ins_p: " << std::setprecision(20)
                  << voxel_ins_p.transpose() << std::endl;
        // exit(0);
        continue;
      }
      if ((voxel_ins_p.array() == 0).all()) {
        std::cout << "tsdf voxel weight: " << std::setprecision(20)
                  << tsdf_voxel.weight << " distance: " << tsdf_voxel.distance
                  << std::endl;
        std::cout << class_voxel.printCount();
        std::cout << "voxel_ins_p all = 0" << std::endl;
        std::cout << "voxel_ins_p: " << std::setprecision(20)
                  << voxel_ins_p.transpose() << std::endl;
        // exit(0);
        continue;
      }
      // check the probabilistic from semantic
      if (!((voxel_sem_p.array() >= 0).all())) {
        std::cout << "tsdf voxel weight: " << std::setprecision(20)
                  << tsdf_voxel.weight << " distance: " << tsdf_voxel.distance
                  << std::endl;
        std::cout << class_voxel.printSemantic();
        std::cout << "voxel_sem_phas probabilstic <0 " << std::endl;
        std::cout << "voxel_sem_p: " << std::setprecision(20)
                  << voxel_sem_p.transpose() << std::endl;
        // exit(0);
        continue;
      }
      if ((voxel_sem_p.array() == 0).all()) {
        std::cout << "tsdf voxel weight: " << std::setprecision(20)
                  << tsdf_voxel.weight << " distance: " << tsdf_voxel.distance
                  << std::endl;
        std::cout << class_voxel.printSemantic();
        std::cout << "voxel_sem_p_ all =0" << std::endl;
        std::cout << "voxel_sem_p: " << std::setprecision(20)
                  << voxel_sem_p.transpose() << std::endl;
        // exit(0);
        continue;
      }
      // assign each voxel's instance_unary and check nan
      std::unique_lock<std::mutex> lock_mat(crf_mat_mutex_);
      std::unique_lock<std::mutex> lock_index(crf_cur_voxel_idx_mutex_);
      std::unique_lock<std::mutex> lock_voxel_indices_in_order(
          crf_voxel_indices_in_order_mutex_);
      instance_unary_mat_.col(cur_voxel_idx_) = -(voxel_ins_p.array().log());
      VectorXf_1col instance_prob_unary =
          instance_unary_mat_.col(cur_voxel_idx_);
      if (!is_finite<VectorXf_1col>(instance_prob_unary))
        LOG(WARNING) << "instance unary vector detect nan input"
                     << instance_prob_unary.transpose();
      semantic_unary_mat_.col(cur_voxel_idx_) = -(voxel_sem_p.array().log());
      VectorXf_1col semantic_prob_unary =
          semantic_unary_mat_.col(cur_voxel_idx_);
      if (!is_finite<VectorXf_1col>(semantic_prob_unary))
        LOG(WARNING) << "semantic unary vector detect nan input"
                     << semantic_prob_unary.transpose();
      Eigen::Vector3f rgbvalue(static_cast<float>(tsdf_voxel.color.r) / 255.0f,
                               static_cast<float>(tsdf_voxel.color.g) / 255.0f,
                               static_cast<float>(tsdf_voxel.color.b) / 255.0f);
      rgb_mat_.col(cur_voxel_idx_) = rgbvalue;

      if (!is_finite<Eigen::Vector3f>(rgbvalue))
        LOG(WARNING) << "rbg_mat vector detect nan input"
                     << rgbvalue.transpose();
      Point vertex = tsdf_block->computeCoordinatesFromLinearIndex(voxel_index);
      Eigen::Vector3f posevalue(vertex);
      pose_mat_.col(cur_voxel_idx_) = posevalue;
      if (!is_finite<Eigen::Vector3f>(posevalue))
        LOG(WARNING) << "pose_mat vector detect nan input"
                     << posevalue.transpose();
      VoxelIndex local_voxel_index =
          tsdf_block->computeVoxelIndexFromLinearIndex(voxel_index);
      GlobalIndex global_voxel_index =
          voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
              block_index, local_voxel_index,
              frozen_submap->getConfig().voxels_per_side);
      voxel_indices_in_order.push_back(global_voxel_index);
      cur_voxel_idx_++;
    } else if (tsdf_voxel.weight >= 1e-6 && (!class_voxel.isObserverd())) {
      std::cout << " tsdf distance: " << std::setprecision(20)
                << tsdf_voxel.distance
                << " tsdf weight: " << std::setprecision(20)
                << tsdf_voxel.weight << std::endl;
      LOG(WARNING) << "class voxel is not observed";
    } else if (tsdf_voxel.weight < 1e-6 && (class_voxel.isObserverd())) {
      std::cout << " tsdf distance: " << std::setprecision(20)
                << tsdf_voxel.distance
                << " tsdf weight: " << std::setprecision(20)
                << tsdf_voxel.weight << std::endl;
      std::cout << class_voxel.printCount();
      LOG(WARNING) << "tsdf voxel is not observed";
    }
  }
}

void MapManager::computeCorners(Point corners[8], Point& origin,
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

void MapManager::setCrfProcessBlock(
    Submap* frozen_submap, SubmapCollection* submaps,
    const voxblox::BlockIndexList& block_indices, const int start_block,
    const int end_block, std::vector<GlobalIndex>& voxel_indices_in_order,
    const std::set<int>& frozen_submaps_id,
    const std::set<int>& frozen_submaps_semantic_id) {
  for (int block_index = start_block; block_index < end_block; ++block_index) {
    voxblox::BlockIndex index = block_indices[block_index];
    if (frozen_submap->hasClassLayer() &&
        frozen_submap->getClassLayer().hasBlock(index) &&
        frozen_submap->getTsdfLayer().hasBlock(index)) {
      ClassBlock::Ptr class_block =
          frozen_submap->getClassLayerPtr()->getBlockPtrByIndex(index);
      TsdfBlock::Ptr tsdf_block =
          frozen_submap->getTsdfLayerPtr()->getBlockPtrByIndex(index);
      // access each voxel in this block
      if (!crf_at_the_end_) {
        //<===first check whether it is inside the bounding volume
        Point block_origin = tsdf_block->origin();
        FloatingPoint block_size = frozen_submap->getTsdfLayer().block_size();
        // get the eight corner of the block
        voxblox::Point corners[8];
        computeCorners(&corners[0], block_origin, block_size);
        // Check if at least one corner is within the bounding volume of the
        // other submap.
        bool has_intersection = false;
        for (auto submap_id : accumulate_deactivated_submap_) {
          Submap* other = submaps->getSubmapPtr(submap_id);
          for (int i = 0; i < 8; ++i) {
            const voxblox::Point& corner = corners[i];
            if (other->getBoundingVolume().contains_M(corner)) {
              // If at least one corner is within the bounding volume, add the
              // block index to the list.
              has_intersection = true;
              break;  // Exit the loop early since we've found an intersection.
            }
          }
        }
        if (!has_intersection) {
          // if this block is not within the other submap skip to check the
          // inside voxels
          continue;
        }
      }

      // Create a vector of futures to hold the results of each voxel thread.
      std::vector<std::future<void>> voxel_threads;
      // const size_t num_voxel_threads = std::thread::hardware_concurrency();
      const size_t num_voxel_threads = config_.threads;
      const size_t voxel_indices = tsdf_block->num_voxels();

      // Divide the voxels among the threads.
      const size_t voxels_per_thread =
          (voxel_indices + num_voxel_threads - 1) / num_voxel_threads;

      for (size_t voxel_thread_index = 0;
           voxel_thread_index < num_voxel_threads; ++voxel_thread_index) {
        size_t start_voxel = voxel_thread_index * voxels_per_thread;
        size_t end_voxel =
            std::min(start_voxel + voxels_per_thread, voxel_indices);
        voxel_threads.push_back(std::async(
            std::launch::async,
            [this, frozen_submap, tsdf_block, class_block, start_voxel,
             end_voxel, &voxel_indices_in_order, frozen_submaps_id,
             frozen_submaps_semantic_id, &index] {
              setCrfProcessVoxel(frozen_submap, tsdf_block, class_block,
                                 start_voxel, end_voxel, voxel_indices_in_order,
                                 frozen_submaps_id, frozen_submaps_semantic_id,
                                 index);
            }));
      }
      // Wait for all voxel threads to finish.
      for (auto& voxel_thread : voxel_threads) {
        voxel_thread.wait();
      }
    } else {
      LOG(WARNING) << "when process crf access the block failed";
    }
  }
}

void MapManager::assignCrfProEachGlobalIndex(
    const std::vector<GlobalIndex>& global_voxel_index_vector,
    int frozen_submap_id, const std::set<int>& frozen_submaps_id,
    const std::set<int>& frozen_submaps_semantic_id, Submap* frozen_submap,
    const Eigen::MatrixXf& instance_crf_output_prob,
    const Eigen::MatrixXf& semantic_crf_output_prob,
    const Eigen::VectorXi& instance_results,
    const Eigen::VectorXi& semantic_results, int start_index, int end_index) {
  for (int i = start_index; i < end_index; ++i) {
    int pro_index = i + cur_map_pro_index_count_;
    const GlobalIndex& global_voxel_index = global_voxel_index_vector[i];
    BlockIndex block_index = voxblox::getBlockIndexFromGlobalVoxelIndex(
        global_voxel_index, 1.0 / frozen_submap->getConfig().voxels_per_side);

    if (frozen_submap->hasClassLayer() &&
        frozen_submap->getClassLayer().hasBlock(block_index) &&
        frozen_submap->getTsdfLayer().hasBlock(block_index)) {
      auto class_layer = frozen_submap->getClassLayerPtr();
      auto tsdf_layer = frozen_submap->getTsdfLayerPtr();
      ClassBlock::Ptr class_block =
          frozen_submap->getClassLayerPtr()->getBlockPtrByIndex(block_index);
      TsdfBlock::Ptr tsdf_block =
          frozen_submap->getTsdfLayerPtr()->getBlockPtrByIndex(block_index);
      Eigen::VectorXf instance_crf_pro_result =
          instance_crf_output_prob.col(pro_index);
      Eigen::VectorXf semantic_crf_pro_result =
          semantic_crf_output_prob.col(pro_index);
      int new_max_instance_label = instance_results(pro_index);
      int new_max_semantic_label = semantic_results(pro_index);
      if (config_.verbosity >= 5) {
        std::cout << "instance_crf_pro_result: "
                  << instance_crf_pro_result.transpose() << std::endl;
        std::cout << "semantic_crf_pro_result: "
                  << semantic_crf_pro_result.transpose() << std::endl;
      }
      float sum_instance_crf_pro = instance_crf_pro_result.sum();
      float sum_semantic_crf_pro = semantic_crf_pro_result.sum();
      if (!is_finite<VectorXf_1col>(instance_crf_pro_result))
        LOG(WARNING) << "instance_crf_pro_result detect nan result"
                     << instance_crf_pro_result.transpose();
      if (fabs(sum_instance_crf_pro - 1.0) > 0.01) {
        LOG(WARNING) << "sum of the instance crf pro results: "
                     << sum_instance_crf_pro;
      }
      if (!is_finite<VectorXf_1col>(semantic_crf_pro_result))
        LOG(WARNING) << "semantic_crf_pro_result detect nan result"
                     << semantic_crf_pro_result.transpose();
      if (fabs(sum_semantic_crf_pro - 1.0) > 0.01) {
        LOG(WARNING) << "sum of the semantic crf pro results: "
                     << sum_semantic_crf_pro;
      }
      ClassVoxel* class_voxel =
          class_layer->getVoxelPtrByGlobalIndex(global_voxel_index);
      TsdfVoxel* tsdf_voxel =
          tsdf_layer->getVoxelPtrByGlobalIndex(global_voxel_index);
      if (tsdf_voxel->weight >= 1e-6 && class_voxel->isObserverd()) {
        int old_max_instance_label = class_voxel->getBelongingID();
        int old_max_semantic_label = class_voxel->getClassId();
        if (old_max_instance_label != 0) {
          LOG(WARNING) << "find error old instance voxel label: "
                       << old_max_instance_label;
        }
        if ((old_max_semantic_label == 0) || (old_max_semantic_label == 1)) {
          LOG(WARNING) << "error old_max_semantic_label: "
                       << old_max_semantic_label;
        }
        std::string old_class_voxel_info = class_voxel->printCount();
        std::stringstream old_class_info_ss(old_class_voxel_info);
        std::string old_semantic_voxel_info = class_voxel->printSemantic();
        std::stringstream old_semantic_info_ss(old_semantic_voxel_info);
        float old_total_count = class_voxel->getTotalCount();
        float old_total_semantic_count = class_voxel->getSemanticTotalCount();
        float new_total_count = 0;
        float new_semantic_total_count = 0;

        // The key point to note here is that crf_pro_result starts from index 0 in eigen,\
        // while the actual label starts from index 1.
        for (int label_id = 0; label_id < instance_crf_pro_result.size();
             ++label_id) {
          // std::set index start 0
          // It is important to note here that `label_id` is not the same as our `frozen_submap_id`,
          // meaning it is not the value of the instance variable we aim to estimate and requires conversion.
          std::set<int>::iterator submap_itr =
              std::next(frozen_submaps_id.begin(), label_id);
          int instance_id = *submap_itr;
          float new_label_id_count =
              instance_crf_pro_result(label_id) * old_total_count;
          // avoid to small value
          if (new_label_id_count < 1e-6 && new_label_id_count != 0) {
            new_label_id_count = 1e-6;
          }
          new_total_count = new_total_count + new_label_id_count;
          class_voxel->setCount(instance_id, new_label_id_count);
        }
        // assign semantic crf pro result
        for (int label_id = 0; label_id < semantic_crf_pro_result.size();
             ++label_id) {
        // It is important to note the number and definition of semantic classes
        // in different datasets, as well as the location of unknown data.
          std::set<int>::iterator class_itr =
              std::next(frozen_submaps_semantic_id.begin(), label_id);
          int semantic_id = *class_itr;
          float new_label_id_semantic_count =
              semantic_crf_pro_result(label_id) * old_total_semantic_count;
          // avoid to small value
          if (new_label_id_semantic_count < 1e-6 &&
              new_label_id_semantic_count != 0) {
            new_label_id_semantic_count = 1e-6;
          }
          new_semantic_total_count =
              new_semantic_total_count + new_label_id_semantic_count;
          class_voxel->setSemanticCount(semantic_id,
                                        new_label_id_semantic_count);
        }

        if (config_.verbosity >= 5) {
          std::cout << "aft assign crf pro result new class voxel count info:"
                    << std::endl;
          std::cout << class_voxel->printCount();
          std::cout << class_voxel->printSemantic();
        }
        bool change_label = false;
        bool change_semantic_class_id = false;
        // This is mainly to address the issue of distinguishing equal probabilities,
        // such as 0.5 and 0.5.
       if (instance_crf_pro_result(new_max_instance_label) >
            instance_crf_pro_result(submap_index_)) {
          if (config_.verbosity >= 5) {
            std::cout << "change label, new: " << new_max_instance_label
                      << " origin: " << submap_index_ << std::endl;
            std::cout << "old class info: \n " << old_class_info_ss.str()
                      << std::endl;
            std::cout << "instance crf pro results: \n"
                      << instance_crf_pro_result.transpose() << std::endl;
          }
          change_label = true;
        }
        int semantic_index;
        auto it =
            std::find(frozen_submaps_semantic_id.begin(),
                      frozen_submaps_semantic_id.end(), old_max_semantic_label);
        if (it != frozen_submaps_semantic_id.end()) {
          semantic_index =
              std::distance(frozen_submaps_semantic_id.begin(), it);
        } else {
          std::cout << "Element " << old_max_semantic_label
                    << " not found in the set" << std::endl;
        }
        if (semantic_crf_pro_result(new_max_semantic_label) >
            semantic_crf_pro_result(semantic_index)) {
          change_semantic_class_id = true;
          if (config_.verbosity >= 5) {
            std::cout << "change semantic class id, new: "
                      << new_max_semantic_label << " , old "
                      << old_max_semantic_label << std::endl;
            std::cout << "crf result for this voxel : "
                      << semantic_crf_pro_result.transpose() << std::endl;
          }
        } else {
          if (config_.verbosity >= 5) {
            std::cout << "not change semantic class_id, new: "
                      << new_max_semantic_label << " , old "
                      << old_max_semantic_label << std::endl;
            std::cout << "crf result for this voxel : "
                      << semantic_crf_pro_result.transpose() << std::endl;
          }
        }
        if (!change_label) {
          // this voxel do not need to change to another submap
          // we set the current index = 0 to indicate that this voxel is
          // still belonging to this submap
          float new_max_current_count =
              instance_crf_pro_result(submap_index_) * old_total_count;
          class_voxel->setAfterCrf(0, new_max_current_count, new_total_count);
        } else {
          // this voxel need further management to be delete from this
          // submap and assign it to the submap with id new_max_count_id
          float new_max_current_count =
              instance_crf_pro_result(new_max_instance_label) * old_total_count;
          std::set<int>::iterator submap_itr =
              std::next(frozen_submaps_id.begin(), new_max_instance_label);
          int new_max_count_id = *submap_itr;
          class_voxel->setAfterCrf(new_max_count_id, new_max_current_count,
                                   new_total_count);
          std::lock_guard<std::mutex> lock_instance_label_voxel_change_record(
              crf_instance_label_voxel_change_record_mutex_);
          if (instance_label_voxel_change_record_.count(frozen_submap_id) > 0 &&
              instance_label_voxel_change_record_[frozen_submap_id].count(
                  new_max_count_id) > 0) {
            instance_label_voxel_change_record_[frozen_submap_id]
                                               [new_max_count_id]++;
            if (config_.verbosity >= 5) {
              std::cout << "change voxel with new prob: "
                        << instance_crf_pro_result.transpose() << std::endl;
              std::cout << "old class info: " << old_class_info_ss.str()
                        << std::endl;
            }
          } else {
            instance_label_voxel_change_record_[frozen_submap_id]
                                               [new_max_count_id] = 1;
          }
          std::lock_guard<std::mutex> lock_instance_change_voxel_num(
              crf_instance_change_voxel_num_mutex_);
          instance_change_voxel_num_++;
        }
        // the semantic class id is changed
        if (change_semantic_class_id) {
          // set semantic
          float new_max_current_semantic_count =
              semantic_crf_pro_result(new_max_semantic_label) *
              old_total_semantic_count;
          std::set<int>::iterator class_itr = std::next(
              frozen_submaps_semantic_id.begin(), new_max_semantic_label);
          int new_max_count_semantic_id = *class_itr;
          class_voxel->setSemanticAfterCrf(new_max_count_semantic_id,
                                           new_max_current_semantic_count,
                                           new_semantic_total_count);
          std::lock_guard<std::mutex> lock_semantic_label_voxel_change_record(
              crf_semantic_label_voxel_change_record_mutex_);
          if (semantic_label_voxel_change_record_.count(
                  old_max_semantic_label) > 0 &&
              semantic_label_voxel_change_record_[old_max_semantic_label].count(
                  new_max_count_semantic_id) > 0) {
            semantic_label_voxel_change_record_[old_max_semantic_label]
                                               [new_max_count_semantic_id]++;
          } else {
            semantic_label_voxel_change_record_[old_max_semantic_label]
                                               [new_max_count_semantic_id] = 1;
          }
          std::lock_guard<std::mutex> lock_semantic_change_voxel_num(
              crf_semantic_change_voxel_num_mutex_);
          semantic_change_voxel_num_++;

          if (config_.verbosity >= 5) {
            std::cout << " semantic class id change " << std::endl;
            std::cout << " semantic pro result: "
                      << semantic_crf_pro_result.transpose() << std::endl;
            std::cout << "old class info: " << old_class_info_ss.str()
                      << std::endl;
            std::cout << "old semantic info: " << old_semantic_info_ss.str()
                      << std::endl;
            std::cout << "new class info: " << class_voxel->printCount()
                      << " \n " << class_voxel->printSemantic() << std::endl;
          }
        } else {
          float new_max_current_semantic_count =
              semantic_crf_pro_result(semantic_index) *
              old_total_semantic_count;
          class_voxel->setSemanticAfterCrf(old_max_semantic_label,
                                           new_max_current_semantic_count,
                                           new_semantic_total_count);
          if (config_.verbosity >= 5) {
            std::cout << "semantic class id preserved" << std::endl;
            std::cout << " semantic pro result: "
                      << semantic_crf_pro_result.transpose() << std::endl;
            std::cout << "old semantic info: " << old_semantic_info_ss.str()
                      << std::endl;
            std::cout << "old class info: " << old_class_info_ss.str()
                      << std::endl;
            std::cout << "new class info: " << class_voxel->printCount()
                      << " \n " << class_voxel->printSemantic() << std::endl;
          }
        }
      } else {
        LOG(WARNING) << "voxel index error!";
      }
    } else {
      LOG(WARNING) << "invalid block index when try to assign crf results";
      exit(0);
    }
  }
}

void MapManager::processVoxels(size_t start_voxel, size_t end_voxel,
                               Submap* frozen_submap, TsdfBlock::Ptr tsdf_block,
                               ClassBlock::Ptr class_block,
                               SubmapCollection* submaps,
                               std::atomic<int>& change_count) {
  for (size_t i = start_voxel; i < end_voxel; ++i) {
    TsdfVoxel& tsdf_voxel = tsdf_block->getVoxelByLinearIndex(i);
    ClassVoxel& class_voxel = class_block->getVoxelByLinearIndex(i);

    if (tsdf_voxel.weight <= 1.0e-6 || !(class_voxel.isObserverd())) {
      if (!class_voxel.belongsToSubmap()) {
        std::cout << class_voxel.printCount();
      }
      continue;
    }

    if (!class_voxel.belongsToSubmap()) {
      change_count++;
      int belonging_id = class_voxel.getBelongingID();
      LOG_IF(INFO, config_.verbosity >= 5)
          << " map id: " << frozen_submap->getID()
          << "class voxel set to belong of other submap " << belonging_id
          << std::endl;
      Submap* belonging_submap = submaps->getSubmapPtr(belonging_id);
      Transformation T_B_R =
          belonging_submap->getT_S_M() * frozen_submap->getT_M_S();
      const Point voxel_center =
          T_B_R * tsdf_block->computeCoordinatesFromLinearIndex(i);

      if (belonging_submap) {
        TsdfBlock::Ptr belonging_block =
            belonging_submap->getTsdfLayerPtr()->allocateBlockPtrByCoordinates(
                voxel_center);
        belonging_block->setUpdatedAll();
        ClassBlock::Ptr belonging_class_block =
            belonging_submap->getClassLayerPtr()->allocateBlockPtrByCoordinates(
                voxel_center);
        ClassVoxel* belonging_class_voxel =
            belonging_submap->getClassLayerPtr()->getVoxelPtrByCoordinates(
                voxel_center);
        if (!belonging_class_voxel) {
          LOG(WARNING) << "belonging_class_voxel invalid";
        }
        TsdfVoxel* belonging_voxel =
            belonging_submap->getTsdfLayerPtr()->getVoxelPtrByCoordinates(
                voxel_center);
        if (!belonging_voxel) {
          LOG(WARNING) << "belonging_voxel invalid";
        }
        if (belonging_class_voxel && belonging_voxel) {
          if (config_.verbosity >= 5) {
            std::cout << " belonging_class_voxel original value: " << std::endl;
            std::cout << belonging_class_voxel->printCount();
            std::cout << belonging_class_voxel->printSemantic();
          }
          belonging_class_voxel->mergeVoxel(class_voxel);
          belonging_class_voxel->setAfterMerge(
              belonging_class_voxel->getCurrentCount());
          voxblox::mergeVoxelAIntoVoxelB(tsdf_voxel, belonging_voxel);
          const int class_id = belonging_class_voxel->getClassId();
          Color voxel_color = id_color_map_.colorLookup(class_id);
          belonging_voxel->vis_color.r = voxel_color.r;
          belonging_voxel->vis_color.b = voxel_color.b;
          belonging_voxel->vis_color.g = voxel_color.g;
          // clear classlayer voxel value
          class_voxel.clearCount();
          // clear tsdflayer voxel value
          tsdf_voxel = TsdfVoxel();
          tsdf_block->setUpdatedAll();
          if (config_.verbosity >= 5) {
            std::cout << "after set belonging_class_voxel value: " << std::endl;
            std::cout << belonging_class_voxel->printCount();
            std::cout << belonging_class_voxel->printSemantic();
            std::cout << "after set class_voxel value: " << std::endl;
            std::cout << class_voxel.printCount();
            std::cout << class_voxel.printSemantic();
          }
        } else {
          LOG(WARNING) << "get belonging tsdf or class voxel failed";
        }
      } else {
        LOG(WARNING) << "invalid belonging submap id";
      }
    }
  }
}

void MapManager::processBlock(Submap* frozen_submap, SubmapCollection* submaps,
                              size_t start_block, size_t end_block,
                              const voxblox::BlockIndexList& block_indices,
                              std::atomic<int>& change_count) {
  for (size_t index = start_block; index < end_block; ++index) {
    const auto& block_index = block_indices[index];

    TsdfBlock::Ptr tsdf_block =
        frozen_submap->getTsdfLayerPtr()->getBlockPtrByIndex(block_index);
    ClassBlock::Ptr class_block =
        frozen_submap->getClassLayerPtr()->getBlockPtrByIndex(block_index);
    if (!class_block || !tsdf_block) {
      continue;
    }

    std::vector<std::future<void>> voxel_threads;
    const size_t num_voxel_threads = std::thread::hardware_concurrency();
    const size_t voxel_indices = tsdf_block->num_voxels();
    const size_t voxels_per_thread =
        (voxel_indices + num_voxel_threads - 1) / num_voxel_threads;

    for (size_t voxel_thread_index = 0; voxel_thread_index < num_voxel_threads;
         ++voxel_thread_index) {
      size_t start_voxel = voxel_thread_index * voxels_per_thread;
      size_t end_voxel =
          std::min(start_voxel + voxels_per_thread, voxel_indices);
      voxel_threads.push_back(std::async(
          std::launch::async, [this, frozen_submap, &tsdf_block, &class_block,
                               start_voxel, end_voxel, submaps, &change_count] {
            processVoxels(start_voxel, end_voxel, frozen_submap, tsdf_block,
                          class_block, submaps, change_count);
          }));
    }

    // Wait for all threads to finish
    for (auto& future : voxel_threads) {
      future.wait();
    }
  }
}

void MapManager::setBelongingsMultiThread(Submap* frozen_submap,
                                          SubmapCollection* submaps) {
  LOG_IF(INFO, config_.verbosity >= 5) << "set belongings multi-thread";
  auto tsdf_layer = frozen_submap->getTsdfLayerPtr();
  std::atomic<int> change_count(0);
  // Parse the tsdf layer.
  voxblox::BlockIndexList block_indices;
  tsdf_layer->getAllAllocatedBlocks(&block_indices);

  std::vector<std::future<void>> block_threads;
  const size_t num_block_threads = std::thread::hardware_concurrency();
  const size_t block_size = block_indices.size();
  const size_t blocks_per_thread =
      (block_size + num_block_threads - 1) / num_block_threads;

  for (size_t thread_index = 0; thread_index < num_block_threads;
       ++thread_index) {
    size_t start_block = thread_index * blocks_per_thread;
    size_t end_block = std::min(start_block + blocks_per_thread, block_size);
    block_threads.push_back(std::async(
        std::launch::async, [this, frozen_submap, submaps, &block_indices,
                             start_block, end_block, &change_count] {
          processBlock(frozen_submap, submaps, start_block, end_block,
                       block_indices, change_count);
        }));
  }

  // Wait for all threads to finish
  for (auto& future : block_threads) {
    future.wait();
  }

}



void MapManager::mergeSubmapVoxelInfo(std::set<int>& frozen_submaps_id,
                                      SubmapCollection* submaps) {
  LOG_IF(INFO, config_.verbosity >= 4) << "Merge Submap Voxel Info";
  for (auto it = frozen_submaps_id.begin(); it != frozen_submaps_id.end();) {
    auto this_id = *it;
    it = frozen_submaps_id.erase(it);
    if (!submaps->submapIdExists(this_id)) {
      LOG(WARNING) << "try to access invalid submap with deactived id: "
                   << this_id;
      continue;
    }
    Submap* this_submap = submaps->getSubmapPtr(this_id);
    if (this_submap->isActive() &&
        this_submap->getLabel() != PanopticLabel::kBackground) {
      // Active submaps need first to be de-activated.
      LOG_IF(INFO, config_.verbosity >= 4)
          << "Active submaps " << this_id << " need first to be de-activated.";
      this_submap->finishActivePeriod();
    }
    // Find all potential matches.
    for (const auto& other_id : frozen_submaps_id) {
      if (!submaps->submapIdExists(other_id)) {
        LOG(WARNING) << "try to access invalid submap with other id: "
                     << other_id;
        continue;
      }
      Submap* other_submap = submaps->getSubmapPtr(other_id);
      // only manage submap pairs whose bounding volume has intersections
      if (!this_submap->getBoundingVolume().intersects(
              other_submap->getBoundingVolume())) {
        continue;
      }
      LOG_IF(INFO, config_.verbosity >= 5)
          << "check same position with " << this_id << "and " << other_id;
      tsdf_registrator_->samePositionMergeMultiThread(this_submap,
                                                      other_submap);
    }
  }
}

void MapManager::performChangeDetection(SubmapCollection* submaps) {
  tsdf_registrator_->checkSubmapCollectionForChange(submaps);
}

void MapManager::finishMapping(SubmapCollection* submaps) {
  // Remove all empty blocks.
  std::stringstream info;
  std::set<int> frozen_submaps_id;
  info << "Finished mapping: ";
  for (Submap& submap : *submaps) {
    info << pruneBlocks(&submap);
  }
  LOG_IF(INFO, config_.verbosity >= 1) << info.str();

  // Deactivate last submaps.
  for (Submap& submap : *submaps) {
    if (submap.isActive()) {
      LOG_IF(INFO, config_.verbosity >= 1)
          << "Deactivating submap " << submap.getID();
      submap.finishActivePeriod();
    }
    frozen_submaps_id.insert(submap.getID());
  }

  // Try to merge the submaps.
  // NOTE(thuaj): frozen_submaps_id is passed by reference and any changes made
  // to it inside the function will only affect the reference within the
  // function scope
  // since we generate the gt we do not merge voxel
  if(config_.voxel_class_management_frequency>0) {
  std::set<int> frozen_submaps_id_copy = frozen_submaps_id;
  mergeSubmapVoxelInfo(frozen_submaps_id_copy, submaps);
  if (config_.verbosity >= 1) {
    std::cout << "when finishing mapping merge frozen submaps:";
    for (const auto& submap_id : frozen_submaps_id) {
      Submap* submap = submaps->getSubmapPtr(submap_id);
      std::cout << submap_id << " " << submap->toString() << std::endl;
    }
    std::cout << "\n";
  }

  std::stringstream remove_info;
  std::vector<int> submaps_to_remove;
  for (const auto& frozen_id : frozen_submaps_id) {
    LOG_IF(INFO, config_.verbosity >= 4)
        << "check submap " << frozen_id
        << " voxels and set to belonging voxels ";
    Submap* frozen_submap = submaps->getSubmapPtr(frozen_id);
    setBelongingsMultiThread(frozen_submap, submaps);
    frozen_submap->updateEverything(false);
    if (frozen_submap->getMeshLayer().getNumberOfAllocatedMeshes() == 0) {
      submaps_to_remove.emplace_back(frozen_submap->getID());
      if (config_.verbosity >= 1) {
        remove_info << "Removed Submap: " << frozen_submap->getID() << " \n";
      }
    }
    // Remove submaps.
    for (int id : submaps_to_remove) {
      submaps->removeSubmap(id);
    }
  }
  LOG_IF(INFO, config_.verbosity >= 1)
      << "Check mesh layer " << remove_info.str();
  }

  for (const auto& frozen_id : frozen_submaps_id) {
    if (frozen_id == 0) continue;
    Submap* submap = submaps->getSubmapPtr(frozen_id);
    if (config_.verbosity >= 1) {
      std::cout << "submap info: " << submap->toString() << std::endl;
    }
    voxblox::BlockIndexList all_class_block_indices;
    submap->getClassLayerPtr()->getAllAllocatedBlocks(&all_class_block_indices);
    // int all_class_block_indices_size = all_class_block_indices.size();
    const int voxel_indices = std::pow(submap->getConfig().voxels_per_side, 3);
    std::map<int, int> semantic_count_records;  // <semantic_id, count>
    for (const auto& block_index : all_class_block_indices) {
      ClassBlock::Ptr class_block =
          submap->getClassLayerPtr()->getBlockPtrByIndex(block_index);
      for (int voxel_index = 0; voxel_index < voxel_indices; ++voxel_index) {
        ClassVoxel& class_voxel =
            class_block->getVoxelByLinearIndex(voxel_index);
        if (class_voxel.isObserverd()) {
          int semantic_id = class_voxel.getClassId();
          semantic_count_records[semantic_id]++;
        }
      }
    }
    // Output the contents of the map
    // for (const auto& record : semantic_count_records) {
    //   std::cout << "Semantic ID " << record.first << ": " << record.second
    //             << " counts" << std::endl;
    // }
    // Create a map to store the max count for each semantic ID
    int max_count = semantic_count_records[submap->getClassID()];
    int max_semantic_id = submap->getClassID();
    // Iterate over the key-value pairs in the map
    for (const auto& record : semantic_count_records) {
      int semantic_id = record.first;
      int count = record.second;

      // Update the max count for the semantic ID
      if (count > max_count) {
        max_count = count;
        max_semantic_id = semantic_id;
      }
    }
    submap->setClassID(max_semantic_id);

    // std::cout << "max semantic id is : " << max_semantic_id << std::endl;
    // std::cout << "max count is : " << max_count << std::endl;
    // std::cout << "submap info aft set max semantic id: " << submap->toString()
    //           << std::endl;

  }
}

bool MapManager::mergeSubmapIfPossible(SubmapCollection* submaps, int submap_id,
                                       int* merged_id) {
  // Use on inactive submaps, checks for possible matches with other inactive
  // submaps.
  if (!submaps->submapIdExists(submap_id)) {
    return false;
  }

  // Setup.
  Submap* submap = submaps->getSubmapPtr(submap_id);
  if (submap->isActive()) {
    // Active submaps need first to be de-activated.
    submap->finishActivePeriod();
  } else if (submap->getChangeState() == ChangeState::kAbsent) {
    return false;
  }

  // Find all potential matches.
  for (Submap& other : *submaps) {
    if (other.isActive() || other.getClassID() != submap->getClassID() ||
        other.getID() == submap->getID() ||
        !submap->getBoundingVolume().intersects(other.getBoundingVolume())) {
      continue;
    }

    bool submaps_match;
    if (!tsdf_registrator_->submapsConflict(*submap, other, &submaps_match)) {
      if (submaps_match) {
        // It's a match, merge the submap into the candidate.

        // Make sure both maps have or don't have class layers.
        if (!(submap->hasClassLayer() && other.hasClassLayer())) {
          submap->applyClassLayer(*layer_manipulator_);
          other.applyClassLayer(*layer_manipulator_);
        }
        layer_manipulator_->mergeSubmapAintoB(*submap, &other);
        LOG_IF(INFO, config_.verbosity >= 4)
            << "Merged Submap " << submap->getID() << " into " << other.getID()
            << ".";
        other.setChangeState(ChangeState::kPersistent);
        submaps->removeSubmap(submap_id);
        if (merged_id) {
          *merged_id = other.getID();
        }
        return true;
      }
    }
  }
  return false;
}

std::string MapManager::pruneBlocks(Submap* submap) const {
  auto t1 = std::chrono::high_resolution_clock::now();
  // Setup.
  ClassLayer* class_layer = nullptr;
  if (submap->hasClassLayer()) {
    class_layer = submap->getClassLayerPtr().get();
  }
  ScoreLayer* score_layer = nullptr;
  if (submap->hasScoreLayer()) {
    score_layer = submap->getScoreLayerPtr().get();
  }
  TsdfLayer* tsdf_layer = submap->getTsdfLayerPtr().get();
  MeshLayer* mesh_layer = submap->getMeshLayerPtr().get();
  const int voxel_indices = std::pow(submap->getConfig().voxels_per_side, 3);
  int count = 0;

  // Remove all blocks that don't have any belonging voxels.
  voxblox::BlockIndexList block_indices;
  tsdf_layer->getAllAllocatedBlocks(&block_indices);
  for (const auto& block_index : block_indices) {
    ClassBlock::Ptr class_block;
    if (class_layer) {
      if (class_layer->hasBlock(block_index)) {
        class_block = class_layer->getBlockPtrByIndex(block_index);
      }
    }
    const TsdfBlock& tsdf_block = tsdf_layer->getBlockByIndex(block_index);
    bool has_beloning_voxels = false;

    // Check all voxels.
    for (int voxel_index = 0; voxel_index < voxel_indices; ++voxel_index) {
      if (tsdf_block.getVoxelByLinearIndex(voxel_index).weight >= 1e-6) {
        if (class_block) {
          if (class_block->getVoxelByLinearIndex(voxel_index)
                  .belongsToSubmap()) {
            has_beloning_voxels = true;
            break;
          }
        } else {
          has_beloning_voxels = true;
          break;
        }
      }
    }

    // Prune blocks.
    if (!has_beloning_voxels) {
      if (class_layer) {
        class_layer->removeBlock(block_index);
      }
      if (score_layer) {
        score_layer->removeBlock(block_index);
      }
      tsdf_layer->removeBlock(block_index);
      mesh_layer->removeMesh(block_index);
      count++;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::stringstream ss;
  if (count > 0 && config_.verbosity >= 1) {
    ss << "\nPruned " << count << " blocks from submap " << submap->getID()
       << " (" << submap->getName() << ") in "
       << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
       << "ms.";
  }
  return ss.str();
}

void MapManager::Ticker::tick(SubmapCollection* submaps) {
  // Perform 'action' every 'max_ticks' ticks.
  current_tick_++;
  if (current_tick_ >= max_ticks_) {
    action_(submaps);
    current_tick_ = 0;
  }
}

}  // namespace panoptic_mapping
