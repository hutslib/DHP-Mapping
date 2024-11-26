#include "panoptic_mapping/integration/raytracing_tsdf_integrator.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pcl/io/ply_io.h>
#include <voxblox/integrator/merge_integration.h>

#include "panoptic_mapping/common/index_getter.h"

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<
    TsdfIntegratorBase, RaytracingTsdfIntegrator, std::shared_ptr<Globals>>
    RaytracingTsdfIntegrator::registration_("raytracing_tsdf");

void RaytracingTsdfIntegrator::Config::checkParams() const {
}

void RaytracingTsdfIntegrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("integrator_threads", &integrator_threads);
  setupParam("global_voxblox_integrator_config",
             &global_voxblox_integrator_config, "global_voxblox_integrator");
  setupParam("use_kitti", &use_kitti);
}

RaytracingTsdfIntegrator::RaytracingTsdfIntegrator(
    const Config& config, std::shared_ptr<Globals> globals)
    : config_(config.checkValid()), TsdfIntegratorBase(std::move(globals)) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
  // Setup all needed inputs.
  setRequiredInputs({InputData::InputType::kSegments});
  frame_ = 0;
}

void RaytracingTsdfIntegrator::processInput(SubmapCollection* submaps,
                                            InputData* input) {

  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(globals_->camera().get());
  CHECK(inputIsValid(*input));
  LOG_IF(INFO, config_.verbosity >= 2) << "\n raytracing tsdf integrator";

  std::vector<int> submap_id_list = input->submapIDList();
  if (config_.verbosity >= 5) {
    std::cout << "current segments will be integrate into submaps with id: ";
    for (int id : submap_id_list) {
      std::cout << id << " ";
    }
    std::cout << "\n";
  }

  SubmapIndexGetter index_getter(submap_id_list);
  std::vector<std::future<void>> threads;
  // NOTE(thuaj): do not consider free space voxel do the 3d reconstruction only
  for (int i = 0; i < config_.integrator_threads; ++i) {
    threads.emplace_back(std::async(std::launch::async, [this, &index_getter,
                                                         input, submaps, i]() {
      int index;
      while (index_getter.getNextIndex(&index)) {
        // NOTE(thuaj): a std::vector start from 0 however index_getter give
        // index start from 1 Segment segment =
        // input->segmentsPtr()->at(index-1); Submap* map =
        // submaps.getSubmapPtr(index);
        if(!submaps->submapIdExists(index)) {
          std::cout<<"we do not have submap id : "<<index<<std::endl;
          continue;
        }
        this->processSubmap(input->segmentsPtr(), submaps->getSubmapPtr(index));
      }
    }));
  }

  // Join all threads.
  for (auto& thread : threads) {
    thread.get();
  }

  frame_++;
}

bool RaytracingTsdfIntegrator::create_directory_if_not_exists(
    const std::string& folder_path) {
  // Check if the folder exists
  if (!std::filesystem::exists(folder_path)) {
    // If it doesn't exist, try to create it
    if (std::filesystem::create_directories(folder_path)) {
      std::cout << "Folder created: " << folder_path << std::endl;
      return true;
    } else {
      std::cerr << "Error creating folder: " << folder_path << std::endl;
      return false;
    }
  } else {
    // If it exists, return true
    std::cout << "Folder already exists: " << folder_path << std::endl;
    return true;
  }
}

void RaytracingTsdfIntegrator::processSubmap(Segments* segments,
                                             Submap* submap) {
  //   std::unique_lock<std::mutex> lock(mtx);
  if (config_.verbosity >= 4) {
    std::cout << yellowHighlight << "integrator map id: " << submap->getID()
              << endColor << std::endl;
  }
  if (!submap->hasClassLayer()) {
    LOG(WARNING) << "do not has class layer !!!";
  }
  // get the segments to integrate for this submap
  Segments this_submap_segments;
  {
    std::lock_guard<std::mutex> lock(segments_mutex_);
    for (auto& segment : *segments) {
      if (segment.getSubmapID() == submap->getID()) {
        this_submap_segments.push_back(segment);
      }
    }
  }

  int segmentation_idx = 0;

  for (auto& segment : this_submap_segments) {
    // merge all the points colors and labels
    Pointcloud all_seg_points;
    Colors all_seg_colors;
    Labels all_seg_labels;
    segmentation_idx++;
    all_seg_points.insert(all_seg_points.end(), segment.points().begin(),
                          segment.points().end());
    all_seg_colors.insert(all_seg_colors.end(), segment.colors().begin(),
                          segment.colors().end());
    all_seg_labels.insert(all_seg_labels.end(), segment.labels().begin(),
                          segment.labels().end());

  Transformation T_S_C =
      submap->getT_M_S().inverse() * this_submap_segments[0].getT();

  Transformation T_C_S = T_S_C.inverse();
  if (submap_geometry_integrtor_ptr_.find(submap->getID()) !=
      submap_geometry_integrtor_ptr_.end()) {
    std::lock_guard<std::mutex> lock(integrator_mutex_);
    submap_geometry_integrtor_ptr_.at(submap->getID())
        ->processIntegration(T_S_C, all_seg_points, all_seg_colors,
                             all_seg_labels, false);
  } else {
    {
      std::lock_guard<std::mutex> lock(integrator_mutex_);
      submap_geometry_integrtor_ptr_.emplace(
          submap->getID(),
          std::make_shared<GlobalVoxbloxIntegrator>(
              config_.global_voxblox_integrator_config, submap));
      submap_geometry_integrtor_ptr_.at(submap->getID())
          ->processIntegration(T_S_C, all_seg_points, all_seg_colors,
                               all_seg_labels, false);
    }
  }
  }
  submap->updateBoundingVolume();
}

void RaytracingTsdfIntegrator::GenerateInputPointCloud(Pointcloud& points,
                                                       Colors& colors,
                                                       Labels& labels,
                                                       std::string ply_path) {
  ColorLidarPointCloud::Ptr cloud_ptr(new ColorLidarPointCloud);
  for (int i = 0; i < points.size(); ++i) {
    ColorLidarPoint pt_rgb;
    pt_rgb.x = points[i](0, 0);
    pt_rgb.y = points[i](1, 0);
    pt_rgb.z = points[i](2, 0);
    LabelEntry label;
    if (globals_->labelHandler()->segmentationIdExists(labels[i].ins_label_)) {
      label = globals_->labelHandler()->getLabelEntry(labels[i].ins_label_);
      const Color color =
          globals_->labelHandler()->getColor(label.segmentation_id);
      pt_rgb.r = color.r;
      pt_rgb.g = color.g;
      pt_rgb.b = color.b;
    } else {
      pt_rgb.r = 255.f;
      pt_rgb.g = 0.f;
      pt_rgb.b = 0.f;
    }

    cloud_ptr->push_back(pt_rgb);
  }
  pcl::io::savePLYFileASCII(ply_path, *cloud_ptr);
}

}  // namespace panoptic_mapping
