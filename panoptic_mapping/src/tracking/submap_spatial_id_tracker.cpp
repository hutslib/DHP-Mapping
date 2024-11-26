#include "panoptic_mapping/tracking/submap_spatial_id_tracker.h"

#include <algorithm>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pcl/io/ply_io.h>
#include <voxblox/core/color.h>

#include "panoptic_mapping/common/index_getter.h"
#include "panoptic_mapping/labels/semantic_kitti_label.h"

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<
    IDTrackerBase, SubmapSpatialIDTracker, std::shared_ptr<Globals>>
    SubmapSpatialIDTracker::registration_("submap_spatial");

void SubmapSpatialIDTracker::Config::checkParams() const {
  // checkParamGT(rendering_threads, 0, "rendering_threads");
  // checkParamNE(depth_tolerance, 0.f, "depth_tolerance");
  // checkParamGT(rendering_subsampling, 0, "rendering_subsampling");
  // checkParamConfig(renderer);
}

void SubmapSpatialIDTracker::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("tracking_metric", &tracking_metric);
  setupParam("match_acceptance_threshold", &match_acceptance_threshold);
  setupParam("use_class_data_for_matching", &use_class_data_for_matching);
  setupParam("min_allocation_size_background", &min_allocation_size_background);
  setupParam("min_allocation_size_small", &min_allocation_size_small);
  setupParam("min_allocation_size_middle", &min_allocation_size_middle);
  setupParam("min_allocation_size_large", &min_allocation_size_large);
  setupParam("rendering_threads", &rendering_threads);
  setupParam("renderer", &renderer);
  setupParam("submap", &submap, "submap");
  setupParam("use_detectron", &use_detectron);
  setupParam("depth_tolerance", &depth_tolerance);
  setupParam("only_active_submaps", &only_active_submaps);
  setupParam("use_kitti", &use_kitti);
  setupParam("generate_gt_maps", &generate_gt_maps);
}

SubmapSpatialIDTracker::SubmapSpatialIDTracker(const Config& config,
                                               std::shared_ptr<Globals> globals,
                                               bool print_config)
    : IDTrackerBase(std::move(globals)),
      config_(config.checkValid()),
      renderer_(config.renderer, globals_->camera()->getConfig(), false) {
  LOG_IF(INFO, config_.verbosity >= 1 && print_config) << "\n"
                                                       << config_.toString();
  if (globals_->points3d()->getConfig().sensor_setup == "rgbd") {
    LOG_IF(INFO, config_.verbosity >= 4)
        << "add requiredInputs for sensor setup rgbd: ";
    addRequiredInputs(
        {InputData::InputType::kPoints, InputData::InputType::kColors,
         InputData::InputType::kLabels, InputData::InputType::kRawImage,
         InputData::InputType::kColorImage, InputData::InputType::kDepthImage,
         InputData::InputType::kSegmentationImage});
  } else if (globals_->points3d()->getConfig().sensor_setup == "lidar") {
    LOG_IF(INFO, config_.verbosity >= 4)
        << "add requiredInputs for sensor setup lidar: ";
    addRequiredInputs(
        {InputData::InputType::kPoints, InputData::InputType::kColors,
         InputData::InputType::kLabels, InputData::InputType::kRawImage,
         InputData::InputType::kColorImage, InputData::InputType::kLidarPoints,
         InputData::InputType::kSegmentationImage});
  } else if (globals_->points3d()->getConfig().sensor_setup ==
             "kitti_lidar_camera") {
    LOG_IF(INFO, config_.verbosity >= 4)
        << "add requiredInputs for sensor setup kitti (use "
           "segmentation directly from 3d)";
    addRequiredInputs(
        {InputData::InputType::kPoints, InputData::InputType::kLabels,
         InputData::InputType::kColors, InputData::InputType::kLidarPoints,
         InputData::InputType::kSegmentationImage,
         InputData::InputType::kRawImage, InputData::InputType::kColorImage,
         InputData::InputType::kKittiLabels});
  } else if (globals_->points3d()->getConfig().sensor_setup == "kitti_lidar") {
    LOG_IF(INFO, config_.verbosity >= 4)
        << "add requiredInputs for sensor setup kitti lidar";
    addRequiredInputs(
        {InputData::InputType::kPoints, InputData::InputType::kLabels,
         InputData::InputType::kColors, InputData::InputType::kLidarPoints,
         InputData::InputType::kRawImage, InputData::InputType::kColorImage,
         InputData::InputType::kKittiLabels});
  }
  if (config_.use_detectron) {
    addRequiredInputs({InputData::InputType::kDetectronLabels});
  }
  frame_count_ = 0;
}

ColorLidarPointCloud::Ptr SubmapSpatialIDTracker::GenerateInputPointCloud(
    InputData* input) {
  ColorLidarPointCloud::Ptr cloud_ptr(new ColorLidarPointCloud);
  for (int i = 0; i < input->points().size(); ++i) {
    ColorLidarPoint pt_rgb;
    pt_rgb.x = input->points()[i](0, 0);
    pt_rgb.y = input->points()[i](1, 0);
    pt_rgb.z = input->points()[i](2, 0);
    LabelEntry label;
    if (globals_->labelHandler()->segmentationIdExists(
            input->labels()[i].ins_label_)) {
      label = globals_->labelHandler()->getLabelEntry(
          input->labels()[i].ins_label_);
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
  return cloud_ptr;
}

bool SubmapSpatialIDTracker::create_directory_if_not_exists(
    const std::string& folder_path) {
  // Check if the folder exists
  if (!std::filesystem::exists(folder_path)) {
    // If it doesn't exist, try to create it recursively
    if (std::filesystem::create_directories(folder_path)) {
      // std::cout << "Folder created: " << folder_path << std::endl;
      return true;
    } else {
      std::cerr << "Error creating folder: " << folder_path << std::endl;
      return false;
    }
  } else {
    // If it exists, return true
    // std::cout << "Folder already exists: " << folder_path << std::endl;
    return true;
  }
}

void SubmapSpatialIDTracker::GenerateSegmentationPointCloud(
    Segments* segments) {
  ColorLidarPointCloud::Ptr cloud_ptr(new ColorLidarPointCloud);
    std::string folder_path =
        "/dataset/results/04/";
    std::string ply_path =
        folder_path + std::to_string(frame_count_)+ ".ply";
    create_directory_if_not_exists(folder_path);

  for (int s = 0; s < segments->size(); ++s) {
    Segment segment = (*segments)[s];
    for (int i = 0; i < segment.points().size(); ++i) {
      const Transformation T_M_C = segment.getT();
      const Point point_i_M = T_M_C * segment.points()[i];
      ColorLidarPoint pt_rgb;
      pt_rgb.x = point_i_M(0, 0);
      pt_rgb.y = point_i_M(1, 0);
      pt_rgb.z = point_i_M(2, 0);
      pt_rgb.r = segment.colors()[i].r;
      pt_rgb.g = segment.colors()[i].g;
      pt_rgb.b = segment.colors()[i].b;      
      cloud_ptr->push_back(pt_rgb);
    }
  }
  std::cout<<"save ply"<<ply_path<<std::endl;
  pcl::io::savePLYFileASCII(ply_path, *cloud_ptr);
}

void SubmapSpatialIDTracker::removeSky(InputData* input) {
  std::vector<int> remove_index;
  int origin_size = input->points().size();
  for (int i = 0; i < input->points().size(); i++) {
    Label this_label = input->labels()[i];
    int input_id = this_label.ins_label_;
    // parse the detectron labels
    if (input_id == 0) {
      // The id 0 is used for no-predictions in detectron and this points is
      // thought as no sky value
      continue;
    }
    // Check whether the instance code is known.
    auto it = labels_->find(input_id);
    if (it == labels_->end()) {
      LOG(ERROR) << "we do not get the corresponding detectron label for the "
                    "input id: "
                 << input_id;
      continue;
    }
    // Parse detectron label.
    LabelEntry label;
    const int class_id = it->second.category_id;
    // NOTE(thuaj): in the detectron the 41 code is reserved for sky
    if (class_id == 41) {
      remove_index.push_back(i);
    }
  }
  // 这里使用了一个倒序循环，以确保删除元素时不会影响后续元素的索引。
  for (int i = remove_index.size() - 1; i >= 0; i--) {
    int index = remove_index[i];
    input->PointcloudPtr()->erase(input->PointcloudPtr()->begin() + index);
    input->colorsPtr()->erase(input->colorsPtr()->begin() + index);
    input->labelsPtr()->erase(input->labelsPtr()->begin() + index);
  }
  if (config_.verbosity >= 4 && remove_index.size() > 0) {
    std::cout << "we remove sky points size: " << remove_index.size()
              << " , origin points size: " << origin_size
              << " , aft remove size: " << input->points().size() << std::endl;
    // exit(0);
  }
}

void SubmapSpatialIDTracker::processInput(SubmapCollection* submaps,
                                          InputData* input) {
  LOG_IF(INFO, config_.verbosity >= 2) << "\n process spatial id tracker ";
  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);
  CHECK(inputIsValid(*input));
  segments_to_integrate_.clear();
  LOG_IF(INFO, config_.verbosity >= 1)
      << " frame: " << frame_count_ << " submap size: " << submaps->size();
  // if(frame_count_>=2) exit(0);
  // Visualization input points3d
  if (config_.verbosity >= 3) {
    ColorLidarPointCloud::Ptr cloud_ptr = GenerateInputPointCloud(input);
    std::string ply_folder =
        "/data/research/3dv_exp/detectron_debug_output/ply/";
    std::string ply_path =
        "/data/research/3dv_exp/detectron_debug_output/ply/" +
        std::to_string(frame_count_) + "_input_points3d.ply";
    create_directory_if_not_exists(ply_folder);
    if (pcl::io::savePLYFileASCII(ply_path, *cloud_ptr) != -1) {
      LOG_IF(INFO, config_.verbosity >= 4)
          << "save ply file " << ply_path << " successfully!";
    }
  }

  // Check whether the map is already allocated.
  if (!is_setup_) {
    setup(submaps);
  }

  // update the existing background_list_;
  for (auto it = background_list_.begin(); it != background_list_.end();) {
    if (!submaps->submapIdExists(it->second)) {
      it = background_list_.erase(it);
      LOG_IF(INFO, config_.verbosity >= 1)
          << "erase the background in the background_list_ because the submap "
             "is delected";
    } else {
      ++it;
    }
  }

  // bool use_detectron = true;
  if (config_.use_detectron) {
    // Cache the input labels for submap allocation.
    labels_ = &(input->detectronLabels());
    if (config_.verbosity >= 4) {
      std::cout << redText << " The input detectron labels Info: " << endColor
                << std::endl;
      for (const auto& label : *labels_) {
        std::cout << " label index: " << label.first << std::endl;
        std::cout << "  ID: " << label.second.id << std::endl;
        std::cout << "  Is Thing: " << label.second.is_thing << std::endl;
        std::cout << "  Category ID: " << label.second.category_id << std::endl;
        std::cout << "  Instance ID: " << label.second.instance_id << std::endl;
        std::cout << "  Score: " << label.second.score << std::endl;
      }
    }
  }

  if (config_.use_kitti && config_.generate_gt_maps) {
    if (config_.verbosity >= 4) {
      std::cout << "we use kitti lidar label to generate the gt, so we need to "
                   "pharse the kitti labels"
                << std::endl;
    }
    for (int i = 0; i < input->labels().size(); i++) {
      auto cur_label = kitti_labels_.find(input->labels()[i].id_label_);
      if (cur_label == kitti_labels_.end())  // not found, add new entry
      {
        LabelEntry new_label;
        new_label.segmentation_id = input->labels()[i].id_label_;
        new_label.class_id = input->labels()[i].sem_label_;

        // only for semantic kitti dataset
        semanticKittiLabelLUT(input->labels()[i].sem_label_, new_label);

        kitti_labels_[input->labels()[i].id_label_] = new_label;
      }
    }
    if (config_.verbosity >= 4) {
      std::cout << redText << " The input kitti labels Info: " << endColor
                << std::endl;
      for (const auto& label : kitti_labels_) {
        std::cout << label.second.toString() << std::endl;
      }
    }
  }

  if (config_.use_kitti && !(config_.generate_gt_maps)) {
    if (config_.verbosity >= 4) {
      std::cout << "we use kitti so we check the input and remove the sky "
                   "points, colors and labels value";
    }
    removeSky(input);
  }
  if (config_.generate_gt_maps) {
    CreateGtNewSubmap(submaps, input);
  } else {
    TrackingInfoAggregator tracking_data =
        MultiThreadcomputeTrackingData(submaps, input);

    MatchingAndCreateNewSubmap(&tracking_data, submaps, input);
  }
  // 根据segments_to_integrate_得到input->kSegments
  Segments segments;
  std::set<int> cur_submap_id_set;
  // std::vector<int> cur_submap_id_vec;
  for (auto it = segments_to_integrate_.begin();
       it != segments_to_integrate_.end(); ++it) {
    int map_id = it->second.getSubmapID();
    int input_segmentation_id = it->first;
    Segment segment = it->second;
    cur_submap_id_set.insert(map_id);
    // cur_submap_id_vec.push_back(map_id);
    // 對每個map_id和Segment對象執行相應操作
    segments.push_back(segment);
    // if (config_.verbosity >= 4) {
    //   std::cout << " insert segment with index id: " << map_id
    //             << " segment input id: " << segment.getSegmentationID()
    //             << " segment class id: " << segment.getClassID()
    //             << " segment submap id: " << segment.getSubmapID()
    //             << " , submap info: " << segment.getSubmapPtr()->toString()
    //             << std::endl;
    // }
    // if (config_.verbosity >= 3) {
    //   std::cout << "segment index: " << cur_submap_id_vec.size() - 1
    //             << " segment input id: " << segment.getSegmentationID()
    //             << " , submap info: " << segment.getSubmapPtr()->toString()
    //             << std::endl;
    // }
  }
  std::vector<int> cur_submap_id_vec(cur_submap_id_set.begin(),
                                     cur_submap_id_set.end());  // set to vector
  input->setSubmapIDList(cur_submap_id_vec);
  input->setSegments(segments);
  if (config_.verbosity >= 0) {
    GenerateSegmentationPointCloud(&segments);
  }
  frame_count_++;

  // always set submap 0 for track and active
  submaps->getSubmapPtr(0)->setIsActive(true);
  submaps->getSubmapPtr(0)->setWasTracked(true);

  for (auto& submap : *submaps) {
    if (config_.verbosity >= 2) {
      std::cout << submap.toString() << std::endl;
    }
  }
}

void SubmapSpatialIDTracker::MatchingAndCreateNewSubmap(
    TrackingInfoAggregator* tracking_data, SubmapCollection* submaps,
    InputData* input) {
  // 根据overlap区分background stuff和foreground thing来得到input_to_output;
  // std::unordered_map<int, MatchInfo> input_to_output;
  // input to output 是为了给input instance 和 map instance id的一个匹配的表
  std::unordered_map<int, MatchInfo> input_to_output;
  int n_matched = 0;
  int n_new = 0;
  for (const int input_id : tracking_data->getInputIDs()) {
    int input_class;
    bool background_class;
    LabelEntry phrased_input_label;
    // phrase input id 判斷是否爲background
    // class，並得到input_class(這個主要是給background class)
    if (config_.use_detectron) {
      // 單獨出離input_id=0
      // TODO(thuaj)其他數據集的void id還需要進行處理
      if (input_id == 0) {
        MatchInfo this_matchinfo(submaps->getSubmapPtr(input_id), 1.);
        input_to_output.emplace(input_id, this_matchinfo);
        LOG_IF(INFO, config_.verbosity >= 4)
            << "Phrase Detectron Labels, the input id is 0";
        continue;
      }
      // parse detectron id
      bool find_detectron_id = false;
      DetectronLabels labels = input->detectronLabels();
      for (auto pair : labels) {
        if (pair.second.id == input_id) {
          input_class = pair.second.category_id;
          background_class =
              globals_->labelHandler()->isBackgroundClass(input_class);
          phrased_input_label =
              globals_->labelHandler()->getLabelEntry(input_class);
          if (input_class == 1) {
            std::cout << "we get thing class input from detectron" << std::endl;
            exit(0);
          }
          LOG_IF(INFO, config_.verbosity >= 4)
              << "Phrase Detectron Labels, the input id is: " << input_id
              << phrased_input_label.toString();
          find_detectron_id = true;
          break;
        }
      }
      if (!find_detectron_id) {
        LOG(WARNING) << "parse detectron id failed, input id: " << input_id
                     << ", frame: " << frame_count_ << std::endl;
      }
    } else {
      input_class = globals_->labelHandler()->getClassID(input_id);
      background_class = globals_->labelHandler()->isBackgroundClass(input_id);
      phrased_input_label = globals_->labelHandler()->getLabelEntry(input_id);
      LOG_IF(INFO, config_.verbosity >= 4)
          << " Phrase Labels, the " << greenText << " id is: " << endColor
          << input_id << phrased_input_label.toString();
    }
    int min_allocation_size = 100;
    switch (phrased_input_label.label) {
      case PanopticLabel::kInstance: {
        if (phrased_input_label.size == "L") {
          min_allocation_size = config_.min_allocation_size_large;
        } else if (phrased_input_label.size == "S") {
          min_allocation_size = config_.min_allocation_size_small;
        } else {
          min_allocation_size = config_.min_allocation_size_middle;
        }
        break;
      }
      case PanopticLabel::kBackground: {
        min_allocation_size = config_.min_allocation_size_background;
        break;
      }
      default: {
        min_allocation_size = 100;
        break;
      }
    }
    LOG_IF(INFO, config_.verbosity >= 4)
        << "min_allocation_size: " << min_allocation_size;
    // 計算overlap 並且得到segments_to_integrate_
    if (background_class) {
      int map_id = 0;
      if (background_list_.find(input_class) == background_list_.end()) {
        LOG_IF(INFO, config_.verbosity >= 4)
            << "Do not find exist background submap with input id: " << input_id
            << " with class id:" << input_class;
        bool allocate_new_submap = tracking_data->getNumberOfInputPixels(
                                       input_id) >= min_allocation_size;
        if (allocate_new_submap) {
          Submap* new_submap = allocateSubmap(input_id, submaps, input);
          LOG_IF(INFO, config_.verbosity >= 4)
              << "new submap info: " << new_submap->toString();
          if (new_submap) {
            map_id = new_submap->getID();
            background_list_.emplace(input_class, map_id);
            // some debug record code
          } else {
            LOG(WARNING) << "create new submap failed.." << std::endl;
          }
        } else {
          // do not need to allocate new submap for background merge to submap 0
          LOG_IF(WARNING, config_.verbosity >= 3)
              << "we do not create submap for background class: " << input_class
              << " , merge to submap 0";
          map_id = 0;
        }
      } else {
        map_id = background_list_.at(input_class);
        submaps->getSubmapPtr(map_id)->setWasTracked(true);
        LOG_IF(INFO, config_.verbosity >= 4)
            << "background input id: " << input_id
            << " with class id:" << input_class
            << "is matched to submap id: " << map_id;
      }
      // TODO(thuaj): give a probability
      MatchInfo this_matchinfo(submaps->getSubmapPtr(map_id), 1.);
      input_to_output.emplace(input_id, this_matchinfo);
      auto record_it = map_to_input_records_.find(map_id);
      if (record_it == map_to_input_records_.end()) {
        std::set<int> new_set;
        new_set.insert(input_id);
        map_to_input_records_.emplace(map_id, new_set);
      } else {
        record_it->second.insert(input_id);
      }
    } else {
      // 對於forground 計算overlap > threshold otherwise test whether to create
      // a new submap
      bool matched = false;
      bool any_overlap;
      float value;  // overlap metric value（iou/ovelap）
      int map_id;
      // TODO(thuaj): refer to projective id tracker to implement whether use
      // class data
      std::vector<std::pair<int, float>>
          ids_values;  // <matched_map_id, metric>
      any_overlap = tracking_data->getAllMetrics(input_id, &ids_values,
                                                 config_.tracking_metric);
      if (!any_overlap && config_.verbosity >= 4) {
        LOG(INFO) << " DO not find any overlap with input id : " << input_id;
      }
      if (any_overlap &&
          ids_values.front().second > config_.match_acceptance_threshold &&
          ids_values.front().first != 0) {
        // check only for the highest match.
        matched = true;
        map_id = ids_values.front().first;
        value = ids_values.front().second;
      } else if (any_overlap && ids_values.front().first == 0) {
        map_id = 0;
        LOG_IF(WARNING, config_.verbosity >= 4)
            << "input id: " << input_id << " matched with submap 0";
      }
      if (config_.verbosity >= 3) {
        LOG(INFO) << pinkText << "input id: " << endColor << input_id
                  << " matching statistics: \n";
        for (const auto& id_value : ids_values) {
          std::cout << " id: " << id_value.first
                    << " overlap metrics: " << id_value.second << std::endl;
        }
      }
      bool allocate_new_submap = tracking_data->getNumberOfInputPixels(
                                     input_id) >= min_allocation_size;
      if (matched) {
        n_matched++;
        MatchInfo this_matchinfo(submaps->getSubmapPtr(map_id), value);
        input_to_output.emplace(input_id, this_matchinfo);
        submaps->getSubmapPtr(map_id)->setWasTracked(true);
        submaps->getSubmapPtr(map_id)->setIsActive(true);
        auto record_it = map_to_input_records_.find(map_id);
        if (record_it == map_to_input_records_.end()) {
          std::set<int> new_set;
          new_set.insert(input_id);
          map_to_input_records_.emplace(map_id, new_set);
        } else {
          record_it->second.insert(input_id);
        }
        LOG_IF(INFO, config_.verbosity >= 4)
            << "find matching information for input id: " << input_id
            << " match to submap id: "
            << input_to_output.at(input_id).submap_info_->getID()
            << " with probabilistic value: "
            << input_to_output.at(input_id).prob_;
      } else if (allocate_new_submap) {
        LOG_IF(INFO, config_.verbosity >= 4)
            << "do not find tracking information for input id: " << input_id;
        n_new++;
        Submap* new_submap = allocateSubmap(input_id, submaps, input);
        LOG_IF(INFO, config_.verbosity >= 4)
            << "new submap info: " << new_submap->toString();
        if (new_submap) {
          map_id = new_submap->getID();
          MatchInfo this_matchinfo(new_submap, 1.);
          input_to_output.emplace(input_id, this_matchinfo);
          auto record_it = map_to_input_records_.find(map_id);
          if (record_it == map_to_input_records_.end()) {
            std::set<int> new_set;
            new_set.insert(input_id);
            map_to_input_records_.emplace(map_id, new_set);
          } else {
            record_it->second.insert(input_id);
          }
        } else {
          LOG(ERROR) << "create new submap failed..." << std::endl;
        }
      } else {
        // TODO(thuaj): 没有match的部分怎么办，需要进一步处理，目前是给id
        // 0然后probabilistic 1
        LOG_IF(WARNING, config_.verbosity >= 3)
            << "do not find tracking information for input id: " << input_id
            << " too fewer size to create new, merge to submap 0";
        map_id = 0;
        MatchInfo this_matchinfo(submaps->getSubmapPtr(0), 1.);
        input_to_output.emplace(input_id, this_matchinfo);
        auto record_it = map_to_input_records_.find(map_id);
        if (record_it == map_to_input_records_.end()) {
          std::set<int> new_set;
          new_set.insert(input_id);
          map_to_input_records_.emplace(map_id, new_set);
        } else {
          record_it->second.insert(input_id);
        }
      }
      }
    }

    // according to the input-output create the segment for this input
    // 根据input_to_output得到segments_to_integrate_
    // std::unordered_map<int, Segment> segments_to_integrate_; //<map_id,
    // Segmentation>
    for (int i = 0; i < input->points().size(); ++i) {
      int input_ins_label = input->labels()[i].ins_label_;
      LabelEntry input_label;
      // std::cout<<input_label.toString()<<std::endl;
      // exit(0);
      int panoptic_id;
      if (config_.use_detectron) {
        // Check whether the instance code is known.
        auto it = labels_->find(input_ins_label);
        int class_id;
        if (it == labels_->end() && input_ins_label != 0) {
          LOG(WARNING) << "unknow detectron input label" << input_ins_label;
        } else if (input_ins_label == 0) {
          class_id = 0;
          panoptic_id = 0;
        } else {
          // Parse detectron label.
          class_id = it->second.category_id;
        }
        if (globals_->labelHandler()->segmentationIdExists(class_id)) {
          input_label = globals_->labelHandler()->getLabelEntry(class_id);
          panoptic_id = static_cast<int>(input_label.label);
          // std::cout << "class id: " << class_id
          //           << " , panoptic id: " << panoptic_id << std::endl;
          // if(class_id == 0) {
          //   std::cout<<input_label.toString()<<std::endl;
          //   std::cout<<"++++"<<panoptic_id<<std::endl;
          //   exit(0);
          // }
          // exit(0);
        }
      } else {
        if (globals_->labelHandler()->segmentationIdExists(input_ins_label)) {
          input_label =
              globals_->labelHandler()->getLabelEntry(input_ins_label);
          panoptic_id = static_cast<int>(input_label.label);
          // std::cout << "input_ins_label: " << input_ins_label
          //           << " , panoptic id: " << panoptic_id << std::endl;
          //           exit(0);
        }
      }
      MatchInfo out_data = input_to_output.at(input_ins_label);  // a MatchInfo
      auto it = segments_to_integrate_.find(input_ins_label);
      if (it == segments_to_integrate_.end()) {
        Segment new_seg(out_data.submap_info_->getID(), input_label.class_id,
                        input_ins_label, out_data.submap_info_, input->T_M_C());
        Label label(input_label.class_id, out_data.submap_info_->getID(),
                    out_data.prob_, panoptic_id);
        new_seg.Pushback(input->points()[i], input->colors()[i], label);
        segments_to_integrate_.emplace(input_ins_label, new_seg);
      } else {
        Label label(input_label.class_id, out_data.submap_info_->getID(),
                    out_data.prob_, panoptic_id);
        it->second.Pushback(input->points()[i], input->colors()[i], label);
      }
    }
    if (config_.verbosity >= 3) {
      std::cout << "input to map id record" << std::endl;
      for (const auto& record : map_to_input_records_) {
        std::cout << "Map ID: " << record.first << ", Input IDs: ";
        for (const auto& value : record.second) {
          std::cout << value << " ";
        }
        std::cout << std::endl;
      }
      // 遍历map_to_input_records_，将每个input_id对应的map_id加入到input_to_map_records中
      for (const auto& record : map_to_input_records_) {
        int map_id = record.first;
        const std::set<int>& input_ids = record.second;
        for (int input_id : input_ids) {
          input_to_map_records_[input_id].insert(map_id);
        }
      }
      // 输出新的结果
      for (const auto& record : input_to_map_records_) {
        std::cout << "Input ID: " << record.first << ", Map IDs: ";
        for (int value : record.second) {
          std::cout << value << " ";
        }
        std::cout << std::endl;
      }
  }
}
  // Utility function.
// void incrementMap(std::unordered_map<int, int>* map, int id, int value = 1) {
//   auto it = map->find(id);
//   if (it == map->end()) {
//     map->emplace(id, value);
//   } else {
//     it->second += value;
//   }
// }
void SubmapSpatialIDTracker::CreateGtNewSubmap(SubmapCollection* submaps,
                                               InputData* input) {
                                                std::cout<<"in create gt new submap"<<std::endl;
  // setp 1: compute tracking data for current input
  TrackingInfoAggregator tracking_data;
  tracking_data.insertInputPoints3d(input->labels(), input->points());
  // 遍历input的points3d的labels和points得到total_input_count_
  // 得到每一帧输入中的segmentation信息

  // for (int i = 0; i < input->points().size(); ++i) {
  //   int input_ins_label;
  //   if (config_.use_kitti) {
  //     std::cout<<"######"<<std::endl;
  //     input_ins_label = input->labels()[i].id_label_;
  //   } else {
  //     input_ins_label = input->labels()[i].ins_label_;
  //   }
  //   std::cout<<"input_ins_label: "<<input_ins_label<<std::endl;
  //   std::cout<<"!!!!!!"<<std::endl;
  //   // 统计这一帧的 input 中每一个 instance id 的pixel 数量
  //   // 主要是为了之后判断是否要生成新的instance in map
  //   // incrementMap(&total_input_count_, input_ins_label);
  //   auto it = tracking_data.total_input_count_.find(input_ins_label);
  //   if (it == tracking_data.total_input_count_.end()) {
  //     std::cout<<"inset new input_ins_label"<<std::endl;
  //     tracking_data.total_input_count_.emplace(input_ins_label, 1);
  //   } else {
  //     std::cout<<"increment input_ins_label"<<std::endl;
  //     it->second += 1;
  //   }
  //   std::cout<<"--------"<<std::endl;
  // }
  // std::cout<<"============"<<std::endl;
  for (const int input_id : tracking_data.getInputIDs()) {
  // std::cout << "create gt submap" << std::endl;
  // step2: pharse the input label
    // LabelEntry input_label;
    LabelEntry phrased_input_label;
    int input_class;
    bool background_class;
    if (config_.use_kitti) {
      auto it = kitti_labels_.find(input_id);
      if (it == kitti_labels_.end()) {
        LOG(WARNING) << "unknown kitti lidar label" << input_id;
      } else {
        // input_label = it->second;
        phrased_input_label = it->second;
        // panoptic_id = static_cast<int>(input_label.label);
        background_class = static_cast<bool>(phrased_input_label.label);
        if (config_.verbosity >= 4) {
          std::cout << phrased_input_label.toString() << std::endl;
        }
      }
    } 
    // step3 set the allocate size
    int min_allocation_size = 100;
    switch (phrased_input_label.label) {
      case PanopticLabel::kInstance: {
        if (phrased_input_label.size == "L") {
          min_allocation_size = config_.min_allocation_size_large;
        } else if (phrased_input_label.size == "S") {
          min_allocation_size = config_.min_allocation_size_small;
        } else {
          min_allocation_size = config_.min_allocation_size_middle;
        }
        break;
      }
      case PanopticLabel::kBackground: {
        min_allocation_size = config_.min_allocation_size_background;
        break;
      }
      default: {
        min_allocation_size = 100;
        break;
      }
    }
    LOG_IF(INFO, config_.verbosity >= 4)
        << "min_allocation_size: " << min_allocation_size;
    // step4 calculatre the input to output
    // since we create the gt submaps, so we do not need to get the tracking
    // information and run the matching algorithms
    // according to the input-output create the segment for this input
    // 根据input_to_output得到segments_to_integrate_ 在这里面， input_to_output
    // 永远都是输入什么就输出什么label不做修改，因为我们是要生成gt的
    // std::unordered_map<int, Segment> segments_to_integrate_; //<map_id,
    // Segmentation>
    auto input_it = input_to_output_.find(input_id);
    if (input_it == input_to_output_.end() || !submaps->submapIdExists(
            input_it->second.submap_info_->getID())){
      // key k does not exist in the unordered_map
      // create a new submap
      bool allocate_new_submap = tracking_data.getNumberOfInputPixels(
                                       input_id) >= min_allocation_size;
      if(!allocate_new_submap) continue;  
      if (config_.verbosity >= 4) {
        std::cout << "we create new submap for input ins id: "
                  << input_id << std::endl;
        // std::cout << " label information, sem:  "
        //           << input->labels()[i].sem_label_
        //           << " ins: " << input->labels()[i].ins_label_ << std::endl;
      }
      Submap* new_submap = allocateSubmap(input_id, submaps, input);
      if (new_submap) {
        LOG_IF(INFO, config_.verbosity >= 2)
            << "new submap info: " << new_submap->toString();
        MatchInfo new_matchinfo(new_submap, 1.);
        // input_to_output_.emplace(input_ins_label, new_matchinfo);
        auto check_input_it = input_to_output_.find(input_id);
        if (check_input_it != input_to_output_.end()) {
          input_to_output_.erase(input_it);
        }
        input_to_output_.emplace(input_id, new_matchinfo);
      }
    } else {
    int map_id = input_it->second.submap_info_->getID();
    submaps->getSubmapPtr(map_id)->setWasTracked(true);
    } 
  } 
  for (int i = 0; i < input->points().size(); ++i) {
    // std::cout<<"generate segments"<<std::endl;
    int input_ins_label;
    if (config_.use_kitti) {
      input_ins_label = input->labels()[i].id_label_;
    } else {
      input_ins_label = input->labels()[i].ins_label_;
    }
    LabelEntry input_label;
    int panoptic_id;
    if (config_.use_kitti) {
      auto it = kitti_labels_.find(input_ins_label);
      if (it == kitti_labels_.end()) {
        LOG(WARNING) << "unknown kitti lidar label" << input_ins_label;
      } else {
        input_label = it->second;
        panoptic_id = static_cast<int>(input_label.label);
      }
    } 
    // MatchInfo out_data = input_to_output_.at(input_ins_label);  // a MatchInfo
    auto out_it = input_to_output_.find(input_ins_label);
    if (out_it == input_to_output_.end()) {
      // LOG(WARNING) << "do not find the input id in the input_to_output_";
      continue;
    }
    auto out_data = out_it->second;
    auto it = segments_to_integrate_.find(input_ins_label);
    if (it == segments_to_integrate_.end()) {
      Segment new_seg(out_data.submap_info_->getID(), input_label.class_id,
                      input_ins_label, out_data.submap_info_, input->T_M_C());
      Label label(input_label.class_id, out_data.submap_info_->getID(),
                  out_data.prob_, panoptic_id);
      new_seg.Pushback(input->points()[i], input->colors()[i], label);
      segments_to_integrate_.emplace(input_ins_label, new_seg);
    } else {
      Label label(input_label.class_id, out_data.submap_info_->getID(),
                  out_data.prob_, panoptic_id);
      it->second.Pushback(input->points()[i], input->colors()[i], label);
    }
  }

  // if (config_.verbosity >= 3) {
  //     std::cout << "input to map id record" << std::endl;
  //     for (const auto& record : map_to_input_records_) {
  //       std::cout << "Map ID: " << record.first << ", Input IDs: ";
  //       for (const auto& value : record.second) {
  //         std::cout << value << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  //     //
  //     遍历map_to_input_records_，将每个input_id对应的map_id加入到input_to_map_records中
  //     for (const auto& record : map_to_input_records_) {
  //       int map_id = record.first;
  //       const std::set<int>& input_ids = record.second;
  //       for (int input_id : input_ids) {
  //         input_to_map_records_[input_id].insert(map_id);
  //       }
  //     }
  //     // 输出新的结果
  //     for (const auto& record : input_to_map_records_) {
  //       std::cout << "Input ID: " << record.first << ", Map IDs: ";
  //       for (int value : record.second) {
  //         std::cout << value << " ";
  //       }
  //       std::cout << std::endl;
  //     }
  // }
}

TrackingInfo SubmapSpatialIDTracker::renderTrackingInfoPoints3d(
    const Submap& submap, const InputData& input) const {
  TrackingInfo result(submap.getID());

  // Compute the maximum extent to lookup vertices.
  //
  const Transformation T_C_S = input.T_M_C().inverse() * submap.getT_M_S();
  const Transformation T_S_C = T_C_S.inverse();
  // sdf tolerance to check whether the point belong to this submap
  const float depth_tolerance =
      config_.depth_tolerance > 0
          ? config_.depth_tolerance
          : -config_.depth_tolerance * submap.getTsdfLayer().voxel_size();

  for (int i = 0; i < input.points().size(); ++i) {
    // NOTE(thuaj): the Point position is in the sensor coordinate
    // needed to be transformed into global map coordinate
    const Point k_point_i_S = T_S_C * input.points()[i];
    const Label k_label_i = input.labels()[i];
    int input_point_segmentation_id = k_label_i.ins_label_;
    if (submap.getBoundingVolume().contains_M(k_point_i_S)) {
      // 得到tsdf block layer 和 class block layer
      auto block_ptr =
          submap.getTsdfLayer().getBlockPtrByCoordinates(k_point_i_S);
      auto class_ptr =
          submap.getClassLayer().getBlockPtrByCoordinates(k_point_i_S);
      if (block_ptr) {
        if (!class_ptr) {
          // std::cout << "block do not has class layer!" << std::endl;
        } else {
          // TODO(thuaj): !!!!这里要仔细进行修改的
          // 根据点的坐标得到voxel
          const TsdfVoxel& voxel =
              block_ptr->getVoxelByCoordinates(k_point_i_S);
          const ClassVoxel& class_voxel =
              class_ptr->getVoxelByCoordinates(k_point_i_S);
          bool classes_match = class_voxel.belongsToSubmap();
          if (voxel.weight > 1e-6 &&
              std::abs(voxel.distance) < depth_tolerance && classes_match) {
            result.insertVertexPoint(input_point_segmentation_id);
          }
        }
      } else {
        // LOG_IF(INFO, config_.verbosity>=4)<< "do not has block with this
        // position" << std::endl;
      }
    }
  }
  return result;
}

// 这里面主要就是得到哪个 counts_ <input_id, count>
TrackingInfoAggregator SubmapSpatialIDTracker::MultiThreadcomputeTrackingData(
    SubmapCollection* submaps, InputData* input) {
  // Render each active submap in parallel to collect overlap statistics.
  SubmapIndexGetter index_getter(globals_->camera()->findVisibleSubmapIDs(
      *submaps, input->T_M_C(), config_.only_active_submaps));
  std::vector<std::future<std::vector<TrackingInfo>>> threads;
  TrackingInfoAggregator tracking_data;
  for (int i = 0; i < config_.rendering_threads; ++i) {
    threads.emplace_back(std::async(
        std::launch::async,
        [this, i, &tracking_data, &index_getter, submaps,
         input]() -> std::vector<TrackingInfo> {
          // Also process the input image.
          if (i == 0) {
            tracking_data.insertInputPoints3d(input->labels(), input->points());
          }
          std::vector<TrackingInfo> result;
          int index;
          while (index_getter.getNextIndex(&index)) {
            submaps->getSubmapPtr(index)->setWasInView(true);
            result.emplace_back(this->renderTrackingInfoPoints3d(
                submaps->getSubmap(index), *input));
          }
          return result;
        }));
  }

  // Join all threads.
  std::vector<TrackingInfo> infos;
  for (auto& thread : threads) {
    for (const TrackingInfo& info : thread.get()) {
      infos.emplace_back(std::move(info));
    }
  }
  // 这里面计算了overlap
  tracking_data.insertTrackingInfos(infos);
  if (config_.verbosity >= 4) {
    tracking_data.printTrackingInfo();
  }
  return tracking_data;
}

Submap* SubmapSpatialIDTracker::allocateSubmap(int input_id,
                                               SubmapCollection* submaps,
                                               InputData* input) {
  if (config_.use_detectron) {
    if (input_id == 0) {
      // The id 0 is used for no-predictions in detectron.
      LOG(ERROR)
          << "we cannot create the submap for Detectron labels input id 0";
      return nullptr;
    }

    // Check whether the instance code is known.
    auto it = labels_->find(input_id);
    if (it == labels_->end()) {
      LOG(ERROR) << "we do not get the corresponding detectron label for the "
                    "input id: "
                 << input_id;
      return nullptr;
    }

    // Parse detectron label.
    LabelEntry label;
    const int class_id = it->second.category_id;
    if (globals_->labelHandler()->segmentationIdExists(class_id)) {
      label = globals_->labelHandler()->getLabelEntry(class_id);
    } else {
      LOG(ERROR) << "allocate submap with error detectron label info"
                 << std::endl;
      return nullptr;
    }

    if (it->second.is_thing) {
      label.label = PanopticLabel::kInstance;
    } else {
      label.label = PanopticLabel::kBackground;
    }
    LOG_IF(INFO, config_.verbosity >= 4)
        << "allocate submap for input id: " << input_id
        << " and corresponding label info: " << label.toString();

    // Allocate new submap.
    return submap_allocator_->allocateSubmap(submaps, input, input_id, label);
  } else if (config_.use_kitti && config_.generate_gt_maps) {
    auto it = kitti_labels_.find(input_id);
    if (it == kitti_labels_.end()) {
      LOG(ERROR)
          << "we do not get the corresponding kitti labels for input id: "
          << input_id;
      return nullptr;
    } else {
      LabelEntry label;
      label = it->second;
      LOG_IF(INFO, config_.verbosity >= 4)
          << "allocate submap for input id: " << input_id
          << "and corresponding label info: " << label.toString();
      return submap_allocator_->allocateSubmap(submaps, input, input_id, label);
    }
  } else {
    LabelEntry label;
    if (globals_->labelHandler()->segmentationIdExists(input_id)) {
      label = globals_->labelHandler()->getLabelEntry(input_id);
      LOG_IF(INFO, config_.verbosity >= 4)
          << "allocate submap for input id: " << input_id
          << " and corresponding label info: " << label.toString();
    }
    return submap_allocator_->allocateSubmap(submaps, input, input_id, label);
  }
}

// 这里就是首先给void/unknown的区域创建一个submap
// id=0存储所有没有融合到submap中的点,unknow 的 submap
// 只是存在color和sdf沒有其他的semantic 信息，但是是有class和scores layer
void SubmapSpatialIDTracker::setup(SubmapCollection* submaps) {
  // Check if there is a loaded map.
  // NOTE(thuaj): we set void submap id = 0 to store unmatched points
  int void_map_id = 0;
  Submap* new_submap = submaps->createSubmap(config_.submap);
  new_submap->setLabel(PanopticLabel::kUnknown);
  new_submap->setClassID(-1);  // void/unknown class is -1
  new_submap->setName("unknow");
  new_submap->setInstanceID(0);
  LOG_IF(INFO, config_.verbosity >= 3)
      << "\n create new submap with id: " << new_submap->getID()
      << " for unknow voxels with params \n"
      << config_.submap.toString()
      << " \n and submap info: " << new_submap->toString();
  is_setup_ = true;
}

}  // namespace panoptic_mapping
