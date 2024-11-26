#include "panoptic_mapping_ros/panoptic_mapper.h"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <panoptic_mapping/common/camera.h>
#include <panoptic_mapping/common/points3d.h>
#include <panoptic_mapping/labels/label_handler_base.h>
#include <panoptic_mapping/map/classification/fixed_count.h>
#include <panoptic_mapping/submap_allocation/freespace_allocator_base.h>
#include <panoptic_mapping/submap_allocation/submap_allocator_base.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

namespace panoptic_mapping {

// Modules that don't have a default type will be required to be explicitly set.
// Entries: <key, <ros_namespace, default type parameter>.
const std::map<std::string, std::pair<std::string, std::string>>
    PanopticMapper::default_names_and_types_ = {
        {"camera", {"camera", ""}},
        {"points3d", {"points3d", ""}},
        {"label_handler", {"labels", "null"}},
        {"submap_allocator", {"submap_allocator", "null"}},
        {"freespace_allocator", {"freespace_allocator", "null"}},
        {"id_tracker", {"id_tracker", ""}},
        {"tsdf_integrator", {"tsdf_integrator", ""}},
        {"map_management", {"map_management", "null"}},
        {"vis_submaps", {"visualization/submaps", "submaps"}},
        {"vis_tracking", {"visualization/tracking", "spatial"}},
        {"vis_planning", {"visualization/planning", ""}},
        {"data_writer", {"data_writer", "null"}}};

void PanopticMapper::Config::checkParams() const {
  checkParamCond(!global_frame_name.empty(),
                 "'global_frame_name' may not be empty.");
  checkParamGT(ros_spinner_threads, 1, "ros_spinnersourc_threads");
  checkParamGT(check_input_interval, 0.f, "check_input_interval");
}

void PanopticMapper::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("global_frame_name", &global_frame_name);
  setupParam("visualization_interval", &visualization_interval, "s");
  setupParam("data_logging_interval", &data_logging_interval, "s");
  setupParam("print_timing_interval", &print_timing_interval, "s");
  setupParam("use_threadsafe_submap_collection",
             &use_threadsafe_submap_collection);
  setupParam("ros_spinner_threads", &ros_spinner_threads);
  setupParam("check_input_interval", &check_input_interval, "s");
  setupParam("load_submaps_conservative", &load_submaps_conservative);
  setupParam("loaded_freespace_stays_active", &loaded_freespace_stays_active);
  setupParam("shutdown_when_finished", &shutdown_when_finished);
  setupParam("save_map_path_when_finished", &save_map_path_when_finished);
  setupParam("display_config_units", &display_config_units);
  setupParam("indicate_default_values", &indicate_default_values);
  setupParam("label_info_print_path", &label_info_print_path);
  setupParam("save_mesh_folder_path", &save_mesh_folder_path);
  setupParam("colormap_print_path", &colormap_print_path);
  setupParam("submap_info_path", &submap_info_path);
  setupParam("vis_colorized_pointcloud", &vis_colorized_pointcloud);
  setupParam("vis_colorized_pointcloud_save_folder",
             &vis_colorized_pointcloud_save_folder);
  setupParam("results_folder", &results_folder);
  setupParam("generate_kimera_pointcloud", &generate_kimera_pointcloud);
  setupParam("generate_gt_merged_ply", &generate_gt_merged_ply);
  setupParam("merged_ply_file", &merged_ply_file);
}

PanopticMapper::PanopticMapper(const ros::NodeHandle& nh,
                               const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      config_(
          config_utilities::getConfigFromRos<PanopticMapper::Config>(nh_private)
              .checkValid()) {
  // Setup printing of configs.
  // NOTE(schmluk): These settings are global so multiple panoptic mappers in
  // the same process might interfere.
  config_utilities::GlobalSettings().indicate_default_values =
      config_.indicate_default_values;
  config_utilities::GlobalSettings().indicate_units =
      config_.display_config_units;
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
  // Setup all components of the panoptic mapper.
  setupMembers();
  setupRos();
}

void PanopticMapper::setupMembers() {
  // merged_cloud
  merged_cloud_ptr = ColorLidarPointCloud::Ptr(new ColorLidarPointCloud);
  // Map.
  submaps_ = std::make_shared<SubmapCollection>();

  // Threadsafe wrapper for the map.
  // thread_safe_submaps_ =
  // std::make_shared<ThreadSafeSubmapCollection>(submaps_);

  // Camera.
  auto camera = std::make_shared<Camera>(
      config_utilities::getConfigFromRos<Camera::Config>(defaultNh("camera")));

  // Lidar
  auto points3d = std::make_shared<Points3d>(
      config_utilities::getConfigFromRos<Points3d::Config>(
          defaultNh("points3d")));

  // Label Handler.
  std::shared_ptr<LabelHandlerBase> label_handler =
      config_utilities::FactoryRos::create<LabelHandlerBase>(
          defaultNh("label_handler"));

  // Setup the number of labels.
  FixedCountVoxel::setNumCounts(label_handler->numberOfLabels());

  if (config_.verbosity >= 3) {
    label_handler->printLabels(config_.label_info_print_path);
  }

  // Globals.
  globals_ = std::make_shared<Globals>(camera, label_handler, points3d);

  // Submap Allocation.
  std::shared_ptr<SubmapAllocatorBase> submap_allocator =
      config_utilities::FactoryRos::create<SubmapAllocatorBase>(
          defaultNh("submap_allocator"));
  std::shared_ptr<FreespaceAllocatorBase> freespace_allocator =
      config_utilities::FactoryRos::create<FreespaceAllocatorBase>(
          defaultNh("freespace_allocator"));

  // ID Tracking.
  id_tracker_ = config_utilities::FactoryRos::create<IDTrackerBase>(
      defaultNh("id_tracker"), globals_);
  id_tracker_->setSubmapAllocator(submap_allocator);
  id_tracker_->setFreespaceAllocator(freespace_allocator);

  // Tsdf Integrator.
  std::unique_lock<std::mutex> label_lock(label_voxel_mutex_);
  tsdf_integrator_ = config_utilities::FactoryRos::create<TsdfIntegratorBase>(
      defaultNh("tsdf_integrator"), globals_);
  label_lock.unlock();

  // Map Manager.
  map_manager_ = config_utilities::FactoryRos::create<MapManagerBase>(
      defaultNh("map_management"));
  // Visualization.
  ros::NodeHandle visualization_nh(nh_private_, "visualization");

  // Submaps.
  submap_visualizer_ = config_utilities::FactoryRos::create<SubmapVisualizer>(
      defaultNh("vis_submaps"), globals_);
  submap_visualizer_->setGlobalFrameName(config_.global_frame_name);

  // Tracking.
  tracking_visualizer_ = std::make_unique<SpatialTrackingVisualizer>(
      config_utilities::getConfigFromRos<SpatialTrackingVisualizer::Config>(
          defaultNh("vis_tracking")));
  tracking_visualizer_->registerIDTracker(id_tracker_.get());

  // Planning.
  setupCollectionDependentMembers();

  // Data Logging.
  data_logger_ = config_utilities::FactoryRos::create<DataWriterBase>(
      defaultNh("data_writer"));

  // Setup all requested inputs from all modules.
  InputData::InputTypes requested_inputs;
  std::vector<InputDataUser*> input_data_users = {
      id_tracker_.get(), tsdf_integrator_.get(), submap_allocator.get(),
      freespace_allocator.get()};
  for (const InputDataUser* input_data_user : input_data_users) {
    requested_inputs.insert(input_data_user->getRequiredInputs().begin(),
                            input_data_user->getRequiredInputs().end());
  }

  compute_vertex_map_ =
      requested_inputs.find(InputData::InputType::kVertexMap) !=
      requested_inputs.end();
  compute_validity_image_ =
      requested_inputs.find(InputData::InputType::kValidityImage) !=
      requested_inputs.end();

  // Setup the input synchronizer.
  input_synchronizer_ = std::make_unique<InputSynchronizer>(
      config_utilities::getConfigFromRos<InputSynchronizer::Config>(
          nh_private_),
      nh_);
  input_synchronizer_->requestInputs(requested_inputs);
}

void PanopticMapper::setupCollectionDependentMembers() {
  // Planning Interface.
  planning_interface_ = std::make_shared<PlanningInterface>(submaps_);

  // Planning Visualizer.
  planning_visualizer_ = std::make_unique<PlanningVisualizer>(
      config_utilities::getConfigFromRos<PlanningVisualizer::Config>(
          defaultNh("vis_planning")),
      planning_interface_);
  planning_visualizer_->setGlobalFrameName(config_.global_frame_name);
}

void PanopticMapper::setupRos() {
  // Setup all input topics.
  input_synchronizer_->advertiseInputTopics();

  // Services.
  save_map_srv_ = nh_private_.advertiseService(
      "save_map", &PanopticMapper::saveMapCallback, this);
  load_map_srv_ = nh_private_.advertiseService(
      "load_map", &PanopticMapper::loadMapCallback, this);
  set_visualization_mode_srv_ = nh_private_.advertiseService(
      "set_visualization_mode", &PanopticMapper::setVisualizationModeCallback,
      this);
  print_timings_srv_ = nh_private_.advertiseService(
      "print_timings", &PanopticMapper::printTimingsCallback, this);
  finish_mapping_srv_ = nh_private_.advertiseService(
      "finish_mapping", &PanopticMapper::finishMappingCallback, this);
  save_mesh_srv_ = nh_private_.advertiseService(
      "save_mesh", &PanopticMapper::saveMeshCallback, this);
  // Timers.
  if (config_.visualization_interval > 0.0) {
    visualization_timer_ = nh_private_.createTimer(
        ros::Duration(config_.visualization_interval),
        &PanopticMapper::publishVisualizationCallback, this);
  }
  if (config_.data_logging_interval > 0.0) {
    data_logging_timer_ =
        nh_private_.createTimer(ros::Duration(config_.data_logging_interval),
                                &PanopticMapper::dataLoggingCallback, this);
  }
  if (config_.print_timing_interval > 0.0) {
    print_timing_timer_ =
        nh_private_.createTimer(ros::Duration(config_.print_timing_interval),
                                &PanopticMapper::dataLoggingCallback, this);
  }
  input_timer_ =
      nh_private_.createTimer(ros::Duration(config_.check_input_interval),
                              &PanopticMapper::inputCallback, this);
}

void PanopticMapper::inputCallback(const ros::TimerEvent&) {
  if (input_synchronizer_->hasInputData()) {
    std::shared_ptr<InputData> data = input_synchronizer_->getInputData();
    if (data) {
      processInput(data.get());
      if (config_.shutdown_when_finished) {
        last_input_ = ros::Time::now();
        got_a_frame_ = true;
      }
    }
  } else {
    if (config_.shutdown_when_finished && got_a_frame_ &&
        (ros::Time::now() - last_input_).toSec() >= 3.0) {
      // No more frames, finish up.
      LOG_IF(INFO, config_.verbosity >= 1)
          << "No more frames received for 3 seconds, shutting down.";
      finishMapping();
      if (config_.visualization_interval < 0.f) {
        Timer vis_timer("input/visualization");
        publishVisualizationCallback(ros::TimerEvent());
      }
      LOG_IF(INFO, config_.verbosity >= 1) << "Finished at the end";
      ros::shutdown();
    }
  }
}

bool PanopticMapper::create_directory_if_not_exists(
    const std::string& folder_path) {
  // Check if the folder exists
  if (!std::filesystem::exists(folder_path)) {
    // If it doesn't exist, try to create it recursively
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

void PanopticMapper::processInput(InputData* input) {
  CHECK_NOTNULL(input);
  Timer timer("input");
  // frame_timer_ = std::make_unique<Timer>("frame");

  {
    std::unique_lock<std::mutex> lock(frame_count_mutex_);
    frame_count_++;
    lock.unlock();
  }

  input->setColorImage(input->rawImage());

  // generate the points3d data structure either from the rgbd or from the
  // lidar-camera setting up
  Colors colors;
  Labels labels;
  Pointcloud points;
  cv::Mat depth_img;
  if (globals_->points3d()->getConfig().sensor_setup == "rgbd") {
    globals_->points3d()->convert2Points3d(
        input->depthImage(), input->colorImage(), globals_->camera(),
        input->idImage(), &points, &colors, &labels);

  } else if (globals_->points3d()->getConfig().sensor_setup ==
                 "kitti_lidar_camera" ||
             globals_->points3d()->getConfig().sensor_setup == "kitti_lidar") {
    globals_->points3d()->convert2Points3d(
        input->lidarPoints(), input->colorImage(), globals_->camera(),
        input->idImage(), input->kittiLabels(), &points, &colors, &labels,
        &depth_img);
  }

  input->setPoints(points);
  input->setColors(colors);
  input->setLabels(labels);

  if (config_.generate_gt_merged_ply) {
    const Transformation T_M_C = input->T_M_C();
    for (int i = 0; i < input->points().size(); ++i) {
      // Transform the point to the map frame (T_M_C * p_C = p_M).
      const Point point_i_M = T_M_C * input->points()[i];
      ColorLidarPoint pt_rgb;
      pt_rgb.x = point_i_M(0, 0);
      pt_rgb.y = point_i_M(1, 0);
      pt_rgb.z = point_i_M(2, 0);
      pt_rgb.r = input->colors()[i].r;
      pt_rgb.g = input->colors()[i].g;
      pt_rgb.b = input->colors()[i].b;
      merged_cloud_ptr->push_back(pt_rgb);
    }
    return;
  }


  std::unique_lock<std::mutex> label_lock(label_voxel_mutex_);
  // Track the segmentation images and allocate new submaps.
  Timer id_timer("input/id_tracking");
  id_tracker_->processInput(submaps_.get(), input);
  ros::WallTime t1 = ros::WallTime::now();
  id_timer.Stop();

  // Integrate the images.
  Timer tsdf_timer("input/tsdf_integration");
  tsdf_integrator_->processInput(submaps_.get(), input);
  ros::WallTime t2 = ros::WallTime::now();
  tsdf_timer.Stop();

  // Perform all requested map management actions.
  Timer management_timer("input/map_management");
  map_manager_->tick(submaps_.get());
  ros::WallTime t3 = ros::WallTime::now();
  management_timer.Stop();

  if (request_finish_mapping_) {
    finishMapping();
  }

  // If requested perform visualization and logging.
  if (config_.visualization_interval < 0.f) {
    Timer vis_timer("input/visualization");
    publishVisualizationCallback(ros::TimerEvent());
  }
  label_lock.unlock();


  // Logging.
  timer.Stop();

  LOG_IF(INFO, config_.print_timing_interval < 0.0) << "\n" << Timing::Print();
  if (request_finish_mapping_) {
    ros::shutdown();
  }
}

void PanopticMapper::finishMapping() {
  if (config_.generate_gt_merged_ply) {
    std::cout<<"save merged ply file"<<std::endl;
    std::cout<<"before filter merged ply size: "<<merged_cloud_ptr->size()<<std::endl;
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(merged_cloud_ptr);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    sor.filter(*cloud_filtered);
    pcl::io::savePLYFileASCII(config_.merged_ply_file, *cloud_filtered);
    std::cout << "successfully saved merged ply file: "
              << config_.merged_ply_file << std::endl;
  }
  map_manager_->finishMapping(submaps_.get());
  // set the visualization mpode to vis inactive submaps since we deactive all
  // submaps this can also help us to republish everything
  SubmapVisualizer::VisualizationMode visualization_mode =
      SubmapVisualizer::visualizationModeFromString("inactive");
  submap_visualizer_->setVisualizationMode(visualization_mode);
  submap_visualizer_->visualizeAll(submaps_.get());
  saveMesh(config_.save_mesh_folder_path);
  LOG_IF(INFO, config_.verbosity >= 1) << "Finished mapping.";
}

void PanopticMapper::publishVisualization() {

  submap_visualizer_->visualizeAll(submaps_.get());

}

bool PanopticMapper::saveMap(const std::string& file_path) {
  bool success = submaps_->saveToFile(file_path);
  LOG_IF(INFO, success) << "Successfully saved " << submaps_->size()
                        << " submaps to '" << file_path << "'.";
  return success;
}

bool PanopticMapper::loadMap(const std::string& file_path) {
  auto loaded_map = std::make_shared<SubmapCollection>();

  // Load the map.
  if (!loaded_map->loadFromFile(file_path, true)) {
    return false;
  }

  // Loaded submaps are 'from the past' so set them to inactive.
  for (Submap& submap : *loaded_map) {
    submap.finishActivePeriod();
    if (config_.load_submaps_conservative) {
      submap.setChangeState(ChangeState::kUnobserved);
    } else {
      submap.setChangeState(ChangeState::kPersistent);
    }
  }

  if (config_.loaded_freespace_stays_active) {
    const int freespace_id = loaded_map->getActiveFreeSpaceSubmapID();
    if (loaded_map->submapIdExists(freespace_id)) {
      Submap* freespace =
          loaded_map->getSubmapPtr(loaded_map->getActiveFreeSpaceSubmapID());
      freespace->setIsActive(true);
    } else {
      LOG(WARNING) << "No active freespace submap found at loading.";
    }
  } else {
    loaded_map->setActiveFreeSpaceSubmapID(-1);
  }

  // Set the map.
  submaps_ = loaded_map;

  // Setup the interfaces that use the new collection.
  setupCollectionDependentMembers();

  // Reproduce the mesh and visualization.
  submap_visualizer_->clearMesh();
  submap_visualizer_->reset();
  submap_visualizer_->visualizeAll(submaps_.get());

  LOG_IF(INFO, config_.verbosity >= 1)
      << "Successfully loaded " << submaps_->size() << " submaps.";
  return true;
}

bool PanopticMapper::saveMesh(const std::string& folder_path) {
  std::vector<int> suc_ids;
  bool success = submaps_->saveMeshToFile(folder_path, suc_ids);
  LOG_IF(INFO, success) << "Successfully saved " << submaps_->size()
                        << " submaps mesh to folder '" << folder_path << "'.";
  std::cout << "we successfully saved submaps: " << std::endl;
  for (auto id : suc_ids) {
    std::cout << id << " , ";
  }

  // save the submap info to a txt file
  // Set output file path
  std::string outputFilePath = config_.submap_info_path;
  // Create output directory and any missing parent directories
  std::filesystem::create_directories(
      std::filesystem::path(outputFilePath).parent_path());
  // Open output file
  std::ofstream outputFile(outputFilePath);

  for (auto id : suc_ids) {
    if (id == 0) continue;
    Submap* submap = submaps_->getSubmapPtr(id);
    // Write output to file
    outputFile << submap->toStringNoCoutColoring() << std::endl;
  }

  // Close output file
  outputFile.close();
  return success;
}

void PanopticMapper::dataLoggingCallback(const ros::TimerEvent&) {
  data_logger_->writeData(ros::Time::now().toSec(), *submaps_);
}

void PanopticMapper::publishVisualizationCallback(const ros::TimerEvent&) {
  publishVisualization();
}

bool PanopticMapper::setVisualizationModeCallback(
    panoptic_mapping_msgs::SetVisualizationMode::Request& request,
    panoptic_mapping_msgs::SetVisualizationMode::Response& response) {
  response.visualization_mode_set = false;
  response.color_mode_set = false;
  bool success = true;

  // Set the visualization mode if requested.
  if (!request.visualization_mode.empty()) {
    SubmapVisualizer::VisualizationMode visualization_mode =
        SubmapVisualizer::visualizationModeFromString(
            request.visualization_mode);
    submap_visualizer_->setVisualizationMode(visualization_mode);
    std::string visualization_mode_is =
        SubmapVisualizer::visualizationModeToString(visualization_mode);
    LOG_IF(INFO, config_.verbosity >= 2)
        << "Set visualization mode to '" << visualization_mode_is << "'.";
    response.visualization_mode_set =
        visualization_mode_is == request.visualization_mode;
    if (!response.visualization_mode_set) {
      success = false;
    }
  }

  // Set the color mode if requested.
  if (!request.color_mode.empty()) {
    SubmapVisualizer::ColorMode color_mode =
        SubmapVisualizer::colorModeFromString(request.color_mode);
    submap_visualizer_->setColorMode(color_mode);
    std::string color_mode_is = SubmapVisualizer::colorModeToString(color_mode);
    LOG_IF(INFO, config_.verbosity >= 2)
        << "Set color mode to '" << color_mode_is << "'.";
    response.color_mode_set = color_mode_is == request.color_mode;
    if (!response.color_mode_set) {
      success = false;
    }
  }

  // Republish the visualization.
  submap_visualizer_->visualizeAll(submaps_.get());
  return success;
}

bool PanopticMapper::saveMapCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  response.success = saveMap(request.file_path);
  return response.success;
}

bool PanopticMapper::loadMapCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  response.success = loadMap(request.file_path);
  return response.success;
}

bool PanopticMapper::saveMeshCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  response.success = saveMesh(request.file_path);
  return response.success;
}

bool PanopticMapper::printTimingsCallback(std_srvs::Empty::Request& request,
                                          std_srvs::Empty::Response& response) {
  printTimings();
  return true;
}

void PanopticMapper::printTimingsCallback(const ros::TimerEvent&) {
  printTimings();
}

void PanopticMapper::printTimings() const { LOG(INFO) << Timing::Print(); }

bool PanopticMapper::finishMappingCallback(
    std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
  LOG(INFO) << "get finish mapping request";
  request_finish_mapping_ = true;
  // finishMapping();
  return true;
}

ros::NodeHandle PanopticMapper::defaultNh(const std::string& key) const {
  // Essentially just read the default namespaces list and type params.
  // NOTE(schmluk): Since these lookups are quasi-static we don't check for
  // correct usage here.
  const std::pair<std::string, std::string>& ns_and_type =
      default_names_and_types_.at(key);
  ros::NodeHandle nh_out(nh_private_, ns_and_type.first);
  if (!ns_and_type.second.empty() && !nh_out.hasParam("type")) {
    nh_out.setParam("type", ns_and_type.second);
  }
  return nh_out;
}

}  // namespace panoptic_mapping
