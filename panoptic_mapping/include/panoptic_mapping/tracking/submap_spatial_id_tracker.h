#ifndef PANOPTIC_MAPPING_TRACKING_SUBMAP_SPATIAL_ID_TRACKER_H_
#define PANOPTIC_MAPPING_TRACKING_SUBMAP_SPATIAL_ID_TRACKER_H_

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/labels/label_handler_base.h"
#include "panoptic_mapping/map/classification/class_layer.h"
#include "panoptic_mapping/map/submap.h"
#include "panoptic_mapping/map/submap_collection.h"
#include "panoptic_mapping/tools/map_renderer.h"
#include "panoptic_mapping/tools/text_colors.h"
#include "panoptic_mapping/tracking/id_tracker_base.h"
#include "panoptic_mapping/tracking/tracking_info.h"
#include "panoptic_mapping/common/segment.h"
namespace panoptic_mapping {

struct MatchInfo {
  MatchInfo(Submap* submap, float prob) : submap_info_(submap), prob_(prob){};
  Submap* submap_info_;
  float prob_;
};

/**
 * @brief Uses the input segmentation images and compares them against the
 * rendered map to track associations.
 */
class SubmapSpatialIDTracker : public IDTrackerBase {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    // Submap allocation config. Set use class_layer to true to perform label
    // integration.
    Submap::Config submap;

    // Which tracking metric to compute. Supported are 'IoU' and 'overlap'.
    std::string tracking_metric = "IoU";

    // Accept matches that have at least this value in the computed trackign
    // metric.
    float match_acceptance_threshold = 0.12;

    // True: Only match submaps and masks that have identical class labels.
    // False: Match any mask to the highest metric submap.
    bool use_class_data_for_matching = true;

    // // Only allocate new submaps for masks that have at least this many
    // pixels.
    int min_allocation_size_background = 0;

    int min_allocation_size_small = 0;

    int min_allocation_size_middle = 0;

    int min_allocation_size_large = 0;

    // Number of threads to use to track submaps in parallel.
    int rendering_threads = std::thread::hardware_concurrency();

    // whether use detectron labels
    bool use_detectron = true;

    // whether use kitti_semantic_panoptic_lidar
    bool use_kitti = false;

    // Count iso-surfaces points as valid whose distance(abs(sdf)) value is
    // within this distance in meters of the measured depth. Negative values
    // indicate multiples of the voxel size.
    float depth_tolerance = -1.0;

    // Renderer settings. The renderer is only used for visualization purposes.
    MapRenderer::Config renderer;

    bool only_active_submaps = false;

    bool generate_gt_maps = false;

    Config() { setConfigName("SubmapSpatialIDTracker"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  SubmapSpatialIDTracker(const Config& config, std::shared_ptr<Globals> globals,
                   bool print_config = true);
  ~SubmapSpatialIDTracker() override = default;

  void processInput(SubmapCollection* submaps, InputData* input) override;

 protected:
  // Setup utility
  void setup(SubmapCollection* submaps);
  // // Internal methods.
  bool classesMatch(int input_id, int submap_class_id);
  // Submap* allocateSubmap(int input_id, SubmapCollection* submaps,
  //                                InputData* input);
  // TrackingInfoAggregator computeTrackingData(SubmapCollection* submaps,
  //                                            InputData* input);
  void computeTrackingData(SubmapCollection* submaps, InputData* input);
  TrackingInfoAggregator MultiThreadcomputeTrackingData(
      SubmapCollection* submaps, InputData* input);
  TrackingInfo renderTrackingInfoPoints3d(const Submap& submap,
                                          const InputData& input) const;
  void spatialMatching(SubmapCollection* submaps, Point position_M,
                       Label label);
  std::vector<int> getInputIDs() const;
  bool getAllMetrics(int input_id, std::vector<std::pair<int, float>>* id_value,
                     const std::string& metric) const;
  float computOverlap(int input_id, int submap_id) const;
  float computIoU(int input_id, int submap_id) const;
  std::function<float(int, int)> getComputeValueFunction(
      const std::string& metric) const;
  int getNumberOfInputPixels(int input_id) const;
  void parseDetectronClasses(InputData* input);
  std::vector<int> getInstanceList() { return instance_tobe_manage_list_; }
  Submap* allocateSubmap(int input_id, SubmapCollection* submaps,
                         InputData* input);
  ColorLidarPointCloud::Ptr GenerateInputPointCloud(InputData* input);
  void GenerateSegmentationPointCloud(Segments* segments);
  bool create_directory_if_not_exists(const std::string& folder_path);
  void MatchingAndCreateNewSubmap(TrackingInfoAggregator* tracking_data,
                                  SubmapCollection* submaps, InputData* input);
  void removeSky(InputData* input);
  void CreateGtNewSubmap(SubmapCollection* submaps, InputData* input);

 private:
  static config_utilities::Factory::RegistrationRos<
      IDTrackerBase, SubmapSpatialIDTracker, std::shared_ptr<Globals>>
      registration_;

  // Members
  const Config config_;

  int map_id_;
  bool is_setup_ = false;
  std::unordered_map<int, std::unordered_map<int, int>>
      overlap_;  // <input_id, <rendered_id, count>>
  std::unordered_map<int, int> total_input_count_;  // <input_id, count>
  std::unordered_map<int, int>
      total_rendered_count_;  // <rendered_id, count> position id in map
  std::vector<int> instance_list_in_map_;
  std::vector<int> instance_tobe_manage_list_;
  int frame_count_;
  // store a vector of input id which can be mapped to this submap id
  std::unordered_map<int, std::set<int>>
      input_to_map_records_;  //<input_id, map_id_vector>
  std::unordered_map<int, std::set<int>> map_to_input_records_;
  // <map_id, input_id_vector> for debug use input id to map instance id
  std::unordered_map<int, int> background_list_;  //  class_id  map_id

  // Cached labels.
  const DetectronLabels* labels_;  // detectron labels
                                   //  Current frame label propagation.
  std::unordered_map<int, LabelEntry> kitti_labels_;

  std::unordered_map<int, Segment>
      segments_to_integrate_;  //<input_segmentation_id, Segmentation>

  TrackingInfoAggregator tracking_data_;
  std::unordered_map<int, MatchInfo>
      input_to_output_;  // reserved for generate gt submap

 protected:
  MapRenderer renderer_;  // The renderer is only used if visualization is on.
  cv::Mat rendered_vis_;  // Store visualization data.
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_TRACKING_SUBMAP_SPATIAL_ID_TRACKER_H_
