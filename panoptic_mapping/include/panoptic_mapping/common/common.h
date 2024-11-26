#ifndef PANOPTIC_MAPPING_COMMON_COMMON_H_
#define PANOPTIC_MAPPING_COMMON_COMMON_H_

#include <deque>
#include <list>
#include <numeric>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>
#include <voxblox/mesh/mesh_layer.h>
#include <voxblox/utils/timing.h>
namespace panoptic_mapping {
/**
 * @brief Common Type definitions for the full framework.
 */

// Types.
// Type used for counting voxels. This stores up to ~65k measurements so should
// never run out. If this parameter is changed double check that all
// serialization still works!
using ClassificationCount = uint16_t;
// aligned eigen vector
// Aligned Eigen containers
template <typename Type>
using AlignedVector = std::vector<Type, Eigen::aligned_allocator<Type>>;
template <typename Type>
using AlignedDeque = std::deque<Type, Eigen::aligned_allocator<Type>>;
template <typename Type>
using AlignedQueue = std::queue<Type, AlignedDeque<Type>>;
template <typename Type>
using AlignedStack = std::stack<Type, AlignedDeque<Type>>;
template <typename Type>
using AlignedList = std::list<Type, Eigen::aligned_allocator<Type>>;

// Wroking with voxblox maps.
using FloatingPoint = voxblox::FloatingPoint;
using VoxelIndex = voxblox::VoxelIndex;
using BlockIndex = voxblox::BlockIndex;
using Color = voxblox::Color;
using Colors = voxblox::Colors;

// Geometry.
using Point = voxblox::Point;
using Transformation = voxblox::Transformation;
using Pointcloud = voxblox::Pointcloud;
using Quaternion = Eigen::Quaternion<FloatingPoint>;
using Vec3 = Eigen::Matrix<FloatingPoint, 3, 1>;

// Tsdf and mesh Maps. Classification maps are defined in class_layer.h
using TsdfVoxel = voxblox::TsdfVoxel;
using TsdfBlock = voxblox::Block<TsdfVoxel>;
using TsdfLayer = voxblox::Layer<TsdfVoxel>;
using MeshLayer = voxblox::MeshLayer;
using GlobalIndex = voxblox::GlobalIndex;
using LongIndexHash = voxblox::LongIndexHash;
// pcl point type
using LidarPoint = pcl::PointXYZ;
using ColorLidarPoint = pcl::PointXYZRGB;
using LidarPointCloud = pcl::PointCloud<LidarPoint>;
using ColorLidarPointCloud = pcl::PointCloud<ColorLidarPoint>;

constexpr int kKITTIMaxIntstance =
    1000; /**< Used for assign an unqiue panoptic label. */

// label
struct Label;
typedef AlignedVector<Label> Labels;

// Panoptic type labels.
enum class PanopticLabel { kUnknown = 0, kInstance, kBackground, kFreeSpace };
inline std::string panopticLabelToString(const PanopticLabel& label) {
  switch (label) {
    case PanopticLabel::kUnknown:
      return "Unknown";
    case PanopticLabel::kInstance:
      return "Instance";
    case PanopticLabel::kBackground:
      return "Background";
    case PanopticLabel::kFreeSpace:
      return "FreeSpace";
  }
  return "UnknownPanopticLabel";
}
// TODO(thuaj): more efficiency for better data structure
struct Label {
  Label() : sem_label_(0), ins_label_(0), probabilistic_(0), panoptic_id_(0) {}
  Label(int sem_label, int ins_label, float probabilistic, int panoptic_id)
      : sem_label_(sem_label),
        ins_label_(ins_label),
        probabilistic_(probabilistic),
        panoptic_id_(panoptic_id) {}
  // label for kitti
  Label(int sem_label, int ins_label, int id_label, float probabilistic,
        int panoptic_id)
      : sem_label_(sem_label),
        ins_label_(ins_label),
        id_label_(id_label),
        probabilistic_(probabilistic),
        panoptic_id_(panoptic_id) {}
  int sem_label_;              // semantic label
  int ins_label_;              // instance label
  float probabilistic_;        // probabilistic
  int matched_map_ins_label_;  // matched label id in map
  int panoptic_id_;  // 0 for unknown 1 for instance 2 for background 3 for
                     // freespace
  int id_label_;
};

// Iso-surface-points are used to check alignment and represent the surface
// of finished submaps.
struct IsoSurfacePoint {
  IsoSurfacePoint(Point _position, FloatingPoint _weight)
      : position(std::move(_position)), weight(_weight) {}
  Point position;
  FloatingPoint weight;
};

// Change detection data stores relevant information for associating submaps.
enum class ChangeState {
  kNew = 0,
  kMatched,
  kUnobserved,
  kAbsent,
  kPersistent
};

inline std::string changeStateToString(const ChangeState& state) {
  switch (state) {
    case ChangeState::kNew:
      return "New";
    case ChangeState::kMatched:
      return "Mathced";
    case ChangeState::kPersistent:
      return "Persistent";
    case ChangeState::kAbsent:
      return "Absent";
    case ChangeState::kUnobserved:
      return "Unobserved";
    default:
      return "UnknownChangeState";
  }
}

/**
 * Frame names are abbreviated consistently (in paranthesesalternative
 * explanations):
 * S - Submap
 * M - Mission (Map / World)
 * C - Camera (Sensor)
 */

// Timing.
#define PANOPTIC_MAPPING_TIMING_ENABLED  // Unset to disable all timers.
#ifdef PANOPTIC_MAPPING_TIMING_ENABLED
using Timer = voxblox::timing::Timer;
#else
using Timer = voxblox::timing::DummyTimer;
#endif  // PANOPTIC_MAPPING_TIMING_ENABLED
using Timing = voxblox::timing::Timing;

// dense crf matrix
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXf_row;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXi_row;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixX8umy;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorXf_1col;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> VectorXf_1row;
}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_COMMON_H_
