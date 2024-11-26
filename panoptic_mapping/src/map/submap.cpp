#include "panoptic_mapping/map/submap.h"

#include <memory>
#include <sstream>
#include <vector>

#include <cblox/QuatTransformation.pb.h>
#include <cblox/utils/quat_transformation_protobuf_utils.h>
#include <voxblox/io/layer_io.h>

#include "panoptic_mapping/map_management/layer_manipulator.h"
#include "panoptic_mapping/tools/serialization.h"

namespace panoptic_mapping {

void Submap::Config::checkParams() const {
  checkParamGT(voxel_size, 0.f, "voxel_size");
  checkParamNE(truncation_distance, 0.f, "truncation_distance");
  checkParamCond(voxels_per_side % 2 == 0,
                 "voxels_per_side is required to be a multiple of 2.");
  checkParamGT(voxels_per_side, 0, "voxels_per_side");
  checkParamConfig(mesh);
  if (classification.isSetup()) {
    checkParamConfig(classification);
  }
  if (scores.isSetup()) {
    checkParamConfig(scores);
  }
}

void Submap::Config::initializeDependentVariableDefaults() {
  if (truncation_distance < 0.f) {
    truncation_distance *= -voxel_size;
  }
}

void Submap::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("voxel_size", &voxel_size);
  setupParam("truncation_distance", &truncation_distance);
  setupParam("voxels_per_side", &voxels_per_side);
  setupParam("classification", &classification, "classification");
  // setupParam("instance_probs",instance_probs,"instance_probs");
  setupParam("scores", &scores, "scores");
  setupParam("mesh", &mesh, "mesh");
}

bool Submap::Config::useClassLayer() const {
  return classification.isSetup() && classification.type() != "null";
}

bool Submap::Config::useScoreLayer() const {
  return scores.isSetup() && scores.type() != "null";
}

Submap::Submap(const Config& config, SubmapIDManager* submap_id_manager,
               InstanceIDManager* instance_id_manager)
    : config_(config.checkValid()),
      bounding_volume_(*this),
      id_(submap_id_manager),
      instance_id_(instance_id_manager) {
  initialize();
}

Submap::Submap(const Config& config, SubmapIDManager* submap_id_manager,
               InstanceIDManager* instance_id_manager, int submap_id)
    : config_(config.checkValid()),
      bounding_volume_(*this),
      id_(submap_id, submap_id_manager),
      instance_id_(instance_id_manager) {
  initialize();
}

void Submap::initialize() {
  // Default values.
  std::stringstream ss;
  ss << "submap_" << static_cast<int>(id_);
  frame_name_ = ss.str();

  // Initialize with identity transformation.
  T_M_S_.setIdentity();
  T_M_S_inv_.setIdentity();

  // Setup layers.
  tsdf_layer_ =
      std::make_shared<TsdfLayer>(config_.voxel_size, config_.voxels_per_side);
  mesh_layer_ =
      std::make_shared<MeshLayer>(config_.voxel_size * config_.voxels_per_side);
  if (config_.useClassLayer()) {
    class_layer_ = config_.classification.create(config_.voxel_size,
                                                 config_.voxels_per_side);
    has_class_layer_ = true;
  }
  if (config_.useScoreLayer()) {
    score_layer_ =
        config_.scores.create(config_.voxel_size, config_.voxels_per_side);
    has_score_layer_ = true;
  }

  // Setup tools.
  mesh_integrator_ = std::make_unique<MeshIntegrator>(
      config_.mesh, tsdf_layer_, mesh_layer_, class_layer_,
      config_.truncation_distance);
}

void Submap::setT_M_S(const Transformation& T_M_S) {
  T_M_S_ = T_M_S;
  T_M_S_inv_ = T_M_S_.inverse();
}

void Submap::getProto(SubmapProto* proto) const {
  CHECK_NOTNULL(proto);
  // Store Submap data.
  proto->set_instance_id(instance_id_);
  proto->set_class_id(class_id_);
  proto->set_panoptic_label(static_cast<int>(label_));
  proto->set_name(name_);
  proto->set_change_state(static_cast<int>(change_state_));

  // Store TSDF data.
  proto->set_num_blocks(tsdf_layer_->getNumberOfAllocatedBlocks());
  proto->set_voxel_size(config_.voxel_size);
  proto->set_voxels_per_side(config_.voxels_per_side);
  proto->set_truncation_distance(config_.truncation_distance);

  // Store classification data.
  if (has_class_layer_) {
    proto->set_class_voxel_type(static_cast<int>(class_layer_->getVoxelType()));
    proto->set_num_class_blocks(class_layer_->getNumberOfAllocatedBlocks());
  } else {
    proto->set_num_class_blocks(0);
  }

  // Store classification data.
  if (has_score_layer_) {
    proto->set_score_voxel_type(static_cast<int>(score_layer_->getVoxelType()));
    proto->set_num_score_blocks(score_layer_->getNumberOfAllocatedBlocks());
  } else {
    proto->set_num_score_blocks(0);
  }

  // Store transformation data.
  auto transformation_proto_ptr = new cblox::QuatTransformationProto();
  cblox::conversions::transformKindrToProto(T_M_S_, transformation_proto_ptr);
  proto->set_allocated_transform(transformation_proto_ptr);
  proto->set_frame_name(frame_name_);
}

bool Submap::saveToStream(std::fstream* outfile_ptr) const {
  CHECK_NOTNULL(outfile_ptr);
  // Saving the submap header.
  SubmapProto submap_proto;
  getProto(&submap_proto);
  if (!voxblox::utils::writeProtoMsgToStream(submap_proto, outfile_ptr)) {
    LOG(ERROR) << "Could not write submap proto message.";
    outfile_ptr->close();
    return false;
  }

  // TSDF Layer.
  constexpr bool kIncludeAllBlocks = true;
  const TsdfLayer& tsdf_layer = *tsdf_layer_;
  if (!tsdf_layer.saveBlocksToStream(kIncludeAllBlocks,
                                     voxblox::BlockIndexList(), outfile_ptr)) {
    LOG(ERROR) << "Could not write submap tsdf blocks to stream.";
    outfile_ptr->close();
    return false;
  }

  // Class Layer.
  if (has_class_layer_) {
    if (!class_layer_->saveBlocksToStream(
            kIncludeAllBlocks, voxblox::BlockIndexList(), outfile_ptr)) {
      LOG(ERROR) << "Could not write submap classification blocks to stream.";
      outfile_ptr->close();
      return false;
    }
  }

  // Score Layer.
  if (has_score_layer_) {
    if (!score_layer_->saveBlocksToStream(
            kIncludeAllBlocks, voxblox::BlockIndexList(), outfile_ptr)) {
      LOG(ERROR) << "Could not write submap score blocks to stream.";
      outfile_ptr->close();
      return false;
    }
  }
  return true;
}

std::unique_ptr<Submap> Submap::loadFromStream(
    std::istream* proto_file_ptr, uint64_t* tmp_byte_offset_ptr,
    SubmapIDManager* id_manager, InstanceIDManager* instance_manager) {
  CHECK_NOTNULL(proto_file_ptr);
  CHECK_NOTNULL(tmp_byte_offset_ptr);

  // Getting the header for this submap.
  SubmapProto submap_proto;
  if (!voxblox::utils::readProtoMsgFromStream(proto_file_ptr, &submap_proto,
                                              tmp_byte_offset_ptr)) {
    LOG(ERROR) << "Could not read tsdf submap protobuf message.";
    return nullptr;
  }

  // Creating a new submap to hold the data.
  Config cfg;
  cfg.voxel_size = submap_proto.voxel_size();
  cfg.voxels_per_side = submap_proto.voxels_per_side();
  cfg.truncation_distance = submap_proto.truncation_distance();
  auto submap = std::make_unique<Submap>(cfg, id_manager, instance_manager);

  // Load the submap data.
  submap->has_class_layer_ = submap_proto.num_class_blocks() > 0;
  submap->has_score_layer_ = submap_proto.num_score_blocks() > 0;
  submap->setInstanceID(submap_proto.instance_id());
  submap->setClassID(submap_proto.class_id());
  submap->setLabel(static_cast<PanopticLabel>(submap_proto.panoptic_label()));
  submap->setName(submap_proto.name());
  submap->setChangeState(static_cast<ChangeState>(submap_proto.change_state()));

  // Load the TSDF layer.
  if (!voxblox::io::LoadBlocksFromStream(
          submap_proto.num_blocks(), TsdfLayer::BlockMergingStrategy::kReplace,
          proto_file_ptr, submap->tsdf_layer_.get(), tmp_byte_offset_ptr)) {
    LOG(ERROR) << "Could not load the tsdf blocks from stream.";
    return nullptr;
  }

  // Load the classification layer.
  if (submap_proto.num_class_blocks() > 0) {
    submap->class_layer_ = loadClassLayerFromStream(
        submap_proto, proto_file_ptr, tmp_byte_offset_ptr);
    if (!submap->class_layer_) {
      LOG(ERROR) << "Could not load the classification layer from stream.";
      return nullptr;
    }
  }

  // Load the score layer.
  if (submap_proto.num_score_blocks() > 0) {
    submap->score_layer_ = loadScoreLayerFromStream(
        submap_proto, proto_file_ptr, tmp_byte_offset_ptr);
    if (!submap->score_layer_) {
      LOG(ERROR) << "Could not load the score layer from stream.";
      return nullptr;
    }
  }

  // Load the transformation.
  Transformation T_M_S;
  cblox::QuatTransformationProto transformation_proto =
      submap_proto.transform();
  cblox::conversions::transformProtoToKindr(transformation_proto, &T_M_S);
  submap->setT_M_S(T_M_S);
  submap->setFrameName(submap_proto.frame_name());

  return submap;
}

void Submap::finishActivePeriod() {
  if (!is_active_) {
    return;
  }
  is_active_ = false;
  // Since the submap was active just before we assume it still exists.
  change_state_ = ChangeState::kPersistent;
  updateEverything();
}

void Submap::updateEverything(bool only_updated_blocks) {
  updateBoundingVolume();
  updateMesh(only_updated_blocks);
  computeIsoSurfacePoints();
}

void Submap::updateMesh(bool only_updated_blocks, bool use_class_layer) {
  // Use the default integrator config to have color always available.
  mesh_integrator_->generateMesh(only_updated_blocks, true,
                                 has_class_layer_ && use_class_layer);
}

void Submap::computeIsoSurfacePoints() {
  iso_surface_points_ = std::vector<IsoSurfacePoint>();

  // Create an interpolator to interpolate the vertex weights from the TSDF.
  voxblox::Interpolator<TsdfVoxel> interpolator(tsdf_layer_.get());

  // Extract the vertices and verify.
  voxblox::BlockIndexList index_list;
  mesh_layer_->getAllAllocatedMeshes(&index_list);
  int ignored_points = 0;
  for (const voxblox::BlockIndex& index : index_list) {
    const Pointcloud& vertices = mesh_layer_->getMeshByIndex(index).vertices;
    iso_surface_points_.reserve(iso_surface_points_.size() + vertices.size());
    for (const Point& vertex : vertices) {
      // Try to interpolate the voxel weight and verify the distance.
      TsdfVoxel voxel;
      if (interpolator.getVoxel(vertex, &voxel, true)) {
        // if (voxel.distance > 0.1 * config_.voxel_size) {
        //   ignored_points++;
        // } else {
        iso_surface_points_.emplace_back(vertex, voxel.weight);
        // }
      }
    }
  }
  if (ignored_points > 0) {
    LOG(WARNING) << "Submap " << static_cast<int>(id_) << " (" << name_
                 << ") has " << ignored_points
                 << " iso-surface points with a distance > "
                 << 0.1 * config_.voxel_size << ", these will be ignored.";
  }
}

void Submap::updateBoundingVolume() { bounding_volume_.update(); }

bool Submap::applyClassLayer(const LayerManipulator& manipulator,
                             bool clear_class_layer) {
  if (!has_class_layer_) {
    return true;
  }
  manipulator.applyClassificationLayer(tsdf_layer_.get(), *class_layer_,
                                       config_.truncation_distance);
  if (clear_class_layer) {
    class_layer_.reset();
    has_class_layer_ = false;
  }
  updateEverything();
  return tsdf_layer_->getNumberOfAllocatedBlocks() != 0;
}

std::unique_ptr<Submap> Submap::clone(
    SubmapIDManager* submap_id_manager,
    InstanceIDManager* instance_id_manager) const {
  auto result = std::unique_ptr<Submap>(
      new Submap(config_, submap_id_manager, instance_id_manager, getID()));

  // Copy all members.
  result->instance_id_ = static_cast<int>(instance_id_);
  result->class_id_ = class_id_;
  result->label_ = label_;
  result->name_ = name_;
  result->is_active_ = is_active_;
  result->was_tracked_ = was_tracked_;
  result->has_class_layer_ = has_class_layer_;
  result->has_score_layer_ = has_score_layer_;
  result->change_state_ = change_state_;
  result->frame_name_ = frame_name_;
  result->T_M_S_ = T_M_S_;
  result->T_M_S_inv_ = T_M_S_inv_;
  result->iso_surface_points_ = iso_surface_points_;

  // Deep copy all pointers.
  result->tsdf_layer_ = std::make_shared<TsdfLayer>(*tsdf_layer_);
  result->mesh_layer_ = std::make_shared<MeshLayer>(*mesh_layer_);
  if (class_layer_) {
    result->class_layer_ = class_layer_->clone();
  }
  if (score_layer_) {
    result->score_layer_ = score_layer_->clone();
  }
  result->mesh_integrator_ = std::make_unique<MeshIntegrator>(
      result->config_.mesh, result->tsdf_layer_, result->mesh_layer_,
      result->class_layer_, result->config_.truncation_distance);

  // The bounding volume can not completely be copied so it's just updated,
  // which should be identical.
  result->bounding_volume_.update();

  return result;
}

}  // namespace panoptic_mapping
