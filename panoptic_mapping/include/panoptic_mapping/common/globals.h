/*
 * @Author: thuaj@connect.ust.hk
 * @Description: DHP-Mapping Global class to manage the camera, label handler and points3d.
 * This part of code is heavily derived from panoptic_mapping.
 * Copyright (c) 2024 by thuaj@connect.ust.hk, All Rights Reserved.
 */

#ifndef PANOPTIC_MAPPING_COMMON_GLOBALS_H_
#define PANOPTIC_MAPPING_COMMON_GLOBALS_H_

#include <memory>
#include <utility>

#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/points3d.h"
#include "panoptic_mapping/labels/label_handler_base.h"

namespace panoptic_mapping {

/**
 * @brief Utility class that provides an interface to globally used components
 * of the system.
 */
class Globals {
 public:
  Globals(std::shared_ptr<Camera> camera,
          std::shared_ptr<LabelHandlerBase> label_handler,
          std::shared_ptr<Points3d> points3d)
      : camera_(std::move(camera)),
        label_handler_(std::move(label_handler)),
        points3d_(std::move(points3d)) {}
  virtual ~Globals() = default;

  // Access.
  const std::shared_ptr<Camera>& camera() const { return camera_; }
  const std::shared_ptr<LabelHandlerBase>& labelHandler() const {
    return label_handler_;
  }
  const std::shared_ptr<Points3d>& points3d() const { return points3d_; }

 private:
  // Components.
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<LabelHandlerBase> label_handler_;
  std::shared_ptr<Points3d> points3d_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_GLOBALS_H_
