/*
 * @Author: thuaj@connect.ust.hk
 * @Date: 2023-07-07 00:52:13
 * @LastEditTime: 2023-09-14 22:03:29
 * @Description:
 * Copyright (c) 2023 by thuaj@connect.ust.hk, All Rights Reserved.
 */
#ifndef VOXBLOX_UTILS_MESHING_UTILS_H_
#define VOXBLOX_UTILS_MESHING_UTILS_H_

#include "voxblox/core/common.h"
#include "voxblox/core/voxel.h"
#include <yaml-cpp/yaml.h>
namespace voxblox {

namespace utils {

template <typename VoxelType>
bool getSdfIfValid(const VoxelType& voxel, const FloatingPoint min_weight,
                   FloatingPoint* sdf);

template <>
inline bool getSdfIfValid(const TsdfVoxel& voxel,
                          const FloatingPoint min_weight, FloatingPoint* sdf) {
  DCHECK(sdf != nullptr);
  if (voxel.weight <= min_weight) {
    return false;
  }
  *sdf = voxel.distance;
  return true;
}

template <>
inline bool getSdfIfValid(const EsdfVoxel& voxel,
                          const FloatingPoint /*min_weight*/,
                          FloatingPoint* sdf) {
  DCHECK(sdf != nullptr);
  if (!voxel.observed) {
    return false;
  }
  *sdf = voxel.distance;
  return true;
}

template <typename VoxelType>
bool getColorIfValid(const VoxelType& voxel, const FloatingPoint min_weight,
                     Color* color);

template <>
inline bool getColorIfValid(const TsdfVoxel& voxel,
                            const FloatingPoint min_weight, Color* color) {
  DCHECK(color != nullptr);
  if (voxel.weight <= min_weight) {
    return false;
  }
  // we change this to vis_color for visualization voxel's semantic class id
  // !!!!CHANGE THIS PART
  // *color = voxel.vis_color;
  *color = voxel.color;
  // exit(0);
  return true;
}

template <>
inline bool getColorIfValid(const EsdfVoxel& voxel,
                            const FloatingPoint /*min_weight*/, Color* color) {
  DCHECK(color != nullptr);
  if (!voxel.observed) {
    return false;
  }
  *color = Color(255u, 255u, 255u);
  return true;
}

}  // namespace utils
}  // namespace voxblox

#endif  // VOXBLOX_UTILS_MESHING_UTILS_H_
