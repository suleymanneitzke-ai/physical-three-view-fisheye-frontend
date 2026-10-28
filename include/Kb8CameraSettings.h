#ifndef STRUCTURE_SLAM_KB8_CAMERA_SETTINGS_H
#define STRUCTURE_SLAM_KB8_CAMERA_SETTINGS_H

#include "camera/KannalaBrandt8Camera.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace StructureSLAM
{

bool IsKannalaBrandt8CameraType(const cv::FileStorage& settings);

bool LoadKb8CameraParametersFromSettings(
    const cv::FileStorage& settings,
    KannalaBrandt8Camera::Parameters& params,
    std::string& error_message);

std::vector<float> Kb8ParametersToFloatVector(
    const KannalaBrandt8Camera::Parameters& params);

cv::Mat Kb8ParametersToPinholeMatrix(
    const KannalaBrandt8Camera::Parameters& params);

bool Kb8ParametersAlmostEqual(
    const KannalaBrandt8Camera::Parameters& lhs,
    const KannalaBrandt8Camera::Parameters& rhs,
    double tolerance);

} // namespace StructureSLAM

#endif
