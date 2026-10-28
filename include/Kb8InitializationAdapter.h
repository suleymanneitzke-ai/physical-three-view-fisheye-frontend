#ifndef STRUCTURE_SLAM_KB8_INITIALIZATION_ADAPTER_H
#define STRUCTURE_SLAM_KB8_INITIALIZATION_ADAPTER_H

#include "Frame.h"
#include "camera/KannalaBrandt8Camera.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace StructureSLAM
{

struct Kb8InitializationView
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<unsigned char> valid;
    cv::Mat K;
};

bool BuildKb8InitializationView(
    const Frame& frame,
    const KannalaBrandt8Camera::Parameters& params,
    Kb8InitializationView& view,
    std::string& error_message);

} // namespace StructureSLAM

#endif
