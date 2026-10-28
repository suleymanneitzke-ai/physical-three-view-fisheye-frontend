#include "Kb8CameraSettings.h"

#include <cmath>
#include <sstream>

namespace StructureSLAM
{
namespace
{

bool ReadDouble(const cv::FileStorage& settings,
                const std::string& key,
                double& value,
                std::string& error_message)
{
    const cv::FileNode node = settings[key];
    if(node.empty())
    {
        error_message = "missing setting: " + key;
        return false;
    }
    value = static_cast<double>(node);
    if(!std::isfinite(value))
    {
        error_message = "non-finite setting: " + key;
        return false;
    }
    return true;
}

bool ReadInt(const cv::FileStorage& settings,
             const std::string& key,
             int& value,
             std::string& error_message)
{
    const cv::FileNode node = settings[key];
    if(node.empty())
    {
        error_message = "missing setting: " + key;
        return false;
    }
    value = static_cast<int>(node);
    return true;
}

bool AlmostEqual(const double lhs,
                 const double rhs,
                 const double tolerance)
{
    return std::abs(lhs - rhs) <= tolerance;
}

} // namespace

bool IsKannalaBrandt8CameraType(const cv::FileStorage& settings)
{
    const cv::FileNode type_node = settings["Camera.type"];
    if(type_node.empty())
        return false;

    std::string camera_type;
    type_node >> camera_type;
    return camera_type == "KannalaBrandt8";
}

bool LoadKb8CameraParametersFromSettings(
    const cv::FileStorage& settings,
    KannalaBrandt8Camera::Parameters& params,
    std::string& error_message)
{
    error_message.clear();

    if(!IsKannalaBrandt8CameraType(settings))
    {
        error_message = "Camera.type is not KannalaBrandt8";
        return false;
    }

    KannalaBrandt8Camera::Parameters loaded;
    if(!ReadDouble(settings, "Camera1.fx", loaded.fx, error_message) ||
       !ReadDouble(settings, "Camera1.fy", loaded.fy, error_message) ||
       !ReadDouble(settings, "Camera1.cx", loaded.cx, error_message) ||
       !ReadDouble(settings, "Camera1.cy", loaded.cy, error_message) ||
       !ReadDouble(settings, "Camera1.k1", loaded.k1, error_message) ||
       !ReadDouble(settings, "Camera1.k2", loaded.k2, error_message) ||
       !ReadDouble(settings, "Camera1.k3", loaded.k3, error_message) ||
       !ReadDouble(settings, "Camera1.k4", loaded.k4, error_message) ||
       !ReadInt(settings, "Camera.width", loaded.width, error_message) ||
       !ReadInt(settings, "Camera.height", loaded.height, error_message))
    {
        return false;
    }

    const KannalaBrandt8Camera camera(loaded);
    if(!camera.valid())
    {
        error_message = camera.validationError();
        return false;
    }

    params = loaded;
    return true;
}

std::vector<float> Kb8ParametersToFloatVector(
    const KannalaBrandt8Camera::Parameters& params)
{
    return {
        static_cast<float>(params.fx),
        static_cast<float>(params.fy),
        static_cast<float>(params.cx),
        static_cast<float>(params.cy),
        static_cast<float>(params.k1),
        static_cast<float>(params.k2),
        static_cast<float>(params.k3),
        static_cast<float>(params.k4)
    };
}

cv::Mat Kb8ParametersToPinholeMatrix(
    const KannalaBrandt8Camera::Parameters& params)
{
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = static_cast<float>(params.fx);
    K.at<float>(1, 1) = static_cast<float>(params.fy);
    K.at<float>(0, 2) = static_cast<float>(params.cx);
    K.at<float>(1, 2) = static_cast<float>(params.cy);
    return K;
}

bool Kb8ParametersAlmostEqual(
    const KannalaBrandt8Camera::Parameters& lhs,
    const KannalaBrandt8Camera::Parameters& rhs,
    const double tolerance)
{
    return AlmostEqual(lhs.fx, rhs.fx, tolerance) &&
           AlmostEqual(lhs.fy, rhs.fy, tolerance) &&
           AlmostEqual(lhs.cx, rhs.cx, tolerance) &&
           AlmostEqual(lhs.cy, rhs.cy, tolerance) &&
           AlmostEqual(lhs.k1, rhs.k1, tolerance) &&
           AlmostEqual(lhs.k2, rhs.k2, tolerance) &&
           AlmostEqual(lhs.k3, rhs.k3, tolerance) &&
           AlmostEqual(lhs.k4, rhs.k4, tolerance) &&
           lhs.width == rhs.width &&
           lhs.height == rhs.height;
}

} // namespace StructureSLAM
