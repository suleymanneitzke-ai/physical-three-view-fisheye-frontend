#include "Kb8InitializationAdapter.h"

#include "Kb8CameraSettings.h"

#include <cmath>

namespace StructureSLAM
{

bool BuildKb8InitializationView(
    const Frame& frame,
    const KannalaBrandt8Camera::Parameters& params,
    Kb8InitializationView& view,
    std::string& error_message)
{
    error_message.clear();
    view = Kb8InitializationView();

    if(!frame.HasKb8PointObservations())
    {
        error_message = "frame_has_no_kb8_point_observations";
        return false;
    }
    if(frame.mvKeysUn.size() != frame.mvKeys.size())
    {
        error_message = "raw_keypoint_size_mismatch";
        return false;
    }
    const KannalaBrandt8Camera camera(params);
    if(!camera.valid())
    {
        error_message = camera.validationError();
        return false;
    }

    view.keypoints = frame.mvKeysUn;
    view.valid.assign(view.keypoints.size(), 0);
    view.K = Kb8ParametersToPinholeMatrix(params);
    if(view.keypoints.empty())
        return true;

    for(size_t i = 0; i < view.keypoints.size(); ++i)
    {
        const cv::Point2f pt = view.keypoints[i].pt;
        if(!std::isfinite(pt.x) || !std::isfinite(pt.y))
        {
            error_message = "non_finite_raw_keypoint";
            return false;
        }
    }

    for(size_t i = 0; i < view.keypoints.size(); ++i)
    {
        const cv::Point2f raw = view.keypoints[i].pt;
        Eigen::Vector3d bearing;
        const bool unprojected =
            camera.isInImage(Eigen::Vector2d(raw.x, raw.y)) &&
            camera.unproject(Eigen::Vector2d(raw.x, raw.y), bearing) &&
            bearing.allFinite() &&
            bearing.norm() > 0.0 &&
            std::abs(bearing.z()) > 1e-9;
        if(!unprojected)
            continue;

        bearing.normalize();
        const double x = bearing.x() / bearing.z();
        const double y = bearing.y() / bearing.z();
        const double u = params.fx * x + params.cx;
        const double v = params.fy * y + params.cy;
        if(!std::isfinite(u) || !std::isfinite(v))
            continue;

        view.keypoints[i].pt =
            cv::Point2f(static_cast<float>(u), static_cast<float>(v));
        view.valid[i] = 1;
    }
    return true;
}

} // namespace StructureSLAM
