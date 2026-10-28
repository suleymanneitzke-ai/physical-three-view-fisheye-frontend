#ifndef FISHEYE_LINE_TRIANGULATOR_H
#define FISHEYE_LINE_TRIANGULATOR_H

#include "FisheyeVirtualCameraGeometry.h"
#include "camera/KannalaBrandt8Camera.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

namespace StructureSLAM
{

class FisheyeLineTriangulator
{
public:
    enum class Config
    {
        Strict,
        Medium,
        Loose
    };

    struct Options
    {
        Config config = Config::Strict;
        int min_track_length = 10;
        double min_mean_score = 0.85;
        double min_score = 0.70;
        int min_pose_observations = 5;
        int min_valid_samples = 8;
        double min_plane_angle_deg = 1.0;
        double min_baseline = 0.005;
        double max_mean_reprojection_px = 8.0;
        double max_p90_reprojection_px = 16.0;
        double max_line_length = 100.0;
        int max_tracks = -1;
    };

    struct Pose
    {
        int frame_id = -1;
        double timestamp = 0.0;
        Eigen::Matrix3d Rwc = Eigen::Matrix3d::Identity();
        Eigen::Vector3d twc = Eigen::Vector3d::Zero();
        Eigen::Matrix3d Rcw = Eigen::Matrix3d::Identity();
        Eigen::Vector3d tcw = Eigen::Vector3d::Zero();
    };

    struct TrackObs
    {
        int track_id = -1;
        int frame_id = -1;
        double timestamp = 0.0;
        int view_id = -1;
        int line_id = -1;
        int track_length = 0;
        double match_score_to_previous = 0.0;
    };

    struct LineInfo
    {
        double virtual_x1 = 0.0;
        double virtual_y1 = 0.0;
        double virtual_x2 = 0.0;
        double virtual_y2 = 0.0;
        double virtual_length_px = 0.0;
        double valid_sample_ratio = 0.0;
    };

    struct Sample
    {
        double virtual_u = 0.0;
        double virtual_v = 0.0;
        double fisheye_u = 0.0;
        double fisheye_v = 0.0;
        bool valid = false;
    };

    struct PlaneObs
    {
        TrackObs obs;
        Pose pose;
        Eigen::Vector3d nc = Eigen::Vector3d::Zero();
        Eigen::Vector3d nw = Eigen::Vector3d::Zero();
        double dw = 0.0;
        int sample_count = 0;
        double mean_ray_plane_residual = 0.0;
        double max_ray_plane_residual = 0.0;
        double smallest_singular_value = 0.0;
        double virtual_plane_angle_difference_deg = 0.0;
        bool virtual_plane_comparison_valid = false;
        bool valid = false;
        std::string rejection_reason;
    };

    struct Line3D
    {
        int line3d_id = -1;
        int track_id = -1;
        int view_id = -1;
        std::string quality_class = "rejected";
        Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
        Eigen::Vector3d dir = Eigen::Vector3d::UnitX();
        Eigen::Vector3d moment = Eigen::Vector3d::Zero();
        Eigen::Vector3d endpoint1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d endpoint2 = Eigen::Vector3d::Zero();
        double length_m = 0.0;
        bool finite_segment_valid = false;
        int observation_count = 0;
        int inlier_observation_count = 0;
        double mean_reprojection_error_px = 0.0;
        double median_reprojection_error_px = 0.0;
        double p90_reprojection_error_px = 0.0;
        double max_reprojection_error_px = 0.0;
        double mean_angular_error_deg = 0.0;
        double plane_fit_residual = 0.0;
        double condition_number = 0.0;
        double baseline_m = 0.0;
        double plane_angle_deg = 0.0;
        double virtual_plane_angle_mean_deg = 0.0;
        double virtual_plane_angle_p90_deg = 0.0;
    };

    struct Candidate
    {
        int track_id = -1;
        int view_id = -1;
        int original_track_length = 0;
        int usable_observation_count = 0;
        double mean_match_score = 0.0;
        double min_match_score = 0.0;
        int selected_frame_1 = -1;
        int selected_frame_2 = -1;
        double baseline_m = 0.0;
        double plane_angle_deg = 0.0;
        double pair_score = 0.0;
        bool candidate_valid = false;
        std::string rejection_reason;
    };

    struct ObsError
    {
        int line3d_id = -1;
        int track_id = -1;
        int frame_id = -1;
        double timestamp = 0.0;
        int view_id = -1;
        int line_id = -1;
        bool inlier = false;
        double mean_pixel_error = 0.0;
        double median_pixel_error = 0.0;
        double p90_pixel_error = 0.0;
        double mean_angular_error_deg = 0.0;
    };

    explicit FisheyeLineTriangulator(const KannalaBrandt8Camera::Parameters& kb8_params,
                                     const Options& options);

    static Options OptionsForConfig(Config config);
    static const char* ConfigName(Config config);

    bool FitPlane(const TrackObs& obs,
                  const Pose& pose,
                  const std::vector<Sample>& samples,
                  const LineInfo* line_info,
                  PlaneObs& plane) const;

    bool IntersectPlanes(const PlaneObs& a,
                         const PlaneObs& b,
                         Eigen::Vector3d& p0,
                         Eigen::Vector3d& dir,
                         double& condition_number) const;

    bool EstimateLineFromPlanes(const std::vector<PlaneObs>& planes,
                                bool robust,
                                Eigen::Vector3d& p0,
                                Eigen::Vector3d& dir,
                                std::vector<int>& inlier_indices,
                                double& condition_number,
                                double& mean_residual) const;

    bool EstimateFiniteSegment(const Line3D& line,
                               const std::vector<PlaneObs>& planes,
                               const std::map<std::string, std::vector<Sample>>& sample_map,
                               Eigen::Vector3d& e1,
                               Eigen::Vector3d& e2,
                               double& length,
                               int& support_count) const;

    ObsError ComputeReprojectionError(int line3d_id,
                                      const Line3D& line,
                                      const PlaneObs& plane,
                                      const std::vector<Sample>& samples) const;

    std::string ObservationKey(int frame_id, int view_id, int line_id) const;

private:
    cv::Point2f Project(const Eigen::Vector3d& pc) const;
    Eigen::Vector3d Unproject(const cv::Point2f& px, bool& ok) const;
    double VirtualPlaneAngleDifferenceDeg(const PlaneObs& plane,
                                          const LineInfo& line_info) const;

    Options options_;
    KannalaBrandt8Camera camera_;
};

} // namespace StructureSLAM

#endif
