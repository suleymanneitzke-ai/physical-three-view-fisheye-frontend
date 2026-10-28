#include "FisheyeLineTriangulator.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace StructureSLAM
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

bool Finite(double x)
{
    return std::isfinite(x);
}

bool FiniteVec(const Eigen::Vector3d& v)
{
    return Finite(v.x()) && Finite(v.y()) && Finite(v.z());
}

double Clamp(double x, double lo, double hi)
{
    return std::max(lo, std::min(hi, x));
}

double Percentile(std::vector<double> values, double q)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = (values.size() - 1) * q;
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if(lo == hi)
        return values[lo];
    return values[lo] * (hi - pos) + values[hi] * (pos - lo);
}

double Median(std::vector<double> values)
{
    return Percentile(values, 0.5);
}

double PointToPolylineDistance(const cv::Point2f& p, const std::vector<cv::Point2f>& poly)
{
    if(poly.empty())
        return std::numeric_limits<double>::infinity();
    double best = std::numeric_limits<double>::infinity();
    for(size_t i = 1; i < poly.size(); ++i)
    {
        const cv::Point2f a = poly[i - 1];
        const cv::Point2f b = poly[i];
        const cv::Point2f ab = b - a;
        const double denom = ab.dot(ab);
        double t = denom > 1e-12 ? (p - a).dot(ab) / denom : 0.0;
        t = Clamp(t, 0.0, 1.0);
        const cv::Point2f q = a + static_cast<float>(t) * ab;
        const cv::Point2f d = p - q;
        best = std::min(best, static_cast<double>(std::sqrt(d.dot(d))));
    }
    return best;
}

double ViewYawRad(int view_id)
{
    return NativeFisheyeVirtualYawRad(view_id);
}

} // namespace

FisheyeLineTriangulator::FisheyeLineTriangulator(
    const KannalaBrandt8Camera::Parameters& kb8_params,
    const Options& options)
    : options_(options), camera_(kb8_params)
{
}

FisheyeLineTriangulator::Options FisheyeLineTriangulator::OptionsForConfig(Config config)
{
    Options o;
    o.config = config;
    if(config == Config::Strict)
    {
        o.min_track_length = 10;
        o.min_mean_score = 0.85;
        o.min_score = 0.70;
        o.min_pose_observations = 5;
    }
    else if(config == Config::Medium)
    {
        o.min_track_length = 5;
        o.min_mean_score = 0.75;
        o.min_score = 0.55;
        o.min_pose_observations = 3;
    }
    else
    {
        o.min_track_length = 3;
        o.min_mean_score = 0.65;
        o.min_score = 0.45;
        o.min_pose_observations = 3;
    }
    return o;
}

const char* FisheyeLineTriangulator::ConfigName(Config config)
{
    if(config == Config::Strict)
        return "strict";
    if(config == Config::Medium)
        return "medium";
    return "loose";
}

std::string FisheyeLineTriangulator::ObservationKey(int frame_id, int view_id, int line_id) const
{
    std::ostringstream oss;
    oss << frame_id << ":" << view_id << ":" << line_id;
    return oss.str();
}

Eigen::Vector3d FisheyeLineTriangulator::Unproject(const cv::Point2f& px, bool& ok) const
{
    Eigen::Vector3d bearing = Eigen::Vector3d::Zero();
    ok = camera_.unproject(Eigen::Vector2d(px.x, px.y), bearing);
    if(ok)
        bearing.normalize();
    return bearing;
}

cv::Point2f FisheyeLineTriangulator::Project(const Eigen::Vector3d& pc) const
{
    Eigen::Vector2d uv;
    if(!camera_.project(pc, uv))
        return cv::Point2f(std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN());
    return cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
}

double FisheyeLineTriangulator::VirtualPlaneAngleDifferenceDeg(
    const PlaneObs& plane,
    const LineInfo& line_info) const
{
    const double fx = 256.0;
    const double fy = 256.0;
    const double cx = 255.5;
    const double cy = 255.5;
    Eigen::Vector3d nv = VirtualPlaneNormalFromEndpoints(
        line_info.virtual_x1,
        line_info.virtual_y1,
        line_info.virtual_x2,
        line_info.virtual_y2,
        fx,
        fy,
        cx,
        cy);
    if(nv.norm() < 1e-12 || !FiniteVec(nv))
        return std::numeric_limits<double>::quiet_NaN();
    const Eigen::Vector3d nc = RotationCvFromYaw(ViewYawRad(plane.obs.view_id)) * nv;
    const Eigen::Vector3d nw = plane.pose.Rwc * nc.normalized();
    const double dot = Clamp(std::abs(nw.normalized().dot(plane.nw.normalized())), -1.0, 1.0);
    return std::acos(dot) * 180.0 / kPi;
}

bool FisheyeLineTriangulator::FitPlane(const TrackObs& obs,
                                       const Pose& pose,
                                       const std::vector<Sample>& samples,
                                       const LineInfo* line_info,
                                       PlaneObs& plane) const
{
    plane = PlaneObs();
    plane.obs = obs;
    plane.pose = pose;

    std::vector<Eigen::Vector3d> rays;
    for(const Sample& s : samples)
    {
        if(!s.valid || !Finite(s.fisheye_u) || !Finite(s.fisheye_v))
            continue;
        bool ok = false;
        Eigen::Vector3d r = Unproject(cv::Point2f(static_cast<float>(s.fisheye_u),
                                                  static_cast<float>(s.fisheye_v)), ok);
        if(ok && FiniteVec(r) && r.norm() > 0.99)
            rays.push_back(r.normalized());
    }

    plane.sample_count = static_cast<int>(rays.size());
    if(static_cast<int>(rays.size()) < options_.min_valid_samples)
    {
        std::ostringstream oss;
        oss << "too_few_valid_samples_" << rays.size();
        plane.rejection_reason = oss.str();
        return false;
    }

    Eigen::MatrixXd R(rays.size(), 3);
    for(size_t i = 0; i < rays.size(); ++i)
        R.row(i) = rays[i].transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeFullV);
    plane.smallest_singular_value = svd.singularValues()(2);
    Eigen::Vector3d n = svd.matrixV().col(2);
    const double nn = n.norm();
    if(nn < 1e-12 || !FiniteVec(n))
    {
        plane.rejection_reason = "invalid_plane_normal";
        return false;
    }
    n /= nn;

    double sum = 0.0;
    double maxr = 0.0;
    for(const Eigen::Vector3d& r : rays)
    {
        const double e = std::abs(n.dot(r));
        sum += e;
        maxr = std::max(maxr, e);
    }

    plane.nc = n;
    plane.mean_ray_plane_residual = sum / rays.size();
    plane.max_ray_plane_residual = maxr;
    plane.nw = PlaneNormalWorldFromCamera(pose.Rcw, plane.nc);
    plane.nw.normalize();
    plane.dw = PlaneOffsetWorldFromCamera(plane.nc, pose.tcw);
    plane.valid = FiniteVec(plane.nw) && Finite(plane.dw);
    if(!plane.valid)
    {
        plane.rejection_reason = "nonfinite_world_plane";
        return false;
    }
    const double center_residual = std::abs(plane.nw.dot(pose.twc) + plane.dw);
    if(center_residual > 1e-8)
    {
        plane.rejection_reason = "camera_center_not_on_plane";
        return false;
    }
    if(line_info)
    {
        const double angle = VirtualPlaneAngleDifferenceDeg(plane, *line_info);
        if(std::isfinite(angle))
        {
            plane.virtual_plane_angle_difference_deg = angle;
            plane.virtual_plane_comparison_valid = true;
        }
    }
    return true;
}

bool FisheyeLineTriangulator::IntersectPlanes(const PlaneObs& a,
                                              const PlaneObs& b,
                                              Eigen::Vector3d& p0,
                                              Eigen::Vector3d& dir,
                                              double& condition_number) const
{
    dir = a.nw.cross(b.nw);
    const double dn = dir.norm();
    if(dn < 1e-10)
        return false;
    dir /= dn;

    Eigen::Matrix3d A;
    A.row(0) = a.nw.transpose();
    A.row(1) = b.nw.transpose();
    A.row(2) = dir.transpose();
    Eigen::Vector3d rhs(-a.dw, -b.dw, 0.0);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto s = svd.singularValues();
    condition_number = s(2) > 1e-12 ? s(0) / s(2) : 1e12;
    if(condition_number > 1e8)
        return false;
    p0 = svd.solve(rhs);
    return FiniteVec(p0) && FiniteVec(dir);
}

bool FisheyeLineTriangulator::EstimateLineFromPlanes(const std::vector<PlaneObs>& planes,
                                                     bool robust,
                                                     Eigen::Vector3d& p0,
                                                     Eigen::Vector3d& dir,
                                                     std::vector<int>& inlier_indices,
                                                     double& condition_number,
                                                     double& mean_residual) const
{
    inlier_indices.clear();
    for(size_t i = 0; i < planes.size(); ++i)
        inlier_indices.push_back(static_cast<int>(i));

    for(int iter = 0; iter < (robust ? 4 : 1); ++iter)
    {
        if(inlier_indices.size() < 2)
            return false;
        Eigen::MatrixXd N(inlier_indices.size(), 3);
        Eigen::VectorXd d(inlier_indices.size());
        for(size_t i = 0; i < inlier_indices.size(); ++i)
        {
            const PlaneObs& p = planes[inlier_indices[i]];
            N.row(i) = p.nw.transpose();
            d(i) = -p.dw;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svdN(N, Eigen::ComputeFullV);
        const auto sN = svdN.singularValues();
        condition_number = sN(sN.size() - 1) > 1e-12
                               ? sN(0) / sN(sN.size() - 1)
                               : 1e12;
        dir = svdN.matrixV().col(2);
        if(dir.norm() < 1e-12)
            return false;
        dir.normalize();

        Eigen::MatrixXd A(inlier_indices.size() + 1, 3);
        Eigen::VectorXd rhs(inlier_indices.size() + 1);
        A.topRows(inlier_indices.size()) = N;
        rhs.head(inlier_indices.size()) = d;
        A.row(inlier_indices.size()) = dir.transpose();
        rhs(inlier_indices.size()) = 0.0;
        Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        p0 = svdA.solve(rhs);
        if(!FiniteVec(p0) || !FiniteVec(dir))
            return false;

        std::vector<double> residuals;
        double sum = 0.0;
        for(int idx : inlier_indices)
        {
            const double r = std::abs(planes[idx].nw.dot(p0) + planes[idx].dw);
            residuals.push_back(r);
            sum += r;
        }
        mean_residual = residuals.empty() ? 0.0 : sum / residuals.size();
        if(!robust || residuals.size() < 5)
            break;
        const double med = Median(residuals);
        std::vector<double> dev;
        for(double r : residuals)
            dev.push_back(std::abs(r - med));
        const double mad = Median(dev);
        const double gate = std::max(0.02, med + 3.0 * 1.4826 * mad);
        std::vector<int> next;
        for(size_t i = 0; i < inlier_indices.size(); ++i)
            if(residuals[i] <= gate)
                next.push_back(inlier_indices[i]);
        if(next.size() == inlier_indices.size())
            break;
        inlier_indices.swap(next);
    }
    return inlier_indices.size() >= 2;
}

bool FisheyeLineTriangulator::EstimateFiniteSegment(
    const Line3D& line,
    const std::vector<PlaneObs>& planes,
    const std::map<std::string, std::vector<Sample>>& sample_map,
    Eigen::Vector3d& e1,
    Eigen::Vector3d& e2,
    double& length,
    int& support_count) const
{
    std::vector<double> lambdas;
    for(const PlaneObs& p : planes)
    {
        const auto it = sample_map.find(ObservationKey(p.obs.frame_id, p.obs.view_id, p.obs.line_id));
        if(it == sample_map.end())
            continue;
        for(const Sample& s : it->second)
        {
            if(!s.valid || !Finite(s.fisheye_u) || !Finite(s.fisheye_v))
                continue;
            bool ok = false;
            Eigen::Vector3d dc = Unproject(cv::Point2f(static_cast<float>(s.fisheye_u),
                                                       static_cast<float>(s.fisheye_v)), ok);
            if(!ok)
                continue;
            Eigen::Vector3d dw = p.pose.Rwc * dc;
            dw.normalize();
            const Eigen::Vector3d w0 = p.pose.twc - line.p0;
            const double b = dw.dot(line.dir);
            const double denom = 1.0 - b * b;
            if(std::abs(denom) < 1e-8)
                continue;
            const double t_line = (w0.dot(dw) * b - w0.dot(line.dir)) / denom;
            if(Finite(t_line))
                lambdas.push_back(t_line);
        }
    }
    support_count = static_cast<int>(lambdas.size());
    if(lambdas.size() < 4)
        return false;
    const double l0 = Percentile(lambdas, 0.10);
    const double l1 = Percentile(lambdas, 0.90);
    length = l1 - l0;
    if(!(length > 1e-4) || length > options_.max_line_length || !Finite(length))
        return false;
    e1 = line.p0 + l0 * line.dir;
    e2 = line.p0 + l1 * line.dir;
    return FiniteVec(e1) && FiniteVec(e2);
}

FisheyeLineTriangulator::ObsError FisheyeLineTriangulator::ComputeReprojectionError(
    int line3d_id,
    const Line3D& line,
    const PlaneObs& plane,
    const std::vector<Sample>& samples) const
{
    ObsError out;
    out.line3d_id = line3d_id;
    out.track_id = line.track_id;
    out.frame_id = plane.obs.frame_id;
    out.timestamp = plane.obs.timestamp;
    out.view_id = plane.obs.view_id;
    out.line_id = plane.obs.line_id;

    if(!line.finite_segment_valid)
        return out;

    std::vector<cv::Point2f> pred;
    for(int i = 0; i < 32; ++i)
    {
        const double a = static_cast<double>(i) / 31.0;
        const Eigen::Vector3d Xw = (1.0 - a) * line.endpoint1 + a * line.endpoint2;
        const Eigen::Vector3d Xc = plane.pose.Rcw * Xw + plane.pose.tcw;
        // CAM_B uses a full fisheye branch; negative camera-frame z is a
        // valid bearing whenever the calibrated projection is finite.
        if(!FiniteVec(Xc))
            continue;
        const cv::Point2f uv = Project(Xc);
        if(Finite(uv.x) && Finite(uv.y))
            pred.push_back(uv);
    }

    std::vector<double> pix;
    std::vector<double> angular;
    if(pred.size() < 2)
    {
        out.mean_pixel_error = 1e9;
        out.median_pixel_error = 1e9;
        out.p90_pixel_error = 1e9;
        out.mean_angular_error_deg = 0.0;
        out.inlier = false;
        return out;
    }
    for(const Sample& s : samples)
    {
        if(!s.valid || !Finite(s.fisheye_u) || !Finite(s.fisheye_v))
            continue;
        const cv::Point2f obs(static_cast<float>(s.fisheye_u),
                              static_cast<float>(s.fisheye_v));
        pix.push_back(PointToPolylineDistance(obs, pred));
        bool ok = false;
        Eigen::Vector3d r = Unproject(obs, ok);
        if(ok)
        {
            const double ae = std::asin(Clamp(std::abs(plane.nc.dot(r)), 0.0, 1.0)) *
                              180.0 / kPi;
            if(Finite(ae))
                angular.push_back(ae);
        }
    }

    if(pix.empty())
        return out;
    double sum = 0.0;
    for(double e : pix)
        sum += e;
    out.mean_pixel_error = sum / pix.size();
    out.median_pixel_error = Median(pix);
    out.p90_pixel_error = Percentile(pix, 0.90);
    double asum = 0.0;
    for(double e : angular)
        asum += e;
    out.mean_angular_error_deg = angular.empty() ? 0.0 : asum / angular.size();
    out.inlier = out.mean_pixel_error <= options_.max_mean_reprojection_px &&
                 out.p90_pixel_error <= options_.max_p90_reprojection_px;
    return out;
}

} // namespace StructureSLAM
