#include "FisheyeLineFrontend.h"

#include "FisheyeVirtualCameraGeometry.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace StructureSLAM
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

KannalaBrandt8Camera::Parameters MakeCameraParameters(
    const std::vector<float>& kb8_params,
    const cv::Size& fisheye_size)
{
    if(kb8_params.size() != 8)
        throw std::runtime_error(
            "FisheyeLineFrontend requires exactly 8 KB8 parameters.");

    KannalaBrandt8Camera::Parameters params;
    params.fx = kb8_params[0];
    params.fy = kb8_params[1];
    params.cx = kb8_params[2];
    params.cy = kb8_params[3];
    params.k1 = kb8_params[4];
    params.k2 = kb8_params[5];
    params.k3 = kb8_params[6];
    params.k4 = kb8_params[7];
    params.width = fisheye_size.width;
    params.height = fisheye_size.height;
    return params;
}

double NowMs()
{
    using Clock = std::chrono::steady_clock;
    static const Clock::time_point start = Clock::now();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool IsFinitePoint(const cv::Point2f& p)
{
    return std::isfinite(p.x) && std::isfinite(p.y);
}

cv::Scalar ViewColor(int view_id)
{
    switch(view_id)
    {
        case 0: return cv::Scalar(0, 255, 0);
        case 1: return cv::Scalar(0, 180, 255);
        case 2: return cv::Scalar(255, 0, 0);
        case 3: return cv::Scalar(255, 0, 255);
        default: return cv::Scalar(255, 255, 255);
    }
}

double Median(std::vector<double> values)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if(values.size() % 2 == 0)
        return 0.5 * (values[mid - 1] + values[mid]);
    return values[mid];
}

} // namespace

FisheyeLineFrontend::FisheyeLineFrontend(const std::vector<float>& kb8_params,
                                         const cv::Size& fisheye_size,
                                         const Options& options)
    : options_(options),
      kb8_params_(kb8_params),
      fisheye_size_(fisheye_size),
      camera_(MakeCameraParameters(kb8_params, fisheye_size))
{
    if(options_.virtual_yaw_deg.empty())
    {
        throw std::runtime_error(
            "FisheyeLineFrontend requires at least one virtual view.");
    }

    const size_t view_count =
        options_.virtual_yaw_deg.size();

    yaw_rad_.reserve(view_count);
    view_names_.reserve(view_count);
    remaps_.resize(view_count);
    valid_masks_.resize(view_count);

    for(size_t i = 0; i < view_count; ++i)
    {
        yaw_rad_.push_back(
            options_.virtual_yaw_deg[i]
            * kPi / 180.0);

        view_names_.push_back(
            "view" + std::to_string(i));
    }

    if(kb8_params_.size() != 8)
        throw std::runtime_error("FisheyeLineFrontend requires 8 KB8 camera parameters.");
    if(fisheye_size_.width <= 0 || fisheye_size_.height <= 0)
        throw std::runtime_error("FisheyeLineFrontend requires a positive fisheye image size.");
    if(options_.virtual_width <= 0 || options_.virtual_height <= 0)
        throw std::runtime_error("FisheyeLineFrontend requires a positive virtual image size.");
    if(options_.samples_per_line < 2)
        throw std::runtime_error("FisheyeLineFrontend requires at least 2 samples per line.");

    BuildRemaps();
}

void FisheyeLineFrontend::BuildRemaps()
{
    int invalid_count = 0;
    int total_count = 0;

    for(int view_id = 0; view_id < static_cast<int>(yaw_rad_.size()); ++view_id)
    {
        RemapView& remap = remaps_[view_id];
        remap.map_x.create(options_.virtual_height, options_.virtual_width, CV_32FC1);
        remap.map_y.create(options_.virtual_height, options_.virtual_width, CV_32FC1);
        valid_masks_[view_id] = cv::Mat::zeros(options_.virtual_height, options_.virtual_width, CV_8UC1);

        for(int y = 0; y < options_.virtual_height; ++y)
        {
            for(int x = 0; x < options_.virtual_width; ++x)
            {
                const cv::Point3f ray = VirtualPixelToFisheyeRay(static_cast<float>(x),
                                                                 static_cast<float>(y),
                                                                 yaw_rad_[view_id]);
                const double ray_norm = std::sqrt(
                    static_cast<double>(ray.x) * ray.x +
                    static_cast<double>(ray.y) * ray.y +
                    static_cast<double>(ray.z) * ray.z);
                const double alpha = ray_norm > 0.0
                    ? std::acos(std::max(
                        -1.0, std::min(1.0,
                        static_cast<double>(ray.z) / ray_norm)))
                    : std::numeric_limits<double>::infinity();
                const bool physical_valid =
                    std::isfinite(alpha) &&
                    alpha <= options_.physical_alpha_max_deg *
                             kPi / 180.0;
                cv::Point2f uv;
                const bool projected = ProjectFullFovRay(ray, uv);
                const bool valid = physical_valid && projected &&
                                   std::isfinite(uv.x) &&
                                   std::isfinite(uv.y) &&
                                   uv.x >= 0.0f &&
                                   uv.x <= static_cast<float>(fisheye_size_.width - 1) &&
                                   uv.y >= 0.0f &&
                                   uv.y <= static_cast<float>(fisheye_size_.height - 1);

                remap.map_x.at<float>(y, x) = valid ? uv.x : -1.0f;
                remap.map_y.at<float>(y, x) = valid ? uv.y : -1.0f;
                valid_masks_[view_id].at<unsigned char>(y, x) = valid ? 255 : 0;

                ++total_count;
                if(!valid)
                    ++invalid_count;
            }
        }
    }

    invalid_mapping_ratio_ = total_count > 0 ? static_cast<double>(invalid_count) / total_count : 0.0;
}

bool FisheyeLineFrontend::ProjectFullFovRay(
    const cv::Point3f& ray,
    cv::Point2f& fisheye_point) const
{
    fisheye_point = cv::Point2f(-1.0f, -1.0f);

    if(!std::isfinite(ray.x) ||
       !std::isfinite(ray.y) ||
       !std::isfinite(ray.z))
    {
        return false;
    }

    const double x = static_cast<double>(ray.x);
    const double y = static_cast<double>(ray.y);
    const double z = static_cast<double>(ray.z);

    const double radius_xy = std::sqrt(x * x + y * y);
    const double theta = std::atan2(radius_xy, z);
    const double psi = std::atan2(y, x);

    const double theta2 = theta * theta;
    const double theta3 = theta * theta2;
    const double theta5 = theta3 * theta2;
    const double theta7 = theta5 * theta2;
    const double theta9 = theta7 * theta2;

    const double distorted_radius =
        theta +
        static_cast<double>(kb8_params_[4]) * theta3 +
        static_cast<double>(kb8_params_[5]) * theta5 +
        static_cast<double>(kb8_params_[6]) * theta7 +
        static_cast<double>(kb8_params_[7]) * theta9;

    const double u =
        static_cast<double>(kb8_params_[0]) *
        distorted_radius * std::cos(psi) +
        static_cast<double>(kb8_params_[2]);

    const double v =
        static_cast<double>(kb8_params_[1]) *
        distorted_radius * std::sin(psi) +
        static_cast<double>(kb8_params_[3]);

    if(!std::isfinite(u) || !std::isfinite(v))
        return false;

    fisheye_point =
        cv::Point2f(static_cast<float>(u),
                    static_cast<float>(v));
    return true;
}

cv::Point3f FisheyeLineFrontend::VirtualPixelToFisheyeRay(float u, float v, double yaw_rad) const
{
    const double fx =
        options_.virtual_width == NativeFisheyeVirtualWidth() &&
                std::abs(options_.horizontal_fov_deg -
                         NativeFisheyeVirtualHorizontalFovDeg()) < 1e-12
            ? NativeFisheyeVirtualFx()
            : (0.5 * options_.virtual_width) /
                  std::tan(0.5 * options_.horizontal_fov_deg *
                           kPi / 180.0);
    const double fy =
        options_.virtual_height == NativeFisheyeVirtualHeight() &&
                std::abs(options_.vertical_fov_deg -
                         NativeFisheyeVirtualVerticalFovDeg()) < 1e-12
            ? NativeFisheyeVirtualFy()
            : (0.5 * options_.virtual_height) /
                  std::tan(0.5 * options_.vertical_fov_deg *
                           kPi / 180.0);
    const double cx = 0.5 * (options_.virtual_width - 1);
    const double cy = 0.5 * (options_.virtual_height - 1);

    const double xv = (static_cast<double>(u) - cx) / fx;
    const double yv = (static_cast<double>(v) - cy) / fy;
    const double zv = 1.0;

    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    const double x = c * xv + s * zv;
    const double y = yv;
    const double z = -s * xv + c * zv;

    return cv::Point3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

bool FisheyeLineFrontend::SampleVirtualPoint(int view_id,
                                             const cv::Point2f& virtual_point,
                                             cv::Point2f& fisheye_point) const
{
    fisheye_point = cv::Point2f(-1.0f, -1.0f);
    if(view_id < 0 || view_id >= static_cast<int>(yaw_rad_.size()) || !IsFinitePoint(virtual_point))
        return false;

    const int x = cvRound(virtual_point.x);
    const int y = cvRound(virtual_point.y);
    if(x < 0 || y < 0 || x >= options_.virtual_width || y >= options_.virtual_height)
        return false;
    if(valid_masks_[view_id].at<unsigned char>(y, x) == 0)
        return false;

    const float fx = remaps_[view_id].map_x.at<float>(y, x);
    const float fy = remaps_[view_id].map_y.at<float>(y, x);
    fisheye_point = cv::Point2f(fx, fy);
    return std::isfinite(fx) && std::isfinite(fy) &&
           fx >= 0.0f && fx <= static_cast<float>(fisheye_size_.width - 1) &&
           fy >= 0.0f && fy <= static_cast<float>(fisheye_size_.height - 1);
}

FisheyeLineFrontend::ViewLine FisheyeLineFrontend::BuildLine(int view_id,
                                                             int line_id,
                                                             const cv::Vec4f& segment,
                                                             double width,
                                                             double precision,
                                                             double response,
                                                             int& nan_inf_count,
                                                             int& out_of_bounds_count) const
{
    ViewLine line;
    line.view_id = view_id;
    line.line_id = line_id;
    line.virtual_line = segment;
    line.width = std::isfinite(width) ? width : 0.0;
    line.precision = std::isfinite(precision) ? precision : 0.0;
    line.response = std::isfinite(response) ? response : 0.0;

    const double dx = static_cast<double>(segment[2] - segment[0]);
    const double dy = static_cast<double>(segment[3] - segment[1]);
    line.virtual_length_px = std::sqrt(dx * dx + dy * dy);
    line.virtual_angle_deg = std::atan2(dy, dx) * 180.0 / kPi;

    int valid_count = 0;
    line.virtual_samples.reserve(options_.samples_per_line);
    line.fisheye_samples.reserve(options_.samples_per_line);
    line.sample_valid.reserve(options_.samples_per_line);

    for(int i = 0; i < options_.samples_per_line; ++i)
    {
        const float alpha = options_.samples_per_line == 1
                                ? 0.0f
                                : static_cast<float>(i) / static_cast<float>(options_.samples_per_line - 1);
        const cv::Point2f virtual_point(segment[0] + alpha * (segment[2] - segment[0]),
                                        segment[1] + alpha * (segment[3] - segment[1]));
        cv::Point2f fisheye_point;
        const bool valid = SampleVirtualPoint(view_id, virtual_point, fisheye_point);
        if(!IsFinitePoint(virtual_point) || !IsFinitePoint(fisheye_point))
            ++nan_inf_count;
        if(valid && IsFinitePoint(fisheye_point) &&
           (fisheye_point.x < 0.0f || fisheye_point.x > static_cast<float>(fisheye_size_.width - 1) ||
            fisheye_point.y < 0.0f || fisheye_point.y > static_cast<float>(fisheye_size_.height - 1)))
            ++out_of_bounds_count;

        if(valid)
            ++valid_count;
        line.virtual_samples.push_back(virtual_point);
        line.fisheye_samples.push_back(fisheye_point);
        line.sample_valid.push_back(valid ? 1 : 0);
    }

    // A line is usable only when every sampled support point remains in the
    // physical domain.  This prevents the artificial remap boundary from
    // becoming a line observation while retaining native calibrated pixels.
    line.valid_sample_ratio = static_cast<double>(valid_count) / options_.samples_per_line;
    if(options_.require_all_physical_samples &&
       valid_count != options_.samples_per_line)
        line.valid_sample_ratio = 0.0;
    if(!line.fisheye_samples.empty())
    {
        line.fisheye_start = line.fisheye_samples.front();
        line.fisheye_end = line.fisheye_samples.back();
    }
    return line;
}

FisheyeLineFrontend::FrameResult FisheyeLineFrontend::ProcessFrame(const cv::Mat& fisheye_gray,
                                                                   bool make_overlays) const
{
    if(fisheye_gray.empty())
        throw std::runtime_error("Cannot process an empty fisheye image.");
    if(fisheye_gray.size() != fisheye_size_)
        throw std::runtime_error("Fisheye image size does not match the configured camera size.");

    FrameResult result;
    const size_t view_count = yaw_rad_.size();

    result.lines.resize(view_count);
    result.virtual_images.resize(view_count);
    result.virtual_overlays.resize(view_count);
    result.raw_line_counts.assign(view_count, 0);

    const double total_start = NowMs();

    const double remap_start = NowMs();
    for(int view_id = 0; view_id < static_cast<int>(yaw_rad_.size()); ++view_id)
    {
        cv::remap(fisheye_gray, result.virtual_images[view_id],
                  remaps_[view_id].map_x, remaps_[view_id].map_y,
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    result.remap_time_ms = NowMs() - remap_start;

    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(
        options_.lsd_refine,
        options_.lsd_scale,
        options_.lsd_sigma_scale,
        options_.lsd_quant,
        options_.lsd_ang_th,
        options_.lsd_log_eps,
        options_.lsd_density_th,
        options_.lsd_n_bins);

    std::vector<double> filtered_lengths;
    const double extraction_start = NowMs();
    int global_line_id = 0;
    for(int view_id = 0; view_id < static_cast<int>(yaw_rad_.size()); ++view_id)
    {
        std::vector<cv::Vec4f> segments;
        std::vector<double> widths;
        std::vector<double> precisions;
        std::vector<double> nfas;
        detector->detect(result.virtual_images[view_id], segments, widths, precisions, nfas);

        result.raw_line_counts[view_id] = static_cast<int>(segments.size());
        result.raw_line_count_total += result.raw_line_counts[view_id];

        for(size_t i = 0; i < segments.size(); ++i)
        {
            const double width = i < widths.size() ? widths[i] : 0.0;
            const double precision = i < precisions.size() ? precisions[i] : 0.0;
            const double response = i < nfas.size() ? nfas[i] : 0.0;
            ViewLine line = BuildLine(view_id, global_line_id, segments[i],
                                      width, precision, response,
                                      result.nan_inf_count, result.out_of_bounds_count);
            if(line.virtual_length_px < options_.min_line_length_px)
                continue;
            if(line.valid_sample_ratio < options_.min_valid_sample_ratio)
                continue;

            filtered_lengths.push_back(line.virtual_length_px);
            result.lines[view_id].push_back(line);
            ++global_line_id;
        }
    }
    result.extraction_time_ms = NowMs() - extraction_start;

    result.filtered_line_count_total = static_cast<int>(filtered_lengths.size());
    if(!filtered_lengths.empty())
    {
        const double sum = std::accumulate(filtered_lengths.begin(), filtered_lengths.end(), 0.0);
        result.mean_length_px = sum / filtered_lengths.size();
        result.median_length_px = Median(filtered_lengths);
    }

    if(make_overlays)
    {
        result.fisheye_overlay = MakeFisheyeOverlay(fisheye_gray, result.lines);
        for(int view_id = 0; view_id < static_cast<int>(yaw_rad_.size()); ++view_id)
            result.virtual_overlays[view_id] =
                MakeVirtualOverlay(result.virtual_images[view_id], result.lines[view_id], view_names_[view_id]);
    }

    result.total_time_ms = NowMs() - total_start;
    return result;
}

cv::Mat FisheyeLineFrontend::MakeFisheyeOverlay(
    const cv::Mat& fisheye_gray,
    const std::vector<std::vector<ViewLine>>& lines) const
{
    cv::Mat overlay;
    cv::cvtColor(fisheye_gray, overlay, cv::COLOR_GRAY2BGR);

    int count = 0;
    for(int view_id = 0; view_id < static_cast<int>(yaw_rad_.size()); ++view_id)
    {
        const cv::Scalar color = ViewColor(view_id);
        for(const ViewLine& line : lines[view_id])
        {
            std::vector<cv::Point> polyline;
            for(size_t i = 0; i < line.fisheye_samples.size(); ++i)
            {
                if(!line.sample_valid[i])
                    continue;
                const cv::Point2f& p = line.fisheye_samples[i];
                polyline.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));
            }
            if(polyline.size() >= 2)
                cv::polylines(overlay, polyline, false, color, 1, cv::LINE_AA);
            ++count;
        }
    }

    cv::putText(overlay, "filtered lines: " + std::to_string(count),
                cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return overlay;
}

cv::Mat FisheyeLineFrontend::MakeVirtualOverlay(const cv::Mat& virtual_gray,
                                                const std::vector<ViewLine>& lines,
                                                const std::string& view_name) const
{
    cv::Mat overlay;
    cv::cvtColor(virtual_gray, overlay, cv::COLOR_GRAY2BGR);
    for(const ViewLine& line : lines)
    {
        cv::line(overlay,
                 cv::Point2f(line.virtual_line[0], line.virtual_line[1]),
                 cv::Point2f(line.virtual_line[2], line.virtual_line[3]),
                 cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    cv::putText(overlay, view_name + " lines: " + std::to_string(lines.size()),
                cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return overlay;
}

} // namespace StructureSLAM
