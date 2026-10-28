#ifndef FISHEYE_LINE_FRONTEND_H
#define FISHEYE_LINE_FRONTEND_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <string>
#include <vector>

#include "camera/KannalaBrandt8Camera.h"

namespace StructureSLAM
{

class FisheyeLineFrontend
{
public:
    struct Options
    {
        int virtual_width = 512;
        int virtual_height = 512;
        double horizontal_fov_deg = 100.0;
        double vertical_fov_deg = 100.0;

        // Yaw angles of the native three virtual perspective views.
        std::vector<double> virtual_yaw_deg{-60.0, -20.0, 20.0};

        double min_line_length_px = 30.0;
        int samples_per_line = 16;
        double min_valid_sample_ratio = 0.70;
        int invalid_mask_radius_px = 1;
        // Physical bearing-domain limit.  The default preserves the native
        // CAM_B branch; Safe89 sets this to 89 degrees.
        double physical_alpha_max_deg = 113.201946;
        bool require_all_physical_samples = false;
        int lsd_refine = cv::LSD_REFINE_STD;
        double lsd_scale = 0.8;
        double lsd_sigma_scale = 0.6;
        double lsd_quant = 2.0;
        double lsd_ang_th = 22.5;
        double lsd_log_eps = 0.0;
        double lsd_density_th = 0.7;
        int lsd_n_bins = 1024;
    };

    struct ViewLine
    {
        int view_id = -1;
        int line_id = -1;
        int duplicate_group_id = -1;
        cv::Vec4f virtual_line = cv::Vec4f(0, 0, 0, 0);
        cv::Point2f fisheye_start = cv::Point2f(-1, -1);
        cv::Point2f fisheye_end = cv::Point2f(-1, -1);
        double virtual_length_px = 0.0;
        double virtual_angle_deg = 0.0;
        double response = 0.0;
        double width = 0.0;
        double precision = 0.0;
        double valid_sample_ratio = 0.0;
        std::vector<cv::Point2f> virtual_samples;
        std::vector<cv::Point2f> fisheye_samples;
        std::vector<unsigned char> sample_valid;
    };

    struct FrameResult
    {
        std::vector<int> raw_line_counts;
        int raw_line_count_total = 0;
        int filtered_line_count_total = 0;
        double mean_length_px = 0.0;
        double median_length_px = 0.0;
        double remap_time_ms = 0.0;
        double extraction_time_ms = 0.0;
        double total_time_ms = 0.0;
        int nan_inf_count = 0;
        int out_of_bounds_count = 0;
        std::vector<std::vector<ViewLine>> lines;
        std::vector<cv::Mat> virtual_images;
        cv::Mat fisheye_overlay;
        std::vector<cv::Mat> virtual_overlays;
    };

    FisheyeLineFrontend(const std::vector<float>& kb8_params,
                        const cv::Size& fisheye_size,
                        const Options& options);

    FrameResult ProcessFrame(const cv::Mat& fisheye_gray, bool make_overlays) const;

    const Options& GetOptions() const { return options_; }
    const std::vector<float>& GetCameraParameters() const { return kb8_params_; }
    cv::Size GetFisheyeSize() const { return fisheye_size_; }
    const std::vector<cv::Mat>& GetValidMasks() const { return valid_masks_; }
    const std::vector<std::string>& GetViewNames() const { return view_names_; }
    double GetInvalidMappingRatio() const { return invalid_mapping_ratio_; }

private:
    struct RemapView
    {
        cv::Mat map_x;
        cv::Mat map_y;
    };

    void BuildRemaps();

    // Stage 3A-compatible full-FOV KB8 projection.
    // This intentionally allows rays with z <= 0, unlike the
    // positive-depth projection used by pose optimization.
    bool ProjectFullFovRay(const cv::Point3f& ray,
                           cv::Point2f& fisheye_point) const;

    cv::Point3f VirtualPixelToFisheyeRay(float u, float v, double yaw_rad) const;
    bool SampleVirtualPoint(int view_id, const cv::Point2f& virtual_point, cv::Point2f& fisheye_point) const;
    ViewLine BuildLine(int view_id,
                       int line_id,
                       const cv::Vec4f& segment,
                       double width,
                       double precision,
                       double response,
                       int& nan_inf_count,
                       int& out_of_bounds_count) const;
    cv::Mat MakeFisheyeOverlay(const cv::Mat& fisheye_gray,
                               const std::vector<std::vector<ViewLine>>& lines) const;
    cv::Mat MakeVirtualOverlay(const cv::Mat& virtual_gray,
                               const std::vector<ViewLine>& lines,
                               const std::string& view_name) const;

    Options options_;
    std::vector<float> kb8_params_;
    cv::Size fisheye_size_;
    KannalaBrandt8Camera camera_;
    std::vector<double> yaw_rad_;
    std::vector<std::string> view_names_;
    std::vector<RemapView> remaps_;
    std::vector<cv::Mat> valid_masks_;
    double invalid_mapping_ratio_ = 0.0;
};

} // namespace StructureSLAM

#endif
