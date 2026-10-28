#include "mono_tum_vi_native_fisheye_plslam_runner.h"

#include "System.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace StructureSLAM
{
namespace
{

struct ImageEntry
{
    double timestamp = 0.0;
    std::string image_path;
};

bool IsAbsolutePath(const std::string& path)
{
    return !path.empty() && path[0] == '/';
}

std::string JoinPath(const std::string& left, const std::string& right)
{
    if(right.empty() || IsAbsolutePath(right))
        return right;
    if(left.empty() || left[left.size() - 1] == '/')
        return left + right;
    return left + "/" + right;
}

bool HasImageExtension(const std::string& name)
{
    const std::string lower_name = [&name]() {
        std::string value = name;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return value;
    }();
    return lower_name.size() >= 4 &&
           (lower_name.rfind(".png") == lower_name.size() - 4 ||
            lower_name.rfind(".jpg") == lower_name.size() - 4 ||
            lower_name.rfind(".jpeg") == lower_name.size() - 5 ||
            lower_name.rfind(".pgm") == lower_name.size() - 4);
}

std::vector<std::string> ListImages(const std::string& image_folder)
{
    std::vector<std::string> images;
    DIR* dir = opendir(image_folder.c_str());
    if(!dir)
        return images;
    while(dirent* entry = readdir(dir))
    {
        const std::string name(entry->d_name);
        if(name == "." || name == ".." || !HasImageExtension(name))
            continue;
        images.push_back(JoinPath(image_folder, name));
    }
    closedir(dir);
    std::sort(images.begin(), images.end());
    return images;
}

double NormalizeTimestamp(double timestamp)
{
    if(std::fabs(timestamp) > 1.0e12)
        return timestamp * 1.0e-9;
    return timestamp;
}

bool MakeDirectoryRecursive(const std::string& path)
{
    if(path.empty())
        return false;
    std::string partial;
    for(size_t i = 0; i < path.size(); ++i)
    {
        partial.push_back(path[i]);
        if(path[i] != '/' && i + 1 != path.size())
            continue;
        if(partial.empty() || partial == "/")
            continue;
        if(mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
            return false;
    }
    return true;
}

bool LoadImageEntries(const std::string& image_folder,
                      const std::string& times_file,
                      std::vector<ImageEntry>* entries,
                      std::string* error)
{
    std::ifstream input(times_file.c_str());
    if(!input.is_open())
    {
        *error = "failed to open times file";
        return false;
    }

    std::vector<double> timestamps;
    std::vector<std::string> explicit_paths;
    std::string line;
    while(std::getline(input, line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        double timestamp = 0.0;
        if(!(ss >> timestamp))
            continue;
        timestamps.push_back(NormalizeTimestamp(timestamp));
        std::string relative_image_path;
        if(ss >> relative_image_path)
            explicit_paths.push_back(relative_image_path);
    }

    if(timestamps.empty())
    {
        *error = "times file contains no frame timestamps";
        return false;
    }

    entries->clear();
    entries->reserve(timestamps.size());
    if(explicit_paths.size() == timestamps.size())
    {
        for(size_t i = 0; i < timestamps.size(); ++i)
        {
            ImageEntry entry;
            entry.timestamp = timestamps[i];
            entry.image_path = JoinPath(image_folder, explicit_paths[i]);
            entries->push_back(entry);
        }
        return true;
    }

    const std::vector<std::string> images = ListImages(image_folder);
    if(images.size() < timestamps.size())
    {
        *error = "not enough images in image_folder for timestamp-only file";
        return false;
    }

    for(size_t i = 0; i < timestamps.size(); ++i)
    {
        ImageEntry entry;
        entry.timestamp = timestamps[i];
        entry.image_path = images[i];
        entries->push_back(entry);
    }
    return true;
}

bool PrintCameraContract(const std::string& settings_path,
                         const std::vector<ImageEntry>& entries)
{
    std::cout << "[CAMERA_CONTRACT]" << std::endl;
    if(entries.empty())
    {
        std::cout << "CAMERA_CONTRACT_FAIL reason=no_input_images"
                  << std::endl;
        return false;
    }

    cv::Mat first_image =
        cv::imread(entries.front().image_path, cv::IMREAD_UNCHANGED);
    if(first_image.empty())
    {
        std::cout << "CAMERA_CONTRACT_FAIL reason=first_image_unreadable"
                  << std::endl;
        return false;
    }

    cv::FileStorage settings(settings_path, cv::FileStorage::READ);
    if(!settings.isOpened())
    {
        std::cout << "CAMERA_CONTRACT_FAIL reason=settings_unreadable"
                  << std::endl;
        return false;
    }

    std::string camera_type = "UNSET";
    if(!settings["Camera.type"].empty())
        settings["Camera.type"] >> camera_type;

    const int config_width = static_cast<int>(settings["Camera.width"]);
    const int config_height = static_cast<int>(settings["Camera.height"]);
    const double fx = static_cast<double>(settings["Camera1.fx"]);
    const double fy = static_cast<double>(settings["Camera1.fy"]);
    const double cx = static_cast<double>(settings["Camera1.cx"]);
    const double cy = static_cast<double>(settings["Camera1.cy"]);
    const double k1 = static_cast<double>(settings["Camera1.k1"]);
    const double k2 = static_cast<double>(settings["Camera1.k2"]);
    const double k3 = static_cast<double>(settings["Camera1.k3"]);
    const double k4 = static_cast<double>(settings["Camera1.k4"]);
    const int native_fisheye_enabled =
        static_cast<int>(settings["NativeFisheye.Enable"]);
    const int point_line_tracking_enabled =
        static_cast<int>(settings["NativeFisheye.PointLineTracking"]);
    const int virtual_views =
        static_cast<int>(settings["NativeFisheye.VirtualViews"]);
    const bool physical_pl_mode =
        native_fisheye_enabled != 0 &&
        point_line_tracking_enabled != 0 &&
        virtual_views > 0;
    const std::string mode =
        physical_pl_mode ? "PHYSICAL_PL" : "POINT_ONLY";

    std::cout << std::setprecision(17)
              << "image_width=" << first_image.cols << std::endl
              << "image_height=" << first_image.rows << std::endl
              << "config_width=" << config_width << std::endl
              << "config_height=" << config_height << std::endl
              << "camera_type=" << camera_type << std::endl
              << "fx=" << fx << std::endl
              << "fy=" << fy << std::endl
              << "cx=" << cx << std::endl
              << "cy=" << cy << std::endl
              << "k1=" << k1 << std::endl
              << "k2=" << k2 << std::endl
              << "k3=" << k3 << std::endl
              << "k4=" << k4 << std::endl
              << "mode=" << mode << std::endl;

    const bool pass =
        first_image.cols == config_width &&
        first_image.rows == config_height &&
        camera_type == "KannalaBrandt8" &&
        std::isfinite(fx) && std::isfinite(fy) &&
        std::isfinite(cx) && std::isfinite(cy) &&
        std::isfinite(k1) && std::isfinite(k2) &&
        std::isfinite(k3) && std::isfinite(k4);
    std::cout << (pass ? "CAMERA_CONTRACT_PASS"
                       : "CAMERA_CONTRACT_FAIL")
              << std::endl;
    return pass;
}

std::set<int> ParseExportFrames(const char* value)
{
    std::set<int> frames;
    if(!value)
        return frames;
    std::stringstream input(value);
    std::string token;
    while(std::getline(input, token, ','))
    {
        if(token.empty())
            continue;
        frames.insert(std::atoi(token.c_str()));
    }
    return frames;
}

std::string CsvBool(const bool value)
{
    return value ? "1" : "0";
}

std::string CsvIntOrNa(const int value)
{
    if(value < 0)
        return "NA";
    return std::to_string(value);
}

std::string MatTypeName(const int type)
{
    switch(type)
    {
        case CV_8UC1: return "CV_8UC1";
        case CV_8UC3: return "CV_8UC3";
        case CV_8UC4: return "CV_8UC4";
        case CV_16UC1: return "CV_16UC1";
        case CV_16UC3: return "CV_16UC3";
        case CV_32FC1: return "CV_32FC1";
        default: return std::to_string(type);
    }
}

void WriteImageStage(std::ofstream& csv,
                     const int frame_index,
                     const double timestamp,
                     const std::string& stage,
                     const int type,
                     const int channels,
                     const int rows,
                     const int cols,
                     const double min_value,
                     const double max_value,
                     const double mean_value)
{
    csv << frame_index << ','
        << std::setprecision(17) << timestamp << ','
        << stage << ','
        << MatTypeName(type) << ','
        << type << ','
        << channels << ','
        << rows << ','
        << cols << ','
        << min_value << ','
        << max_value << ','
        << mean_value << '\n';
}

void MatValueRange(const cv::Mat& image,
                   double* min_value,
                   double* max_value,
                   double* mean_value)
{
    if(image.empty())
    {
        *min_value = 0.0;
        *max_value = 0.0;
        *mean_value = 0.0;
        return;
    }
    cv::minMaxLoc(image, min_value, max_value);
    *mean_value = cv::mean(image)[0];
}

void WriteCsvHeader(std::ofstream& csv)
{
    csv << "frame_id,timestamp,tracking_state,native_observations,"
        << "active_tracks,pending_tracks,formal_maplines,local_maplines,"
        << "projected_maplines,matched_maplines,point_edges,line_edges,"
        << "physical_observations,physical_group_size_1,"
        << "physical_group_size_2,physical_group_size_3,physical_matches,"
        << "physical_residual_created,physical_residual_active,"
        << "physical_residual_accepted,physical_residual_rejected,"
        << "duplicate_physical_residual_violations,"
        << "line_projection_candidates,line_rejected_bad,"
        << "line_rejected_geometry,line_rejected_temporal,"
        << "line_rejected_descriptor,line_rejected_duplicate,"
        << "line_gate_input_pairs,line_fail_input_invalid,"
        << "line_fail_invalid_yaw_or_pose,line_fail_invalid_world_segment,"
        << "line_fail_invalid_camera_segment,line_fail_depth,"
        << "line_fail_range,line_fail_distance_invariance,"
        << "line_fail_fisheye_projection,line_fail_fisheye_fov,"
        << "line_fail_virtual_depth,line_fail_virtual_fov,"
        << "line_fail_angle,line_fail_overlap,line_fail_temporal,"
        << "line_fail_descriptor,line_fail_duplicate,"
        << "line_view0_input,line_view1_input,line_view2_input,"
        << "line_view0_projection_pass,line_view1_projection_pass,"
        << "line_view2_projection_pass,line_view0_angle_pass,"
        << "line_view1_angle_pass,line_view2_angle_pass,"
        << "line_view0_overlap_pass,line_view1_overlap_pass,"
        << "line_view2_overlap_pass,line_view0_descriptor_pass,"
        << "line_view1_descriptor_pass,line_view2_descriptor_pass,"
        << "line_view0_association,line_view1_association,"
        << "line_view2_association,line_angle_fail_mean,"
        << "line_angle_pass_mean,line_angle_fail_count,"
        << "line_angle_pass_count,line_angle_fail_within_5pct,"
        << "line_angle_fail_within_10pct,line_angle_fail_within_20pct,"
        << "line_overlap_fail_mean,line_overlap_pass_mean,"
        << "line_overlap_fail_count,line_overlap_pass_count,"
        << "line_overlap_fail_within_5pct,"
        << "line_overlap_fail_within_10pct,"
        << "line_overlap_fail_within_20pct,"
        << "point_inliers,line_inliers,point_cost_before,point_cost_after,"
        << "line_cost_before,line_cost_after,pose_delta_translation,"
        << "pose_delta_rotation,joint_accepted,pose_source,"
        << "fallback_reason,new_keyframe,"
        << "new_maplines,native_enabled,point_line_tracking_enabled,"
        << "virtual_views,kb8_parameters_available,formal_system_path\n";
}

void WriteCsvRow(std::ofstream& csv,
                 const NativeLoopFrameDiagnostics& diagnostics)
{
    csv << diagnostics.frame_id << ','
        << std::setprecision(17) << diagnostics.timestamp << ','
        << diagnostics.tracking_state << ','
        << diagnostics.native_observations << ','
        << diagnostics.active_tracks << ','
        << CsvIntOrNa(diagnostics.pending_tracks) << ','
        << diagnostics.formal_maplines << ','
        << diagnostics.local_maplines << ','
        << diagnostics.projected_maplines << ','
        << diagnostics.matched_maplines << ','
        << diagnostics.point_edges << ','
        << diagnostics.line_edges << ','
        << diagnostics.physical_observations << ','
        << diagnostics.physical_group_size_1 << ','
        << diagnostics.physical_group_size_2 << ','
        << diagnostics.physical_group_size_3 << ','
        << diagnostics.physical_matches << ','
        << diagnostics.physical_residual_created << ','
        << diagnostics.physical_residual_active << ','
        << diagnostics.physical_residual_accepted << ','
        << diagnostics.physical_residual_rejected << ','
        << diagnostics.duplicate_physical_residual_violations << ','
        << diagnostics.line_projection_candidates << ','
        << diagnostics.line_rejected_bad << ','
        << diagnostics.line_rejected_geometry << ','
        << diagnostics.line_rejected_temporal << ','
        << diagnostics.line_rejected_descriptor << ','
        << diagnostics.line_rejected_duplicate << ','
        << diagnostics.line_gate_input_pairs << ','
        << diagnostics.line_fail_input_invalid << ','
        << diagnostics.line_fail_invalid_yaw_or_pose << ','
        << diagnostics.line_fail_invalid_world_segment << ','
        << diagnostics.line_fail_invalid_camera_segment << ','
        << diagnostics.line_fail_depth << ','
        << diagnostics.line_fail_range << ','
        << diagnostics.line_fail_distance_invariance << ','
        << diagnostics.line_fail_fisheye_projection << ','
        << diagnostics.line_fail_fisheye_fov << ','
        << diagnostics.line_fail_virtual_depth << ','
        << diagnostics.line_fail_virtual_fov << ','
        << diagnostics.line_fail_angle << ','
        << diagnostics.line_fail_overlap << ','
        << diagnostics.line_fail_temporal << ','
        << diagnostics.line_fail_descriptor << ','
        << diagnostics.line_fail_duplicate << ','
        << diagnostics.line_view0_input << ','
        << diagnostics.line_view1_input << ','
        << diagnostics.line_view2_input << ','
        << diagnostics.line_view0_projection_pass << ','
        << diagnostics.line_view1_projection_pass << ','
        << diagnostics.line_view2_projection_pass << ','
        << diagnostics.line_view0_angle_pass << ','
        << diagnostics.line_view1_angle_pass << ','
        << diagnostics.line_view2_angle_pass << ','
        << diagnostics.line_view0_overlap_pass << ','
        << diagnostics.line_view1_overlap_pass << ','
        << diagnostics.line_view2_overlap_pass << ','
        << diagnostics.line_view0_descriptor_pass << ','
        << diagnostics.line_view1_descriptor_pass << ','
        << diagnostics.line_view2_descriptor_pass << ','
        << diagnostics.line_view0_association << ','
        << diagnostics.line_view1_association << ','
        << diagnostics.line_view2_association << ','
        << diagnostics.line_angle_fail_mean << ','
        << diagnostics.line_angle_pass_mean << ','
        << diagnostics.line_angle_fail_count << ','
        << diagnostics.line_angle_pass_count << ','
        << diagnostics.line_angle_fail_within_5pct << ','
        << diagnostics.line_angle_fail_within_10pct << ','
        << diagnostics.line_angle_fail_within_20pct << ','
        << diagnostics.line_overlap_fail_mean << ','
        << diagnostics.line_overlap_pass_mean << ','
        << diagnostics.line_overlap_fail_count << ','
        << diagnostics.line_overlap_pass_count << ','
        << diagnostics.line_overlap_fail_within_5pct << ','
        << diagnostics.line_overlap_fail_within_10pct << ','
        << diagnostics.line_overlap_fail_within_20pct << ','
        << diagnostics.point_inliers << ','
        << diagnostics.line_inliers << ','
        << diagnostics.point_cost_before << ','
        << diagnostics.point_cost_after << ','
        << diagnostics.line_cost_before << ','
        << diagnostics.line_cost_after << ','
        << diagnostics.pose_delta_translation << ','
        << diagnostics.pose_delta_rotation << ','
        << CsvBool(diagnostics.joint_accepted) << ','
        << diagnostics.pose_source << ','
        << diagnostics.fallback_reason << ','
        << CsvBool(diagnostics.new_keyframe) << ','
        << CsvIntOrNa(diagnostics.new_maplines) << ','
        << CsvBool(diagnostics.native_enabled) << ','
        << CsvBool(diagnostics.point_line_tracking_enabled) << ','
        << diagnostics.virtual_views << ','
        << CsvBool(diagnostics.kb8_parameters_available) << ','
        << CsvBool(diagnostics.formal_system_path) << '\n';
}

void WritePointFeatureHeader(std::ofstream& csv)
{
    csv << "frame_id,timestamp,tracking_state,orb_keypoints_raw,"
        << "undistorted_keypoints,descriptors_rows,"
        << "initial_frame_keypoints,initial_matches,"
        << "valid_match_count,median_displacement,median_parallax\n";
}

void WriteSafe89AuditHeader(std::ofstream& csv)
{
    csv << "frame_id,timestamp,tracking_state,"
        << "total_detected_features,valid_features_after_mask,"
        << "accepted_orb_features,feature_density_per_valid_pixel,"
        << "orb_bin_0_30,orb_bin_30_60,orb_bin_60_80,orb_bin_80_89,"
        << "orb_max_alpha_deg,orb_alpha_leaks,"
        << "line_support_max_alpha_deg,line_support_alpha_leaks,"
        << "matched_point_alpha_leaks,optimizer_point_alpha_leaks\n";
}

void WriteSafe89AuditRow(std::ofstream& csv,
                         const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << std::setprecision(17) << d.timestamp << ','
        << d.tracking_state << ','
        << d.safe89_orb_detected_before_mask << ','
        << d.safe89_orb_valid_after_mask << ','
        << d.orb_keypoints_raw << ','
        << d.safe89_orb_feature_density << ','
        << d.safe89_orb_bin_0_30 << ','
        << d.safe89_orb_bin_30_60 << ','
        << d.safe89_orb_bin_60_80 << ','
        << d.safe89_orb_bin_80_89 << ','
        << d.safe89_orb_max_alpha_deg << ','
        << d.safe89_orb_alpha_leaks << ','
        << d.safe89_line_support_max_alpha_deg << ','
        << d.safe89_line_support_alpha_leaks << ','
        << d.safe89_matched_point_alpha_leaks << ','
        << d.safe89_optimizer_point_alpha_leaks << '\n';
}

void WritePointFeatureRow(std::ofstream& csv,
                          const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << std::setprecision(17) << d.timestamp << ','
        << d.tracking_state << ','
        << d.orb_keypoints_raw << ','
        << d.undistorted_keypoints << ','
        << d.descriptors_rows << ','
        << d.initial_frame_keypoints << ','
        << d.initial_matches << ','
        << d.valid_match_count << ','
        << d.median_displacement << ','
        << d.median_parallax << '\n';
}

void WriteInitializationAttemptHeader(std::ofstream& csv)
{
    csv << "frame_id,timestamp,entered,initializer_created,"
        << "initializer_reset,initialization_failure_reason,"
        << "initializer_model,h_score,f_score,h_ratio,"
        << "tracking_track_called,early_return,"
        << "initial_frame_id,search_center_mode,search_center_reset,"
        << "search_center_median_shift_from_initial,"
        << "search_center_p95_shift_from_initial,matches,solver_success,"
        << "legacy_solver_success,kb8_postvalidation_success,"
        << "initialization_final_success,failure_reason,"
        << "candidate_count,best_candidate_id,best_good_points,"
        << "second_best_good_points,best_positive_depth_cam1,"
        << "best_positive_depth_cam2,best_both_positive,"
        << "best_triangulated,best_reprojection_pass,"
        << "best_median_parallax_deg,required_min_good_points,"
        << "required_min_parallax_deg,ambiguity_ratio_or_threshold,"
        << "exact_failure_branch\n";
}

void WriteInitializationAttemptRow(std::ofstream& csv,
                                   const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << std::setprecision(17) << d.timestamp << ','
        << CsvBool(d.monocular_initialization_entered) << ','
        << CsvBool(d.initializer_created) << ','
        << CsvBool(d.initializer_reset) << ','
        << (d.initialization_failure_reason.empty()
                ? "unknown"
                : d.initialization_failure_reason) << ','
        << (d.initializer_model.empty()
                ? "none"
                : d.initializer_model) << ','
        << d.initializer_h_score << ','
        << d.initializer_f_score << ','
        << d.initializer_h_ratio << ','
        << CsvBool(d.tracking_track_called) << ','
        << CsvBool(d.early_return) << ','
        << d.initial_frame_id << ','
        << (d.search_center_mode.empty()
                ? "NA"
                : d.search_center_mode) << ','
        << CsvBool(d.search_center_reset) << ','
        << d.search_center_median_shift_from_initial << ','
        << d.search_center_p95_shift_from_initial << ','
        << d.initial_matches << ','
        << CsvBool(d.monocular_initialization_entered &&
                   d.initialization_failure_reason.empty() &&
                   !d.early_return) << ','
        << CsvBool(d.legacy_solver_success) << ','
        << CsvBool(d.kb8_postvalidation_success) << ','
        << CsvBool(d.initialization_final_success) << ','
        << (d.initialization_failure_reason.empty()
                ? "unknown"
                : d.initialization_failure_reason) << ','
        << d.initialization_candidate_count << ','
        << d.initialization_best_candidate_id << ','
        << d.initialization_best_good_points << ','
        << d.initialization_second_best_good_points << ','
        << d.initialization_best_positive_depth_cam1 << ','
        << d.initialization_best_positive_depth_cam2 << ','
        << d.initialization_best_both_positive << ','
        << d.initialization_best_triangulated << ','
        << d.initialization_best_reprojection_pass << ','
        << d.initialization_best_median_parallax_deg << ','
        << d.initialization_required_min_good_points << ','
        << d.initialization_required_min_parallax_deg << ','
        << d.initialization_ambiguity_ratio_or_threshold << ','
        << (d.initialization_exact_failure_branch.empty()
                ? "NA"
                : d.initialization_exact_failure_branch) << '\n';
}

void WritePoseCandidateHeader(std::ofstream& csv)
{
    csv << "reference_frame_id,current_frame_id,model,candidate_id,"
        << "R00,R01,R02,R10,R11,R12,R20,R21,R22,tx,ty,tz,"
        << "valid_match_count,positive_lambda_count,positive_lambda_ratio,"
        << "negative_lambda1_count,negative_lambda2_count,"
        << "ray_miss_median,ray_miss_p95,angular_error_median_deg,"
        << "angular_error_p95_deg,triangulation_condition_median,"
        << "triangulation_condition_p95,positive_depth_cam1,"
        << "positive_depth_cam2,both_positive,reprojection_pass,"
        << "triangulated,median_parallax_deg,bin_0_30_total,"
        << "bin_0_30_positive,bin_30_60_total,bin_30_60_positive,"
        << "bin_60_90_total,bin_60_90_positive,bin_90_113p2_total,"
        << "bin_90_113p2_positive,"
        << "accepted_by_check_rt\n";
}

void WritePoseCandidateRows(std::ofstream& csv,
                            const NativeLoopFrameDiagnostics& d)
{
    for(size_t i = 0;
        i < d.initialization_pose_candidates.size();
        ++i)
    {
        const InitializationPoseCandidateDiagnostics& c =
            d.initialization_pose_candidates[i];
        csv << d.initial_frame_id << ','
            << d.frame_id << ','
            << c.model << ','
            << c.candidate_id << ','
            << c.R21[0] << ',' << c.R21[1] << ',' << c.R21[2] << ','
            << c.R21[3] << ',' << c.R21[4] << ',' << c.R21[5] << ','
            << c.R21[6] << ',' << c.R21[7] << ',' << c.R21[8] << ','
            << c.t21[0] << ',' << c.t21[1] << ',' << c.t21[2] << ','
            << c.valid_match_count << ',' << c.positive_lambda_count << ','
            << c.positive_lambda_ratio << ',' << c.negative_lambda1_count << ','
            << c.negative_lambda2_count << ',' << c.ray_miss_median << ','
            << c.ray_miss_p95 << ',' << c.angular_error_median_deg << ','
            << c.angular_error_p95_deg << ','
            << c.triangulation_condition_median << ','
            << c.triangulation_condition_p95 << ','
            << c.positive_depth_cam1 << ','
            << c.positive_depth_cam2 << ','
            << c.both_positive << ','
            << c.reprojection_pass << ','
            << c.triangulated << ','
            << c.median_parallax_deg << ','
            << c.angular_bin_total[0] << ',' << c.angular_bin_positive[0] << ','
            << c.angular_bin_total[1] << ',' << c.angular_bin_positive[1] << ','
            << c.angular_bin_total[2] << ',' << c.angular_bin_positive[2] << ','
            << c.angular_bin_total[3] << ',' << c.angular_bin_positive[3] << ','
            << CsvBool(c.accepted_by_check_rt) << '\n';
    }
}

void WriteReferenceLifecycleHeader(std::ofstream& csv)
{
    csv << "current_frame_id,timestamp,initial_frame_id,"
        << "initializer_created,initializer_reset,initializer_age,"
        << "failure_reason,reference_retained_after_failure,"
        << "reference_replaced_after_failure,replacement_condition\n";
}

void WriteReferenceLifecycleRow(std::ofstream& csv,
                                const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << std::setprecision(17) << d.timestamp << ','
        << d.initial_frame_id << ','
        << CsvBool(d.initializer_created) << ','
        << CsvBool(d.initializer_reset) << ','
        << d.initializer_age << ','
        << (d.initialization_failure_reason.empty()
                ? "unknown"
                : d.initialization_failure_reason) << ','
        << CsvBool(d.reference_frame_retained_after_failure) << ','
        << CsvBool(d.reference_frame_replaced_after_failure) << ','
        << (d.reference_frame_replacement_condition.empty()
                ? "NA"
                : d.reference_frame_replacement_condition) << '\n';
}

void WriteFrameTrackingMetricsHeader(std::ofstream& csv)
{
    csv << "frame_id,tracking_state_before,tracking_state_after,"
        << "reference_keyframe_id,map_points_total,local_map_points,"
        << "mappoint_projection_input,valid_camera_points,"
        << "valid_kb8_projections,inside_image,track_in_view,"
        << "search_by_projection_attempts,accepted_2d3d_matches,"
        << "pose_point_edges,pose_initial_inliers,pose_final_inliers,"
        << "pose_outliers,reprojection_error_before,"
        << "reprojection_error_after,pose_optimization_success,"
        << "keyframe_created,guard_triggered,guard_reason,"
        << "tracking_failure_reason\n";
}

void WriteFrameTrackingMetricsRow(std::ofstream& csv,
                                  const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << d.tracking_state_before << ','
        << d.tracking_state_after << ','
        << d.reference_keyframe_id << ','
        << d.map_points_total << ','
        << d.local_map_points << ','
        << d.mappoint_projection_input << ','
        << d.valid_camera_points << ','
        << d.valid_kb8_projections << ','
        << d.inside_image << ','
        << d.track_in_view << ','
        << d.search_by_projection_attempts << ','
        << d.accepted_2d3d_matches << ','
        << d.pose_point_edges << ','
        << d.pose_initial_inliers << ','
        << d.pose_final_inliers << ','
        << d.pose_outliers << ','
        << d.reprojection_error_before << ','
        << d.reprojection_error_after << ','
        << CsvBool(d.pose_optimization_success) << ','
        << CsvBool(d.new_keyframe) << ','
        << CsvBool(d.guard_triggered) << ','
        << (d.guard_reason.empty() ? "NA" : d.guard_reason) << ','
        << (d.tracking_failure_reason.empty()
                ? "NA"
                : d.tracking_failure_reason) << '\n';
}

void WriteKb8ProjectionRuntimeHeader(std::ofstream& csv)
{
    csv << "frame_id,timestamp,mappoint_projection_input,"
        << "valid_camera_points,valid_kb8_projections,inside_image,"
        << "track_in_view\n";
}

void WriteKb8ProjectionRuntimeRow(std::ofstream& csv,
                                  const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << std::setprecision(17) << d.timestamp << ','
        << d.mappoint_projection_input << ','
        << d.valid_camera_points << ','
        << d.valid_kb8_projections << ','
        << d.inside_image << ','
        << d.track_in_view << '\n';
}

void WriteSearchByProjectionRuntimeHeader(std::ofstream& csv)
{
    csv << "frame_id,timestamp,search_by_projection_attempts,"
        << "accepted_2d3d_matches\n";
}

void WriteSearchByProjectionRuntimeRow(std::ofstream& csv,
                                       const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << std::setprecision(17) << d.timestamp << ','
        << d.search_by_projection_attempts << ','
        << d.accepted_2d3d_matches << '\n';
}

void WritePoseOptimizationRuntimeHeader(std::ofstream& csv)
{
    csv << "frame_id,timestamp,pose_point_edges,"
        << "pose_initial_inliers,pose_final_inliers,pose_outliers,"
        << "reprojection_error_before,reprojection_error_after,"
        << "pose_optimization_success\n";
}

void WritePoseOptimizationRuntimeRow(std::ofstream& csv,
                                     const NativeLoopFrameDiagnostics& d)
{
    csv << d.frame_id << ','
        << std::setprecision(17) << d.timestamp << ','
        << d.pose_point_edges << ','
        << d.pose_initial_inliers << ','
        << d.pose_final_inliers << ','
        << d.pose_outliers << ','
        << d.reprojection_error_before << ','
        << d.reprojection_error_after << ','
        << CsvBool(d.pose_optimization_success) << '\n';
}

} // namespace

int RunNativeFisheyePlslamSequence(const std::string& vocabulary_path,
                                   const std::string& settings_path,
                                   const std::string& image_folder,
                                   const std::string& times_file,
                                   const std::string& output_dir,
                                   int max_frames)
{
    if(!MakeDirectoryRecursive(output_dir))
    {
        std::cerr << "Failed to create output directory: "
                  << output_dir << std::endl;
        return 1;
    }

    std::vector<ImageEntry> entries;
    std::string load_error;
    if(!LoadImageEntries(image_folder, times_file, &entries, &load_error))
    {
        std::cerr << "Failed to load sequence entries: "
                  << load_error << std::endl;
        return 1;
    }

    if(max_frames > 0 &&
       max_frames < static_cast<int>(entries.size()))
    {
        entries.resize(static_cast<size_t>(max_frames));
    }

    if(!PrintCameraContract(settings_path, entries))
        return 1;

    const std::string csv_path =
        JoinPath(output_dir, "native_loop_per_frame.csv");
    std::ofstream csv(csv_path.c_str());
    if(!csv.is_open())
    {
        std::cerr << "Failed to open native loop CSV at: "
                  << csv_path << std::endl;
        return 1;
    }
    WriteCsvHeader(csv);

    std::ofstream image_csv(
        JoinPath(output_dir, "image_type_value_range.csv").c_str());
    if(!image_csv.is_open())
    {
        std::cerr << "Failed to open image type CSV" << std::endl;
        return 1;
    }
    image_csv << "frame_index,timestamp,stage,type,type_id,channels,"
              << "rows,cols,min,max,mean\n";

    std::ofstream point_csv(
        JoinPath(output_dir, "point_feature_per_frame.csv").c_str());
    if(!point_csv.is_open())
    {
        std::cerr << "Failed to open point feature CSV" << std::endl;
        return 1;
    }
    WritePointFeatureHeader(point_csv);

    std::ofstream safe89_audit_csv(
        JoinPath(output_dir, "safe89_domain_audit.csv").c_str());
    if(!safe89_audit_csv.is_open())
    {
        std::cerr << "Failed to open Safe89 domain audit CSV"
                  << std::endl;
        return 1;
    }
    WriteSafe89AuditHeader(safe89_audit_csv);

    std::ofstream init_csv(
        JoinPath(output_dir, "initialization_attempts.csv").c_str());
    if(!init_csv.is_open())
    {
        std::cerr << "Failed to open initialization attempts CSV"
                  << std::endl;
        return 1;
    }
    WriteInitializationAttemptHeader(init_csv);

    std::ofstream pose_candidate_csv(
        JoinPath(output_dir, "pose_candidate_checks.csv").c_str());
    if(!pose_candidate_csv.is_open())
    {
        std::cerr << "Failed to open pose candidate CSV"
                  << std::endl;
        return 1;
    }
    WritePoseCandidateHeader(pose_candidate_csv);

    std::ofstream reference_lifecycle_csv(
        JoinPath(output_dir, "reference_lifecycle_log.csv").c_str());
    if(!reference_lifecycle_csv.is_open())
    {
        std::cerr << "Failed to open reference lifecycle CSV"
                  << std::endl;
        return 1;
    }
    WriteReferenceLifecycleHeader(reference_lifecycle_csv);

    const std::string localmapping_guard_log_path =
        JoinPath(output_dir, "localmapping_guard_runtime_log.csv");
    std::ofstream localmapping_guard_log(
        localmapping_guard_log_path.c_str());
    if(!localmapping_guard_log.is_open())
    {
        std::cerr << "Failed to open LocalMapping guard log CSV"
                  << std::endl;
        return 1;
    }
    localmapping_guard_log
        << "frame_id,keyframe_id,localmapping_stage,guard_triggered,"
        << "guard_reason,map_points_before,map_points_after,"
        << "map_lines_before,map_lines_after,tracking_state,"
        << "thread_alive,capabilities\n";
    localmapping_guard_log.close();
    setenv("STRUCTURE_SLAM_LOCALMAPPING_GUARD_LOG",
           localmapping_guard_log_path.c_str(),
           1);

    const std::string kb8_new_mappoint_stats_path =
        JoinPath(output_dir, "kb8_new_mappoint_runtime_stats.csv");
    std::ofstream kb8_new_mappoint_stats(
        kb8_new_mappoint_stats_path.c_str());
    if(!kb8_new_mappoint_stats.is_open())
    {
        std::cerr << "Failed to open KB8 new MapPoint stats CSV"
                  << std::endl;
        return 1;
    }
    kb8_new_mappoint_stats
        << "current_keyframe_id,neighbor_keyframe_id,candidate_matches,"
        << "valid_bearing_pairs,epipolar_gate_pass,"
        << "triangulation_attempted,triangulation_finite,"
        << "positive_or_valid_depth,parallax_pass,"
        << "reprojection_pass_kf1,reprojection_pass_kf2,"
        << "scale_consistency_pass,formal_mappoint_created,"
        << "reject_invalid_bearing,reject_epipolar,"
        << "reject_triangulation,reject_depth,reject_parallax,"
        << "reject_reprojection,reject_scale\n";
    kb8_new_mappoint_stats.close();
    setenv("STRUCTURE_SLAM_KB8_NEW_MAPPOINT_STATS",
           kb8_new_mappoint_stats_path.c_str(),
           1);

    const std::string kb8_new_mappoint_created_path =
        JoinPath(output_dir, "kb8_new_mappoint_created_points.csv");
    std::ofstream kb8_new_mappoint_created(
        kb8_new_mappoint_created_path.c_str());
    if(!kb8_new_mappoint_created.is_open())
    {
        std::cerr << "Failed to open KB8 created MapPoint CSV"
                  << std::endl;
        return 1;
    }
    kb8_new_mappoint_created
        << "kf1_id,kf2_id,idx1,idx2,parallax_deg,"
        << "reprojection_error_kf1,reprojection_error_kf2,"
        << "octave1,octave2,Pw_x,Pw_y,Pw_z\n";
    kb8_new_mappoint_created.close();
    setenv("STRUCTURE_SLAM_KB8_NEW_MAPPOINT_CREATED",
           kb8_new_mappoint_created_path.c_str(),
           1);

    const std::string kb8_fusion_stats_path =
        JoinPath(output_dir, "kb8_fusion_runtime_stats.csv");
    std::ofstream kb8_fusion_stats(kb8_fusion_stats_path.c_str());
    if(!kb8_fusion_stats.is_open())
    {
        std::cerr << "Failed to open KB8 fusion stats CSV"
                  << std::endl;
        return 1;
    }
    kb8_fusion_stats
        << "current_keyframe_id,target_keyframe_id,direction,"
        << "input_mappoints,bad_mappoints_skipped,"
        << "valid_camera_points,valid_kb8_projections,inside_image,"
        << "distance_gate_pass,view_angle_pass,scale_gate_pass,"
        << "feature_area_queries,feature_candidates,"
        << "descriptor_compared,descriptor_pass,new_observations,"
        << "replacement_candidates,replacements_performed,"
        << "already_same_mappoint,fusion_fail_no_feature,"
        << "fusion_fail_descriptor,fusion_fail_geometry\n";
    kb8_fusion_stats.close();
    setenv("STRUCTURE_SLAM_KB8_FUSION_STATS",
           kb8_fusion_stats_path.c_str(),
           1);

    const std::string kb8_fusion_replacements_path =
        JoinPath(output_dir, "kb8_fusion_replacements.csv");
    std::ofstream kb8_fusion_replacements(
        kb8_fusion_replacements_path.c_str());
    if(!kb8_fusion_replacements.is_open())
    {
        std::cerr << "Failed to open KB8 fusion replacements CSV"
                  << std::endl;
        return 1;
    }
    kb8_fusion_replacements
        << "kept_mappoint_id,replaced_mappoint_id,"
        << "kept_observations_before,replaced_observations_before,"
        << "kept_observations_after,affected_keyframes\n";
    kb8_fusion_replacements.close();
    setenv("STRUCTURE_SLAM_KB8_FUSION_REPLACEMENTS",
           kb8_fusion_replacements_path.c_str(),
           1);

    const std::string kb8_local_ba_stats_path =
        JoinPath(output_dir, "kb8_local_ba_runtime_stats.csv");
    std::ofstream kb8_local_ba_stats(
        kb8_local_ba_stats_path.c_str());
    if(!kb8_local_ba_stats.is_open())
    {
        std::cerr << "Failed to open KB8 Local BA stats CSV"
                  << std::endl;
        return 1;
    }
    kb8_local_ba_stats
        << "current_keyframe_id,local_keyframes,fixed_keyframes,"
        << "local_mappoints,pose_vertices,point_vertices,"
        << "kb8_point_edges,legacy_pinhole_point_edges,"
        << "stereo_point_edges,line_edges,"
        << "point_edges_with_robust_kernel,"
        << "point_edges_without_robust_kernel,"
        << "optimization_round1_ran,optimization_round2_ran,"
        << "point_inliers_before,point_outliers_before,"
        << "point_inliers_after,point_outliers_after,"
        << "point_cost_before,point_cost_after,"
        << "line_cost_before,line_cost_after,"
        << "keyframe_poses_updated,mappoint_positions_updated,"
        << "observations_erased,stop_flag_triggered,"
        << "optimizer_success\n";
    kb8_local_ba_stats.close();
    setenv("STRUCTURE_SLAM_KB8_LOCAL_BA_STATS",
           kb8_local_ba_stats_path.c_str(),
           1);

    const std::string kb8_local_ba_erasure_path =
        JoinPath(output_dir, "kb8_local_ba_observation_erasure.csv");
    std::ofstream kb8_local_ba_erasure(
        kb8_local_ba_erasure_path.c_str());
    if(!kb8_local_ba_erasure.is_open())
    {
        std::cerr << "Failed to open KB8 Local BA erasure CSV"
                  << std::endl;
        return 1;
    }
    kb8_local_ba_erasure
        << "keyframe_id,mappoint_id,edge_type,chi2,depth_positive\n";
    kb8_local_ba_erasure.close();
    setenv("STRUCTURE_SLAM_KB8_LOCAL_BA_ERASURES",
           kb8_local_ba_erasure_path.c_str(),
           1);

    const std::string kb8_global_ba_stats_path =
        JoinPath(output_dir, "kb8_global_ba_runtime_stats.csv");
    std::ofstream kb8_global_ba_stats(
        kb8_global_ba_stats_path.c_str());
    if(!kb8_global_ba_stats.is_open())
    {
        std::cerr << "Failed to open KB8 Global BA stats CSV"
                  << std::endl;
        return 1;
    }
    kb8_global_ba_stats
        << "keyframes,mappoints,maplines,pose_vertices,"
        << "point_vertices,line_vertices,kb8_point_edges,"
        << "legacy_pinhole_point_edges,stereo_point_edges,"
        << "line_edges,robust_point_edges,robust_line_edges,"
        << "iterations_requested,optimization_ran,"
        << "point_inliers_before,point_outliers_before,"
        << "point_inliers_after,point_outliers_after,"
        << "point_cost_before,point_cost_after,"
        << "line_cost_before,line_cost_after,"
        << "keyframe_pose_writebacks,mappoint_position_writebacks,"
        << "mapline_position_writebacks,stop_flag_triggered,"
        << "loop_keyframe_id,robust_enabled,with_line_feature,"
        << "optimizer_success\n";
    kb8_global_ba_stats.close();
    setenv("STRUCTURE_SLAM_KB8_GLOBAL_BA_STATS",
           kb8_global_ba_stats_path.c_str(),
           1);

    {
        const std::string path =
            JoinPath(output_dir, "natural_kf_generation.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open natural KF generation CSV"
                      << std::endl;
            return 1;
        }
        out << "timestamp,frame_id,kf_id,pose,map_points\n";
        out.close();
        setenv("KB8_NATURAL_KF_GENERATION_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "natural_loop_detection.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open natural loop detection CSV"
                      << std::endl;
            return 1;
        }
        out << "current_kf,candidate_kf,bow_score,candidate_count,accepted\n";
        out.close();
        setenv("KB8_NATURAL_LOOP_DETECTION_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "natural_loop_call_chain.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open natural loop call chain CSV"
                      << std::endl;
            return 1;
        }
        out << "event,timestamp,thread,result\n";
        out.close();
        setenv("KB8_NATURAL_LOOP_CALL_CHAIN_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "natural_loop_thread_state.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open natural loop thread state CSV"
                      << std::endl;
            return 1;
        }
        out << "timestamp,thread_id,event,keyframe_id,state\n";
        out.close();
        setenv("KB8_LOOPCLOSING_THREAD_EVENTS_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "compute_sim3_runtime.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open ComputeSim3 runtime CSV"
                      << std::endl;
            return 1;
        }
        out << "current_kf,matched_kf,initial_matches,map_points_current,"
               "map_points_loop,sim3_solver_called,optimize_sim3_called,"
               "accepted,failure_reason\n";
        out.close();
        setenv("KB8_LOOPCLOSING_COMPUTE_SIM3_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "correctloop_runtime_call_chain.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open CorrectLoop call chain CSV"
                      << std::endl;
            return 1;
        }
        out << "function,thread_id,caller,timestamp,keyframe_id,matched_kf,"
               "state_before,state_after,kb8_dispatch,result\n";
        out.close();
        setenv("KB8_CORRECTLOOP_CALL_CHAIN_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "correctloop_localmapping_coordination.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open CorrectLoop LocalMapping CSV"
                      << std::endl;
            return 1;
        }
        out << "event,timestamp,thread_id,localmapping_state,"
               "map_write_allowed,result\n";
        out.close();
        setenv("KB8_CORRECTLOOP_LOCALMAPPING_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "searchandfuse_runtime_call_chain.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open SearchAndFuse call chain CSV"
                      << std::endl;
            return 1;
        }
        out << "timestamp,thread_id,current_kf,target_kf,function,"
               "direction,result\n";
        out.close();
        setenv("KB8_SEARCHANDFUSE_CALL_CHAIN_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "loop_fuse_scw_internal_stats.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open loop Fuse(Scw) stats CSV"
                      << std::endl;
            return 1;
        }
        out << "target_keyframe_id,input_loop_mappoints,"
               "scw_projection_attempts,kb8_projection_calls,"
               "legacy_pinhole_projection_calls,valid_projection,"
               "inside_image,already_observed,scale_gate_pass,"
               "descriptor_candidates,matches,new_observations,"
               "replacements,rejected_bad_point,rejected_invalid_depth,"
               "rejected_outside_image,rejected_distance,"
               "rejected_view_angle,rejected_scale,rejected_descriptor\n";
        out.close();
        setenv("STRUCTURE_SLAM_KB8_LOOP_FUSE_SCW_STATS", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "essential_graph_runtime_call_chain.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open Essential Graph runtime CSV"
                      << std::endl;
            return 1;
        }
        out << "timestamp,thread_id,event,vertex_count,edge_count,result\n";
        out.close();
        setenv("KB8_ESSENTIAL_GRAPH_RUNTIME_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "loop_gba_runtime_call_chain.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open loop GBA runtime CSV"
                      << std::endl;
            return 1;
        }
        out << "timestamp,thread_id,event,kf_count,mappoint_count,result\n";
        out.close();
        setenv("KB8_LOOP_GBA_RUNTIME_CSV", path.c_str(), 1);
    }
    {
        const std::string path =
            JoinPath(output_dir, "loop_gba_thread_lifecycle.csv");
        std::ofstream out(path.c_str());
        if(!out.is_open())
        {
            std::cerr << "Failed to open loop GBA lifecycle CSV"
                      << std::endl;
            return 1;
        }
        out << "thread,event,timestamp,state\n";
        out.close();
        setenv("KB8_LOOP_GBA_THREAD_LIFECYCLE_CSV", path.c_str(), 1);
    }

    std::ofstream frame_tracking_csv(
        JoinPath(output_dir, "frame_tracking_metrics.csv").c_str());
    if(!frame_tracking_csv.is_open())
    {
        std::cerr << "Failed to open frame tracking metrics CSV"
                  << std::endl;
        return 1;
    }
    WriteFrameTrackingMetricsHeader(frame_tracking_csv);

    std::ofstream kb8_projection_csv(
        JoinPath(output_dir, "kb8_projection_runtime_stats.csv").c_str());
    if(!kb8_projection_csv.is_open())
    {
        std::cerr << "Failed to open KB8 projection runtime CSV"
                  << std::endl;
        return 1;
    }
    WriteKb8ProjectionRuntimeHeader(kb8_projection_csv);

    std::ofstream search_projection_csv(
        JoinPath(output_dir,
                 "search_by_projection_runtime_stats.csv").c_str());
    if(!search_projection_csv.is_open())
    {
        std::cerr << "Failed to open SearchByProjection runtime CSV"
                  << std::endl;
        return 1;
    }
    WriteSearchByProjectionRuntimeHeader(search_projection_csv);

    std::ofstream pose_optimization_csv(
        JoinPath(output_dir,
                 "pose_optimization_runtime_stats.csv").c_str());
    if(!pose_optimization_csv.is_open())
    {
        std::cerr << "Failed to open pose optimization runtime CSV"
                  << std::endl;
        return 1;
    }
    WritePoseOptimizationRuntimeHeader(pose_optimization_csv);

    const char* export_dir_env =
        std::getenv("STRUCTURE_SLAM_KB8_MAP_EXPORT_DIR");
    const std::string map_export_dir =
        export_dir_env ? std::string(export_dir_env) : std::string();
    const char* colmap_export_dir_env =
        std::getenv("STRUCTURE_SLAM_KB8_COLMAP_EXPORT_DIR");
    const std::string colmap_export_dir =
        colmap_export_dir_env ? std::string(colmap_export_dir_env) :
                                std::string();
    const std::set<int> export_frames = ParseExportFrames(
        std::getenv("STRUCTURE_SLAM_KB8_MAP_EXPORT_FRAMES"));

    System slam(
        vocabulary_path,
        settings_path,
        System::MONOCULAR,
        false);

    for(size_t i = 0; i < entries.size(); ++i)
    {
        const ImageEntry& entry = entries[i];
        cv::Mat image =
            cv::imread(entry.image_path, cv::IMREAD_UNCHANGED);
        if(image.empty())
        {
            std::cerr << "Failed to load image at: "
                      << entry.image_path << std::endl;
            slam.Shutdown();
            return 1;
        }

        double min_value = 0.0;
        double max_value = 0.0;
        double mean_value = 0.0;
        if(i < 10)
        {
            MatValueRange(image, &min_value, &max_value, &mean_value);
            WriteImageStage(image_csv,
                            static_cast<int>(i),
                            entry.timestamp,
                            "runner_read",
                            image.type(),
                            image.channels(),
                            image.rows,
                            image.cols,
                            min_value,
                            max_value,
                            mean_value);
            WriteImageStage(image_csv,
                            static_cast<int>(i),
                            entry.timestamp,
                            "TrackMonocularWithPL_input",
                            image.type(),
                            image.channels(),
                            image.rows,
                            image.cols,
                            min_value,
                            max_value,
                            mean_value);
        }
        slam.TrackMonocularWithPL(image, entry.timestamp);
        NativeLoopFrameDiagnostics diagnostics =
            slam.GetNativeLoopFrameDiagnostics();
        WriteCsvRow(csv, diagnostics);
        WritePointFeatureRow(point_csv, diagnostics);
        WriteSafe89AuditRow(safe89_audit_csv, diagnostics);
        WriteFrameTrackingMetricsRow(frame_tracking_csv, diagnostics);
        WriteKb8ProjectionRuntimeRow(kb8_projection_csv, diagnostics);
        WriteSearchByProjectionRuntimeRow(search_projection_csv,
                                          diagnostics);
        WritePoseOptimizationRuntimeRow(pose_optimization_csv,
                                        diagnostics);
        const int processed_frame_count = static_cast<int>(i) + 1;
        if(!map_export_dir.empty() &&
           export_frames.find(processed_frame_count) != export_frames.end())
        {
            std::string export_error;
            const std::string stage_dir = JoinPath(
                map_export_dir,
                std::string("stage_") +
                    std::to_string(processed_frame_count));
            if(!slam.ExportKb8MapSnapshot(stage_dir, &export_error))
            {
                std::cerr << "[KB8 map export] failed for "
                          << stage_dir << ": " << export_error << std::endl;
            }
        }
        if(diagnostics.monocular_initialization_entered)
        {
            WriteInitializationAttemptRow(init_csv, diagnostics);
            WritePoseCandidateRows(pose_candidate_csv, diagnostics);
            WriteReferenceLifecycleRow(reference_lifecycle_csv,
                                       diagnostics);
        }
        if(i < 10)
        {
            WriteImageStage(image_csv,
                            static_cast<int>(i),
                            entry.timestamp,
                            "Frame_constructor_input",
                            diagnostics.frame_constructor_input_type,
                            diagnostics.image_channels,
                            diagnostics.image_rows,
                            diagnostics.image_cols,
                            diagnostics.image_min,
                            diagnostics.image_max,
                            diagnostics.image_mean);
            WriteImageStage(image_csv,
                            static_cast<int>(i),
                            entry.timestamp,
                            "ORBextractor_input",
                            diagnostics.orb_input_type,
                            diagnostics.image_channels,
                            diagnostics.image_rows,
                            diagnostics.image_cols,
                            diagnostics.image_min,
                            diagnostics.image_max,
                            diagnostics.image_mean);
        }
    }

    csv.close();
    image_csv.close();
    point_csv.close();
    init_csv.close();
    pose_candidate_csv.close();
    reference_lifecycle_csv.close();
    frame_tracking_csv.close();
    kb8_projection_csv.close();
    search_projection_csv.close();
    pose_optimization_csv.close();
    slam.Shutdown();
    slam.SaveKeyFrameTrajectoryTUM(
        JoinPath(output_dir, "KeyFrameTrajectory.txt"));
    slam.SaveTrajectoryTUM(
        JoinPath(output_dir, "MonoTrajectory.txt"));
    if(!map_export_dir.empty())
    {
        std::string export_error;
        if(!slam.ExportKb8MapSnapshot(map_export_dir, &export_error))
        {
            std::cerr << "[KB8 map export] final export failed: "
                      << export_error << std::endl;
            return 2;
        }
        std::string stage_error;
        const std::string stage_final = JoinPath(map_export_dir,
                                                 "stage_final");
        if(!slam.ExportKb8MapSnapshot(stage_final, &stage_error))
        {
            std::cerr << "[KB8 map export] stage_final export failed: "
                      << stage_error << std::endl;
            return 2;
        }
    }
    if(!colmap_export_dir.empty())
    {
        std::string colmap_error;
        if(!slam.ExportKb8ColmapSnapshot(colmap_export_dir, &colmap_error))
        {
            std::cerr << "[KB8 COLMAP export] failed: "
                      << colmap_error << std::endl;
            return 3;
        }
    }
    return 0;
}

} // namespace StructureSLAM
