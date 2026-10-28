#ifndef STRUCTURE_SLAM_NATIVE_LOOP_FRAME_DIAGNOSTICS_H
#define STRUCTURE_SLAM_NATIVE_LOOP_FRAME_DIAGNOSTICS_H

#include <string>
#include <vector>
#include <array>

namespace StructureSLAM
{

struct InitializationPoseCandidateDiagnostics
{
    long unsigned int reference_frame_id =
        static_cast<long unsigned int>(-1);
    long unsigned int current_frame_id =
        static_cast<long unsigned int>(-1);
    std::string model;
    int candidate_id = -1;
    std::array<double,9> R21{{0,0,0,0,0,0,0,0,0}};
    std::array<double,3> t21{{0,0,0}};
    int valid_match_count = 0;
    int positive_lambda_count = 0;
    double positive_lambda_ratio = 0.0;
    int negative_lambda1_count = 0;
    int negative_lambda2_count = 0;
    double ray_miss_median = 0.0;
    double ray_miss_p95 = 0.0;
    double angular_error_median_deg = 0.0;
    double angular_error_p95_deg = 0.0;
    double triangulation_condition_median = 0.0;
    double triangulation_condition_p95 = 0.0;
    std::array<int,4> angular_bin_total{{0,0,0,0}};
    std::array<int,4> angular_bin_positive{{0,0,0,0}};
    int positive_depth_cam1 = 0;
    int positive_depth_cam2 = 0;
    int both_positive = 0;
    int reprojection_pass = 0;
    int triangulated = 0;
    double median_parallax_deg = 0.0;
    bool accepted_by_check_rt = false;
};

struct NativeLoopFrameDiagnostics
{
    long unsigned int frame_id = static_cast<long unsigned int>(-1);
    double timestamp = 0.0;
    int tracking_state = -1;
    int tracking_state_before = -1;
    int tracking_state_after = -1;
    long unsigned int reference_keyframe_id =
        static_cast<long unsigned int>(-1);
    long unsigned int map_points_total = 0;
    int local_map_points = 0;
    int mappoint_projection_input = 0;
    int valid_camera_points = 0;
    int valid_kb8_projections = 0;
    int inside_image = 0;
    int track_in_view = 0;
    int search_by_projection_attempts = 0;
    int accepted_2d3d_matches = 0;
    int pose_point_edges = 0;
    int pose_initial_inliers = 0;
    int pose_final_inliers = 0;
    int pose_outliers = 0;
    double reprojection_error_before = 0.0;
    double reprojection_error_after = 0.0;
    bool pose_optimization_success = false;
    bool guard_triggered = false;
    std::string guard_reason;
    std::string tracking_failure_reason;
    int initial_mappoints_created = 0;
    int initial_keyframes_created = 0;
    std::string last_tracking_function;
    int native_observations = 0;
    int active_tracks = 0;
    int pending_tracks = -1;
    long unsigned int formal_maplines = 0;
    int local_maplines = 0;
    int projected_maplines = 0;
    int matched_maplines = 0;
    int line_projection_candidates = 0;
    int line_rejected_bad = 0;
    int line_rejected_geometry = 0;
    int line_rejected_temporal = 0;
    int line_rejected_descriptor = 0;
    int line_rejected_duplicate = 0;
    long long line_gate_input_pairs = 0;
    long long line_fail_input_invalid = 0;
    long long line_fail_invalid_yaw_or_pose = 0;
    long long line_fail_invalid_world_segment = 0;
    long long line_fail_invalid_camera_segment = 0;
    long long line_fail_depth = 0;
    long long line_fail_range = 0;
    long long line_fail_distance_invariance = 0;
    long long line_fail_fisheye_projection = 0;
    long long line_fail_fisheye_fov = 0;
    long long line_fail_virtual_depth = 0;
    long long line_fail_virtual_fov = 0;
    long long line_fail_angle = 0;
    long long line_fail_overlap = 0;
    long long line_fail_temporal = 0;
    long long line_fail_descriptor = 0;
    long long line_fail_duplicate = 0;
    long long line_view0_input = 0;
    long long line_view1_input = 0;
    long long line_view2_input = 0;
    long long line_view0_projection_pass = 0;
    long long line_view1_projection_pass = 0;
    long long line_view2_projection_pass = 0;
    long long line_view0_angle_pass = 0;
    long long line_view1_angle_pass = 0;
    long long line_view2_angle_pass = 0;
    long long line_view0_overlap_pass = 0;
    long long line_view1_overlap_pass = 0;
    long long line_view2_overlap_pass = 0;
    long long line_view0_descriptor_pass = 0;
    long long line_view1_descriptor_pass = 0;
    long long line_view2_descriptor_pass = 0;
    long long line_view0_association = 0;
    long long line_view1_association = 0;
    long long line_view2_association = 0;
    double line_angle_fail_mean = 0.0;
    double line_angle_pass_mean = 0.0;
    long long line_angle_fail_count = 0;
    long long line_angle_pass_count = 0;
    long long line_angle_fail_within_5pct = 0;
    long long line_angle_fail_within_10pct = 0;
    long long line_angle_fail_within_20pct = 0;
    double line_overlap_fail_mean = 0.0;
    double line_overlap_pass_mean = 0.0;
    long long line_overlap_fail_count = 0;
    long long line_overlap_pass_count = 0;
    long long line_overlap_fail_within_5pct = 0;
    long long line_overlap_fail_within_10pct = 0;
    long long line_overlap_fail_within_20pct = 0;
    int point_edges = 0;
    int line_edges = 0;
    int physical_observations = 0;
    int physical_group_size_1 = 0;
    int physical_group_size_2 = 0;
    int physical_group_size_3 = 0;
    int physical_matches = 0;
    int physical_residual_created = 0;
    int physical_residual_active = 0;
    int physical_residual_accepted = 0;
    int physical_residual_rejected = 0;
    int duplicate_physical_residual_violations = 0;
    int point_inliers = 0;
    int line_inliers = 0;
    double point_cost_before = 0.0;
    double point_cost_after = 0.0;
    double line_cost_before = 0.0;
    double line_cost_after = 0.0;
    double pose_delta_translation = 0.0;
    double pose_delta_rotation = 0.0;
    bool joint_accepted = false;
    std::string pose_source = "POINT_ONLY";
    std::string fallback_reason;
    bool new_keyframe = false;
    int new_maplines = -1;
    bool native_enabled = false;
    bool point_line_tracking_enabled = false;
    int virtual_views = 0;
    bool kb8_parameters_available = false;
    bool formal_system_path = true;
    int runner_input_type = -1;
    int track_input_type = -1;
    int frame_constructor_input_type = -1;
    int orb_input_type = -1;
    int image_channels = 0;
    int image_rows = 0;
    int image_cols = 0;
    double image_min = 0.0;
    double image_max = 0.0;
    double image_mean = 0.0;
    int orb_keypoints_raw = 0;
    int undistorted_keypoints = 0;
    int descriptors_rows = 0;
    int safe89_orb_detected_before_mask = 0;
    int safe89_orb_valid_after_mask = 0;
    int safe89_orb_bin_0_30 = 0;
    int safe89_orb_bin_30_60 = 0;
    int safe89_orb_bin_60_80 = 0;
    int safe89_orb_bin_80_89 = 0;
    double safe89_orb_feature_density = 0.0;
    double safe89_orb_max_alpha_deg = -1.0;
    int safe89_orb_alpha_leaks = 0;
    double safe89_line_support_max_alpha_deg = -1.0;
    int safe89_line_support_alpha_leaks = 0;
    int safe89_matched_point_alpha_leaks = 0;
    int safe89_optimizer_point_alpha_leaks = 0;
    long unsigned int initial_frame_id =
        static_cast<long unsigned int>(-1);
    int initial_frame_keypoints = 0;
    int initial_matches = 0;
    int valid_match_count = 0;
    double median_displacement = -1.0;
    double median_parallax = -1.0;
    std::string search_center_mode;
    bool search_center_reset = false;
    double search_center_median_shift_from_initial = -1.0;
    double search_center_p95_shift_from_initial = -1.0;
    bool tracking_track_called = false;
    bool monocular_initialization_entered = false;
    bool initializer_created = false;
    bool initializer_reset = false;
    bool early_return = false;
    std::string initialization_failure_reason;
    std::string initializer_model;
    double initializer_h_score = 0.0;
    double initializer_f_score = 0.0;
    double initializer_h_ratio = 0.0;
    bool legacy_solver_success = false;
    bool kb8_postvalidation_success = false;
    bool initialization_final_success = false;
    int initialization_candidate_count = 0;
    int initialization_best_candidate_id = -1;
    int initialization_best_good_points = 0;
    int initialization_second_best_good_points = 0;
    int initialization_best_positive_depth_cam1 = 0;
    int initialization_best_positive_depth_cam2 = 0;
    int initialization_best_both_positive = 0;
    int initialization_best_triangulated = 0;
    int initialization_best_reprojection_pass = 0;
    double initialization_best_median_parallax_deg = 0.0;
    int initialization_required_min_good_points = 0;
    double initialization_required_min_parallax_deg = 0.0;
    double initialization_ambiguity_ratio_or_threshold = 0.0;
    std::string initialization_exact_failure_branch;
    int initializer_age = -1;
    bool reference_frame_retained_after_failure = false;
    bool reference_frame_replaced_after_failure = false;
    std::string reference_frame_replacement_condition;
    std::vector<InitializationPoseCandidateDiagnostics>
        initialization_pose_candidates;
};

} // namespace StructureSLAM

#endif
