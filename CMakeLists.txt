cmake_minimum_required(VERSION 2.8)
project(StructureSLAM)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Keep Eigen ABI compatible between Structure-SLAM and the bundled legacy g2o.
# The legacy g2o Hessian storage is not guaranteed to satisfy Eigen's AVX
# 32-byte alignment requirement. Both targets must use the same 16-byte ABI.
add_definitions(
    -DEIGEN_MAX_ALIGN_BYTES=16
    -DEIGEN_MAX_STATIC_ALIGN_BYTES=16
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++14 support.
# PCL 1.10 and the installed Pangolin headers require at least C++14.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# This P9-C frontend is built against the installed OpenCV 4.x package.
# OpenCV's config-version file does not treat 4.x as compatible with a
# find_package request for major version 3, so request version 4 explicitly.
find_package(OpenCV 4 REQUIRED CONFIG)

find_package(Eigen3 3.1.0 REQUIRED CONFIG)
find_package(Pangolin REQUIRED)
find_package( PCL REQUIRED )
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

add_definitions( ${PCL_DEFINITIONS} )
# PCL exports 16-byte Eigen alignment definitions on this platform, while the
# bundled prebuilt g2o library was compiled with Eigen's native alignment.
# g2o vertex types cross this shared-library boundary, so their object layout
# must remain identical in both translation units.
remove_definitions(-DEIGEN_MAX_ALIGN_BYTES=16)
remove_definitions(-DEIGEN_MAX_STATIC_ALIGN_BYTES=16)
link_directories( ${PCL_LIBRARY_DIRS} )


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/CorrectLoopHelpers.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Initializer.cc
src/Sim3Solver.cc
src/Kb8PnPSolver.cc
src/Kb8MapExporter.cc
src/Kb8ColmapExporter.cc
src/Viewer.cc
src/Lineextractor.cpp
src/CornerFrameDatebase.cpp
src/ExtractLineSegment.cpp 
src/MapLine.cpp 
src/LSDmatcher.cpp
src/Kb8CameraSettings.cc
src/Kb8InitializationAdapter.cc
src/camera/KannalaBrandt8Camera.cpp
src/FisheyeLineFrontend.cc
src/FisheyeLineShadowMatcher.cc
src/FisheyeLineMatcher.cc
src/FisheyeLineTriangulator.cc
src/LineMapExporter.cc
src/NativeLineIdentityBridge.cc
src/NativeFisheyeMapLineProjector.cc
src/PhysicalLineObservationWrapper.cc
src/PhysicalLineObservationBuilder.cc
src/CandidateFingerprint.cc
src/ReplaySnapshotWriter.cc
src/OfflineReplaySnapshot.cc
src/VirtualViewLinePixelResidual.cc
src/line_geometry/EndpointPluckerConverter.cpp
src/optimization/FisheyeBearingPlaneLineEdge.cpp
src/optimization/FisheyeJointPoseOptimizer.cpp
src/optimization/FisheyeJointSafetyConfig.cpp
src/optimization/FisheyePointPoseEdge.cpp
src/optimization/JointSafetyFilter.cpp
src/optimization/Kb8PointPoseEdge.cpp
src/optimization/NativeFisheyeLinePoseEdge.cpp
src/optimization/PhysicalLinePlanePoseEdge.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PCL_LIBRARIES}
        )
# Build examples

option(BUILD_MONO_NATIVE_ONLY "Build only the CAM_B native fisheye runner target" OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples)

if(NOT BUILD_MONO_NATIVE_ONLY)
# ABI probe compiled in the same directory context as the application, after
# importing PCL and removing its conflicting Eigen alignment definitions.
add_executable(abi_layout_diagnostic Examples/abi_layout_diagnostic.cc)
target_link_libraries(abi_layout_diagnostic
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so)

add_executable(frame_mode_contract_test Examples/frame_mode_contract_test.cc)
target_link_libraries(frame_mode_contract_test StructureSLAM ${OpenCV_LIBS})

add_executable(Structure-SLAM
Examples/Structure-SLAM.cc)
target_link_libraries(Structure-SLAM ${PROJECT_NAME})

add_executable(offline_replay_snapshot Examples/offline_replay_snapshot.cc)
target_link_libraries(offline_replay_snapshot ${PROJECT_NAME} ${OpenCV_LIBS})
add_executable(physical_line_wrapper_unit_test Examples/physical_line_wrapper_unit_test.cc)
target_link_libraries(physical_line_wrapper_unit_test ${PROJECT_NAME} ${OpenCV_LIBS})
add_executable(physical_line_builder_unit_test Examples/physical_line_builder_unit_test.cc)
target_link_libraries(physical_line_builder_unit_test ${PROJECT_NAME} ${OpenCV_LIBS})
endif()

# Formal TUM-VI native fisheye point-line SLAM monocular entry.
add_executable(mono_tum_vi_native_fisheye_plslam
    Examples/Monocular/mono_tum_vi_native_fisheye_plslam.cc
    Examples/Monocular/mono_tum_vi_native_fisheye_plslam_runner.cc
)

target_link_libraries(mono_tum_vi_native_fisheye_plslam
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_native_fisheye_plslam PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

add_executable(physical_three_view_core_smoke Examples/physical_three_view_core_smoke.cc)
target_link_libraries(physical_three_view_core_smoke StructureSLAM ${OpenCV_LIBS})
set_target_properties(physical_three_view_core_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples)

if(BUILD_MONO_NATIVE_ONLY)
    return()
endif()

add_executable(export_kb8_map_points
    Examples/export_kb8_map_points.cc
    Examples/Monocular/mono_tum_vi_native_fisheye_plslam_runner.cc
)

target_link_libraries(export_kb8_map_points
    StructureSLAM
    ${OpenCV_LIBS}
)

target_include_directories(export_kb8_map_points
    PRIVATE
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

set_target_properties(export_kb8_map_points PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_map_export_smoke
    Examples/kb8_map_export_smoke.cc
)

set_target_properties(kb8_map_export_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(export_kb8_to_colmap
    Examples/export_kb8_to_colmap.cc
    Examples/Monocular/mono_tum_vi_native_fisheye_plslam_runner.cc
)

target_link_libraries(export_kb8_to_colmap
    StructureSLAM
    ${OpenCV_LIBS}
)

target_include_directories(export_kb8_to_colmap
    PRIVATE
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

set_target_properties(export_kb8_to_colmap PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_colmap_export_smoke
    Examples/kb8_colmap_export_smoke.cc
)

set_target_properties(kb8_colmap_export_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(mono_tum_vi_native_fisheye_plslam_smoke
    Examples/Monocular/mono_tum_vi_native_fisheye_plslam_smoke.cc
    Examples/Monocular/mono_tum_vi_native_fisheye_plslam_runner.cc
)

target_link_libraries(mono_tum_vi_native_fisheye_plslam_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

target_compile_definitions(mono_tum_vi_native_fisheye_plslam_smoke
    PRIVATE
    STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

set_target_properties(mono_tum_vi_native_fisheye_plslam_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

add_executable(mono_tum_vi_initialization_diagnostic_smoke
    Examples/Monocular/mono_tum_vi_initialization_diagnostic_smoke.cc
)

target_include_directories(mono_tum_vi_initialization_diagnostic_smoke
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(mono_tum_vi_initialization_diagnostic_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

add_executable(kb8_point_camera_parameter_unification_smoke
    Examples/kb8_point_camera_parameter_unification_smoke.cc
)

target_link_libraries(kb8_point_camera_parameter_unification_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(kb8_point_camera_parameter_unification_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_frame_point_bearing_storage_smoke
    Examples/kb8_frame_point_bearing_storage_smoke.cc
)

target_link_libraries(kb8_frame_point_bearing_storage_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(kb8_frame_point_bearing_storage_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_two_view_initialization_smoke
    Examples/kb8_two_view_initialization_smoke.cc
)

target_link_libraries(kb8_two_view_initialization_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(kb8_two_view_initialization_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_initial_map_creation_smoke
    Examples/kb8_initial_map_creation_smoke.cc
)

target_link_libraries(kb8_initial_map_creation_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(kb8_initial_map_creation_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_rectified_real_pair_diagnostic
    Examples/kb8_rectified_real_pair_diagnostic.cc
)

target_link_libraries(kb8_rectified_real_pair_diagnostic
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(kb8_rectified_real_pair_diagnostic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_frame_extractor_narrowing_diag
    Examples/kb8_frame_extractor_narrowing_diag.cc
)

target_link_libraries(kb8_frame_extractor_narrowing_diag
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(kb8_frame_extractor_narrowing_diag PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

foreach(_alignment_smoke
    legacy_vs_native_tracking_call_chain_smoke
    mature_plslam_lifecycle_reuse_smoke
    kb8_rectified_initialization_adapter_smoke
    formal_mapline_lifecycle_reuse_smoke
    optimizer_single_lifecycle_smoke
    kb8_initialization_search_center_rollback_smoke
    kb8_direct_vs_formal_pair_parity_smoke
    legacy_initialization_propagation_parity_smoke
)
    add_executable(${_alignment_smoke}
        Examples/${_alignment_smoke}.cc
    )
    target_compile_definitions(${_alignment_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
    set_target_properties(${_alignment_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_optimize_sim3_smoke
    kb8_optimize_sim3_forward_projection_smoke
    kb8_optimize_sim3_inverse_projection_smoke
    kb8_optimize_sim3_coordinate_contract_smoke
    kb8_optimize_sim3_bidirectional_smoke
    kb8_optimize_sim3_recovery_smoke
    kb8_optimize_sim3_outlier_rejection_smoke
    kb8_optimize_sim3_runtime_no_pinhole_smoke
    kb8_optimize_sim3_single_optimizer_smoke
    legacy_optimize_sim3_parity_smoke
)
    add_executable(${_kb8_optimize_sim3_smoke}
        Examples/kb8_optimize_sim3_smoke.cc
    )
    target_compile_definitions(${_kb8_optimize_sim3_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_optimize_sim3_smoke}"
    )
    target_link_libraries(${_kb8_optimize_sim3_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_optimize_sim3_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_loop_searchbyprojection_scw_smoke
    kb8_loop_searchbyprojection_scw_projection_smoke
    kb8_loop_searchbyprojection_scw_coordinate_contract_smoke
    kb8_loop_searchbyprojection_scw_transform_direction_smoke
    kb8_loop_searchbyprojection_scw_target_camera_smoke
    kb8_loop_searchbyprojection_scw_match_smoke
    kb8_loop_searchbyprojection_scw_reject_wrong_feature_smoke
    kb8_loop_searchbyprojection_scw_runtime_no_pinhole_smoke
    legacy_loop_searchbyprojection_scw_parity_smoke
)
    add_executable(${_kb8_loop_searchbyprojection_scw_smoke}
        Examples/kb8_loop_searchbyprojection_scw_smoke.cc
    )
    target_compile_definitions(${_kb8_loop_searchbyprojection_scw_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_loop_searchbyprojection_scw_smoke}"
    )
    target_link_libraries(${_kb8_loop_searchbyprojection_scw_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_loop_searchbyprojection_scw_smoke}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_loop_fuse_scw_smoke
    kb8_loop_fuse_scw_projection_smoke
    kb8_loop_fuse_scw_coordinate_contract_smoke
    kb8_loop_fuse_scw_transform_direction_smoke
    kb8_loop_fuse_scw_target_camera_smoke
    kb8_loop_fuse_scw_match_smoke
    kb8_loop_fuse_scw_replacement_smoke
    kb8_loop_fuse_scw_no_duplicate_observation_smoke
    kb8_loop_searchandfuse_multikeyframe_smoke
    kb8_loop_fuse_scw_runtime_no_pinhole_smoke
    legacy_loop_fuse_scw_parity_smoke
)
    add_executable(${_kb8_loop_fuse_scw_smoke}
        Examples/kb8_loop_fuse_scw_smoke.cc
    )
    target_compile_definitions(${_kb8_loop_fuse_scw_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_loop_fuse_scw_smoke}"
    )
    target_link_libraries(${_kb8_loop_fuse_scw_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_loop_fuse_scw_smoke}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_correctloop_eg_smoke
    kb8_correctloop_sim3_to_se3_smoke
    kb8_correctloop_pose_writeback_smoke
    kb8_correctloop_mappoint_correction_smoke
    kb8_correctloop_no_duplicate_correction_smoke
    kb8_correctloop_bookkeeping_smoke
    kb8_correctloop_graph_update_smoke
    kb8_correctloop_loopconnections_difference_smoke
    kb8_essential_graph_single_optimizer_smoke
    kb8_essential_graph_no_camera_math_smoke
    kb8_essential_graph_objective_reduction_smoke
    kb8_essential_graph_formal_writeback_smoke
    kb8_correctloop_runtime_guard_smoke
    kb8_loop_triggered_gba_guard_smoke
    legacy_correctloop_lifecycle_parity_smoke
)
    add_executable(${_kb8_correctloop_eg_smoke}
        Examples/kb8_correctloop_essential_graph_smoke.cc
    )
    target_compile_definitions(${_kb8_correctloop_eg_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_correctloop_eg_smoke}"
    )
    target_link_libraries(${_kb8_correctloop_eg_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_correctloop_eg_smoke}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_loop_triggered_gba_smoke
    kb8_loop_gba_single_optimizer_smoke
    kb8_loop_gba_runtime_edge_dispatch_smoke
    kb8_loop_gba_staging_smoke
    kb8_loop_gba_commit_smoke
    kb8_loop_gba_spanning_tree_propagation_smoke
    kb8_loop_gba_direct_mappoint_writeback_smoke
    kb8_loop_gba_indirect_mappoint_writeback_smoke
    kb8_loop_gba_no_partial_commit_smoke
    kb8_loop_gba_stale_result_guard_smoke
    kb8_loop_gba_stop_cleanup_smoke
    kb8_loop_gba_mutex_guard_smoke
    kb8_loop_gba_flag_lifecycle_smoke
    legacy_loop_gba_runtime_parity_smoke
)
    add_executable(${_kb8_loop_triggered_gba_smoke}
        Examples/kb8_loop_triggered_gba_smoke.cc
    )
    target_compile_definitions(${_kb8_loop_triggered_gba_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_loop_triggered_gba_smoke}"
    )
    target_link_libraries(${_kb8_loop_triggered_gba_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_loop_triggered_gba_smoke}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_loopclosing_runtime_smoke
    kb8_loopclosing_queue_smoke
    kb8_loopclosing_no_loop_runtime_smoke
    kb8_loopclosing_valid_loop_runtime_smoke
    kb8_loopclosing_localmapping_coordination_smoke
    kb8_loopclosing_map_mutex_smoke
    kb8_loopclosing_flag_lifecycle_smoke
    kb8_loopclosing_reset_smoke
    kb8_loopclosing_finish_smoke
    kb8_loopclosing_thread_cleanup_smoke
    kb8_loopclosing_full_kb8_dispatch_smoke
    kb8_loopclosing_formal_map_integrity_smoke
    kb8_loopclosing_error_reduction_smoke
    kb8_loopclosing_relocalization_guard_smoke
    legacy_loopclosing_runtime_parity_smoke
)
    add_executable(${_kb8_loopclosing_runtime_smoke}
        Examples/kb8_loopclosing_controlled_runtime_smoke.cc
    )
    target_compile_definitions(${_kb8_loopclosing_runtime_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_loopclosing_runtime_smoke}"
    )
    target_link_libraries(${_kb8_loopclosing_runtime_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_loopclosing_runtime_smoke}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

add_executable(kb8_pnp_solver_audit_smoke
    Examples/kb8_pnp_solver_audit_smoke.cc
)
target_compile_definitions(kb8_pnp_solver_audit_smoke
    PRIVATE
    STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
set_target_properties(kb8_pnp_solver_audit_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_pnp_adapter_prototype_smoke
    Examples/kb8_pnp_adapter_prototype_smoke.cc
)
target_compile_definitions(kb8_pnp_adapter_prototype_smoke
    PRIVATE
    STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(kb8_pnp_adapter_prototype_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)
set_target_properties(kb8_pnp_adapter_prototype_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_loopclosing_formal_candidate_injection_smoke
    Examples/kb8_loopclosing_formal_candidate_injection_smoke.cc
)
target_link_libraries(kb8_loopclosing_formal_candidate_injection_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)
set_target_properties(kb8_loopclosing_formal_candidate_injection_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_correctloop_formal_runtime_smoke
    Examples/kb8_correctloop_formal_runtime_smoke.cc
)
target_link_libraries(kb8_correctloop_formal_runtime_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)
set_target_properties(kb8_correctloop_formal_runtime_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_searchandfuse_formal_runtime_smoke
    Examples/kb8_searchandfuse_formal_runtime_smoke.cc
)
target_link_libraries(kb8_searchandfuse_formal_runtime_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)
set_target_properties(kb8_searchandfuse_formal_runtime_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_essential_graph_formal_runtime_smoke
    Examples/kb8_essential_graph_formal_runtime_smoke.cc
)
target_link_libraries(kb8_essential_graph_formal_runtime_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)
set_target_properties(kb8_essential_graph_formal_runtime_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_loop_triggered_gba_formal_runtime_smoke
    Examples/kb8_loop_triggered_gba_formal_runtime_smoke.cc
)
target_link_libraries(kb8_loop_triggered_gba_formal_runtime_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)
set_target_properties(kb8_loop_triggered_gba_formal_runtime_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

add_executable(kb8_natural_loopclosing_runtime_smoke
    Examples/kb8_natural_loopclosing_runtime_smoke.cc
    Examples/Monocular/mono_tum_vi_native_fisheye_plslam_runner.cc
)
target_link_libraries(kb8_natural_loopclosing_runtime_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)
set_target_properties(kb8_natural_loopclosing_runtime_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

foreach(_kb8_searchbysim3_smoke
    kb8_searchbysim3_projection_smoke
    kb8_searchbysim3_coordinate_contract_smoke
    kb8_searchbysim3_bidirectional_smoke
    legacy_searchbysim3_parity_smoke
    kb8_searchbysim3_runtime_no_pinhole_smoke
)
    add_executable(${_kb8_searchbysim3_smoke}
        Examples/kb8_searchbysim3_smoke.cc
    )
    target_compile_definitions(${_kb8_searchbysim3_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_searchbysim3_smoke}"
    )
    target_link_libraries(${_kb8_searchbysim3_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_searchbysim3_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_sim3solver_smoke
    kb8_sim3solver_projection_smoke
    kb8_sim3solver_inlier_check_smoke
    kb8_sim3solver_coordinate_contract_smoke
    legacy_sim3solver_parity_smoke
)
    add_executable(${_kb8_sim3solver_smoke}
        Examples/kb8_sim3solver_projection_smoke.cc
    )
    target_compile_definitions(${_kb8_sim3solver_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_sim3solver_smoke}"
    )
    target_link_libraries(${_kb8_sim3solver_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_sim3solver_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_initializer_contract_smoke
    kb8_initializer_reference_current_domain_parity_smoke
    kb8_initializer_virtual_k_contract_smoke
    kb8_initializer_match_index_contract_smoke
    kb8_initializer_output_index_contract_smoke
    kb8_legacy_success_vs_postvalidation_smoke
)
    add_executable(${_kb8_initializer_contract_smoke}
        Examples/kb8_initializer_interface_contract_smoke.cc
    )
    target_compile_definitions(${_kb8_initializer_contract_smoke}
        PRIVATE
        TEST_LABEL="${_kb8_initializer_contract_smoke}"
    )
    target_link_libraries(${_kb8_initializer_contract_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_initializer_contract_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_full_framework_smoke
    single_tracking_state_machine_smoke
    single_formal_map_lifecycle_smoke
    single_formal_optimizer_lifecycle_smoke
    kb8_point_adapter_only_smoke
    native_line_legacy_mapline_lifecycle_smoke
    diagnostic_formal_state_isolation_smoke
    mode_switch_contract_smoke
)
    add_executable(${_full_framework_smoke}
        Examples/full_framework_compatibility_smoke.cc
    )
    target_compile_definitions(${_full_framework_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_full_framework_smoke}"
    )
    target_link_libraries(${_full_framework_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_full_framework_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_point_tracking_smoke
    kb8_camera_orbslam3_project_equivalence_smoke
    kb8_camera_orbslam3_unproject_equivalence_smoke
    kb8_camera_orbslam3_project_jac_equivalence_smoke
    kb8_mappoint_projection_smoke
    kb8_mappoint_frustum_lifecycle_smoke
    kb8_search_by_projection_adapter_smoke
    kb8_point_pose_edge_smoke
    kb8_point_pose_jacobian_finite_difference_smoke
    kb8_point_pose_optimizer_lifecycle_smoke
    kb8_post_initialization_capability_guard_smoke
    legacy_pinhole_point_tracking_parity_smoke
)
    add_executable(${_kb8_point_tracking_smoke}
        Examples/kb8_point_tracking_port_smoke.cc
    )
    target_compile_definitions(${_kb8_point_tracking_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_point_tracking_smoke}"
    )
    target_link_libraries(${_kb8_point_tracking_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_point_tracking_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

add_executable(kb8_joint_point_line_formal_smoke
    Examples/kb8_joint_point_line_formal_smoke.cc
)

target_link_libraries(kb8_joint_point_line_formal_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(kb8_joint_point_line_formal_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

foreach(_kb8_localmapping_guard_smoke
    kb8_localmapping_capability_guard_smoke
    kb8_localmapping_thread_safe_stop_smoke
    kb8_new_mappoint_call_chain_smoke
    kb8_triangulation_adapter_contract_smoke
    kb8_localmapping_no_pinhole_runtime_smoke
    legacy_localmapping_parity_smoke
)
    add_executable(${_kb8_localmapping_guard_smoke}
        Examples/kb8_localmapping_guard_smoke.cc
    )
    target_compile_definitions(${_kb8_localmapping_guard_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_localmapping_guard_smoke}"
    )
    set_target_properties(${_kb8_localmapping_guard_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_new_mappoint_smoke
    kb8_new_mappoint_unproject_smoke
    kb8_two_keyframe_triangulation_smoke
    kb8_new_mappoint_raw_reprojection_smoke
    kb8_new_mappoint_scale_consistency_smoke
    kb8_new_mappoint_index_contract_smoke
    kb8_new_mappoint_formal_lifecycle_smoke
    kb8_create_new_mappoints_runtime_no_pinhole_smoke
    kb8_new_mappoint_runtime_gate_contract_smoke
    legacy_create_new_mappoints_parity_smoke
)
    add_executable(${_kb8_new_mappoint_smoke}
        Examples/kb8_new_mappoint_triangulation_smoke.cc
    )
    target_compile_definitions(${_kb8_new_mappoint_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_new_mappoint_smoke}"
    )
    target_link_libraries(${_kb8_new_mappoint_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_new_mappoint_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_fusion_smoke
    kb8_fuse_projection_smoke
    kb8_fuse_raw_image_grid_smoke
    kb8_fuse_scale_level_smoke
    kb8_fuse_descriptor_lifecycle_smoke
    kb8_fuse_new_observation_smoke
    kb8_fuse_replacement_smoke
    kb8_search_in_neighbors_bidirectional_smoke
    kb8_fusion_runtime_no_pinhole_smoke
    kb8_fusion_capability_guard_smoke
    legacy_fusion_parity_smoke
)
    add_executable(${_kb8_fusion_smoke}
        Examples/kb8_mappoint_fusion_smoke.cc
    )
    target_compile_definitions(${_kb8_fusion_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_fusion_smoke}"
    )
    target_link_libraries(${_kb8_fusion_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_fusion_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

foreach(_kb8_local_ba_smoke
    kb8_local_ba_point_edge_projection_smoke
    kb8_local_ba_pose_jacobian_smoke
    kb8_local_ba_point_jacobian_smoke
    kb8_local_ba_joint_pose_point_recovery_smoke
    kb8_local_ba_outlier_rejection_smoke
    kb8_local_ba_formal_writeback_smoke
    kb8_local_ba_single_optimizer_lifecycle_smoke
    kb8_local_ba_with_existing_line_edge_smoke
    kb8_local_ba_runtime_no_pinhole_point_edge_smoke
    kb8_local_ba_capability_guard_smoke
    legacy_local_ba_parity_smoke
)
    add_executable(${_kb8_local_ba_smoke}
        Examples/kb8_local_ba_point_edge_smoke.cc
    )
    target_compile_definitions(${_kb8_local_ba_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_local_ba_smoke}"
    )
    target_link_libraries(${_kb8_local_ba_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_local_ba_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()

# P9-C1 standalone monocular fisheye line frontend smoke
add_executable(fisheye_line_frontend_smoke
    Examples/fisheye_line_frontend_smoke.cc
)

target_link_libraries(fisheye_line_frontend_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(fisheye_line_frontend_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# P9-C1 batch monocular fisheye line frontend smoke
add_executable(fisheye_line_frontend_batch_smoke
    Examples/fisheye_line_frontend_batch_smoke.cc
)

target_link_libraries(fisheye_line_frontend_batch_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(fisheye_line_frontend_batch_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# P9-C2 fisheye line Frame storage smoke
add_executable(fisheye_line_frame_storage_smoke
    Examples/fisheye_line_frame_storage_smoke.cc
)

target_link_libraries(fisheye_line_frame_storage_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(fisheye_line_frame_storage_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# Native temporal-LBD track <-> formal MapLine identity bridge smoke
add_executable(native_line_identity_bridge_smoke
    Examples/native_line_identity_bridge_smoke.cc
)

target_link_libraries(native_line_identity_bridge_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(native_line_identity_bridge_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# Native fisheye LocalMapping formal MapLine creation smoke
add_executable(native_fisheye_localmapping_mapline_creation_smoke
    Examples/native_fisheye_localmapping_mapline_creation_smoke.cc
)

target_link_libraries(native_fisheye_localmapping_mapline_creation_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(native_fisheye_localmapping_mapline_creation_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# Native fisheye formal MapLine projection and Frame association smoke
add_executable(native_fisheye_mapline_projection_association_smoke
    Examples/native_fisheye_mapline_projection_association_smoke.cc
)

target_link_libraries(native_fisheye_mapline_projection_association_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(native_fisheye_mapline_projection_association_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# Native fisheye formal MapLine point-line tracking pose smoke
add_executable(native_fisheye_formal_tracking_pose_smoke
    Examples/native_fisheye_formal_tracking_pose_smoke.cc
)

target_link_libraries(native_fisheye_formal_tracking_pose_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(native_fisheye_formal_tracking_pose_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# Native fisheye joint point-line tracking pose graph smoke
add_executable(native_fisheye_joint_tracking_pose_smoke
    Examples/native_fisheye_joint_tracking_pose_smoke.cc
)

target_link_libraries(native_fisheye_joint_tracking_pose_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(native_fisheye_joint_tracking_pose_smoke
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)


# P9-C3a standalone adjacent-frame fisheye line matcher smoke
add_executable(fisheye_line_shadow_matcher_smoke
    Examples/fisheye_line_shadow_matcher_smoke.cc
)

target_link_libraries(fisheye_line_shadow_matcher_smoke
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(fisheye_line_shadow_matcher_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# P9-C3a.2 fisheye line association visualization
add_executable(fisheye_line_shadow_matcher_visualize
    Examples/fisheye_line_shadow_matcher_visualize.cc
)

target_link_libraries(fisheye_line_shadow_matcher_visualize
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(fisheye_line_shadow_matcher_visualize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples
)

# VPL V3 strict three-view Stage 3A exporter
add_executable(mono_tum_vi_line_frontend_v3
    Examples/Monocular/mono_tum_vi_line_frontend_v3.cc
)

target_link_libraries(mono_tum_vi_line_frontend_v3
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_line_frontend_v3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# VPL V3 strict three-view Stage 3B line matching experiment
add_executable(mono_tum_vi_line_matching_v3
    Examples/Monocular/mono_tum_vi_line_matching_v3.cc
)

target_link_libraries(mono_tum_vi_line_matching_v3
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_line_matching_v3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# V3-P9B1-D native virtual-view LBD descriptor replay
add_executable(mono_tum_vi_virtual_view_line_lbd_replay
    Examples/Monocular/mono_tum_vi_virtual_view_line_lbd_replay.cc
)

target_link_libraries(mono_tum_vi_virtual_view_line_lbd_replay
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_virtual_view_line_lbd_replay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# V3-P9B2 native temporal-LBD shadow pose replay
add_executable(mono_tum_vi_native_temporal_lbd_shadow_pose
    Examples/Monocular/mono_tum_vi_native_temporal_lbd_shadow_pose.cc
)

target_link_libraries(mono_tum_vi_native_temporal_lbd_shadow_pose
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_native_temporal_lbd_shadow_pose PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# VPL V3 virtual pose geometry validation
add_executable(fisheye_virtual_pose_geometry_test
    Examples/Monocular/fisheye_virtual_pose_geometry_test.cc
)

target_link_libraries(fisheye_virtual_pose_geometry_test
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(fisheye_virtual_pose_geometry_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# VPL V3 strict three-view Stage 3C line triangulation experiment
add_executable(mono_tum_vi_line_triangulation_v3
    Examples/Monocular/mono_tum_vi_line_triangulation_v3.cc
)

target_link_libraries(mono_tum_vi_line_triangulation_v3
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_line_triangulation_v3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# VPL V3 auxiliary virtual-view to physical-camera unification smoke
add_executable(fisheye_auxiliary_view_unification_test
    Examples/Monocular/fisheye_auxiliary_view_unification_test.cc
)

target_link_libraries(fisheye_auxiliary_view_unification_test
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(fisheye_auxiliary_view_unification_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# VPL V3 virtual-view endpoint-to-line pixel residual validation
add_executable(virtual_view_line_pixel_residual_test
    Examples/Monocular/virtual_view_line_pixel_residual_test.cc
)

target_link_libraries(virtual_view_line_pixel_residual_test
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(virtual_view_line_pixel_residual_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# VPL V3 real-data virtual-view endpoint-to-line pixel residual audit
add_executable(mono_tum_vi_virtual_view_line_pixel_residual_audit
    Examples/Monocular/mono_tum_vi_virtual_view_line_pixel_residual_audit.cc
)

target_link_libraries(mono_tum_vi_virtual_view_line_pixel_residual_audit
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_virtual_view_line_pixel_residual_audit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

# VPL V3 shadow MapLine offline single-pose optimization
add_executable(mono_tum_vi_virtual_view_line_shadow_pose_optimization
    Examples/Monocular/mono_tum_vi_virtual_view_line_shadow_pose_optimization.cc
)

target_link_libraries(mono_tum_vi_virtual_view_line_shadow_pose_optimization
    StructureSLAM
    ${OpenCV_LIBS}
)

set_target_properties(mono_tum_vi_virtual_view_line_shadow_pose_optimization PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_SOURCE_DIR}/Examples/Monocular
)

foreach(_kb8_global_ba_smoke
    kb8_global_ba_point_edge_smoke
    kb8_global_ba_pose_jacobian_smoke
    kb8_global_ba_point_jacobian_smoke
    kb8_global_ba_writeback_smoke
    kb8_global_ba_single_optimizer_smoke
    kb8_global_ba_no_pinhole_runtime_smoke
    legacy_global_ba_parity_smoke
)
    add_executable(${_kb8_global_ba_smoke}
        Examples/kb8_global_ba_point_edge_smoke.cc
    )
    target_compile_definitions(${_kb8_global_ba_smoke}
        PRIVATE
        STRUCTURE_SLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        TEST_LABEL="${_kb8_global_ba_smoke}"
    )
    target_link_libraries(${_kb8_global_ba_smoke}
        StructureSLAM
        ${OpenCV_LIBS}
    )
    set_target_properties(${_kb8_global_ba_smoke} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/Examples
    )
endforeach()
