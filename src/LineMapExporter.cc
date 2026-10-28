#include "LineMapExporter.h"

#include <fstream>

namespace StructureSLAM
{
namespace
{

void Color(const std::string& q, int& r, int& g, int& b)
{
    if(q == "high")
    {
        r = 0;
        g = 255;
        b = 0;
    }
    else if(q == "medium")
    {
        r = 255;
        g = 220;
        b = 0;
    }
    else
    {
        r = 255;
        g = 0;
        b = 0;
    }
}

} // namespace

bool LineMapExporter::WriteLineMapCsv(
    const std::string& path,
    const std::vector<FisheyeLineTriangulator::Line3D>& lines)
{
    std::ofstream f(path.c_str());
    if(!f.is_open())
        return false;
    f << "line3d_id,track_id,view_id,quality_class,p0_x,p0_y,p0_z,dir_x,dir_y,dir_z,"
      << "plucker_m_x,plucker_m_y,plucker_m_z,endpoint1_x,endpoint1_y,endpoint1_z,"
      << "endpoint2_x,endpoint2_y,endpoint2_z,length_m,finite_segment_valid,"
      << "observation_count,inlier_observation_count,mean_reprojection_error_px,"
      << "median_reprojection_error_px,p90_reprojection_error_px,max_reprojection_error_px,"
      << "mean_angular_error_deg,plane_fit_residual,condition_number,baseline_m,"
      << "plane_angle_deg,virtual_vs_fisheye_plane_angle_mean_deg,"
      << "virtual_vs_fisheye_plane_angle_p90_deg\n";
    for(const auto& l : lines)
    {
        f << l.line3d_id << "," << l.track_id << "," << l.view_id << "," << l.quality_class << ","
          << l.p0.x() << "," << l.p0.y() << "," << l.p0.z() << ","
          << l.dir.x() << "," << l.dir.y() << "," << l.dir.z() << ","
          << l.moment.x() << "," << l.moment.y() << "," << l.moment.z() << ","
          << l.endpoint1.x() << "," << l.endpoint1.y() << "," << l.endpoint1.z() << ","
          << l.endpoint2.x() << "," << l.endpoint2.y() << "," << l.endpoint2.z() << ","
          << l.length_m << "," << (l.finite_segment_valid ? 1 : 0) << ","
          << l.observation_count << "," << l.inlier_observation_count << ","
          << l.mean_reprojection_error_px << "," << l.median_reprojection_error_px << ","
          << l.p90_reprojection_error_px << "," << l.max_reprojection_error_px << ","
          << l.mean_angular_error_deg << "," << l.plane_fit_residual << ","
          << l.condition_number << "," << l.baseline_m << "," << l.plane_angle_deg << ","
          << l.virtual_plane_angle_mean_deg << "," << l.virtual_plane_angle_p90_deg << "\n";
    }
    return f.good();
}

bool LineMapExporter::WriteObj(
    const std::string& path,
    const std::vector<FisheyeLineTriangulator::Line3D>& lines)
{
    std::ofstream f(path.c_str());
    if(!f.is_open())
        return false;
    int idx = 1;
    for(const auto& l : lines)
    {
        if(!l.finite_segment_valid)
            continue;
        f << "o line_" << l.line3d_id << "_track_" << l.track_id << "_" << l.quality_class << "\n";
        f << "v " << l.endpoint1.x() << " " << l.endpoint1.y() << " " << l.endpoint1.z() << "\n";
        f << "v " << l.endpoint2.x() << " " << l.endpoint2.y() << " " << l.endpoint2.z() << "\n";
        f << "l " << idx << " " << idx + 1 << "\n";
        idx += 2;
    }
    return f.good();
}

bool LineMapExporter::WritePlySamples(
    const std::string& path,
    const std::vector<FisheyeLineTriangulator::Line3D>& lines,
    int samples_per_line)
{
    int count = 0;
    for(const auto& l : lines)
        if(l.finite_segment_valid)
            count += samples_per_line;
    std::ofstream f(path.c_str());
    if(!f.is_open())
        return false;
    f << "ply\nformat ascii 1.0\n";
    f << "comment line3d_id track_id quality are encoded by sample order and color\n";
    f << "element vertex " << count << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "end_header\n";
    for(const auto& l : lines)
    {
        if(!l.finite_segment_valid)
            continue;
        int r, g, b;
        Color(l.quality_class, r, g, b);
        for(int i = 0; i < samples_per_line; ++i)
        {
            const double a = samples_per_line == 1 ? 0.0 : static_cast<double>(i) / (samples_per_line - 1);
            const Eigen::Vector3d p = (1.0 - a) * l.endpoint1 + a * l.endpoint2;
            f << p.x() << " " << p.y() << " " << p.z() << " " << r << " " << g << " " << b << "\n";
        }
    }
    return f.good();
}

} // namespace StructureSLAM
