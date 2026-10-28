#include "Kb8MapExporter.h"

#include "Converter.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapLine.h"
#include "MapPoint.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace StructureSLAM
{
namespace
{

struct PointRow
{
    unsigned long id = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    int observations = 0;
    float found_ratio = 0.0f;
    bool is_bad = false;
};

struct LineRow
{
    unsigned long id = 0;
    double x0 = 0.0;
    double y0 = 0.0;
    double z0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    double z1 = 0.0;
    int observations = 0;
    bool is_bad = false;
    double length = 0.0;
};

struct ObservationAudit
{
    size_t rows = 0;
    size_t finite = 0;
    size_t valid_forward = 0;
    size_t in_image = 0;
    size_t duplicate_pixels = 0;
    size_t conflicting_pixels = 0;
    double reprojection_sum = 0.0;
    std::vector<double> reprojection_errors;
};

bool IsFinite3(const double x, const double y, const double z)
{
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
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

std::string JoinPath(const std::string& left, const std::string& right)
{
    if(left.empty() || right.empty() || right[0] == '/')
        return right;
    if(left[left.size() - 1] == '/')
        return left + right;
    return left + "/" + right;
}

void PointColor(const int observations, int* r, int* g, int* b)
{
    if(observations <= 1)
    {
        *r = 220; *g = 70; *b = 70;
    }
    else if(observations <= 5)
    {
        *r = 70; *g = 170; *b = 230;
    }
    else
    {
        *r = 80; *g = 210; *b = 120;
    }
}

std::string VecToString(const double x, const double y, const double z)
{
    std::ostringstream out;
    out << std::setprecision(9) << x << " " << y << " " << z;
    return out.str();
}

std::string TimestampFilename(const double timestamp)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << timestamp * 1e9 << ".png";
    return out.str();
}

double Percentile(std::vector<double> values, const double q)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if(lo == hi)
        return values[lo];
    const double t = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

bool GetRawPointPixel(const KeyFrame* pKF,
                      const size_t idx,
                      double* u,
                      double* v,
                      std::string* source)
{
    if(!pKF || !u || !v || !source)
        return false;
    if(pKF->mbUseKb8PointObservations &&
       idx < pKF->mvPointRawPixels.size())
    {
        const cv::Point2f& raw = pKF->mvPointRawPixels[idx];
        if(std::isfinite(raw.x) && std::isfinite(raw.y))
        {
            *u = raw.x;
            *v = raw.y;
            *source = "mvPointRawPixels";
            return true;
        }
    }
    if(idx < pKF->mvKeys.size())
    {
        const cv::KeyPoint& kp = pKF->mvKeys[idx];
        if(std::isfinite(kp.pt.x) && std::isfinite(kp.pt.y))
        {
            *u = kp.pt.x;
            *v = kp.pt.y;
            *source = "mvKeys_fallback";
            return true;
        }
    }
    if(idx < pKF->mvKeysUn.size())
    {
        const cv::KeyPoint& kp = pKF->mvKeysUn[idx];
        if(std::isfinite(kp.pt.x) && std::isfinite(kp.pt.y))
        {
            *u = kp.pt.x;
            *v = kp.pt.y;
            *source = "mvKeysUn_fallback";
            return true;
        }
    }
    return false;
}

bool WritePointObservations(const std::string& path,
                            const std::vector<MapPoint*>& map_points,
                            ObservationAudit* audit)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;

    out
        << "frame_timestamp,image_filename,keyframe_id,mappoint_id,"
        << "observation_index,keypoint_index,u_raw,v_raw,"
        << "u_projected,v_projected,reprojection_error,"
        << "Pw_x,Pw_y,Pw_z,Pc_x,Pc_y,Pc_z,ray_depth,"
        << "observation_count,octave,response,"
        << "pose_source,pose_convention,camera_model,"
        << "fx,fy,cx,cy,k1,k2,k3,k4,"
        << "pixel_source,valid_forward,in_image,is_bad_mappoint,is_bad_keyframe\n";
    out << std::fixed << std::setprecision(12);

    std::map<std::string, std::vector<unsigned long> > pixel_to_mappoints;
    if(audit)
        *audit = ObservationAudit();

    for(MapPoint* pMP : map_points)
    {
        if(!pMP || pMP->isBad())
            continue;
        const cv::Mat Pw = pMP->GetWorldPos();
        if(Pw.rows < 3 || Pw.cols < 1)
            continue;
        const double pwx = Pw.at<float>(0);
        const double pwy = Pw.at<float>(1);
        const double pwz = Pw.at<float>(2);
        if(!IsFinite3(pwx, pwy, pwz))
            continue;

        const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
        const int observation_count = pMP->Observations();
        for(std::map<KeyFrame*, size_t>::const_iterator it = observations.begin();
            it != observations.end(); ++it)
        {
            KeyFrame* pKF = it->first;
            const size_t idx = it->second;
            if(!pKF || pKF->isBad())
                continue;

            double u = std::numeric_limits<double>::quiet_NaN();
            double v = std::numeric_limits<double>::quiet_NaN();
            std::string pixel_source;
            if(!GetRawPointPixel(pKF, idx, &u, &v, &pixel_source))
                continue;

            const cv::Mat Rcw = pKF->GetRotation();
            const cv::Mat tcw = pKF->GetTranslation();
            const double pcx =
                Rcw.at<float>(0, 0) * pwx + Rcw.at<float>(0, 1) * pwy +
                Rcw.at<float>(0, 2) * pwz + tcw.at<float>(0);
            const double pcy =
                Rcw.at<float>(1, 0) * pwx + Rcw.at<float>(1, 1) * pwy +
                Rcw.at<float>(1, 2) * pwz + tcw.at<float>(1);
            const double pcz =
                Rcw.at<float>(2, 0) * pwx + Rcw.at<float>(2, 1) * pwy +
                Rcw.at<float>(2, 2) * pwz + tcw.at<float>(2);
            const double ray_depth = std::sqrt(pcx * pcx + pcy * pcy + pcz * pcz);
            const bool finite = IsFinite3(pcx, pcy, pcz) && std::isfinite(ray_depth);
            const bool valid_forward = finite && pcz > 0.0 && ray_depth > 0.0;

            const KannalaBrandt8Camera::Parameters& params =
                pKF->mKb8PointCameraParameters;
            const KannalaBrandt8Camera camera(params);
            Eigen::Vector2d uv_projected;
            const bool projected =
                camera.project(Eigen::Vector3d(pcx, pcy, pcz), uv_projected);
            const bool in_image = projected && camera.isInImage(uv_projected);
            const double reproj =
                projected
                    ? std::sqrt((uv_projected.x() - u) * (uv_projected.x() - u) +
                                (uv_projected.y() - v) * (uv_projected.y() - v))
                    : std::numeric_limits<double>::quiet_NaN();

            int octave = -1;
            double response = std::numeric_limits<double>::quiet_NaN();
            if(idx < pKF->mvKeys.size())
            {
                octave = pKF->mvKeys[idx].octave;
                response = pKF->mvKeys[idx].response;
            }

            out << pKF->mTimeStamp << ","
                << TimestampFilename(pKF->mTimeStamp) << ","
                << pKF->mnId << ","
                << pMP->mnId << ","
                << idx << ","
                << idx << ","
                << u << "," << v << ","
                << (projected ? uv_projected.x() : std::numeric_limits<double>::quiet_NaN()) << ","
                << (projected ? uv_projected.y() : std::numeric_limits<double>::quiet_NaN()) << ","
                << reproj << ","
                << pwx << "," << pwy << "," << pwz << ","
                << pcx << "," << pcy << "," << pcz << ","
                << ray_depth << ","
                << observation_count << ","
                << octave << ","
                << response << ","
                << "KeyFrame::GetRotation/GetTranslation,"
                << "Tcw_world_to_camera,"
                << "KannalaBrandt8,"
                << params.fx << "," << params.fy << ","
                << params.cx << "," << params.cy << ","
                << params.k1 << "," << params.k2 << ","
                << params.k3 << "," << params.k4 << ","
                << pixel_source << ","
                << (valid_forward ? "true" : "false") << ","
                << (in_image ? "true" : "false") << ","
                << (pMP->isBad() ? "true" : "false") << ","
                << (pKF->isBad() ? "true" : "false") << "\n";

            if(audit)
            {
                ++audit->rows;
                if(finite)
                    ++audit->finite;
                if(valid_forward)
                    ++audit->valid_forward;
                if(in_image)
                    ++audit->in_image;
                if(std::isfinite(reproj))
                {
                    audit->reprojection_sum += reproj;
                    audit->reprojection_errors.push_back(reproj);
                }
            }
            std::ostringstream key;
            key << pKF->mnId << ":"
                << static_cast<int>(std::floor(u + 0.5)) << ":"
                << static_cast<int>(std::floor(v + 0.5));
            pixel_to_mappoints[key.str()].push_back(pMP->mnId);
        }
    }

    if(audit)
    {
        for(std::map<std::string, std::vector<unsigned long> >::const_iterator it =
                pixel_to_mappoints.begin();
            it != pixel_to_mappoints.end(); ++it)
        {
            if(it->second.size() > 1)
            {
                ++audit->duplicate_pixels;
                bool conflicting = false;
                for(size_t i = 1; i < it->second.size(); ++i)
                {
                    if(it->second[i] != it->second[0])
                    {
                        conflicting = true;
                        break;
                    }
                }
                if(conflicting)
                    ++audit->conflicting_pixels;
            }
        }
    }
    return true;
}

bool WritePointObservationAudit(const std::string& path,
                                const ObservationAudit& audit)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    const double denom = static_cast<double>(std::max<size_t>(audit.rows, 1));
    const double finite_ratio = static_cast<double>(audit.finite) / denom;
    const double forward_ratio = static_cast<double>(audit.valid_forward) / denom;
    const double in_image_ratio = static_cast<double>(audit.in_image) / denom;
    const double reproj_mean =
        audit.reprojection_errors.empty()
            ? 0.0
            : audit.reprojection_sum /
                  static_cast<double>(audit.reprojection_errors.size());
    out << "observation_count,finite_ratio,valid_forward_ratio,in_image_ratio,"
        << "reprojection_error_mean,reprojection_error_median,"
        << "reprojection_error_p95,reprojection_error_max,"
        << "duplicate_pixel_count,conflicting_pixel_count\n";
    out << std::fixed << std::setprecision(12)
        << audit.rows << ","
        << finite_ratio << ","
        << forward_ratio << ","
        << in_image_ratio << ","
        << reproj_mean << ","
        << Percentile(audit.reprojection_errors, 0.50) << ","
        << Percentile(audit.reprojection_errors, 0.95) << ","
        << Percentile(audit.reprojection_errors, 1.00) << ","
        << audit.duplicate_pixels << ","
        << audit.conflicting_pixels << "\n";
    return true;
}

bool WritePointObservationSummary(const std::string& path,
                                  const ObservationAudit& audit,
                                  const size_t keyframe_count,
                                  const size_t mappoint_count)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    const double denom = static_cast<double>(std::max<size_t>(audit.rows, 1));
    out << std::fixed << std::setprecision(12);
    out << "status=FORMAL_PLSLAM_POINT_OBSERVATIONS_EXPORTED\n";
    out << "keyframe_count=" << keyframe_count << "\n";
    out << "mappoint_count=" << mappoint_count << "\n";
    out << "observation_count=" << audit.rows << "\n";
    out << "finite_ratio=" << static_cast<double>(audit.finite) / denom << "\n";
    out << "valid_forward_ratio="
        << static_cast<double>(audit.valid_forward) / denom << "\n";
    out << "in_image_ratio=" << static_cast<double>(audit.in_image) / denom << "\n";
    out << "reprojection_error_median="
        << Percentile(audit.reprojection_errors, 0.50) << "\n";
    out << "reprojection_error_p95="
        << Percentile(audit.reprojection_errors, 0.95) << "\n";
    out << "duplicate_pixel_count=" << audit.duplicate_pixels << "\n";
    out << "conflicting_pixel_count=" << audit.conflicting_pixels << "\n";
    out << "depth_semantics=ray_depth_is_norm_Pc\n";
    out << "pose_convention=Tcw_world_to_camera\n";
    out << "pixel_contract=raw_KB8_mvvPointRawPixels_when_available\n";
    out << "slam_math_modified=NO\n";
    return true;
}

bool WriteMapPointsPly(const std::string& path,
                       const std::vector<PointRow>& points)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << points.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";
    out << std::fixed << std::setprecision(9);
    for(const PointRow& p : points)
    {
        int r = 255, g = 255, b = 255;
        PointColor(p.observations, &r, &g, &b);
        out << p.x << " " << p.y << " " << p.z << " "
            << r << " " << g << " " << b << "\n";
    }
    return true;
}

bool WriteMapLinesPly(const std::string& path,
                      const std::vector<LineRow>& lines)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << lines.size() * 2 << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "element edge " << lines.size() << "\n";
    out << "property int vertex1\n";
    out << "property int vertex2\n";
    out << "property uint line_id\n";
    out << "property int observations\n";
    out << "end_header\n";
    out << std::fixed << std::setprecision(9);
    for(const LineRow& line : lines)
    {
        out << line.x0 << " " << line.y0 << " " << line.z0 << "\n";
        out << line.x1 << " " << line.y1 << " " << line.z1 << "\n";
    }
    for(size_t i = 0; i < lines.size(); ++i)
    {
        out << i * 2 << " " << i * 2 + 1 << " "
            << lines[i].id << " " << lines[i].observations << "\n";
    }
    return true;
}

bool WritePointStatistics(const std::string& path,
                          const std::vector<PointRow>& points)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "id,x,y,z,observations,found_ratio,is_bad\n";
    out << std::fixed << std::setprecision(9);
    for(const PointRow& p : points)
    {
        out << p.id << "," << p.x << "," << p.y << "," << p.z << ","
            << p.observations << "," << p.found_ratio << ","
            << (p.is_bad ? "true" : "false") << "\n";
    }
    return true;
}

bool WriteTrajectory(const std::string& path, std::vector<KeyFrame*> keyframes)
{
    std::sort(keyframes.begin(), keyframes.end(), KeyFrame::lId);
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << std::fixed;
    for(KeyFrame* pKF : keyframes)
    {
        if(!pKF || pKF->isBad())
            continue;
        const cv::Mat Rwc = pKF->GetRotation().t();
        const cv::Mat twc = pKF->GetCameraCenter();
        const std::vector<float> q = Converter::toQuaternion(Rwc);
        out << std::setprecision(6) << pKF->mTimeStamp << " "
            << std::setprecision(9)
            << twc.at<float>(0) << " "
            << twc.at<float>(1) << " "
            << twc.at<float>(2) << " "
            << q[0] << " " << q[1] << " "
            << q[2] << " " << q[3] << "\n";
    }
    return true;
}

bool WriteQualityReport(const std::string& path,
                        const std::vector<PointRow>& points,
                        const std::vector<LineRow>& lines,
                        const size_t keyframe_count)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    int obs1 = 0, obs2_5 = 0, obs_gt5 = 0;
    double obs_sum = 0.0;
    int finite_points = 0;
    int abnormal_points = 0;
    for(const PointRow& p : points)
    {
        if(!IsFinite3(p.x, p.y, p.z))
            continue;
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        min_z = std::min(min_z, p.z);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
        max_z = std::max(max_z, p.z);
        if(p.observations <= 1)
            ++obs1;
        else if(p.observations <= 5)
            ++obs2_5;
        else
            ++obs_gt5;
        obs_sum += p.observations;
        ++finite_points;
        const double norm =
            std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if(norm > 1000.0)
            ++abnormal_points;
    }
    if(finite_points == 0)
        min_x = min_y = min_z = max_x = max_y = max_z = 0.0;

    double min_len = std::numeric_limits<double>::infinity();
    double max_len = 0.0;
    double sum_len = 0.0;
    int finite_lines = 0;
    for(const LineRow& line : lines)
    {
        if(!std::isfinite(line.length))
            continue;
        min_len = std::min(min_len, line.length);
        max_len = std::max(max_len, line.length);
        sum_len += line.length;
        ++finite_lines;
    }
    if(finite_lines == 0)
        min_len = 0.0;

    out << std::fixed << std::setprecision(9);
    out << "KeyFrame count: " << keyframe_count << "\n";
    out << "MapPoint count: " << points.size() << "\n";
    out << "MapLine count: " << lines.size() << "\n";
    out << "Spatial range min xyz: " << VecToString(min_x, min_y, min_z)
        << "\n";
    out << "Spatial range max xyz: " << VecToString(max_x, max_y, max_z)
        << "\n";
    out << "Observation count 1: " << obs1 << "\n";
    out << "Observation count 2-5: " << obs2_5 << "\n";
    out << "Observation count >5: " << obs_gt5 << "\n";
    out << "Average MapPoint observations: "
        << (points.empty() ? 0.0 : obs_sum / points.size()) << "\n";
    out << "Abnormal point norm >1000: " << abnormal_points << "\n";
    out << "MapLine length min: " << min_len << "\n";
    out << "MapLine length max: " << max_len << "\n";
    out << "MapLine length mean: "
        << (finite_lines == 0 ? 0.0 : sum_len / finite_lines) << "\n";
    return true;
}

bool WriteVisualizationNotes(const std::string& path)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out
        << "CloudCompare:\n"
        << "1. Open mappoints.ply and maplines.ply.\n"
        << "2. Color classes: red means single observation, blue means 2-5 observations, green means more than 5 observations.\n"
        << "3. Inspect point cloud continuity, isolated floating points, wall planes, and line alignment.\n\n"
        << "MeshLab:\n"
        << "1. Import mappoints.ply first, then maplines.ply.\n"
        << "2. Use layer visibility to compare line segments against the point cloud.\n\n"
        << "Open3D:\n"
        << "1. Use open3d.io.read_point_cloud for mappoints.ply.\n"
        << "2. Use open3d.io.read_line_set for maplines.ply.\n"
        << "3. Load trajectory.txt separately as camera centers to check trajectory continuity.\n";
    return true;
}

} // namespace

bool Kb8MapExporter::Export(const Map& map,
                            const std::string& output_dir,
                            std::string* error)
{
    if(!MakeDirectoryRecursive(output_dir))
    {
        if(error)
            *error = "failed to create output directory";
        return false;
    }

    Map& mutable_map = const_cast<Map&>(map);
    const std::vector<MapPoint*> map_points = mutable_map.GetAllMapPoints();
    const std::vector<MapLine*> map_lines = mutable_map.GetAllMapLines();
    const std::vector<KeyFrame*> keyframes = mutable_map.GetAllKeyFrames();

    std::vector<PointRow> points;
    points.reserve(map_points.size());
    for(MapPoint* pMP : map_points)
    {
        if(!pMP)
            continue;
        PointRow row;
        row.id = pMP->mnId;
        row.is_bad = pMP->isBad();
        const cv::Mat pos = pMP->GetWorldPos();
        if(pos.rows >= 3 && pos.cols >= 1)
        {
            row.x = pos.at<float>(0);
            row.y = pos.at<float>(1);
            row.z = pos.at<float>(2);
        }
        row.observations = pMP->Observations();
        row.found_ratio = pMP->GetFoundRatio();
        if(!row.is_bad && IsFinite3(row.x, row.y, row.z))
            points.push_back(row);
    }
    std::sort(points.begin(), points.end(),
              [](const PointRow& a, const PointRow& b) {
                  return a.id < b.id;
              });

    std::vector<LineRow> lines;
    lines.reserve(map_lines.size());
    for(MapLine* pML : map_lines)
    {
        if(!pML)
            continue;
        LineRow row;
        row.id = pML->mnId;
        row.is_bad = pML->isBad();
        const Vector6d pos = pML->GetWorldPos();
        row.x0 = pos(0);
        row.y0 = pos(1);
        row.z0 = pos(2);
        row.x1 = pos(3);
        row.y1 = pos(4);
        row.z1 = pos(5);
        row.observations = pML->Observations();
        const double dx = row.x1 - row.x0;
        const double dy = row.y1 - row.y0;
        const double dz = row.z1 - row.z0;
        row.length = std::sqrt(dx * dx + dy * dy + dz * dz);
        if(!row.is_bad &&
           IsFinite3(row.x0, row.y0, row.z0) &&
           IsFinite3(row.x1, row.y1, row.z1))
            lines.push_back(row);
    }
    std::sort(lines.begin(), lines.end(),
              [](const LineRow& a, const LineRow& b) {
                  return a.id < b.id;
              });

    ObservationAudit observation_audit;

    const bool ok =
        WriteMapPointsPly(JoinPath(output_dir, "mappoints.ply"), points) &&
        WriteMapLinesPly(JoinPath(output_dir, "maplines.ply"), lines) &&
        WritePointStatistics(JoinPath(output_dir, "mappoint_statistics.csv"),
                             points) &&
        WritePointObservations(JoinPath(output_dir, "point_observations.csv"),
                               map_points, &observation_audit) &&
        WritePointObservationAudit(JoinPath(output_dir, "point_observation_projection_audit.csv"),
                                   observation_audit) &&
        WritePointObservationSummary(JoinPath(output_dir, "point_observation_summary.txt"),
                                     observation_audit, keyframes.size(),
                                     points.size()) &&
        WriteTrajectory(JoinPath(output_dir, "trajectory.txt"), keyframes) &&
        WriteQualityReport(JoinPath(output_dir, "map_quality_report.txt"),
                           points, lines, keyframes.size()) &&
        WriteVisualizationNotes(JoinPath(output_dir, "visualization_notes.txt"));

    if(!ok && error)
        *error = "failed to write one or more export files";
    return ok;
}

} // namespace StructureSLAM
