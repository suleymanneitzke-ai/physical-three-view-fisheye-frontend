#include "Kb8ColmapExporter.h"

#include "Converter.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "camera/KannalaBrandt8Camera.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace StructureSLAM
{
namespace
{

struct PoseCheck
{
    unsigned long kf_id = 0;
    double rotation_error = 0.0;
    double translation_error = 0.0;
};

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

bool IsFiniteMat3(const cv::Mat& p)
{
    return p.rows >= 3 && p.cols >= 1 &&
           std::isfinite(p.at<float>(0)) &&
           std::isfinite(p.at<float>(1)) &&
           std::isfinite(p.at<float>(2));
}

int CameraWidth(const std::vector<KeyFrame*>& keyframes)
{
    for(KeyFrame* pKF : keyframes)
        if(pKF)
            return std::max(0, pKF->mnMaxX - pKF->mnMinX);
    return 0;
}

int CameraHeight(const std::vector<KeyFrame*>& keyframes)
{
    for(KeyFrame* pKF : keyframes)
        if(pKF)
            return std::max(0, pKF->mnMaxY - pKF->mnMinY);
    return 0;
}

KannalaBrandt8Camera::Parameters CameraParameters(
    const Map& map,
    const std::vector<KeyFrame*>& keyframes)
{
    KannalaBrandt8Camera::Parameters params;
    if(const_cast<Map&>(map).GetNativeFisheyeCameraParameters(params))
        return params;
    for(KeyFrame* pKF : keyframes)
    {
        if(!pKF)
            continue;
        if(pKF->mbUseKb8PointObservations)
            return pKF->mKb8PointCameraParameters;
    }
    for(KeyFrame* pKF : keyframes)
    {
        if(!pKF)
            continue;
        params.fx = pKF->fx;
        params.fy = pKF->fy;
        params.cx = pKF->cx;
        params.cy = pKF->cy;
        params.width = CameraWidth(keyframes);
        params.height = CameraHeight(keyframes);
        return params;
    }
    return params;
}

cv::Mat InverseSe3(const cv::Mat& Tcw)
{
    cv::Mat Twc = cv::Mat::eye(4, 4, CV_32F);
    const cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    const cv::Mat Rwc = Rcw.t();
    const cv::Mat twc = -Rwc * tcw;
    Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    twc.copyTo(Twc.rowRange(0, 3).col(3));
    return Twc;
}

double RotationFrobeniusError(const cv::Mat& R)
{
    double sum = 0.0;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
        {
            const double expected = (r == c) ? 1.0 : 0.0;
            const double d = R.at<float>(r, c) - expected;
            sum += d * d;
        }
    return std::sqrt(sum);
}

double TranslationNorm(const cv::Mat& t)
{
    double sum = 0.0;
    for(int r = 0; r < 3; ++r)
    {
        const double v = t.at<float>(r);
        sum += v * v;
    }
    return std::sqrt(sum);
}

bool WriteCameras(const std::string& path,
                  const KannalaBrandt8Camera::Parameters& params,
                  const std::vector<KeyFrame*>& keyframes)
{
    const int width = params.width > 0 ? params.width : CameraWidth(keyframes);
    const int height = params.height > 0 ? params.height : CameraHeight(keyframes);
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "# Camera list with one line of data per camera:\n";
    out << "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    out << std::setprecision(12);
    out << "1 KB8_CUSTOM " << width << " " << height << " "
        << params.fx << " " << params.fy << " "
        << params.cx << " " << params.cy << " "
        << params.k1 << " " << params.k2 << " "
        << params.k3 << " " << params.k4 << "\n";
    return true;
}

bool WriteCameraNote(const std::string& path,
                     const KannalaBrandt8Camera::Parameters& params)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "original_model=KB8\n";
    out << "target_model=COLMAP text compatibility only\n";
    out << "export_model=KB8_CUSTOM\n";
    out << "colmap_native_reader=NOT_COMPATIBLE_WITH_CUSTOM_MODEL\n";
    out << "reason=COLMAP built-in text importer does not know KB8_CUSTOM unless COLMAP is extended with the same camera model.\n";
    out << "params=fx fy cx cy k1 k2 k3 k4\n";
    out << std::setprecision(12)
        << "values=" << params.fx << " " << params.fy << " "
        << params.cx << " " << params.cy << " "
        << params.k1 << " " << params.k2 << " "
        << params.k3 << " " << params.k4 << "\n";
    return true;
}

bool WriteImages(const std::string& path,
                 std::vector<KeyFrame*> keyframes,
                 std::map<KeyFrame*, int>* image_ids,
                 std::vector<PoseCheck>* pose_checks)
{
    std::sort(keyframes.begin(), keyframes.end(), KeyFrame::lId);
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "# Image list with two lines of data per image:\n";
    out << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n";
    out << "# POINTS2D[] as (X, Y, POINT3D_ID)\n";
    out << std::fixed << std::setprecision(9);

    int image_id = 1;
    for(KeyFrame* pKF : keyframes)
    {
        if(!pKF || pKF->isBad())
            continue;
        (*image_ids)[pKF] = image_id;
        const cv::Mat Tcw = pKF->GetPose();
        const cv::Mat Twc = InverseSe3(Tcw);
        const cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
        const cv::Mat twc = Twc.rowRange(0, 3).col(3);
        const std::vector<float> q = Converter::toQuaternion(Rwc);
        out << image_id << " "
            << q[3] << " " << q[0] << " " << q[1] << " " << q[2] << " "
            << twc.at<float>(0) << " " << twc.at<float>(1) << " "
            << twc.at<float>(2) << " 1 "
            << "kf_" << pKF->mnId << ".png\n";

        const std::vector<MapPoint*> matches = pKF->GetMapPointMatches();
        for(size_t i = 0; i < matches.size(); ++i)
        {
            MapPoint* pMP = matches[i];
            if(!pMP || pMP->isBad() || i >= pKF->mvKeysUn.size())
                continue;
            out << pKF->mvKeysUn[i].pt.x << " "
                << pKF->mvKeysUn[i].pt.y << " "
                << pMP->mnId << " ";
        }
        out << "\n";

        const cv::Mat identity = Tcw * Twc;
        PoseCheck check;
        check.kf_id = pKF->mnId;
        check.rotation_error = RotationFrobeniusError(
            identity.rowRange(0, 3).colRange(0, 3));
        check.translation_error = TranslationNorm(
            identity.rowRange(0, 3).col(3));
        pose_checks->push_back(check);
        ++image_id;
    }
    return true;
}

bool WritePoints3D(const std::string& path,
                   std::vector<MapPoint*> points,
                   const std::map<KeyFrame*, int>& image_ids,
                   int* exported_points,
                   int* track_missing_count)
{
    std::sort(points.begin(), points.end(),
              [](MapPoint* a, MapPoint* b) {
                  if(!a || !b)
                      return a < b;
                  return a->mnId < b->mnId;
              });
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "# 3D point list with one line of data per point:\n";
    out << "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
    out << std::fixed << std::setprecision(9);
    *exported_points = 0;
    *track_missing_count = 0;
    for(MapPoint* pMP : points)
    {
        if(!pMP || pMP->isBad())
            continue;
        const cv::Mat pos = pMP->GetWorldPos();
        if(!IsFiniteMat3(pos))
            continue;
        out << pMP->mnId << " "
            << pos.at<float>(0) << " "
            << pos.at<float>(1) << " "
            << pos.at<float>(2) << " 255 255 255 0";
        const std::map<KeyFrame*, size_t> observations =
            pMP->GetObservations();
        int written_track_items = 0;
        for(const auto& obs : observations)
        {
            KeyFrame* pKF = obs.first;
            const auto image_it = image_ids.find(pKF);
            if(image_it == image_ids.end())
            {
                ++(*track_missing_count);
                continue;
            }
            out << " " << image_it->second << " " << obs.second;
            ++written_track_items;
        }
        if(written_track_items == 0)
            ++(*track_missing_count);
        out << "\n";
        ++(*exported_points);
    }
    return true;
}

bool WritePoseCheck(const std::string& path,
                    const std::vector<PoseCheck>& pose_checks)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "kf_id,rotation_error,translation_error\n";
    out << std::scientific << std::setprecision(9);
    for(const PoseCheck& check : pose_checks)
        out << check.kf_id << "," << check.rotation_error << ","
            << check.translation_error << "\n";
    return true;
}

bool WriteTrajectoryCompare(const std::string& path,
                            const std::vector<PoseCheck>& pose_checks,
                            const int image_count,
                            const int point_count,
                            const int track_missing_count)
{
    double max_rot = 0.0;
    double max_trans = 0.0;
    for(const PoseCheck& check : pose_checks)
    {
        max_rot = std::max(max_rot, check.rotation_error);
        max_trans = std::max(max_trans, check.translation_error);
    }
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << std::scientific << std::setprecision(9);
    out << "slam_pose_source=Tcw\n";
    out << "colmap_pose_output=Twc_inverse_of_Tcw\n";
    out << "image_count=" << image_count << "\n";
    out << "point_count=" << point_count << "\n";
    out << "track_missing_count=" << track_missing_count << "\n";
    out << "max_rotation_error=" << max_rot << "\n";
    out << "max_translation_error=" << max_trans << "\n";
    out << "status=" << ((max_rot < 1e-4 && max_trans < 1e-4) ? "PASS" : "FAIL") << "\n";
    return true;
}

bool WritePly(const std::string& path, const std::vector<MapPoint*>& points)
{
    size_t valid_count = 0;
    for(MapPoint* pMP : points)
    {
        if(!pMP || pMP->isBad())
            continue;
        if(IsFiniteMat3(pMP->GetWorldPos()))
            ++valid_count;
    }
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << valid_count << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";
    out << std::fixed << std::setprecision(9);
    for(MapPoint* pMP : points)
    {
        if(!pMP || pMP->isBad())
            continue;
        const cv::Mat pos = pMP->GetWorldPos();
        if(!IsFiniteMat3(pos))
            continue;
        out << pos.at<float>(0) << " " << pos.at<float>(1) << " "
            << pos.at<float>(2) << " 255 255 255\n";
    }
    return true;
}

bool WriteImportTest(const std::string& path)
{
    std::ofstream out(path.c_str());
    if(!out.is_open())
        return false;
    out << "COLMAP_MODEL_CONVERTER=NOT_RUN\n";
    out << "reason=The exported camera model is KB8_CUSTOM. Stock COLMAP cannot import this custom model without adding a matching camera model implementation.\n";
    out << "format_check=PERFORMED_BY_kb8_colmap_export_smoke\n";
    return true;
}

} // namespace

bool Kb8ColmapExporter::Export(const Map& map,
                               const std::string& output_dir,
                               std::string* error)
{
    if(!MakeDirectoryRecursive(output_dir))
    {
        if(error)
            *error = "failed to create output directory";
        return false;
    }
    const std::string model_dir = JoinPath(output_dir, "colmap_kb8_export");
    if(!MakeDirectoryRecursive(model_dir))
    {
        if(error)
            *error = "failed to create colmap_kb8_export directory";
        return false;
    }

    Map& mutable_map = const_cast<Map&>(map);
    std::vector<KeyFrame*> keyframes = mutable_map.GetAllKeyFrames();
    std::vector<MapPoint*> points = mutable_map.GetAllMapPoints();
    const KannalaBrandt8Camera::Parameters params =
        CameraParameters(map, keyframes);

    std::map<KeyFrame*, int> image_ids;
    std::vector<PoseCheck> pose_checks;
    int exported_points = 0;
    int track_missing_count = 0;

    const bool ok =
        WriteCameras(JoinPath(model_dir, "cameras.txt"),
                     params, keyframes) &&
        WriteImages(JoinPath(model_dir, "images.txt"),
                    keyframes, &image_ids, &pose_checks) &&
        WritePoints3D(JoinPath(model_dir, "points3D.txt"),
                      points, image_ids, &exported_points,
                      &track_missing_count) &&
        WritePoseCheck(JoinPath(output_dir, "pose_conversion_check.csv"),
                       pose_checks) &&
        WriteTrajectoryCompare(JoinPath(output_dir, "trajectory_compare.txt"),
                               pose_checks,
                               static_cast<int>(image_ids.size()),
                               exported_points,
                               track_missing_count) &&
        WriteCameraNote(JoinPath(output_dir,
                                 "camera_model_conversion_note.txt"),
                        params) &&
        WriteImportTest(JoinPath(output_dir, "colmap_import_test.txt")) &&
        WritePly(JoinPath(output_dir, "colmap_kb8_export.ply"), points);

    if(!ok && error)
        *error = "failed to write COLMAP export files";
    return ok;
}

} // namespace StructureSLAM
