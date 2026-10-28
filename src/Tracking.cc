/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "NativeFisheyeMapLineProjector.h"
#include "PhysicalLineObservationWrapper.h"
#include "PhysicalLineObservationBuilder.h"
#include "NativeLineIdentityBridge.h"
#include "Kb8InitializationAdapter.h"
#include "Kb8CameraSettings.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include "Kb8PnPSolver.h"
#include <iostream>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <thread>

using namespace std;
namespace StructureSLAM
{

namespace
{

int ReadEnvIntClamped(const char* name, const int default_value,
                      const int min_value)
{
    const char* value = std::getenv(name);
    if(!value)
        return default_value;
    return std::max(min_value, std::atoi(value));
}

}
namespace
{
std::atomic<unsigned long> gTrackingLocalSearchEventSeq(0);
std::mutex gTrackingLocalSearchEventMutex;
std::mutex gRelocalizationDiagnosticMutex;

void AppendTrackingLocalSearchThreadEvent(
    const std::string& event,
    const long unsigned int frame_id,
    const std::thread::id& thread_id,
    const std::string& worker_type,
    const void* current_frame_address,
    const void* map_address,
    const size_t local_mappoint_count,
    const size_t local_mapline_count,
    const size_t frame_mappoint_vector_size,
    const size_t frame_mapline_vector_size,
    const bool loopclosing_thread_running,
    const std::string& localmapping_state,
    const bool exception_caught,
    const std::string& exception_message)
{
    const char* path =
        std::getenv("STRUCTURE_SLAM_TRACKING_LOCAL_SEARCH_THREAD_EVENTS");
    if(!path || path[0] == '\0')
        return;

    std::lock_guard<std::mutex> lock(gTrackingLocalSearchEventMutex);
    std::ofstream out(path, std::ios::app);
    if(!out.is_open())
        return;

    out << ++gTrackingLocalSearchEventSeq << ','
        << frame_id << ','
        << thread_id << ','
        << worker_type << ','
        << event << ','
        << exception_caught << ','
        << current_frame_address << ','
        << map_address << ','
        << local_mappoint_count << ','
        << local_mapline_count << ','
        << frame_mappoint_vector_size << ','
        << frame_mapline_vector_size << ','
        << loopclosing_thread_running << ','
        << localmapping_state << ','
        << exception_message << '\n';
}

bool UseSerialLocalSearchDiagnostic()
{
    const char* value =
        std::getenv("STRUCTURE_SLAM_TRACKING_LOCAL_SEARCH_SERIAL");
    return value && value[0] != '\0' && std::string(value) != "0";
}

std::string RelocalizationOutputPath(const std::string& filename)
{
    const char* dir = std::getenv("STRUCTURE_SLAM_RELOCALIZATION_OUTPUT_DIR");
    if(!dir || dir[0] == '\0')
        return std::string();
    std::string path(dir);
    if(!path.empty() && path[path.size() - 1] != '/')
        path += '/';
    return path + filename;
}

void AppendRelocalizationCsv(const std::string& filename,
                             const std::string& header,
                             const std::string& row)
{
    const std::string path = RelocalizationOutputPath(filename);
    if(path.empty())
        return;
    std::lock_guard<std::mutex> lock(gRelocalizationDiagnosticMutex);
    std::ifstream existing(path.c_str());
    const bool needs_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
    existing.close();
    std::ofstream out(path.c_str(), std::ios::app);
    if(!out.is_open())
        return;
    if(needs_header)
        out << header << '\n';
    out << row << '\n';
}

bool ControlledLostFrameRequested(const long unsigned int frame_id)
{
    const char* value =
        std::getenv("STRUCTURE_SLAM_KB8_RELOCALIZATION_CONTROLLED_LOST_FRAME");
    if(!value || value[0] == '\0')
        return false;
    char* end = NULL;
    const unsigned long requested = std::strtoul(value, &end, 10);
    return end != value && requested == frame_id;
}

const char* TrackingStateName(const Tracking::eTrackingState state)
{
    switch(state)
    {
    case Tracking::SYSTEM_NOT_READY:
        return "SYSTEM_NOT_READY";
    case Tracking::NO_IMAGES_YET:
        return "NO_IMAGES_YET";
    case Tracking::NOT_INITIALIZED:
        return "NOT_INITIALIZED";
    case Tracking::OK:
        return "OK";
    case Tracking::LOST:
        return "LOST";
    default:
        return "UNKNOWN";
    }
}

bool ReadBoolSetting(const cv::FileStorage& settings,
                     const std::string& primary_key,
                     const std::string& legacy_key,
                     const bool default_value)
{
    const cv::FileNode primary_node = settings[primary_key];
    if(!primary_node.empty())
        return static_cast<int>(primary_node) != 0;
    const cv::FileNode legacy_node = settings[legacy_key];
    if(!legacy_node.empty())
        return static_cast<int>(legacy_node) != 0;
    return default_value;
}

int ReadIntSetting(const cv::FileStorage& settings,
                   const std::string& key,
                   const int default_value)
{
    const cv::FileNode node = settings[key];
    if(!node.empty())
        return static_cast<int>(node);
    return default_value;
}

double MedianDiagnosticValue(std::vector<double> values)
{
    if(values.empty())
        return -1.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if(values.size() % 2 == 1)
        return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

double PhysicalAlphaDeg(const KannalaBrandt8Camera& camera,
                        const cv::Point2f& pixel)
{
    Eigen::Vector3d bearing;
    if(!camera.unproject(Eigen::Vector2d(pixel.x, pixel.y), bearing) ||
       !bearing.allFinite() || bearing.norm() <= 0.0)
        return std::numeric_limits<double>::infinity();
    const double z = std::max(-1.0, std::min(1.0, bearing.z() / bearing.norm()));
    return std::acos(z) * 180.0 / 3.14159265358979323846;
}

double PercentileDiagnosticValue(std::vector<double> values,
                                 const double percentile)
{
    if(values.empty())
        return -1.0;
    std::sort(values.begin(), values.end());
    const double clamped =
        std::max(0.0, std::min(1.0, percentile));
    const double index = clamped *
        static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(index));
    const size_t hi = static_cast<size_t>(std::ceil(index));
    if(lo == hi)
        return values[lo];
    return values[lo] * (static_cast<double>(hi) - index) +
           values[hi] * (index - static_cast<double>(lo));
}

double BearingParallaxDegrees(const cv::KeyPoint& a,
                              const cv::KeyPoint& b,
                              const cv::Mat& K)
{
    const double fx = K.at<float>(0,0);
    const double fy = K.at<float>(1,1);
    const double cx = K.at<float>(0,2);
    const double cy = K.at<float>(1,2);
    if(fx == 0.0 || fy == 0.0)
        return -1.0;
    cv::Vec3d va((a.pt.x - cx) / fx, (a.pt.y - cy) / fy, 1.0);
    cv::Vec3d vb((b.pt.x - cx) / fx, (b.pt.y - cy) / fy, 1.0);
    const double na = cv::norm(va);
    const double nb = cv::norm(vb);
    if(na == 0.0 || nb == 0.0)
        return -1.0;
    double cos_angle = va.dot(vb) / (na * nb);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    return std::acos(cos_angle) * 180.0 / CV_PI;
}

}

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0),
    mbEnableNativeFisheyeLines(false),
    mbEnablePointLineTracking(false),
    mPointLineMode(LEGACY_PL),
    mbEnableFisheyeLineShadow(false),
    mpFisheyeLineFrontend(),
    mpFisheyeLineShadowMatcher(),
    mvPreviousFisheyeLineObservations(),
    mLastFisheyeLineShadowMatchResult(),
    mbHasPreviousFisheyeLineObservations(false),
    mnPreviousFisheyeLineFrameId(0),
    mvPreviousFisheyeLineTrackIds(),
    mvTwoFramesAgoFisheyeLineObservations(),
    mvTwoFramesAgoFisheyeLineTrackIds(),
    mbHasTwoFramesAgoFisheyeLineObservations(false),
    mnTwoFramesAgoFisheyeLineFrameId(0),
    mvCurrentFisheyeLineTrackIds(),
    mvFisheyeLineTrackLengths(),
    mvFisheyeLineTrackMissingFrameCounts(),
    mvFisheyeLineTrackStartFrames(),
    mvFisheyeLineTrackLastFrames(),
    mnNextFisheyeLineTrackId(0),
    mnConfiguredNativeVirtualViews(0),
    mnMonocularInitializationMinMatches(100),
    mnMonocularInitializationSearchWindow(100),
    mnMonocularInitializationMaxFrameGap(40),
    mbUseKb8PointObservations(false),
    mLastNativeLoopFrameDiagnostics()
{
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    std::string kb8_error_message;
    mbUseKb8PointObservations =
        LoadKb8CameraParametersFromSettings(
            fSettings,
            mKb8PointCameraParameters,
            kb8_error_message);

    if(IsKannalaBrandt8CameraType(fSettings) &&
       !mbUseKb8PointObservations)
    {
        throw std::runtime_error(
            "Invalid KannalaBrandt8 camera settings: " +
            kb8_error_message);
    }
    if(mbUseKb8PointObservations)
    {
        const cv::FileNode alpha_node =
            fSettings["NativeFisheye.PhysicalAlphaMaxDeg"];
        if(!alpha_node.empty())
        {
            const double alpha_limit = static_cast<double>(alpha_node);
            if(std::isfinite(alpha_limit) &&
               alpha_limit > 0.0 && alpha_limit < 180.0)
            {
                mKb8PointCameraParameters.physical_alpha_max_deg =
                    alpha_limit;
            }
        }
    }

    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    cv::Mat K;
    cv::Mat DistCoef(4,1,CV_32F);

    if(mbUseKb8PointObservations)
    {
        K = Kb8ParametersToPinholeMatrix(
            mKb8PointCameraParameters);
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        DistCoef.setTo(0.0f);
        if(mpMap)
            mpMap->SetNativeFisheyeCameraParameters(
                mKb8PointCameraParameters);
    }
    else
    {
        fx = fSettings["Camera.fx"];
        fy = fSettings["Camera.fy"];
        cx = fSettings["Camera.cx"];
        cy = fSettings["Camera.cy"];

        K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;

        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
    }
    K.copyTo(mK);
    DistCoef.copyTo(mDistCoef);

    int img_width = fSettings["Camera.width"];
    int img_height = fSettings["Camera.height"];

    mbEnableNativeFisheyeLines =
        ReadBoolSetting(
            fSettings,
            "NativeFisheye.Enable",
            "EnableNativeFisheyeLines",
            false);

    mbEnablePointLineTracking =
        ReadBoolSetting(
            fSettings,
            "NativeFisheye.PointLineTracking",
            "EnablePointLineTracking",
            false);

    mPointLineMode =
        (mbEnableNativeFisheyeLines && mbEnablePointLineTracking)
            ? PHYSICAL_PIXEL_PL
            : LEGACY_PL;
    const char* mode_env = std::getenv("T265_POINT_LINE_MODE");
    if(mode_env && std::strcmp(mode_env, "POINT_ONLY") == 0)
        mPointLineMode = POINT_ONLY;
    else if(mode_env && std::strcmp(mode_env, "LEGACY_PL") == 0)
        mPointLineMode = LEGACY_PL;
    else if(mode_env && std::strcmp(mode_env, "PHYSICAL_ANGULAR_PL") == 0)
        mPointLineMode = PHYSICAL_ANGULAR_PL;
    else if(mode_env && std::strcmp(mode_env, "PHYSICAL_PIXEL_PL") == 0)
        mPointLineMode = PHYSICAL_PIXEL_PL;
    else if(mode_env && std::strcmp(mode_env, "PHYSICAL_PIXEL_COV_PL") == 0)
        mPointLineMode = PHYSICAL_PIXEL_COV_PL;
    else if(mode_env && std::strcmp(mode_env, "PHYSICAL_PL") == 0)
        mPointLineMode = PHYSICAL_PL;

    mnConfiguredNativeVirtualViews =
        ReadIntSetting(
            fSettings,
            "NativeFisheye.VirtualViews",
            0);

    const cv::FileNode fisheye_enable_node =
        fSettings["FisheyeLine.enable"];

    if(!fisheye_enable_node.empty())
        mbEnableFisheyeLineShadow =
            static_cast<int>(fisheye_enable_node) != 0;
    if(mbEnableNativeFisheyeLines)
        mbEnableFisheyeLineShadow = true;

    if(mbEnableFisheyeLineShadow)
    {
        std::vector<float> kb8_params;
        int fisheye_width = 0;
        int fisheye_height = 0;

        KannalaBrandt8Camera::Parameters native_camera_params;
        if(mbUseKb8PointObservations)
        {
            native_camera_params = mKb8PointCameraParameters;
            kb8_params =
                Kb8ParametersToFloatVector(
                    native_camera_params);
            fisheye_width = native_camera_params.width;
            fisheye_height = native_camera_params.height;
        }
        else
        {
            kb8_params = {
                static_cast<float>(fSettings["FisheyeLine.fx"]),
                static_cast<float>(fSettings["FisheyeLine.fy"]),
                static_cast<float>(fSettings["FisheyeLine.cx"]),
                static_cast<float>(fSettings["FisheyeLine.cy"]),
                static_cast<float>(fSettings["FisheyeLine.k1"]),
                static_cast<float>(fSettings["FisheyeLine.k2"]),
                static_cast<float>(fSettings["FisheyeLine.k3"]),
                static_cast<float>(fSettings["FisheyeLine.k4"])
            };

            fisheye_width =
                static_cast<int>(fSettings["FisheyeLine.width"]);
            fisheye_height =
                static_cast<int>(fSettings["FisheyeLine.height"]);

            native_camera_params.fx = kb8_params[0];
            native_camera_params.fy = kb8_params[1];
            native_camera_params.cx = kb8_params[2];
            native_camera_params.cy = kb8_params[3];
            native_camera_params.k1 = kb8_params[4];
            native_camera_params.k2 = kb8_params[5];
            native_camera_params.k3 = kb8_params[6];
            native_camera_params.k4 = kb8_params[7];
            native_camera_params.width = fisheye_width;
            native_camera_params.height = fisheye_height;
        }
        const cv::FileNode alpha_node =
            fSettings["NativeFisheye.PhysicalAlphaMaxDeg"];
        if(!alpha_node.empty())
        {
            const double alpha_limit = static_cast<double>(alpha_node);
            if(std::isfinite(alpha_limit) &&
               alpha_limit > 0.0 && alpha_limit < 180.0)
            {
                native_camera_params.physical_alpha_max_deg = alpha_limit;
            }
        }
        if(mpMap)
            mpMap->SetNativeFisheyeCameraParameters(native_camera_params);

        FisheyeLineFrontend::Options line_options;
        line_options.physical_alpha_max_deg =
            native_camera_params.physical_alpha_max_deg;
        line_options.require_all_physical_samples =
            native_camera_params.physical_alpha_max_deg < 113.201945;

        std::vector<double> configured_yaws;
        for(int yaw_index = 0; yaw_index < 16; ++yaw_index)
        {
            const std::string key =
                "FisheyeLine.yaw" + std::to_string(yaw_index);
            const cv::FileNode yaw_node = fSettings[key];
            if(yaw_node.empty())
                break;
            configured_yaws.push_back(static_cast<double>(yaw_node));
        }
        if(mnConfiguredNativeVirtualViews > 0)
        {
            line_options.virtual_yaw_deg.clear();
            for(int yaw_index = 0;
                yaw_index < mnConfiguredNativeVirtualViews &&
                    yaw_index < static_cast<int>(configured_yaws.size());
                ++yaw_index)
            {
                line_options.virtual_yaw_deg.push_back(
                    configured_yaws[yaw_index]);
            }
            if(line_options.virtual_yaw_deg.empty())
                mnConfiguredNativeVirtualViews =
                    static_cast<int>(
                        FisheyeLineFrontend::Options().
                            virtual_yaw_deg.size());
        }
        else if(!configured_yaws.empty())
        {
            line_options.virtual_yaw_deg = configured_yaws;
            mnConfiguredNativeVirtualViews =
                static_cast<int>(configured_yaws.size());
        }
        if(line_options.virtual_yaw_deg.empty())
            line_options.virtual_yaw_deg =
                FisheyeLineFrontend::Options().virtual_yaw_deg;
        mnConfiguredNativeVirtualViews =
            static_cast<int>(line_options.virtual_yaw_deg.size());

        mpFisheyeLineFrontend =
            std::make_shared<FisheyeLineFrontend>(
                kb8_params,
                cv::Size(fisheye_width, fisheye_height),
                line_options);

        FisheyeLineShadowMatcher::Options
            matcher_options;

        matcher_options.require_same_view = true;
        matcher_options.min_valid_sample_ratio = 0.60;
        matcher_options.min_virtual_length_px = 15.0;
        matcher_options.min_length_ratio = 0.50;
        matcher_options.max_angle_deg = 15.0;
        matcher_options.max_midpoint_distance_px = 50.0;
        matcher_options.max_endpoint_distance_px = 50.0;
        matcher_options.max_sample_distance_px = 15.0;
        matcher_options.max_total_cost = 1.0;

        mpFisheyeLineShadowMatcher =
            std::make_shared<FisheyeLineShadowMatcher>(
                matcher_options);

        cout << endl
             << "P9-C2b Fisheye Line Shadow Frontend: ENABLED"
             << endl;
        cout << "- EnableNativeFisheyeLines: "
             << (mbEnableNativeFisheyeLines ? 1 : 0) << endl;
        cout << "- EnablePointLineTracking: "
             << (mbEnablePointLineTracking ? 1 : 0) << endl;
        cout << "- NativeFisheye.VirtualViews: "
             << mnConfiguredNativeVirtualViews << endl;

        cout << "- image size: "
             << fisheye_width << " x "
             << fisheye_height << endl;

        cout << "- yaw layout:";
        for(size_t yaw_index = 0;
            yaw_index < line_options.virtual_yaw_deg.size();
            ++yaw_index)
        {
            cout << (yaw_index == 0 ? " " : ", ")
                 << line_options.virtual_yaw_deg[yaw_index];
        }
        cout << endl;

        cout << "- invalid mapping ratio: "
             << mpFisheyeLineFrontend->
                    GetInvalidMappingRatio()
             << endl;
    }
    else
    {
        cout << endl
             << "P9-C2b Fisheye Line Shadow Frontend: DISABLED"
             << endl;
    }

    cout << "img_width = " << img_width << endl;
    cout << "img_height = " << img_height << endl;
    initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3,3), mK, Size(img_width, img_height), CV_32F, mUndistX, mUndistY);
    cout << "mUndistX size = " << mUndistX.size << endl;
    cout << "mUndistY size = " << mUndistY.size << endl;
    mbf = fSettings["Camera.bf"];
    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    mnMonocularInitializationMinMatches = ReadIntSetting(
        fSettings,
        "Initialization.MinMatches",
        100);
    if(mnMonocularInitializationMinMatches < 8)
        mnMonocularInitializationMinMatches = 8;
    mnMonocularInitializationSearchWindow = ReadIntSetting(
        fSettings,
        "Initialization.SearchWindow",
        100);
    if(mnMonocularInitializationSearchWindow < 10)
        mnMonocularInitializationSearchWindow = 10;
    mnMonocularInitializationMaxFrameGap = ReadIntSetting(
        fSettings,
        "Initialization.MaxFrameGap",
        40);
    if(mnMonocularInitializationMaxFrameGap < 1)
        mnMonocularInitializationMaxFrameGap = 1;

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    cout << "- Initialization Min Matches: "
         << mnMonocularInitializationMinMatches << endl;
    cout << "- Initialization Search Window: "
         << mnMonocularInitializationSearchWindow << endl;
    cout << "- Initialization Max Frame Gap: "
         << mnMonocularInitializationMaxFrameGap << endl;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

NativeLoopFrameDiagnostics
Tracking::GetNativeLoopFrameDiagnostics() const
{
    return mLastNativeLoopFrameDiagnostics;
}

cv::Mat Tracking::GrabImageMonocularWithPL(const cv::Mat &im,  const double &timestamp)
{
    mLastNativeLoopFrameDiagnostics = NativeLoopFrameDiagnostics();
    mLastNativeLoopFrameDiagnostics.timestamp = timestamp;
    mLastNativeLoopFrameDiagnostics.track_input_type = im.type();

    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    // The monocular fisheye dataset stores grayscale images as CV_16UC1.
    // Use a fixed mapping to preserve cross-frame photometric consistency.
    if(mImGray.type() == CV_16UC1)
    {
        cv::Mat gray8;
        mImGray.convertTo(
            gray8,
            CV_8UC1,
            1.0 / 256.0);
        mImGray = gray8;
    }

    if(mImGray.type() != CV_8UC1)
    {
        cerr << "[Tracking] Unsupported monocular image type: "
             << mImGray.type() << endl;

        return cv::Mat();
    }

    double image_min = 0.0;
    double image_max = 0.0;
    cv::Scalar image_mean;
    cv::minMaxLoc(mImGray, &image_min, &image_max);
    image_mean = cv::mean(mImGray);
    mLastNativeLoopFrameDiagnostics.frame_constructor_input_type =
        mImGray.type();
    mLastNativeLoopFrameDiagnostics.orb_input_type = mImGray.type();
    mLastNativeLoopFrameDiagnostics.image_channels = mImGray.channels();
    mLastNativeLoopFrameDiagnostics.image_rows = mImGray.rows;
    mLastNativeLoopFrameDiagnostics.image_cols = mImGray.cols;
    mLastNativeLoopFrameDiagnostics.image_min = image_min;
    mLastNativeLoopFrameDiagnostics.image_max = image_max;
    mLastNativeLoopFrameDiagnostics.image_mean = image_mean[0];

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    {
        if(mbUseKb8PointObservations)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mKb8PointCameraParameters,mPointLineMode != POINT_ONLY);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mPointLineMode != POINT_ONLY);
    }
    else
    {
        if(mbUseKb8PointObservations)
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mKb8PointCameraParameters,mPointLineMode != POINT_ONLY);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mPointLineMode != POINT_ONLY);
    }

    mLastNativeLoopFrameDiagnostics.frame_id = mCurrentFrame.mnId;
    mLastNativeLoopFrameDiagnostics.timestamp = timestamp;
    mLastNativeLoopFrameDiagnostics.tracking_state = mState;
    mLastNativeLoopFrameDiagnostics.tracking_state_before = mState;
    mLastNativeLoopFrameDiagnostics.orb_keypoints_raw =
        static_cast<int>(mCurrentFrame.mvKeys.size());
    mLastNativeLoopFrameDiagnostics.undistorted_keypoints =
        static_cast<int>(mCurrentFrame.mvKeysUn.size());
    mLastNativeLoopFrameDiagnostics.descriptors_rows =
        mCurrentFrame.mDescriptors.rows;
    mLastNativeLoopFrameDiagnostics.safe89_orb_detected_before_mask =
        mCurrentFrame.mKb8FeatureCountBeforeAlphaFilter;
    mLastNativeLoopFrameDiagnostics.safe89_orb_valid_after_mask =
        mCurrentFrame.mKb8FeatureCountAfterAlphaFilter;
    mLastNativeLoopFrameDiagnostics.safe89_orb_bin_0_30 =
        mCurrentFrame.mKb8FeatureAlphaBins[0];
    mLastNativeLoopFrameDiagnostics.safe89_orb_bin_30_60 =
        mCurrentFrame.mKb8FeatureAlphaBins[1];
    mLastNativeLoopFrameDiagnostics.safe89_orb_bin_60_80 =
        mCurrentFrame.mKb8FeatureAlphaBins[2];
    mLastNativeLoopFrameDiagnostics.safe89_orb_bin_80_89 =
        mCurrentFrame.mKb8FeatureAlphaBins[3];
    mLastNativeLoopFrameDiagnostics.safe89_orb_feature_density =
        static_cast<double>(mCurrentFrame.mKb8FeatureCountAfterAlphaFilter) /
        (static_cast<double>(mImGray.cols) * static_cast<double>(mImGray.rows) *
         0.563732638889);
    if(mbUseKb8PointObservations)
    {
        const KannalaBrandt8Camera camera(mKb8PointCameraParameters);
        for(const cv::Point2f& pixel : mCurrentFrame.mvPointRawPixels)
        {
            const double alpha = PhysicalAlphaDeg(camera, pixel);
            mLastNativeLoopFrameDiagnostics.safe89_orb_max_alpha_deg =
                std::max(
                    mLastNativeLoopFrameDiagnostics.safe89_orb_max_alpha_deg,
                    alpha);
            if(alpha > mKb8PointCameraParameters.physical_alpha_max_deg)
                ++mLastNativeLoopFrameDiagnostics.safe89_orb_alpha_leaks;
        }
    }
    mLastNativeLoopFrameDiagnostics.native_enabled =
        mbEnableNativeFisheyeLines;
    mLastNativeLoopFrameDiagnostics.point_line_tracking_enabled =
        mbEnablePointLineTracking;
    mLastNativeLoopFrameDiagnostics.virtual_views =
        mnConfiguredNativeVirtualViews;
    if(mpMap)
    {
        KannalaBrandt8Camera::Parameters native_camera_params;
        mLastNativeLoopFrameDiagnostics.kb8_parameters_available =
            mpMap->GetNativeFisheyeCameraParameters(
                native_camera_params);
    }

    // P9-C2b shadow-only fisheye line extraction.
    // This does not modify NL, mvKeylinesUn, descriptors,
    // mvpMapLines, matching, pose estimation or optimization.
    if(mbEnableFisheyeLineShadow &&
       mpFisheyeLineFrontend)
    {
        const FisheyeLineFrontend::FrameResult line_result =
            mpFisheyeLineFrontend->ProcessFrame(
                mImGray,
                false);

        PhysicalLineObservationBuilder::Stats builder_stats;
        mCurrentFrame.SetPhysicalLineObservations(
            PhysicalLineObservationBuilder::Build(
                line_result, mpFisheyeLineFrontend->GetValidMasks(),
                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp,
                &builder_stats));
        mLastNativeLoopFrameDiagnostics.physical_observations =
            builder_stats.physical;
        mLastNativeLoopFrameDiagnostics.physical_group_size_1 =
            builder_stats.single;
        mLastNativeLoopFrameDiagnostics.physical_group_size_2 =
            builder_stats.two;
        mLastNativeLoopFrameDiagnostics.physical_group_size_3 =
            builder_stats.three;
        PhysicalLineObservationBuilder::AccumulateAndFlushAtExit(
            builder_stats);

        if(std::getenv("T265_PHYSICAL_WRAPPER_STATS"))
        {
            PhysicalLineObservationWrapper::FrameStats wrapper_stats;
            PhysicalLineObservationWrapper::Analyze(
                line_result, mpFisheyeLineFrontend->GetValidMasks(),
                mCurrentFrame.mnId, &wrapper_stats);
            PhysicalLineObservationWrapper::AccumulateAndFlushAtExit(
                wrapper_stats);
        }

        mCurrentFrame.SetFisheyeLineObservations(
            ConvertFisheyeLineObservations(
                line_result));
        if(mbUseKb8PointObservations)
        {
            const KannalaBrandt8Camera camera(mKb8PointCameraParameters);
            for(const FisheyeLineObservation& observation :
                mCurrentFrame.mvFisheyeLineObservations)
            {
                for(size_t i = 0; i < observation.fisheye_samples.size(); ++i)
                {
                    if(i < observation.sample_valid.size() &&
                       observation.sample_valid[i] == 0)
                        continue;
                    const double alpha =
                        PhysicalAlphaDeg(camera, observation.fisheye_samples[i]);
                    mLastNativeLoopFrameDiagnostics.safe89_line_support_max_alpha_deg =
                        std::max(
                            mLastNativeLoopFrameDiagnostics.safe89_line_support_max_alpha_deg,
                            alpha);
                    if(alpha > mKb8PointCameraParameters.physical_alpha_max_deg)
                        ++mLastNativeLoopFrameDiagnostics.safe89_line_support_alpha_leaks;
                }
            }
        }

        cout << "[P9-C2b][Frame "
             << mCurrentFrame.mnId
             << "] fisheye_lines="
             << mCurrentFrame.NFisheyeLines
             << " legacy_NL="
             << mCurrentFrame.NL
             << " nan_inf="
             << line_result.nan_inf_count
             << " oob="
             << line_result.out_of_bounds_count
             << " total_ms="
             << line_result.total_time_ms
             << endl;

        if(mpFisheyeLineShadowMatcher &&
           mbHasPreviousFisheyeLineObservations)
        {
            mLastFisheyeLineShadowMatchResult =
                mpFisheyeLineShadowMatcher->
                    MatchAdjacentFrames(
                        mvPreviousFisheyeLineObservations,
                        mCurrentFrame.
                            mvFisheyeLineObservations);

            int strong_matches = 0;
            int medium_matches = 0;
            int weak_matches = 0;

            // P9-C3j diagnostic only:
            // accumulate cost components for strong / medium / weak
            // mutual matches without changing matching decisions.
            double strong_cost_sum = 0.0;
            double strong_angle_sum = 0.0;
            double strong_midpoint_sum = 0.0;
            double strong_endpoint_sum = 0.0;
            double strong_sample_sum = 0.0;
            double strong_length_ratio_sum = 0.0;

            double medium_cost_sum = 0.0;
            double medium_angle_sum = 0.0;
            double medium_midpoint_sum = 0.0;
            double medium_endpoint_sum = 0.0;
            double medium_sample_sum = 0.0;
            double medium_length_ratio_sum = 0.0;

            double weak_cost_sum = 0.0;
            double weak_angle_sum = 0.0;
            double weak_midpoint_sum = 0.0;
            double weak_endpoint_sum = 0.0;
            double weak_sample_sum = 0.0;
            double weak_length_ratio_sum = 0.0;

            // P9-C3k diagnostic only:
            // signed midpoint displacement of every mutual match.
            std::vector<double> mutual_dx_values;
            std::vector<double> mutual_dy_values;
            std::vector<double> mutual_motion_values;

            // P9-C3l diagnostic only:
            // split signed displacement by virtual view.
            const int fisheye_view_count = 4;

            std::vector<std::vector<double>>
                view_dx_values(fisheye_view_count);
            std::vector<std::vector<double>>
                view_dy_values(fisheye_view_count);
            std::vector<std::vector<double>>
                view_motion_values(fisheye_view_count);

            // P9-C3m diagnostic only:
            // previous midpoint position for spatial motion analysis.
            std::vector<std::vector<double>>
                view_midpoint_x_values(fisheye_view_count);
            std::vector<std::vector<double>>
                view_midpoint_y_values(fisheye_view_count);

            mutual_dx_values.reserve(
                mLastFisheyeLineShadowMatchResult.matches.size());
            mutual_dy_values.reserve(
                mLastFisheyeLineShadowMatchResult.matches.size());
            mutual_motion_values.reserve(
                mLastFisheyeLineShadowMatchResult.matches.size());

            for(const auto& match :
                mLastFisheyeLineShadowMatchResult.matches)
            {
                if(match.previous_index >= 0 &&
                   match.current_index >= 0 &&
                   static_cast<std::size_t>(
                       match.previous_index) <
                       mvPreviousFisheyeLineObservations.size() &&
                   static_cast<std::size_t>(
                       match.current_index) <
                       mCurrentFrame.
                           mvFisheyeLineObservations.size())
                {
                    const FisheyeLineObservation&
                        previous_observation =
                            mvPreviousFisheyeLineObservations[
                                match.previous_index];

                    const FisheyeLineObservation&
                        current_observation =
                            mCurrentFrame.
                                mvFisheyeLineObservations[
                                    match.current_index];

                    const double previous_midpoint_x =
                        0.5 * (
                            previous_observation.
                                fisheye_start.x +
                            previous_observation.
                                fisheye_end.x);

                    const double previous_midpoint_y =
                        0.5 * (
                            previous_observation.
                                fisheye_start.y +
                            previous_observation.
                                fisheye_end.y);

                    const double current_midpoint_x =
                        0.5 * (
                            current_observation.
                                fisheye_start.x +
                            current_observation.
                                fisheye_end.x);

                    const double current_midpoint_y =
                        0.5 * (
                            current_observation.
                                fisheye_start.y +
                            current_observation.
                                fisheye_end.y);

                    const double displacement_x =
                        current_midpoint_x -
                        previous_midpoint_x;

                    const double displacement_y =
                        current_midpoint_y -
                        previous_midpoint_y;

                    const double displacement_magnitude =
                        std::sqrt(
                            displacement_x *
                                displacement_x +
                            displacement_y *
                                displacement_y);

                    if(std::isfinite(displacement_x) &&
                       std::isfinite(displacement_y) &&
                       std::isfinite(
                           displacement_magnitude))
                    {
                        const char* match_class =
                            match.total_cost <= 0.10
                                ? "strong"
                                : (
                                    match.total_cost <= 0.12
                                        ? "medium"
                                        : "weak"
                                  );

                        // P9-C3n diagnostic only:
                        // one record for every valid mutual match.
                        cout
                            << "[P9-C3n][Match]"
                            << " frame="
                            << mCurrentFrame.mnId
                            << " previous_frame="
                            << mnPreviousFisheyeLineFrameId
                            << " view="
                            << previous_observation.view_id
                            << " previous_index="
                            << match.previous_index
                            << " current_index="
                            << match.current_index
                            << " previous_x="
                            << previous_midpoint_x
                            << " previous_y="
                            << previous_midpoint_y
                            << " current_x="
                            << current_midpoint_x
                            << " current_y="
                            << current_midpoint_y
                            << " dx="
                            << displacement_x
                            << " dy="
                            << displacement_y
                            << " motion="
                            << displacement_magnitude
                            << " angle="
                            << match.angle_deg
                            << " midpoint="
                            << match.midpoint_distance_px
                            << " endpoint="
                            << match.endpoint_distance_px
                            << " sample="
                            << match.sample_distance_px
                            << " length_ratio="
                            << match.length_ratio
                            << " total_cost="
                            << match.total_cost
                            << " class="
                            << match_class
                            << endl;

                        mutual_dx_values.push_back(
                            displacement_x);
                        mutual_dy_values.push_back(
                            displacement_y);
                        mutual_motion_values.push_back(
                            displacement_magnitude);

                        const int previous_view_id =
                            previous_observation.view_id;

                        const int current_view_id =
                            current_observation.view_id;

                        if(previous_view_id ==
                               current_view_id &&
                           previous_view_id >= 0 &&
                           previous_view_id <
                               fisheye_view_count)
                        {
                            view_dx_values[
                                previous_view_id].
                                    push_back(
                                        displacement_x);

                            view_dy_values[
                                previous_view_id].
                                    push_back(
                                        displacement_y);

                            view_motion_values[
                                previous_view_id].
                                    push_back(
                                        displacement_magnitude);

                            view_midpoint_x_values[
                                previous_view_id].
                                    push_back(
                                        previous_midpoint_x);

                            view_midpoint_y_values[
                                previous_view_id].
                                    push_back(
                                        previous_midpoint_y);
                        }
                    }
                }

                if(match.total_cost <= 0.10)
                {
                    ++strong_matches;

                    strong_cost_sum +=
                        match.total_cost;
                    strong_angle_sum +=
                        match.angle_deg;
                    strong_midpoint_sum +=
                        match.midpoint_distance_px;
                    strong_endpoint_sum +=
                        match.endpoint_distance_px;
                    strong_sample_sum +=
                        match.sample_distance_px;
                    strong_length_ratio_sum +=
                        match.length_ratio;
                }
                else if(match.total_cost <= 0.12)
                {
                    ++medium_matches;

                    medium_cost_sum +=
                        match.total_cost;
                    medium_angle_sum +=
                        match.angle_deg;
                    medium_midpoint_sum +=
                        match.midpoint_distance_px;
                    medium_endpoint_sum +=
                        match.endpoint_distance_px;
                    medium_sample_sum +=
                        match.sample_distance_px;
                    medium_length_ratio_sum +=
                        match.length_ratio;
                }
                else
                {
                    ++weak_matches;

                    weak_cost_sum +=
                        match.total_cost;
                    weak_angle_sum +=
                        match.angle_deg;
                    weak_midpoint_sum +=
                        match.midpoint_distance_px;
                    weak_endpoint_sum +=
                        match.endpoint_distance_px;
                    weak_sample_sum +=
                        match.sample_distance_px;
                    weak_length_ratio_sum +=
                        match.length_ratio;
                }
            }

            const auto safe_mean =
                [](const double sum, const int count)
                {
                    return count > 0
                        ? sum /
                              static_cast<double>(count)
                        : 0.0;
                };

            const auto median_value =
                [](std::vector<double> values)
                {
                    if(values.empty())
                        return 0.0;

                    std::sort(
                        values.begin(),
                        values.end());

                    const std::size_t middle =
                        values.size() / 2;

                    if(values.size() % 2 == 1)
                        return values[middle];

                    return 0.5 * (
                        values[middle - 1] +
                        values[middle]);
                };

            const double median_dx =
                median_value(mutual_dx_values);

            const double median_dy =
                median_value(mutual_dy_values);

            const double median_motion =
                median_value(mutual_motion_values);

            std::vector<double> dx_deviations;
            std::vector<double> dy_deviations;

            dx_deviations.reserve(
                mutual_dx_values.size());
            dy_deviations.reserve(
                mutual_dy_values.size());

            for(const double value :
                mutual_dx_values)
            {
                dx_deviations.push_back(
                    std::abs(value - median_dx));
            }

            for(const double value :
                mutual_dy_values)
            {
                dy_deviations.push_back(
                    std::abs(value - median_dy));
            }

            const double mad_dx =
                median_value(dx_deviations);

            const double mad_dy =
                median_value(dy_deviations);

            const double median_direction_norm =
                std::sqrt(
                    median_dx * median_dx +
                    median_dy * median_dy);

            int same_direction_count = 0;

            if(median_direction_norm > 1e-9)
            {
                for(std::size_t index = 0;
                    index < mutual_dx_values.size();
                    ++index)
                {
                    const double dot_product =
                        mutual_dx_values[index] *
                            median_dx +
                        mutual_dy_values[index] *
                            median_dy;

                    if(dot_product > 0.0)
                        ++same_direction_count;
                }
            }

            const double same_direction_ratio =
                mutual_dx_values.empty()
                    ? 0.0
                    : static_cast<double>(
                          same_direction_count) /
                          static_cast<double>(
                              mutual_dx_values.size());

            struct ViewMotionStatistics
            {
                std::size_t count = 0;
                double median_dx = 0.0;
                double median_dy = 0.0;
                double median_motion = 0.0;
                double mad_dx = 0.0;
                double mad_dy = 0.0;
                double same_direction_ratio = 0.0;

                double positive_dy_ratio = 0.0;
                double corr_x_dx = 0.0;
                double corr_y_dx = 0.0;
                double corr_x_dy = 0.0;
                double corr_y_dy = 0.0;
            };

            const auto compute_view_motion_statistics =
                [&median_value](
                    const std::vector<double>& midpoint_x_values,
                    const std::vector<double>& midpoint_y_values,
                    const std::vector<double>& dx_values,
                    const std::vector<double>& dy_values,
                    const std::vector<double>& motion_values)
                {
                    ViewMotionStatistics statistics;

                    statistics.count =
                        dx_values.size();

                    if(dx_values.empty() ||
                       dx_values.size() !=
                           dy_values.size() ||
                       dx_values.size() !=
                           motion_values.size() ||
                       dx_values.size() !=
                           midpoint_x_values.size() ||
                       dx_values.size() !=
                           midpoint_y_values.size())
                    {
                        return statistics;
                    }

                    statistics.median_dx =
                        median_value(dx_values);

                    statistics.median_dy =
                        median_value(dy_values);

                    statistics.median_motion =
                        median_value(motion_values);

                    std::vector<double> dx_deviations;
                    std::vector<double> dy_deviations;

                    dx_deviations.reserve(
                        dx_values.size());
                    dy_deviations.reserve(
                        dy_values.size());

                    for(const double value : dx_values)
                    {
                        dx_deviations.push_back(
                            std::abs(
                                value -
                                statistics.median_dx));
                    }

                    for(const double value : dy_values)
                    {
                        dy_deviations.push_back(
                            std::abs(
                                value -
                                statistics.median_dy));
                    }

                    statistics.mad_dx =
                        median_value(dx_deviations);

                    statistics.mad_dy =
                        median_value(dy_deviations);

                    const double direction_norm =
                        std::sqrt(
                            statistics.median_dx *
                                statistics.median_dx +
                            statistics.median_dy *
                                statistics.median_dy);

                    int same_direction_count = 0;
                    int positive_dy_count = 0;

                    for(std::size_t index = 0;
                        index < dx_values.size();
                        ++index)
                    {
                        if(dy_values[index] > 0.0)
                        {
                            ++positive_dy_count;
                        }

                        if(direction_norm > 1e-9)
                        {
                            const double dot_product =
                                dx_values[index] *
                                    statistics.median_dx +
                                dy_values[index] *
                                    statistics.median_dy;

                            if(dot_product > 0.0)
                            {
                                ++same_direction_count;
                            }
                        }
                    }

                    statistics.same_direction_ratio =
                        static_cast<double>(
                            same_direction_count) /
                        static_cast<double>(
                            dx_values.size());

                    statistics.positive_dy_ratio =
                        static_cast<double>(
                            positive_dy_count) /
                        static_cast<double>(
                            dy_values.size());

                    const auto correlation =
                        [](
                            const std::vector<double>& first,
                            const std::vector<double>& second)
                        {
                            if(first.size() < 2 ||
                               first.size() !=
                                   second.size())
                            {
                                return 0.0;
                            }

                            double first_sum = 0.0;
                            double second_sum = 0.0;

                            for(std::size_t index = 0;
                                index < first.size();
                                ++index)
                            {
                                first_sum += first[index];
                                second_sum += second[index];
                            }

                            const double first_mean =
                                first_sum /
                                static_cast<double>(
                                    first.size());

                            const double second_mean =
                                second_sum /
                                static_cast<double>(
                                    second.size());

                            double covariance = 0.0;
                            double first_variance = 0.0;
                            double second_variance = 0.0;

                            for(std::size_t index = 0;
                                index < first.size();
                                ++index)
                            {
                                const double first_centered =
                                    first[index] -
                                    first_mean;

                                const double second_centered =
                                    second[index] -
                                    second_mean;

                                covariance +=
                                    first_centered *
                                    second_centered;

                                first_variance +=
                                    first_centered *
                                    first_centered;

                                second_variance +=
                                    second_centered *
                                    second_centered;
                            }

                            const double denominator =
                                std::sqrt(
                                    first_variance *
                                    second_variance);

                            if(denominator <= 1e-12)
                            {
                                return 0.0;
                            }

                            const double value =
                                covariance /
                                denominator;

                            if(!std::isfinite(value))
                            {
                                return 0.0;
                            }

                            return std::max(
                                -1.0,
                                std::min(
                                    1.0,
                                    value));
                        };

                    statistics.corr_x_dx =
                        correlation(
                            midpoint_x_values,
                            dx_values);

                    statistics.corr_y_dx =
                        correlation(
                            midpoint_y_values,
                            dx_values);

                    statistics.corr_x_dy =
                        correlation(
                            midpoint_x_values,
                            dy_values);

                    statistics.corr_y_dy =
                        correlation(
                            midpoint_y_values,
                            dy_values);

                    return statistics;
                };

            std::vector<ViewMotionStatistics>
                view_motion_statistics(
                    fisheye_view_count);

            for(int view_id = 0;
                view_id < fisheye_view_count;
                ++view_id)
            {
                view_motion_statistics[view_id] =
                    compute_view_motion_statistics(
                        view_midpoint_x_values[
                            view_id],
                        view_midpoint_y_values[
                            view_id],
                        view_dx_values[view_id],
                        view_dy_values[view_id],
                        view_motion_values[view_id]);
            }

            cout
                << "[P9-C3b][Pair "
                << mnPreviousFisheyeLineFrameId
                << "->"
                << mCurrentFrame.mnId
                << "] previous="
                << mLastFisheyeLineShadowMatchResult.
                       previous_count
                << " current="
                << mLastFisheyeLineShadowMatchResult.
                       current_count
                << " gated="
                << mLastFisheyeLineShadowMatchResult.
                       gated_candidate_count
                << " mutual="
                << mLastFisheyeLineShadowMatchResult.
                       mutual_match_count
                << " strong="
                << strong_matches
                << " medium="
                << medium_matches
                << " weak="
                << weak_matches
                << " duplicate_previous="
                << mLastFisheyeLineShadowMatchResult.
                       duplicate_previous_count
                << " duplicate_current="
                << mLastFisheyeLineShadowMatchResult.
                       duplicate_current_count
                << " nan_inf="
                << mLastFisheyeLineShadowMatchResult.
                       nan_inf_count
                << " legacy_NL="
                << mCurrentFrame.NL
                << endl;

            cout
                << "[P9-C3j][Pair "
                << mnPreviousFisheyeLineFrameId
                << "->"
                << mCurrentFrame.mnId
                << "]"
                << " strong_n="
                << strong_matches
                << " strong_cost="
                << safe_mean(
                       strong_cost_sum,
                       strong_matches)
                << " strong_angle="
                << safe_mean(
                       strong_angle_sum,
                       strong_matches)
                << " strong_midpoint="
                << safe_mean(
                       strong_midpoint_sum,
                       strong_matches)
                << " strong_endpoint="
                << safe_mean(
                       strong_endpoint_sum,
                       strong_matches)
                << " strong_sample="
                << safe_mean(
                       strong_sample_sum,
                       strong_matches)
                << " strong_length_ratio="
                << safe_mean(
                       strong_length_ratio_sum,
                       strong_matches)
                << " medium_n="
                << medium_matches
                << " medium_cost="
                << safe_mean(
                       medium_cost_sum,
                       medium_matches)
                << " medium_angle="
                << safe_mean(
                       medium_angle_sum,
                       medium_matches)
                << " medium_midpoint="
                << safe_mean(
                       medium_midpoint_sum,
                       medium_matches)
                << " medium_endpoint="
                << safe_mean(
                       medium_endpoint_sum,
                       medium_matches)
                << " medium_sample="
                << safe_mean(
                       medium_sample_sum,
                       medium_matches)
                << " medium_length_ratio="
                << safe_mean(
                       medium_length_ratio_sum,
                       medium_matches)
                << " weak_n="
                << weak_matches
                << " weak_cost="
                << safe_mean(
                       weak_cost_sum,
                       weak_matches)
                << " weak_angle="
                << safe_mean(
                       weak_angle_sum,
                       weak_matches)
                << " weak_midpoint="
                << safe_mean(
                       weak_midpoint_sum,
                       weak_matches)
                << " weak_endpoint="
                << safe_mean(
                       weak_endpoint_sum,
                       weak_matches)
                << " weak_sample="
                << safe_mean(
                       weak_sample_sum,
                       weak_matches)
                << " weak_length_ratio="
                << safe_mean(
                       weak_length_ratio_sum,
                       weak_matches)
                << endl;

            cout
                << "[P9-C3k][Pair "
                << mnPreviousFisheyeLineFrameId
                << "->"
                << mCurrentFrame.mnId
                << "]"
                << " mutual_n="
                << mutual_dx_values.size()
                << " median_dx="
                << median_dx
                << " median_dy="
                << median_dy
                << " median_motion="
                << median_motion
                << " mad_dx="
                << mad_dx
                << " mad_dy="
                << mad_dy
                << " same_direction_ratio="
                << same_direction_ratio
                << endl;

            for(int view_id = 0;
                view_id < fisheye_view_count;
                ++view_id)
            {
                const ViewMotionStatistics&
                    statistics =
                        view_motion_statistics[
                            view_id];

                cout
                    << "[P9-C3l][Pair "
                    << mnPreviousFisheyeLineFrameId
                    << "->"
                    << mCurrentFrame.mnId
                    << "][View "
                    << view_id
                    << "]"
                    << " count="
                    << statistics.count
                    << " median_dx="
                    << statistics.median_dx
                    << " median_dy="
                    << statistics.median_dy
                    << " median_motion="
                    << statistics.median_motion
                    << " mad_dx="
                    << statistics.mad_dx
                    << " mad_dy="
                    << statistics.mad_dy
                    << " same_direction_ratio="
                    << statistics.
                           same_direction_ratio
                    << " positive_dy_ratio="
                    << statistics.
                           positive_dy_ratio
                    << " corr_x_dx="
                    << statistics.corr_x_dx
                    << " corr_y_dx="
                    << statistics.corr_y_dx
                    << " corr_x_dy="
                    << statistics.corr_x_dy
                    << " corr_y_dy="
                    << statistics.corr_y_dy
                    << endl;
            }
        }
        else
        {
            cout
                << "[P9-C3b][Frame "
                << mCurrentFrame.mnId
                << "] previous_shadow_observations=NONE"
                << endl;
        }

        const long unsigned int invalid_track_id =
            std::numeric_limits<long unsigned int>::max();

        mvCurrentFisheyeLineTrackIds.assign(
            mCurrentFrame.mvFisheyeLineObservations.size(),
            invalid_track_id);

        int continued_tracks = 0;
        int strong_continued_tracks = 0;
        int medium_continued_tracks = 0;
        int weak_rejected_matches = 0;
        int invalid_track_indices = 0;
        int duplicate_track_assignments = 0;

        std::set<long unsigned int> assigned_track_ids;

        if(mbHasPreviousFisheyeLineObservations)
        {
            for(const auto& match :
                mLastFisheyeLineShadowMatchResult.matches)
            {
                // Weak associations remain visible in P9-C3b,
                // but do not propagate a multi-frame track ID.
                if(match.total_cost > 0.12)
                {
                    ++weak_rejected_matches;
                    continue;
                }

                if(match.previous_index < 0 ||
                   match.current_index < 0 ||
                   static_cast<std::size_t>(
                       match.previous_index) >=
                       mvPreviousFisheyeLineTrackIds.size() ||
                   static_cast<std::size_t>(
                       match.current_index) >=
                       mvCurrentFisheyeLineTrackIds.size())
                {
                    ++invalid_track_indices;
                    continue;
                }

                const long unsigned int track_id =
                    mvPreviousFisheyeLineTrackIds[
                        static_cast<std::size_t>(
                            match.previous_index)];

                if(track_id == invalid_track_id ||
                   track_id >=
                       mvFisheyeLineTrackLengths.size())
                {
                    ++invalid_track_indices;
                    continue;
                }

                const std::size_t current_index =
                    static_cast<std::size_t>(
                        match.current_index);

                if(mvCurrentFisheyeLineTrackIds[
                       current_index] != invalid_track_id)
                {
                    ++duplicate_track_assignments;
                    continue;
                }

                if(!assigned_track_ids.insert(
                        track_id).second)
                {
                    ++duplicate_track_assignments;
                    continue;
                }

                mvCurrentFisheyeLineTrackIds[
                    current_index] = track_id;

                ++mvFisheyeLineTrackLengths[
                    static_cast<std::size_t>(track_id)];

                mvFisheyeLineTrackLastFrames[
                    static_cast<std::size_t>(track_id)] =
                        mCurrentFrame.mnId;

                ++continued_tracks;

                if(match.total_cost <= 0.10)
                    ++strong_continued_tracks;
                else
                    ++medium_continued_tracks;
            }
        }

        // P9-C3f: strong-only one-missing-frame reconnection.
        // Only currently unassigned observations are considered.
        int gap1_strong_candidates = 0;
        int gap1_reconnected = 0;
        int gap1_reconnect_conflicts = 0;
        int gap1_reconnect_invalid = 0;
        int gap1_short_guard_rejected = 0;

        // P9-C3g: quality statistics for accepted strong
        // one-frame-gap reconnections.
        double gap1_cost_sum = 0.0;
        double gap1_angle_sum = 0.0;
        double gap1_midpoint_sum = 0.0;
        double gap1_endpoint_sum = 0.0;
        double gap1_sample_sum = 0.0;
        double gap1_length_ratio_sum = 0.0;

        double gap1_cost_max = 0.0;
        double gap1_angle_max = 0.0;
        double gap1_midpoint_max = 0.0;
        double gap1_endpoint_max = 0.0;
        double gap1_sample_max = 0.0;

        int gap1_old_len1 = 0;
        int gap1_old_len2 = 0;
        int gap1_old_len3_to_5 = 0;
        int gap1_old_len6_to_10 = 0;
        int gap1_old_len_gt10 = 0;
        int gap1_repeated_gap_tracks = 0;

        if(mbHasTwoFramesAgoFisheyeLineObservations &&
           mbHasPreviousFisheyeLineObservations)
        {
            std::set<long unsigned int> previous_track_ids(
                mvPreviousFisheyeLineTrackIds.begin(),
                mvPreviousFisheyeLineTrackIds.end());

            std::vector<FisheyeLineObservation>
                gap1_old_observations;

            std::vector<long unsigned int>
                gap1_old_track_ids;

            for(std::size_t old_index = 0;
                old_index <
                    mvTwoFramesAgoFisheyeLineObservations.size();
                ++old_index)
            {
                if(old_index >=
                   mvTwoFramesAgoFisheyeLineTrackIds.size())
                {
                    ++gap1_reconnect_invalid;
                    continue;
                }

                const long unsigned int old_track_id =
                    mvTwoFramesAgoFisheyeLineTrackIds[
                        old_index];

                // The track must truly be absent at t-1.
                if(previous_track_ids.count(
                       old_track_id) != 0)
                {
                    continue;
                }

                if(old_track_id >=
                       mvFisheyeLineTrackLengths.size() ||
                   old_track_id >=
                       mvFisheyeLineTrackMissingFrameCounts.size() ||
                   old_track_id >=
                       mvFisheyeLineTrackLastFrames.size())
                {
                    ++gap1_reconnect_invalid;
                    continue;
                }

                gap1_old_observations.push_back(
                    mvTwoFramesAgoFisheyeLineObservations[
                        old_index]);

                gap1_old_track_ids.push_back(
                    old_track_id);
            }

            std::vector<FisheyeLineObservation>
                gap1_current_unassigned_observations;

            std::vector<std::size_t>
                gap1_current_original_indices;

            for(std::size_t current_index = 0;
                current_index <
                    mvCurrentFisheyeLineTrackIds.size();
                ++current_index)
            {
                if(mvCurrentFisheyeLineTrackIds[
                       current_index] != invalid_track_id)
                {
                    continue;
                }

                gap1_current_unassigned_observations.push_back(
                    mCurrentFrame.mvFisheyeLineObservations[
                        current_index]);

                gap1_current_original_indices.push_back(
                    current_index);
            }

            if(!gap1_old_observations.empty() &&
               !gap1_current_unassigned_observations.empty())
            {
                const FisheyeLineShadowMatcher::Result
                    gap1_reconnect_result =
                        mpFisheyeLineShadowMatcher->
                            MatchAdjacentFrames(
                                gap1_old_observations,
                                gap1_current_unassigned_observations);

                gap1_reconnect_invalid +=
                    gap1_reconnect_result.nan_inf_count +
                    gap1_reconnect_result.
                        duplicate_previous_count +
                    gap1_reconnect_result.
                        duplicate_current_count;

                for(const auto& match :
                    gap1_reconnect_result.matches)
                {
                    if(match.previous_index < 0 ||
                       match.current_index < 0 ||
                       static_cast<std::size_t>(
                           match.previous_index) >=
                           gap1_old_track_ids.size() ||
                       static_cast<std::size_t>(
                           match.current_index) >=
                           gap1_current_original_indices.size())
                    {
                        ++gap1_reconnect_invalid;
                        continue;
                    }

                    const long unsigned int old_track_id =
                        gap1_old_track_ids[
                            static_cast<std::size_t>(
                                match.previous_index)];

                    if(old_track_id >=
                       mvFisheyeLineTrackLengths.size())
                    {
                        ++gap1_reconnect_invalid;
                        continue;
                    }

                    const unsigned int
                        old_track_length_for_guard =
                            mvFisheyeLineTrackLengths[
                                static_cast<std::size_t>(
                                    old_track_id)];

                    // P9-C3h: short-track adaptive guard.
                    // Tracks with only one or two observations
                    // require stronger geometric evidence.
                    const double gap1_acceptance_threshold =
                        old_track_length_for_guard <= 2
                            ? 0.09
                            : 0.10;

                    if(match.total_cost >
                       gap1_acceptance_threshold)
                    {
                        if(old_track_length_for_guard <= 2 &&
                           match.total_cost <= 0.10)
                        {
                            ++gap1_short_guard_rejected;
                        }

                        continue;
                    }

                    ++gap1_strong_candidates;

                    const std::size_t current_index =
                        gap1_current_original_indices[
                            static_cast<std::size_t>(
                                match.current_index)];

                    if(current_index >=
                       mvCurrentFisheyeLineTrackIds.size())
                    {
                        ++gap1_reconnect_invalid;
                        continue;
                    }

                    if(mvCurrentFisheyeLineTrackIds[
                           current_index] != invalid_track_id)
                    {
                        ++gap1_reconnect_conflicts;
                        continue;
                    }

                    if(!assigned_track_ids.insert(
                            old_track_id).second)
                    {
                        ++gap1_reconnect_conflicts;
                        continue;
                    }

                    const unsigned int
                        old_track_length_before =
                            mvFisheyeLineTrackLengths[
                                static_cast<std::size_t>(
                                    old_track_id)];

                    const unsigned int
                        old_missing_count_before =
                            mvFisheyeLineTrackMissingFrameCounts[
                                static_cast<std::size_t>(
                                    old_track_id)];

                    gap1_cost_sum += match.total_cost;
                    gap1_angle_sum += match.angle_deg;
                    gap1_midpoint_sum +=
                        match.midpoint_distance_px;
                    gap1_endpoint_sum +=
                        match.endpoint_distance_px;
                    gap1_sample_sum +=
                        match.sample_distance_px;
                    gap1_length_ratio_sum +=
                        match.length_ratio;

                    gap1_cost_max =
                        std::max(
                            gap1_cost_max,
                            match.total_cost);

                    gap1_angle_max =
                        std::max(
                            gap1_angle_max,
                            match.angle_deg);

                    gap1_midpoint_max =
                        std::max(
                            gap1_midpoint_max,
                            match.midpoint_distance_px);

                    gap1_endpoint_max =
                        std::max(
                            gap1_endpoint_max,
                            match.endpoint_distance_px);

                    gap1_sample_max =
                        std::max(
                            gap1_sample_max,
                            match.sample_distance_px);

                    if(old_track_length_before == 1)
                        ++gap1_old_len1;
                    else if(old_track_length_before == 2)
                        ++gap1_old_len2;
                    else if(old_track_length_before <= 5)
                        ++gap1_old_len3_to_5;
                    else if(old_track_length_before <= 10)
                        ++gap1_old_len6_to_10;
                    else
                        ++gap1_old_len_gt10;

                    if(old_missing_count_before > 0)
                        ++gap1_repeated_gap_tracks;

                    cout
                        << "[P9-C3g][Reconnect]"
                        << " frame="
                        << mCurrentFrame.mnId
                        << " old_frame="
                        << mnTwoFramesAgoFisheyeLineFrameId
                        << " old_track_id="
                        << old_track_id
                        << " current_index="
                        << current_index
                        << " old_length_before="
                        << old_track_length_before
                        << " old_missing_before="
                        << old_missing_count_before
                        << " total_cost="
                        << match.total_cost
                        << " angle_deg="
                        << match.angle_deg
                        << " midpoint_px="
                        << match.midpoint_distance_px
                        << " endpoint_px="
                        << match.endpoint_distance_px
                        << " sample_px="
                        << match.sample_distance_px
                        << " length_ratio="
                        << match.length_ratio
                        << endl;

                    mvCurrentFisheyeLineTrackIds[
                        current_index] = old_track_id;

                    ++mvFisheyeLineTrackLengths[
                        static_cast<std::size_t>(
                            old_track_id)];

                    ++mvFisheyeLineTrackMissingFrameCounts[
                        static_cast<std::size_t>(
                            old_track_id)];

                    mvFisheyeLineTrackLastFrames[
                        static_cast<std::size_t>(
                            old_track_id)] =
                            mCurrentFrame.mnId;

                    ++gap1_reconnected;
                }
            }
        }

        int new_tracks = 0;

        for(std::size_t current_index = 0;
            current_index <
                mvCurrentFisheyeLineTrackIds.size();
            ++current_index)
        {
            if(mvCurrentFisheyeLineTrackIds[
                   current_index] != invalid_track_id)
            {
                continue;
            }

            const long unsigned int new_track_id =
                mnNextFisheyeLineTrackId++;

            mvCurrentFisheyeLineTrackIds[
                current_index] = new_track_id;

            mvFisheyeLineTrackLengths.push_back(1);

            mvFisheyeLineTrackMissingFrameCounts.push_back(0);

            mvFisheyeLineTrackStartFrames.push_back(
                mCurrentFrame.mnId);

            mvFisheyeLineTrackLastFrames.push_back(
                mCurrentFrame.mnId);

            ++new_tracks;
        }

        int duplicate_current_track_ids = 0;
        std::set<long unsigned int> unique_current_track_ids;

        unsigned int max_active_track_length = 0;
        int active_length_ge_3 = 0;
        int active_length_ge_5 = 0;
        int active_length_ge_10 = 0;

        for(const long unsigned int track_id :
            mvCurrentFisheyeLineTrackIds)
        {
            if(!unique_current_track_ids.insert(
                    track_id).second)
            {
                ++duplicate_current_track_ids;
            }

            if(track_id >=
               mvFisheyeLineTrackLengths.size())
            {
                ++invalid_track_indices;
                continue;
            }

            const unsigned int track_length =
                mvFisheyeLineTrackLengths[
                    static_cast<std::size_t>(track_id)];

            max_active_track_length =
                std::max(
                    max_active_track_length,
                    track_length);

            if(track_length >= 3)
                ++active_length_ge_3;

            if(track_length >= 5)
                ++active_length_ge_5;

            if(track_length >= 10)
                ++active_length_ge_10;
        }

        int global_length_1 = 0;
        int global_length_2 = 0;
        int global_length_3_to_5 = 0;
        int global_length_6_to_10 = 0;
        int global_length_over_10 = 0;
        int global_invalid_lifecycle = 0;
        int full_span_tracks = 0;
        int tracks_with_gap = 0;
        unsigned int total_missing_frames = 0;

        for(std::size_t track_index = 0;
            track_index <
                mvFisheyeLineTrackLengths.size();
            ++track_index)
        {
            if(track_index >=
                   mvFisheyeLineTrackMissingFrameCounts.size() ||
               track_index >=
                   mvFisheyeLineTrackStartFrames.size() ||
               track_index >=
                   mvFisheyeLineTrackLastFrames.size())
            {
                ++global_invalid_lifecycle;
                continue;
            }

            const unsigned int track_length =
                mvFisheyeLineTrackLengths[track_index];

            if(track_length == 1)
                ++global_length_1;
            else if(track_length == 2)
                ++global_length_2;
            else if(track_length <= 5)
                ++global_length_3_to_5;
            else if(track_length <= 10)
                ++global_length_6_to_10;
            else
                ++global_length_over_10;

            const long unsigned int start_frame =
                mvFisheyeLineTrackStartFrames[
                    track_index];

            const long unsigned int last_frame =
                mvFisheyeLineTrackLastFrames[
                    track_index];

            if(last_frame < start_frame)
            {
                ++global_invalid_lifecycle;
                continue;
            }

            const long unsigned int observed_span =
                last_frame - start_frame + 1;

            const unsigned int missing_frame_count =
                mvFisheyeLineTrackMissingFrameCounts[
                    track_index];

            if(missing_frame_count > 0)
                ++tracks_with_gap;

            total_missing_frames += missing_frame_count;

            if(observed_span !=
               static_cast<long unsigned int>(
                   track_length + missing_frame_count))
            {
                ++global_invalid_lifecycle;
            }

            if(start_frame == 0 &&
               last_frame == mCurrentFrame.mnId &&
               track_length + missing_frame_count ==
                   mCurrentFrame.mnId + 1)
            {
                ++full_span_tracks;
            }
        }

        const int active_tracks =
            static_cast<int>(
                unique_current_track_ids.size());

        const int ended_tracks =
            static_cast<int>(
                mvFisheyeLineTrackLengths.size()) -
            active_tracks;

        // P9-C3e: diagnose tracks that were visible at t-2,
        // absent at t-1, and appear as a new track at t.
        // This block never changes or merges track identities.
        int gap1_old_ended = 0;
        int gap1_current_new = 0;
        int gap1_mutual = 0;
        int gap1_strong = 0;
        int gap1_medium = 0;
        int gap1_weak = 0;
        int gap1_reconnectable_old_len1 = 0;
        int gap1_reconnectable_old_len2 = 0;
        int gap1_reconnectable_old_len3_to_5 = 0;
        int gap1_reconnectable_old_len6_to_10 = 0;
        int gap1_reconnectable_old_len_gt10 = 0;
        int gap1_invalid_indices = 0;

        if(mbHasTwoFramesAgoFisheyeLineObservations &&
           mbHasPreviousFisheyeLineObservations)
        {
            std::set<long unsigned int> previous_track_ids(
                mvPreviousFisheyeLineTrackIds.begin(),
                mvPreviousFisheyeLineTrackIds.end());

            std::vector<FisheyeLineObservation>
                ended_two_frames_ago_observations;

            std::vector<long unsigned int>
                ended_two_frames_ago_track_ids;

            for(std::size_t index = 0;
                index <
                    mvTwoFramesAgoFisheyeLineObservations.size();
                ++index)
            {
                if(index >=
                   mvTwoFramesAgoFisheyeLineTrackIds.size())
                {
                    ++gap1_invalid_indices;
                    continue;
                }

                const long unsigned int track_id =
                    mvTwoFramesAgoFisheyeLineTrackIds[index];

                if(previous_track_ids.count(track_id) != 0)
                    continue;

                ended_two_frames_ago_observations.push_back(
                    mvTwoFramesAgoFisheyeLineObservations[index]);

                ended_two_frames_ago_track_ids.push_back(
                    track_id);
            }

            std::vector<FisheyeLineObservation>
                current_new_observations;

            std::vector<long unsigned int>
                current_new_track_ids;

            for(std::size_t index = 0;
                index <
                    mCurrentFrame.mvFisheyeLineObservations.size();
                ++index)
            {
                if(index >=
                   mvCurrentFisheyeLineTrackIds.size())
                {
                    ++gap1_invalid_indices;
                    continue;
                }

                const long unsigned int track_id =
                    mvCurrentFisheyeLineTrackIds[index];

                if(track_id >=
                   mvFisheyeLineTrackStartFrames.size())
                {
                    ++gap1_invalid_indices;
                    continue;
                }

                if(mvFisheyeLineTrackStartFrames[
                       static_cast<std::size_t>(track_id)] !=
                   mCurrentFrame.mnId)
                {
                    continue;
                }

                current_new_observations.push_back(
                    mCurrentFrame.mvFisheyeLineObservations[
                        index]);

                current_new_track_ids.push_back(
                    track_id);
            }

            gap1_old_ended =
                static_cast<int>(
                    ended_two_frames_ago_observations.size());

            gap1_current_new =
                static_cast<int>(
                    current_new_observations.size());

            if(!ended_two_frames_ago_observations.empty() &&
               !current_new_observations.empty())
            {
                const FisheyeLineShadowMatcher::Result
                    gap1_result =
                        mpFisheyeLineShadowMatcher->
                            MatchAdjacentFrames(
                                ended_two_frames_ago_observations,
                                current_new_observations);

                gap1_mutual =
                    gap1_result.mutual_match_count;

                gap1_invalid_indices +=
                    gap1_result.nan_inf_count +
                    gap1_result.duplicate_previous_count +
                    gap1_result.duplicate_current_count;

                for(const auto& match :
                    gap1_result.matches)
                {
                    if(match.previous_index < 0 ||
                       match.current_index < 0 ||
                       static_cast<std::size_t>(
                           match.previous_index) >=
                           ended_two_frames_ago_track_ids.size() ||
                       static_cast<std::size_t>(
                           match.current_index) >=
                           current_new_track_ids.size())
                    {
                        ++gap1_invalid_indices;
                        continue;
                    }

                    if(match.total_cost <= 0.10)
                        ++gap1_strong;
                    else if(match.total_cost <= 0.12)
                        ++gap1_medium;
                    else
                        ++gap1_weak;

                    const long unsigned int old_track_id =
                        ended_two_frames_ago_track_ids[
                            static_cast<std::size_t>(
                                match.previous_index)];

                    if(old_track_id >=
                       mvFisheyeLineTrackLengths.size())
                    {
                        ++gap1_invalid_indices;
                        continue;
                    }

                    const unsigned int old_length =
                        mvFisheyeLineTrackLengths[
                            static_cast<std::size_t>(
                                old_track_id)];

                    // Only strong and medium associations are
                    // considered reconnectable.
                    if(match.total_cost <= 0.12)
                    {
                        if(old_length == 1)
                            ++gap1_reconnectable_old_len1;
                        else if(old_length == 2)
                            ++gap1_reconnectable_old_len2;
                        else if(old_length <= 5)
                            ++gap1_reconnectable_old_len3_to_5;
                        else if(old_length <= 10)
                            ++gap1_reconnectable_old_len6_to_10;
                        else
                            ++gap1_reconnectable_old_len_gt10;
                    }
                }
            }

            cout
                << "[P9-C3e][Frame "
                << mCurrentFrame.mnId
                << "] pair="
                << mnTwoFramesAgoFisheyeLineFrameId
                << "->"
                << mCurrentFrame.mnId
                << " missing_frames=1"
                << " old_ended="
                << gap1_old_ended
                << " current_new="
                << gap1_current_new
                << " mutual="
                << gap1_mutual
                << " strong="
                << gap1_strong
                << " medium="
                << gap1_medium
                << " weak="
                << gap1_weak
                << " reconnectable="
                << gap1_strong + gap1_medium
                << " reconnect_old_len1="
                << gap1_reconnectable_old_len1
                << " reconnect_old_len2="
                << gap1_reconnectable_old_len2
                << " reconnect_old_len3_5="
                << gap1_reconnectable_old_len3_to_5
                << " reconnect_old_len6_10="
                << gap1_reconnectable_old_len6_to_10
                << " reconnect_old_len_gt10="
                << gap1_reconnectable_old_len_gt10
                << " invalid="
                << gap1_invalid_indices
                << endl;
        }
        else
        {
            cout
                << "[P9-C3e][Frame "
                << mCurrentFrame.mnId
                << "] history=INSUFFICIENT"
                << endl;
        }

        const double gap1_quality_denominator =
            gap1_reconnected > 0
                ? static_cast<double>(
                      gap1_reconnected)
                : 1.0;

        cout
            << "[P9-C3g][Frame "
            << mCurrentFrame.mnId
            << "] accepted="
            << gap1_reconnected
            << " cost_mean="
            << gap1_cost_sum /
                   gap1_quality_denominator
            << " cost_max="
            << gap1_cost_max
            << " angle_mean="
            << gap1_angle_sum /
                   gap1_quality_denominator
            << " angle_max="
            << gap1_angle_max
            << " midpoint_mean="
            << gap1_midpoint_sum /
                   gap1_quality_denominator
            << " midpoint_max="
            << gap1_midpoint_max
            << " endpoint_mean="
            << gap1_endpoint_sum /
                   gap1_quality_denominator
            << " endpoint_max="
            << gap1_endpoint_max
            << " sample_mean="
            << gap1_sample_sum /
                   gap1_quality_denominator
            << " sample_max="
            << gap1_sample_max
            << " length_ratio_mean="
            << gap1_length_ratio_sum /
                   gap1_quality_denominator
            << " old_len1="
            << gap1_old_len1
            << " old_len2="
            << gap1_old_len2
            << " old_len3_5="
            << gap1_old_len3_to_5
            << " old_len6_10="
            << gap1_old_len6_to_10
            << " old_len_gt10="
            << gap1_old_len_gt10
            << " repeated_gap="
            << gap1_repeated_gap_tracks
            << endl;

        cout
            << "[P9-C3f][Frame "
            << mCurrentFrame.mnId
            << "] strong_candidates="
            << gap1_strong_candidates
            << " reconnected="
            << gap1_reconnected
            << " short_guard_rejected="
            << gap1_short_guard_rejected
            << " conflicts="
            << gap1_reconnect_conflicts
            << " invalid="
            << gap1_reconnect_invalid
            << " new_tracks_after_reconnect="
            << new_tracks
            << " tracks_with_gap="
            << tracks_with_gap
            << " total_missing_frames="
            << total_missing_frames
            << endl;

        const double continuation_ratio =
            mvCurrentFisheyeLineTrackIds.empty()
                ? 0.0
                : static_cast<double>(
                      continued_tracks) /
                  static_cast<double>(
                      mvCurrentFisheyeLineTrackIds.size());

        cout
            << "[P9-C3c][Frame "
            << mCurrentFrame.mnId
            << "] lines="
            << mvCurrentFisheyeLineTrackIds.size()
            << " continued="
            << continued_tracks
            << " new_tracks="
            << new_tracks
            << " strong_continued="
            << strong_continued_tracks
            << " medium_continued="
            << medium_continued_tracks
            << " weak_rejected="
            << weak_rejected_matches
            << " continuation_ratio="
            << continuation_ratio
            << " active_ge3="
            << active_length_ge_3
            << " active_ge5="
            << active_length_ge_5
            << " active_ge10="
            << active_length_ge_10
            << " max_active_length="
            << max_active_track_length
            << " total_tracks="
            << mnNextFisheyeLineTrackId
            << " duplicate_assignments="
            << duplicate_track_assignments
            << " duplicate_current_ids="
            << duplicate_current_track_ids
            << " invalid_indices="
            << invalid_track_indices
            << " global_len1="
            << global_length_1
            << " global_len2="
            << global_length_2
            << " global_len3_5="
            << global_length_3_to_5
            << " global_len6_10="
            << global_length_6_to_10
            << " global_len_gt10="
            << global_length_over_10
            << " active_tracks="
            << active_tracks
            << " ended_tracks="
            << ended_tracks
            << " full_span_tracks="
            << full_span_tracks
            << " invalid_lifecycle="
            << global_invalid_lifecycle
            << " tracks_with_gap="
            << tracks_with_gap
            << " total_missing_frames="
            << total_missing_frames
            << endl;

        if(mbHasPreviousFisheyeLineObservations)
        {
            mvTwoFramesAgoFisheyeLineObservations =
                mvPreviousFisheyeLineObservations;

            mvTwoFramesAgoFisheyeLineTrackIds =
                mvPreviousFisheyeLineTrackIds;

            mnTwoFramesAgoFisheyeLineFrameId =
                mnPreviousFisheyeLineFrameId;

            mbHasTwoFramesAgoFisheyeLineObservations =
                true;
        }

        for(std::size_t index = 0;
            index < mvCurrentFisheyeLineTrackIds.size() &&
            index < mCurrentFrame.mvFisheyeLineObservations.size();
            ++index)
        {
            mCurrentFrame.SetNativeFisheyeLineTrackId(
                index, mvCurrentFisheyeLineTrackIds[index]);
        }

        mvPreviousFisheyeLineObservations =
            mCurrentFrame.mvFisheyeLineObservations;

        mvPreviousFisheyeLineTrackIds =
            mvCurrentFisheyeLineTrackIds;

        mnPreviousFisheyeLineFrameId =
            mCurrentFrame.mnId;

        mbHasPreviousFisheyeLineObservations =
            true;
    }

    TrackWithPL();

    mLastNativeLoopFrameDiagnostics.tracking_state = mState;
    mLastNativeLoopFrameDiagnostics.tracking_state_after = mState;
    if(mpReferenceKF)
        mLastNativeLoopFrameDiagnostics.reference_keyframe_id =
            mpReferenceKF->mnId;
    mLastNativeLoopFrameDiagnostics.local_map_points =
        static_cast<int>(mvpLocalMapPoints.size());
    mLastNativeLoopFrameDiagnostics.native_observations =
        mCurrentFrame.NFisheyeLines;
    mLastNativeLoopFrameDiagnostics.local_maplines =
        static_cast<int>(mvpLocalMapLines.size());
    if(mpMap)
    {
        mLastNativeLoopFrameDiagnostics.formal_maplines =
            mpMap->MapLinesInMap();
        mLastNativeLoopFrameDiagnostics.map_points_total =
            mpMap->MapPointsInMap();
        mLastNativeLoopFrameDiagnostics.pending_tracks =
            static_cast<int>(
                mpMap->NativeLineBridge().
                    PendingObservationCount());
    }

    std::set<long unsigned int> active_track_ids;
    for(const FisheyeLineObservation& observation :
        mCurrentFrame.mvFisheyeLineObservations)
    {
        if(observation.temporal_track_id !=
           FisheyeLineObservation::kInvalidTrackId)
        {
            active_track_ids.insert(observation.temporal_track_id);
        }
    }
    mLastNativeLoopFrameDiagnostics.active_tracks =
        static_cast<int>(active_track_ids.size());

    int projected_maplines = 0;
    int matched_maplines = 0;
    for(size_t index = 0;
        index < mCurrentFrame.mvFisheyeLineProjectionDiagnostics.size();
        ++index)
    {
        const NativeFisheyeLineProjectionDiagnostic& diagnostic =
            mCurrentFrame.mvFisheyeLineProjectionDiagnostics[index];
        if(diagnostic.visible)
            ++projected_maplines;
        if(diagnostic.matched)
            ++matched_maplines;
    }
    mLastNativeLoopFrameDiagnostics.projected_maplines =
        projected_maplines;
    mLastNativeLoopFrameDiagnostics.matched_maplines =
        matched_maplines;

    if(mLastNativeLoopFrameDiagnostics.fallback_reason.empty())
    {
        if(!mbEnableNativeFisheyeLines)
            mLastNativeLoopFrameDiagnostics.fallback_reason =
                "NATIVE_DISABLED";
        else if(!mbEnablePointLineTracking)
            mLastNativeLoopFrameDiagnostics.fallback_reason =
                "POINT_LINE_TRACKING_DISABLED";
        else
            mLastNativeLoopFrameDiagnostics.fallback_reason =
                "NOT_ATTEMPTED";
    }
    return mCurrentFrame.mTcw.clone();
}

void Tracking::TrackWithPL()
{
    mLastNativeLoopFrameDiagnostics.tracking_track_called = true;
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }
    bool bIni_Manhattan = true;
    bool bManhattanGood = false;
    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    if(mState==NOT_INITIALIZED)
    {
        MonocularInitialization();
        mpFrameDrawer->Update(this);
        if(mState!=OK)
        {
            mLastNativeLoopFrameDiagnostics.early_return = true;
            return;
        }
    }
    else  //after initialization
    {
        if(mbUseKb8PointObservations && mState == OK &&
           ControlledLostFrameRequested(mCurrentFrame.mnId))
        {
            const eTrackingState before_state = mState;
            mState = LOST;
            mLastNativeLoopFrameDiagnostics.fallback_reason =
                "KB8_CONTROLLED_RELOCALIZATION_LOST";
            AppendRelocalizationCsv(
                "formal_lost_recovery.csv",
                "frame,lost_reason,relocalization_called,solver,recovered",
                std::to_string(mCurrentFrame.mnId) +
                    ",KB8_CONTROLLED_RELOCALIZATION_LOST,false,NOT_SELECTED,false");
            AppendRelocalizationCsv(
                "relocalization_solver_dispatch.csv",
                "frame,state_before,solver_selected,candidate_kf,success",
                std::to_string(mCurrentFrame.mnId) + "," +
                    TrackingStateName(before_state) +
                    ",CONTROLLED_LOST_INJECTION,-1,true");
        }

        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // mapping and tracking
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

            if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
            {
                mLastNativeLoopFrameDiagnostics.last_tracking_function =
                    "TrackReferenceKeyFrame";
                bOK = TrackReferenceKeyFrame();
            }
            else
            {
                mLastNativeLoopFrameDiagnostics.last_tracking_function =
                    "TrackWithMotionModel";
                bOK = TrackWithMotionModel();
                if(!bOK)
                {
                    mLastNativeLoopFrameDiagnostics.last_tracking_function =
                        "TrackReferenceKeyFrame_after_motion_model_failed";
                    bOK = TrackReferenceKeyFrame();
                }
            }
            }
            else
            {
                mLastNativeLoopFrameDiagnostics.last_tracking_function =
                    "Relocalization";
                bOK = Relocalization();
            }
        }


        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                if(mbUseKb8PointObservations)
                {
                    mLastNativeLoopFrameDiagnostics.last_tracking_function +=
                        "+TrackLocalMap";
                    bOK = TrackLocalMap();
                }
                else if(mbEnableNativeFisheyeLines &&
                   mbEnablePointLineTracking)
                    bOK = TrackLocalMap();
                else
                    bOK = TrackLocalMapWithLines();
            }
        }

        if(bOK)
            mState = OK;
        else
        {
            mState=LOST;
            if(mLastNativeLoopFrameDiagnostics.tracking_failure_reason.empty())
                mLastNativeLoopFrameDiagnostics.tracking_failure_reason =
                    mLastNativeLoopFrameDiagnostics.last_tracking_function.empty()
                    ? "tracking_failed"
                    : mLastNativeLoopFrameDiagnostics.last_tracking_function;
        }

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                if(mLastNativeLoopFrameDiagnostics.tracking_failure_reason.empty())
                    mLastNativeLoopFrameDiagnostics.tracking_failure_reason =
                        "lost_soon_after_initialisation_reset";
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }


}

void Tracking::MonocularInitialization()
{
    mLastNativeLoopFrameDiagnostics.monocular_initialization_entered = true;
    mLastNativeLoopFrameDiagnostics.initial_frame_keypoints =
        static_cast<int>(mInitialFrame.mvKeysUn.size());

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()+mCurrentFrame.mvKeylinesUn.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mLastNativeLoopFrameDiagnostics.initial_frame_id =
                mInitialFrame.mnId;
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            if(mCurrentFrame.HasKb8PointObservations())
            {
                Kb8InitializationView initial_view;
                std::string adapter_error;
                if(!BuildKb8InitializationView(
                       mCurrentFrame,
                       mKb8PointCameraParameters,
                       initial_view,
                       adapter_error) ||
                   initial_view.keypoints.size() !=
                       mCurrentFrame.mvKeysUn.size())
                {
                    mLastNativeLoopFrameDiagnostics.
                        initialization_failure_reason =
                        "kb8_initialization_view_failed:" +
                        adapter_error;
                    return;
                }
                mpInitializer =
                    new Initializer(mCurrentFrame,
                                    initial_view.keypoints,
                                    initial_view.K,
                                    1.0,
                                    200);
            }
            else
            {
                mpInitializer =  new Initializer(mCurrentFrame,1.0,200);
            }
            mLastNativeLoopFrameDiagnostics.initializer_created = true;
            mLastNativeLoopFrameDiagnostics.initial_frame_keypoints =
                static_cast<int>(mInitialFrame.mvKeysUn.size());
            mLastNativeLoopFrameDiagnostics.initializer_age = 0;
            mLastNativeLoopFrameDiagnostics.initialization_failure_reason =
                "initializer_created";

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvLineMatches.begin(),mvLineMatches.end(), make_pair(-1, -1));
            return;
        }
        mLastNativeLoopFrameDiagnostics.initialization_failure_reason =
            "not_enough_keypoints";
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvLineMatches.begin(),mvLineMatches.end(), make_pair(-1, -1));
            mLastNativeLoopFrameDiagnostics.initializer_reset = true;
            mLastNativeLoopFrameDiagnostics.initialization_failure_reason =
                "not_enough_keypoints";
            mLastNativeLoopFrameDiagnostics.reference_frame_replaced_after_failure =
                true;
            mLastNativeLoopFrameDiagnostics.reference_frame_replacement_condition =
                "current_frame_keypoints<=100";
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        const bool use_kb8_bearing_initialization =
            mInitialFrame.HasKb8PointObservations() &&
            mCurrentFrame.HasKb8PointObservations();
        mLastNativeLoopFrameDiagnostics.search_center_mode =
            use_kb8_bearing_initialization ?
            "KB8_INITIAL_FRAME_RESET" :
            "LEGACY_PROPAGATED";
        mLastNativeLoopFrameDiagnostics.search_center_reset =
            use_kb8_bearing_initialization;
        std::vector<double> search_center_shifts;
        const size_t center_count = std::min(
            mvbPrevMatched.size(),
            mInitialFrame.mvKeysUn.size());
        search_center_shifts.reserve(center_count);
        for(size_t i = 0; i < center_count; ++i)
        {
            const double dx = mvbPrevMatched[i].x -
                mInitialFrame.mvKeysUn[i].pt.x;
            const double dy = mvbPrevMatched[i].y -
                mInitialFrame.mvKeysUn[i].pt.y;
            search_center_shifts.push_back(
                std::sqrt(dx * dx + dy * dy));
        }
        mLastNativeLoopFrameDiagnostics.
            search_center_median_shift_from_initial =
            MedianDiagnosticValue(search_center_shifts);
        mLastNativeLoopFrameDiagnostics.
            search_center_p95_shift_from_initial =
            PercentileDiagnosticValue(search_center_shifts, 0.95);
        if(use_kb8_bearing_initialization)
        {
            mvbPrevMatched.resize(mInitialFrame.mvKeysUn.size());
            for(size_t i = 0; i < mInitialFrame.mvKeysUn.size(); ++i)
                mvbPrevMatched[i] = mInitialFrame.mvKeysUn[i].pt;
        }
        int nmatches = matcher.SearchForInitialization(
            mInitialFrame,
            mCurrentFrame,
            mvbPrevMatched,
            mvIniMatches,
            mnMonocularInitializationSearchWindow);
        int nlineMatches = 0;
        // POINT_ONLY frames deliberately have no line extractor or line
        // observations.  The legacy initializer must not enter the LSD
        // matcher, which assumes non-empty line descriptor containers.
        if(mPointLineMode != POINT_ONLY)
        {
            LSDmatcher lmatcher;   //建立线特征之间的匹配
            nlineMatches = lmatcher.SerachForInitialize(
                mInitialFrame, mCurrentFrame, mvLineMatches);
        }
        mLastNativeLoopFrameDiagnostics.initial_frame_keypoints =
            static_cast<int>(mInitialFrame.mvKeysUn.size());
        mLastNativeLoopFrameDiagnostics.initial_frame_id =
            mInitialFrame.mnId;
        mLastNativeLoopFrameDiagnostics.initializer_age =
            static_cast<int>(mCurrentFrame.mnId) -
            static_cast<int>(mInitialFrame.mnId);
        mLastNativeLoopFrameDiagnostics.initial_matches = nmatches;
        int valid_match_count = 0;
        std::vector<double> displacements;
        std::vector<double> parallaxes;
        for(size_t i = 0; i < mvIniMatches.size(); ++i)
        {
            const int matched_index = mvIniMatches[i];
            if(matched_index < 0 ||
               i >= mInitialFrame.mvKeysUn.size() ||
               static_cast<size_t>(matched_index) >=
                   mCurrentFrame.mvKeysUn.size())
            {
                continue;
            }
            ++valid_match_count;
            const cv::KeyPoint& initial_keypoint =
                mInitialFrame.mvKeysUn[i];
            const cv::KeyPoint& current_keypoint =
                mCurrentFrame.mvKeysUn[matched_index];
            const double dx =
                current_keypoint.pt.x - initial_keypoint.pt.x;
            const double dy =
                current_keypoint.pt.y - initial_keypoint.pt.y;
            displacements.push_back(std::sqrt(dx * dx + dy * dy));
            double parallax = -1.0;
            if(mInitialFrame.HasKb8PointObservations() &&
               mCurrentFrame.HasKb8PointObservations() &&
               i < mInitialFrame.mvPointUnitBearings.size() &&
               i < mInitialFrame.mvbPointBearingValid.size() &&
               static_cast<size_t>(matched_index) <
                   mCurrentFrame.mvPointUnitBearings.size() &&
               static_cast<size_t>(matched_index) <
                   mCurrentFrame.mvbPointBearingValid.size() &&
               mInitialFrame.mvbPointBearingValid[i] &&
               mCurrentFrame.mvbPointBearingValid[matched_index])
            {
                double cos_angle =
                    mInitialFrame.mvPointUnitBearings[i].dot(
                        mCurrentFrame.
                            mvPointUnitBearings[matched_index]);
                cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
                parallax = std::acos(cos_angle) * 180.0 / CV_PI;
            }
            else
            {
                parallax =
                    BearingParallaxDegrees(initial_keypoint,
                                           current_keypoint,
                                           mK);
            }
            if(parallax >= 0.0)
                parallaxes.push_back(parallax);
        }
        mLastNativeLoopFrameDiagnostics.valid_match_count =
            valid_match_count;
        mLastNativeLoopFrameDiagnostics.median_displacement =
            MedianDiagnosticValue(displacements);
        mLastNativeLoopFrameDiagnostics.median_parallax =
            MedianDiagnosticValue(parallaxes);

        // Check if there are enough correspondences
        if((use_kb8_bearing_initialization &&
            nmatches < mnMonocularInitializationMinMatches) ||
           (!use_kb8_bearing_initialization &&
            nmatches + nlineMatches < mnMonocularInitializationMinMatches))
        {
            mLastNativeLoopFrameDiagnostics.initialization_failure_reason =
                "not_enough_matches";
            if(use_kb8_bearing_initialization &&
               mLastNativeLoopFrameDiagnostics.initializer_age <
                   mnMonocularInitializationMaxFrameGap)
            {
                mLastNativeLoopFrameDiagnostics.
                    reference_frame_retained_after_failure = true;
                mLastNativeLoopFrameDiagnostics.
                    reference_frame_replacement_condition =
                    "kb8_not_enough_matches_keeps_existing_initializer";
            }
            else
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer*>(NULL);
                mLastNativeLoopFrameDiagnostics.initializer_reset = true;
                mLastNativeLoopFrameDiagnostics.
                    reference_frame_replaced_after_failure = true;
                mLastNativeLoopFrameDiagnostics.
                    reference_frame_replacement_condition =
                    use_kb8_bearing_initialization ?
                    "kb8_nmaches<Initialization.MinMatches_age>=Initialization.MaxFrameGap" :
                    "legacy_nmatches+nlineMatches<Initialization.MinMatches";
            }
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        vector<bool> mvbLineTriangulated;   //匹配的线特征是否能够三角化
        const bool initialized =
            use_kb8_bearing_initialization ?
            mpInitializer->InitializeKb8Bearing(
                mCurrentFrame, mvIniMatches, Rcw, tcw,
                mvIniP3D, vbTriangulated) :
            mpInitializer->Initialize(
                mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D,
                vbTriangulated, mvLineMatches, mvLineS3D, mvLineE3D,
                mvbLineTriangulated);
        mLastNativeLoopFrameDiagnostics.legacy_solver_success =
            mpInitializer->LastLegacySolverSuccess();
        mLastNativeLoopFrameDiagnostics.kb8_postvalidation_success =
            use_kb8_bearing_initialization ?
            mpInitializer->LastKb8ValidationSuccess() :
            initialized;
        mLastNativeLoopFrameDiagnostics.initialization_final_success =
            initialized;
        mLastNativeLoopFrameDiagnostics.initialization_candidate_count =
            mpInitializer->LastCandidateCount();
        mLastNativeLoopFrameDiagnostics.initialization_best_candidate_id =
            mpInitializer->LastBestCandidateId();
        mLastNativeLoopFrameDiagnostics.initialization_best_good_points =
            mpInitializer->LastBestGoodPoints();
        mLastNativeLoopFrameDiagnostics.initialization_second_best_good_points =
            mpInitializer->LastSecondBestGoodPoints();
        mLastNativeLoopFrameDiagnostics.
            initialization_best_positive_depth_cam1 =
            mpInitializer->LastBestPositiveDepthCam1();
        mLastNativeLoopFrameDiagnostics.
            initialization_best_positive_depth_cam2 =
            mpInitializer->LastBestPositiveDepthCam2();
        mLastNativeLoopFrameDiagnostics.
            initialization_best_both_positive =
            mpInitializer->LastBestBothPositive();
        mLastNativeLoopFrameDiagnostics.initialization_best_triangulated =
            mpInitializer->LastBestTriangulated();
        mLastNativeLoopFrameDiagnostics.
            initialization_best_reprojection_pass =
            mpInitializer->LastBestReprojectionPass();
        mLastNativeLoopFrameDiagnostics.
            initialization_best_median_parallax_deg =
            mpInitializer->LastBestMedianParallaxDeg();
        mLastNativeLoopFrameDiagnostics.
            initialization_required_min_good_points =
            mpInitializer->LastRequiredMinGoodPoints();
        mLastNativeLoopFrameDiagnostics.
            initialization_required_min_parallax_deg =
            mpInitializer->LastRequiredMinParallaxDeg();
        mLastNativeLoopFrameDiagnostics.
            initialization_ambiguity_ratio_or_threshold =
            mpInitializer->LastAmbiguityRatioOrThreshold();
        mLastNativeLoopFrameDiagnostics.
            initialization_exact_failure_branch =
            mpInitializer->LastExactFailureBranch();
        mLastNativeLoopFrameDiagnostics.initialization_pose_candidates =
            mpInitializer->LastPoseCandidateDiagnostics();
        if(initialized)
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }
            if(use_kb8_bearing_initialization)
            {
                mvLineS3D.clear();
                mvLineE3D.clear();
                mvbLineTriangulated.assign(mvLineMatches.size(), false);
                for(size_t i=0, iend=mvLineMatches.size(); i<iend; ++i)
                    mvLineMatches[i] = make_pair(-1, -1);
                nlineMatches = 0;
            }
            for(size_t i=0, iend=mvLineMatches.size(); i<iend;i++)
            {
                if(mvLineMatches[i].second>=0 && !mvbLineTriangulated[i])
                {
                    mvLineMatches[i].second = -1;
                    nlineMatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);
            CreateInitialMapMonoWithPL();
        }
        else
        {
            mLastNativeLoopFrameDiagnostics.initialization_failure_reason =
                mpInitializer->LastFailureReason();
            mLastNativeLoopFrameDiagnostics.initializer_model =
                mpInitializer->LastModel();
            mLastNativeLoopFrameDiagnostics.initializer_h_score =
                mpInitializer->LastHomographyScore();
            mLastNativeLoopFrameDiagnostics.initializer_f_score =
                mpInitializer->LastFundamentalScore();
            mLastNativeLoopFrameDiagnostics.initializer_h_ratio =
                mpInitializer->LastHomographyRatio();
            if(mLastNativeLoopFrameDiagnostics.
                   initialization_failure_reason.empty())
                mLastNativeLoopFrameDiagnostics.
                    initialization_failure_reason = "unknown";
            mLastNativeLoopFrameDiagnostics.
                reference_frame_retained_after_failure = true;
            mLastNativeLoopFrameDiagnostics.
                reference_frame_replacement_condition =
                "solver_failure_keeps_existing_initializer";
            if(use_kb8_bearing_initialization &&
               mLastNativeLoopFrameDiagnostics.initializer_age >=
                   mnMonocularInitializationMaxFrameGap)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer*>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                fill(mvLineMatches.begin(), mvLineMatches.end(),
                     make_pair(-1, -1));
                mLastNativeLoopFrameDiagnostics.initializer_reset = true;
                mLastNativeLoopFrameDiagnostics.
                    reference_frame_replaced_after_failure = true;
                mLastNativeLoopFrameDiagnostics.
                    reference_frame_retained_after_failure = false;
                mLastNativeLoopFrameDiagnostics.
                    reference_frame_replacement_condition =
                    "kb8_solver_failure_age>=Initialization.MaxFrameGap";
            }
        }
    }
}

cv::Mat Tracking::SeekManhattanFrame(vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection)
{

    vector<cv::Mat> vRotaionMatrix;
    vector<cv::Mat> vRotaionMatrix_good;
    cv::RNG rnger(cv::getTickCount());
    vector<cv::Mat> vSN_good;
    vector<double> lambda_good;
    vector<cv::Point2d> m_j_selected;
    // R_cm_update matrix
    cv::Mat R_cm_update=cv::Mat::eye(cv::Size(3,3),CV_32F);
    cv::Mat R_cm_new=cv::Mat::eye(cv::Size(3,3),CV_32F);
// initialization with random matrix
#if 1
    cv::Mat qu = cv::Mat::zeros(cv::Size(4,1),CV_32F);
    rnger.fill(qu, cv::RNG::UNIFORM, cv::Scalar::all(0.01), cv::Scalar::all(1));
    Eigen::Quaterniond qnorm;
    Eigen::Quaterniond q(qu.at<float>(0,0),qu.at<float>(1,0),qu.at<float>(2,0),qu.at<float>(3,0));//=Eigen::MatrixXd::Random(1, 4);
    qnorm.x()=q.x()/q.norm();qnorm.y()=q.y()/q.norm();
    qnorm.z()=q.z()/q.norm();qnorm.w()=q.w()/q.norm();
    cv::eigen2cv(qnorm.matrix(),R_cm_update);//	eigen2cv(m, img);;*/
    //cout<<R_cm_update<<endl;
    cv::SVD svd; cv::Mat U,W,VT;
    svd.compute(R_cm_update,W,U,VT);
    R_cm_update=U*VT;
    //cout<<000<<R_cm_update<<endl;
    // R_cm_Rec matrix
    cv::Mat R_cm_Rec=cv::Mat::zeros(cv::Size(3,3),CV_32F);

    cv::Mat R_cm_initial;
    int  validMF=0;
    //cout<<R_cm_update<<endl;
    R_cm_new.at<float>(0,0) = R_cm_update.at<double>(0,0);
    R_cm_new.at<float>(0,1) = R_cm_update.at<double>(0,1);
    R_cm_new.at<float>(0,2) = R_cm_update.at<double>(0,2);
    R_cm_new.at<float>(1,0) = R_cm_update.at<double>(1,0);
    R_cm_new.at<float>(1,1) = R_cm_update.at<double>(1,1);
    R_cm_new.at<float>(1,2) = R_cm_update.at<double>(1,2);
    R_cm_new.at<float>(2,0) = R_cm_update.at<double>(2,0);
    R_cm_new.at<float>(2,1) = R_cm_update.at<double>(2,1);
    R_cm_new.at<float>(2,2) = R_cm_update.at<double>(2,2);
    //cout<<R_cm_new<<endl;
    //matTemp.convertTo(MatTemp2, CV_8U)
#endif
    //cout<<R_cm_update<<endl;
    R_cm_new = TrackManhattanFrame(R_cm_new, vTempSurfaceNormal,vVanishingDirection);
    return R_cm_new;//vRotaionMatrix_good[0];
}

ResultOfMS Tracking::ProjectSN2MF(int a,const cv::Mat &R_mc,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection,const int numOfSN)
{
    vector<cv::Point2d> m_j_selected;
    cv::Mat R_cm_Rec=cv::Mat::zeros(cv::Size(1,3),CV_32F);
    cv::Mat R_cm_NULL=cv::Mat::zeros(cv::Size(1,3),CV_32F);
    ResultOfMS RandDen;
    RandDen.axis=a;

    size_t sizeOfSurfaceNormal=vTempSurfaceNormal.size()+vVanishingDirection.size();
    m_j_selected.reserve(sizeOfSurfaceNormal);

    for(size_t i=0;i<sizeOfSurfaceNormal;i++)
    {
        //cv::Mat temp=cv::Mat::zeros(cv::Size(1,3),CV_32F);

        cv::Point3f n_ini;
        int tepSize=i-vTempSurfaceNormal.size();
        if(i>=vTempSurfaceNormal.size())
        {


            n_ini.x = R_mc.at<float>(0,0) * vVanishingDirection[tepSize].direction.x + R_mc.at<float>(0,1) * vVanishingDirection[tepSize].direction.y + R_mc.at<float>(0,2) * vVanishingDirection[tepSize].direction.z;
            n_ini.y = R_mc.at<float>(1,0) * vVanishingDirection[tepSize].direction.x + R_mc.at<float>(1,1) * vVanishingDirection[tepSize].direction.y + R_mc.at<float>(1,2) * vVanishingDirection[tepSize].direction.z;
            n_ini.z = R_mc.at<float>(2,0) * vVanishingDirection[tepSize].direction.x + R_mc.at<float>(2,1) * vVanishingDirection[tepSize].direction.y + R_mc.at<float>(2,2) * vVanishingDirection[tepSize].direction.z;
        }
        else{

            n_ini.x = R_mc.at<float>(0,0) * vTempSurfaceNormal[i].normal.x + R_mc.at<float>(0,1) * vTempSurfaceNormal[i].normal.y + R_mc.at<float>(0,2) * vTempSurfaceNormal[i].normal.z;
            n_ini.y = R_mc.at<float>(1,0) * vTempSurfaceNormal[i].normal.x + R_mc.at<float>(1,1) * vTempSurfaceNormal[i].normal.y + R_mc.at<float>(1,2) * vTempSurfaceNormal[i].normal.z;
            n_ini.z = R_mc.at<float>(2,0) * vTempSurfaceNormal[i].normal.x + R_mc.at<float>(2,1) * vTempSurfaceNormal[i].normal.y + R_mc.at<float>(2,2) * vTempSurfaceNormal[i].normal.z;
        }


        double lambda=sqrt(n_ini.x*n_ini.x+n_ini.y*n_ini.y);//at(k).y*n_a.at(k).y);
        //cout<<lambda<<endl;
        //inside the cone
        if(lambda<sin(0.2518)) //0.25
        {
            double tan_alfa=lambda/std::abs(n_ini.z);
            double alfa=asin(lambda);
            double m_j_x=alfa/tan_alfa*n_ini.x/n_ini.z;
            double m_j_y=alfa/tan_alfa*n_ini.y/n_ini.z;
            if(!std::isnan(m_j_x)&&!std::isnan(m_j_y))
                m_j_selected.push_back(cv::Point2d(m_j_x,m_j_y));
            if(i<vTempSurfaceNormal.size())
            {
                if(a==1)
                {
                    mCurrentFrame.vSurfaceNormalx.push_back(vTempSurfaceNormal[i].FramePosition);
                    mCurrentFrame.vSurfacePointx.push_back(vTempSurfaceNormal[i].cameraPosition);
                }
                else if(a==2)
                {
                    mCurrentFrame.vSurfaceNormaly.push_back(vTempSurfaceNormal[i].FramePosition);
                    mCurrentFrame.vSurfacePointy.push_back(vTempSurfaceNormal[i].cameraPosition);
                }
                else if(a==3)
                {
                    mCurrentFrame.vSurfaceNormalz.push_back(vTempSurfaceNormal[i].FramePosition);
                    mCurrentFrame.vSurfacePointz.push_back(vTempSurfaceNormal[i].cameraPosition);
                }
            }
            else
            {
                if(a==1)
                {
                    cv::Point2d endPoint=vVanishingDirection[tepSize].p;
                    cv::Point2d startPoint=vVanishingDirection[tepSize].q;
                    vector<cv::Point2d> pointPair(2);
                    pointPair.push_back(endPoint);pointPair.push_back(startPoint);
                    mCurrentFrame.vVanishingLinex.push_back(pointPair);
                    for(int k=0;k<vVanishingDirection[tepSize].rndpts3d.size();k++)
                        mCurrentFrame.vVaishingLinePCx.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                }
                else if(a==2)
                {
                    cv::Point2d endPoint=vVanishingDirection[tepSize].p;
                    cv::Point2d startPoint=vVanishingDirection[tepSize].q;
                    vector<cv::Point2d> pointPair(2);
                    pointPair.push_back(endPoint);pointPair.push_back(startPoint);
                    mCurrentFrame.vVanishingLiney.push_back(pointPair);
                    for(int k=0;k<vVanishingDirection[tepSize].rndpts3d.size();k++)
                        mCurrentFrame.vVaishingLinePCy.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                }
                else if(a==3)
                {
                    cv::Point2d endPoint=vVanishingDirection[tepSize].p;
                    cv::Point2d startPoint=vVanishingDirection[tepSize].q;
                    vector<cv::Point2d> pointPair(2);
                    pointPair.push_back(endPoint);pointPair.push_back(startPoint);
                    mCurrentFrame.vVanishingLinez.push_back(pointPair);
                    for(int k=0;k<vVanishingDirection[tepSize].rndpts3d.size();k++)
                        mCurrentFrame.vVaishingLinePCz.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                }
            }


        }
    }
    //cout<<"a=1:"<<mCurrentFrame.vSurfaceNormalx.size()<<",a =2:"<<mCurrentFrame.vSurfaceNormaly.size()<<", a=3:"<<mCurrentFrame.vSurfaceNormalz.size()<<endl;
    //cout<<"m_j_selected.push_back(temp)"<<m_j_selected.size()<<endl;

    if(m_j_selected.size()>numOfSN)
    {
        sMS tempMeanShift = MeanShift(m_j_selected);
        cv::Point2d s_j = tempMeanShift.centerOfShift;// MeanShift(m_j_selected);
        float s_j_density=tempMeanShift.density;
        //cout<<"tracking:s_j"<<s_j.x<<","<<s_j.y<<endl;
        float alfa=norm(s_j);
        float ma_x=tan(alfa)/alfa*s_j.x;
        float ma_y=tan(alfa)/alfa*s_j.y;
        cv::Mat temp1=cv::Mat::zeros(cv::Size(1,3),CV_32F);
        temp1.at<float>(0,0)=ma_x;
        temp1.at<float>(1,0)=ma_y;
        temp1.at<float>(2,0)=1;
        cv::Mat rtemp=R_mc.t();
        R_cm_Rec=rtemp*temp1;
        R_cm_Rec=R_cm_Rec/norm(R_cm_Rec); //列向量
        RandDen.R_cm_Rec=R_cm_Rec;
        RandDen.s_j_density=s_j_density;

        return RandDen;
    }
    RandDen.R_cm_Rec=R_cm_NULL;
    return RandDen;

}

axiSNV Tracking::ProjectSN2Conic(int a,const cv::Mat &R_mc,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection)
{
    int numInConic=0;
    vector<cv::Point2d> m_j_selected;
    cv::Mat R_cm_Rec=cv::Mat::zeros(cv::Size(3,3),CV_32F);
    cv::Mat R_cm_NULL=cv::Mat::zeros(cv::Size(1,3),CV_32F);
    //cv::Mat R_mc=cv::Mat::zeros(cv::Size(3,3),CV_32F);
    vector<SurfaceNormal> vSNCadidate;
    axiSNV tempaxiSNV;
    tempaxiSNV.axis=a;

    size_t sizeOfSurfaceNormal=vTempSurfaceNormal.size();//TODO 先去掉线 +vVanishingDirection.size();
    tempaxiSNV.SNVector.reserve(sizeOfSurfaceNormal);
    //cout<<"size of SN"<<sizeOfSurfaceNormal<<endl;
    for(size_t i=0;i<sizeOfSurfaceNormal;i++)
    {
        cv::Point3f n_ini;
        if(i<vTempSurfaceNormal.size())
        {
            n_ini.x = R_mc.at<float>(0,0) * vTempSurfaceNormal[i].normal.x + R_mc.at<float>(0,1) * vTempSurfaceNormal[i].normal.y + R_mc.at<float>(0,2) * vTempSurfaceNormal[i].normal.z;
            n_ini.y = R_mc.at<float>(1,0) * vTempSurfaceNormal[i].normal.x + R_mc.at<float>(1,1) * vTempSurfaceNormal[i].normal.y + R_mc.at<float>(1,2) * vTempSurfaceNormal[i].normal.z;
            n_ini.z = R_mc.at<float>(2,0) * vTempSurfaceNormal[i].normal.x + R_mc.at<float>(2,1) * vTempSurfaceNormal[i].normal.y + R_mc.at<float>(2,2) * vTempSurfaceNormal[i].normal.z;

            double lambda=sqrt(n_ini.x*n_ini.x+n_ini.y*n_ini.y);//at(k).y*n_a.at(k).y);
            //cout<<lambda<<endl;
            if(lambda<sin(0.2518)) //0.25
            {
                //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                //numInConic++;
                tempaxiSNV.SNVector.push_back(vTempSurfaceNormal[i]);
            }
        }
        else
        {   //cout<<"vanishing"<<endl;
            int tepSize=i-vTempSurfaceNormal.size();
            //cout<<vVanishingDirection[tepSize].direction.x<<"vanishing"<<endl;

            n_ini.x = R_mc.at<float>(0,0) * vVanishingDirection[tepSize].direction.x + R_mc.at<float>(0,1) * vVanishingDirection[tepSize].direction.y + R_mc.at<float>(0,2) * vVanishingDirection[tepSize].direction.z;
            n_ini.y = R_mc.at<float>(1,0) * vVanishingDirection[tepSize].direction.x + R_mc.at<float>(1,1) * vVanishingDirection[tepSize].direction.y + R_mc.at<float>(1,2) * vVanishingDirection[tepSize].direction.z;
            n_ini.z = R_mc.at<float>(2,0) * vVanishingDirection[tepSize].direction.x + R_mc.at<float>(2,1) * vVanishingDirection[tepSize].direction.y + R_mc.at<float>(2,2) * vVanishingDirection[tepSize].direction.z;

            double lambda=sqrt(n_ini.x*n_ini.x+n_ini.y*n_ini.y);//at(k).y*n_a.at(k).y);
            //cout<<lambda<<endl;
            if(lambda<sin(0.1518)) //0.25
            {
                //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                //numInConic++;
                tempaxiSNV.Linesvector.push_back(vVanishingDirection[tepSize]);
            }
        }
    }

    return tempaxiSNV;//numInConic;

}

cv::Mat Tracking::TrackManhattanFrame(cv::Mat &mLastRcm,vector<SurfaceNormal> &vSurfaceNormal,vector<FrameLine> &vVanishingDirection)
{

    cv::Mat R_cm_update=mLastRcm.clone();

    int isTracked = 0;
    vector<double>denTemp(3, 0.00001);
    for (int i = 0; i < 6; i++) {

        cv::Mat R_cm = R_cm_update;//cv::Mat::eye(cv::Size(3,3),CV_32FC1);  // 对角线为1的对角矩阵(3, 3, CV_32FC1);
        //cout<<"R_cm"<<R_cm<<endl;
        int directionFound1 = 0;
        int directionFound2 = 0;
        int directionFound3 = 0; //三个方向
        int numDirectionFound = 0;
        vector<axiSNVector>vaxiSNV(4);
        vector<int> numInCone = vector<int>(3, 0);
        vector<cv::Point2f> vDensity ;
        //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        for (int a = 1; a < 4; a++) {
            //在每个conic有多少 点
            cv::Mat R_mc=cv::Mat::zeros(cv::Size(3,3),CV_32F);
            int c1=(a+3)%3; int c2=(a+4)%3; int c3=(a+5)%3;
            //cout<<"R_cm a"<<R_cm<<endl;
            R_mc.at<float>(0,0)=R_cm.at<float>(0,c1);R_mc.at<float>(0,1)=R_cm.at<float>(0,c2);
            R_mc.at<float>(0,2)=R_cm.at<float>(0,c3);R_mc.at<float>(1,0)=R_cm.at<float>(1,c1);
            R_mc.at<float>(1,1)=R_cm.at<float>(1,c2);R_mc.at<float>(1,2)=R_cm.at<float>(1,c3);
            R_mc.at<float>(2,0)=R_cm.at<float>(2,c1);R_mc.at<float>(2,1)=R_cm.at<float>(2,c2);
            R_mc.at<float>(2,2)=R_cm.at<float>(2,c3);
            //cout<<"R_cm b"<<R_mc<<endl;
            cv::Mat R_mc_new=R_mc.t();
            //cout<<"R_mc_new"<<R_mc_new<<endl;
            vaxiSNV[a-1] = ProjectSN2Conic(a, R_mc_new, vSurfaceNormal,vVanishingDirection);
            numInCone[a - 1] = vaxiSNV[a-1].SNVector.size();
            //cout<<"2 a:"<<vaxiSNV[a-1].axis<<",vector:"<<numInCone[a - 1]<<endl;
        }
        //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        //cout << "first sN time: " << time_used.count() << endl;
        int minNumOfSN = vSurfaceNormal.size() / 20;
        //cout<<"minNumOfSN"<<minNumOfSN<<endl;
        //排序  a<b<c
        int a = numInCone[0];
        int b = numInCone[1];
        int c = numInCone[2];
        //cout<<"a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
        int temp = 0;
        if (a > b) temp = a, a = b, b = temp;
        if (b > c) temp = b, b = c, c = temp;
        if (a > b) temp = a, a = b, b = temp;
        //cout<<"sequence  a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
        //打印排序后的三个数
        if (b < minNumOfSN)
        {
            minNumOfSN = (b + a) / 2;
            //cout <<"thr"<<minNumOfSN<<endl;
        }

        //cout<<"new  minNumOfSN"<<minNumOfSN<<endl;
        //chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        for (int a = 1; a < 4; a++)
        {
            cv::Mat R_mc=cv::Mat::zeros(cv::Size(3,3),CV_32F);
            int c1=(a+3)%3; int c2=(a+4)%3; int c3=(a+5)%3;
            R_mc.at<float>(0,0)=R_cm.at<float>(0,c1);R_mc.at<float>(0,1)=R_cm.at<float>(0,c2);
            R_mc.at<float>(0,2)=R_cm.at<float>(0,c3);R_mc.at<float>(1,0)=R_cm.at<float>(1,c1);
            R_mc.at<float>(1,1)=R_cm.at<float>(1,c2);R_mc.at<float>(1,2)=R_cm.at<float>(1,c3);
            R_mc.at<float>(2,0)=R_cm.at<float>(2,c1);R_mc.at<float>(2,1)=R_cm.at<float>(2,c2);
            R_mc.at<float>(2,2)=R_cm.at<float>(2,c3);
            cv::Mat R_mc_new=R_mc.t();
            vector<SurfaceNormal>* tempVVSN;
            vector<FrameLine>* tempLineDirection;
            for(int i=0;i<3;i++)
            {
                if(vaxiSNV[i].axis==a)
                {

                    tempVVSN=&vaxiSNV[i].SNVector;
                    tempLineDirection=&vaxiSNV[i].Linesvector;
                    //cout<<"2 a:"<<vaxiSNV[i].axis<<",vector:"<<tempVVSN->size()<<endl;
                    break;
                }

            }
            //cout<<"1 a:"<<a<<endl;

            ResultOfMS RD_temp  = ProjectSN2MF(a, R_mc_new, *tempVVSN,*tempLineDirection, minNumOfSN);

            //chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
            //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t4-t3);
            //cout << "second SN time: " << time_used.count() << endl;

            //cout << "test projectSN2MF" << ra << endl;
            //如果 ra不为0
            if (sum(RD_temp.R_cm_Rec)[0] != 0) {
                numDirectionFound += 1;
                if (a == 1) directionFound1 = 1;//第一个轴
                else if (a == 2) directionFound2 = 1;
                else if (a == 3) directionFound3 = 1;
                R_cm_update.at<float>(0, a - 1) = RD_temp.R_cm_Rec.at<float>(0, 0);
                R_cm_update.at<float>(1, a - 1) = RD_temp.R_cm_Rec.at<float>(1, 0);
                R_cm_update.at<float>(2, a - 1) = RD_temp.R_cm_Rec.at<float>(2, 0);
                //RD_temp.s_j_density;

                vDensity.push_back(cv::Point2f(RD_temp.axis,RD_temp.s_j_density));

            }
        }
        //cout<<"numDirectionFound:"<<numDirectionFound<<endl;

        if (numDirectionFound < 2) {
            cout<<"oh, it has happened"<<endl;
            R_cm_update = R_cm;
            numDirectionFound = 0;
            isTracked = 0;
            directionFound1 = 0;
            directionFound2 = 0;
            directionFound3 = 0;
            break;
        } else if (numDirectionFound == 2) {
            if (directionFound1 && directionFound2) {
                cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                //cout << 1 << v1 << endl;
                cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                cv::Mat v3 = v1.cross(v2);
                R_cm_update.at<float>(0, 2) = v3.at<float>(0, 0);
                R_cm_update.at<float>(1, 2) = v3.at<float>(1, 0);
                R_cm_update.at<float>(2, 2) = v3.at<float>(2, 0);
                if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                    R_cm_update.at<float>(0, 2) = -v3.at<float>(0, 0);
                    R_cm_update.at<float>(1, 2) = -v3.at<float>(1, 0);
                    R_cm_update.at<float>(2, 2) = -v3.at<float>(2, 0);
                }

            } else if (directionFound2 && directionFound3) {
                cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                //cout << 2 << v2 << endl;
                cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                cv::Mat v1 = v3.cross(v2);
                R_cm_update.at<float>(0, 0) = v1.at<float>(0, 0);
                R_cm_update.at<float>(1, 0) = v1.at<float>(1, 0);
                R_cm_update.at<float>(2, 0) = v1.at<float>(2, 0);
                if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                    R_cm_update.at<float>(0, 0) = -v1.at<float>(0, 0);
                    R_cm_update.at<float>(1, 0) = -v1.at<float>(1, 0);
                    R_cm_update.at<float>(2, 0) = -v1.at<float>(2, 0);
                }
            } else if (directionFound1 && directionFound3) {
                cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                //cout << "3-v3" << v1 << endl;
                cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                //cout << "3-v3" << v3 << endl;
                cv::Mat v2 = v1.cross(v3);
                R_cm_update.at<float>(0, 1) = v2.at<float>(0, 0);
                R_cm_update.at<float>(1, 1) = v2.at<float>(1, 0);
                R_cm_update.at<float>(2, 1) = v2.at<float>(2, 0);
                if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                    R_cm_update.at<float>(0, 1) = -v2.at<float>(0, 0);
                    R_cm_update.at<float>(1, 1) = -v2.at<float>(1, 0);
                    R_cm_update.at<float>(2, 1) = -v2.at<float>(2, 0);
                }

            }
        }

        //cout << "direction:" << directionFound1 << "," << directionFound2 << "," << directionFound3 << endl;
        //cout<<"svd before"<<R_cm_update<<endl;
        SVD svd;
        cv::Mat U, W, VT;


        svd.compute(R_cm_update, W, U, VT);

        R_cm_update = U * VT;
        //cout<<"svd after"<<R_cm_update<<endl;

        //validMF=1;
        //判断是否收敛
        vDensity.clear();
        if (acos((trace(R_cm.t() * R_cm_update)[0] - 1.0)) / 2 < 0.001)
        {cout<<"go outside"<<endl;break;}
    }
    isTracked=1;
    return R_cm_update.clone();
}

sMS Tracking::MeanShift(vector<cv::Point2d> & v2D)
{
    sMS tempMS;
    int numPoint=v2D.size();
    float density;
    cv::Point2d nominator;
    double denominator=0;
    double nominator_x=0;
    double nominator_y=0;
    for(int i=0;i<numPoint;i++)
    {
        double k = exp(-20*norm(v2D.at(i))*norm(v2D.at(i)));
        nominator.x+=k*v2D.at(i).x;
        nominator.y+=k*v2D.at(i).y;
        denominator+=k;
    }
    tempMS.centerOfShift=nominator/denominator;
    tempMS.density=denominator/numPoint;

    return  tempMS;
}

void Tracking::CreateInitialMapMonoWithPL()
{

        KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
        mLastNativeLoopFrameDiagnostics.new_keyframe = true;
        mLastNativeLoopFrameDiagnostics.initial_keyframes_created = 2;
        int created_initial_mappoints = 0;
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);
        for(size_t i=0; i<mvIniMatches.size(); i++)
        {
            if(mvIniMatches[i]<0)
                continue;

            // Create MapPoint
            cv::Mat worldPos(mvIniP3D[i]);
            MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);
            pMP->UpdateNormalAndDepth();
            // Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;
            // Add to Map
            mpMap->AddMapPoint(pMP);
            ++created_initial_mappoints;
        }
        mLastNativeLoopFrameDiagnostics.initial_mappoints_created =
            created_initial_mappoints;

        // step5：将特征线包装成MapLines
        for(size_t i=0; i<mvLineMatches.size(); i++)
        {
            if(mvLineMatches[i].first<0||mvLineMatches[i].second<0)
                continue;

            // Create MapLine
            Vector6d worldPos;
            worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;
            MapLine* pML = new MapLine(worldPos, pKFcur, mpMap);
            pML->SetCreationOrigin(
                MapLine::ORIGIN_LEGACY,
                pKFcur ? static_cast<long>(pKFcur->mnFrameId) : -1,
                pKFcur ? static_cast<long>(pKFcur->mnId) : -1,
                pKFcur ? pKFcur->mTimeStamp : -1.0,
                "Tracking::CreateInitialMapMonocular");
            pKFini->AddMapLine(pML,i);
            pKFcur->AddMapLine(pML,i);

            pML->AddObservation(pKFini, i);
            pML->AddObservation(pKFcur, mvLineMatches[i].second);

            //b.MapPoint中是选取区分度最高的描述子，pl-slam直接采用前一帧的描述子,这里先按照ORB-SLAM的过程来
            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapLines[i] = pML;
            mCurrentFrame.mvbLineOutlier[i] = false;
            // step: Add to Map
            mpMap->AddMapLine(pML);
        }

        //cout << "this Map created with " << mpMap->MapPointsInMap() << " points, and "<< mpMap->MapLinesInMap() << " lines." << endl;
        //Optimizer::GlobalBundleAdjustemnt(mpMap, 20, true); //true代表使用有线特征的BA

        // step8：将MapPoints的中值深度归一化到1，并归一化两帧之间的变换
        // Q：MapPoints的中值深度归一化为1，MapLine是否也归一化？
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f/medianDepth;
        int min_initial_tracked_mappoints = 80;
        if(const char* value =
               std::getenv("CAM_B_INIT_MIN_TRACKED_MAPPOINTS"))
        {
            min_initial_tracked_mappoints =
                std::max(8, std::atoi(value));
        }
        if(medianDepth<0 ||
           pKFcur->TrackedMapPoints(1) <
               min_initial_tracked_mappoints)
        {
            cout << "Wrong initialization, reseting ... " << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale Points
        vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP)
        {
            if(vpAllMapPoints[iMP])
            {
                MapPoint* pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            }
        }

        // Scale Line Segments
        vector<MapLine*> vpAllMapLines = pKFini->GetMapLineMatches();
        for(size_t iML=0; iML < vpAllMapLines.size(); iML++)
        {
            if(vpAllMapLines[iML])
            {
                MapLine* pML = vpAllMapLines[iML];
                pML->SetWorldPos(pML->GetWorldPos()*invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mvpLocalMapLines = mpMap->GetAllMapLines();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);
        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);
        mState = OK;
    }

void Tracking::CheckReplacedInLastFrame()
{
        // points
        for(int i =0; i<mLastFrame.N; i++)
        {
            MapPoint* pMP = mLastFrame.mvpMapPoints[i];

            if(pMP)
            {
                MapPoint* pRep = pMP->GetReplaced();
                if(pRep)
                {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
        // lines
        for(int i=0; i<mLastFrame.NL; i++)
        {
            MapLine* pML = mLastFrame.mvpMapLines[i];

            if(pML)
            {
                MapLine* pReL = pML->GetReplaced();
                if(pReL)
                {
                    mLastFrame.mvpMapLines[i] = pReL;
                }
            }
        }
    }

bool Tracking::TrackReferenceKeyFrame()
{
        //cout<<"Tracking: with reference keyframe"<<endl;
        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();
        int lmatches=0;
        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.9,true);
        LSDmatcher lmatcher;

        vector<MapPoint*> vpMapPointMatches;
        vector<MapLine*> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
        cout<<"Tracking REFERENCE mmatches:"<<nmatches<<endl;

        const bool useLegacyLinePath =
            mCurrentFrame.dealWithLine &&
            !mCurrentFrame.HasKb8PointObservations();

        if(useLegacyLinePath)
        {   lmatches = lmatcher.SearchByProjection(mpReferenceKF,mCurrentFrame,vpMapLineMatches);
            mCurrentFrame.mvpMapLines=vpMapLineMatches;
            cout<<"Tracking REFERENCE lmatches:"<<lmatches<<endl;
        }
        else
        {
            if(nmatches<15)
                return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;

        for(size_t i=0;i<mCurrentFrame.mvpMapLines.size();i++)
        {
            MapLine * ml=mCurrentFrame.mvpMapLines[i];
            if(ml)
            {
                Eigen::Vector3d line_obs=mCurrentFrame.mvKeyLineFunctions[i];
            }
        }

        mCurrentFrame.SetPose(mLastFrame.mTcw);
        if(mCurrentFrame.HasKb8PointObservations())
            Optimizer::PoseOptimization(&mCurrentFrame);
        else
            Optimizer::PoseOptimization(&mCurrentFrame,true);
        mLastNativeLoopFrameDiagnostics.pose_point_edges =
            mCurrentFrame.mPoseOptimizationPointEdges;
        mLastNativeLoopFrameDiagnostics.pose_initial_inliers =
            mCurrentFrame.mPoseOptimizationInitialInliers;
        mLastNativeLoopFrameDiagnostics.pose_final_inliers =
            mCurrentFrame.mPoseOptimizationFinalInliers;
        mLastNativeLoopFrameDiagnostics.pose_outliers =
            mCurrentFrame.mPoseOptimizationOutliers;
        mLastNativeLoopFrameDiagnostics.reprojection_error_before =
            mCurrentFrame.mPoseOptimizationReprojectionBefore;
        mLastNativeLoopFrameDiagnostics.reprojection_error_after =
            mCurrentFrame.mPoseOptimizationReprojectionAfter;
        mLastNativeLoopFrameDiagnostics.pose_optimization_success =
            mCurrentFrame.mbPoseOptimizationSuccess;
        mLastNativeLoopFrameDiagnostics.pose_source = "POINT_ONLY";
        cout<<"tracking reference,pose after opti"<<mCurrentFrame.mTcw<<endl;
        // Discard outliers
        // 剔除优化后的outlier匹配点（MapPoints）
        int nmatchesMap = 0;
        int nmatchesLineMap=0;
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }

        //当 cMurrentFrame.dealWithLine为true，使用Pose去除错误的线
        for(int i=0;i<mCurrentFrame.NL;i++)
        {
            if(mCurrentFrame.mvpMapLines[i])
            {
                if(mCurrentFrame.mvbLineOutlier[i])
                {
                    MapLine* pML=mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
                    mCurrentFrame.mvbLineOutlier[i]=false;
                    pML->mbTrackInView=false;
                    pML->mnLastFrameSeen=mCurrentFrame.mnId;
                    lmatches--;
                    cout<<"tracking,lmatches"<<lmatches<<endl;
                }
                else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                    nmatchesLineMap++;
            }
        }
        if(useLegacyLinePath)
            return  nmatchesMap>=10||nmatchesLineMap>=5;
        else
            return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
        // Update pose according to reference keyframe
        // step1: 更新最近一帧的位姿
        KeyFrame* pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr*pRef->GetPose());

        // 如果上一帧为关键帧，或者单目的情况，则退出
        if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for(int i=0; i<mLastFrame.N;i++)
        {
            float z = mLastFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }


        sort(vDepthIdx.begin(),vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for(size_t j=0; j<vDepthIdx.size();j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint* pMP = mLastFrame.mvpMapPoints[i];
            if(!pMP)
                bCreateNew = true;
            else if(pMP->Observations()<1)
            {
                bCreateNew = true;
            }

            if(bCreateNew)
            {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

                mLastFrame.mvpMapPoints[i]=pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if(vDepthIdx[j].first>mThDepth && nPoints>100)
                break;
        }


        // Create "visual odometry" MapLines
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float,int> > vLineDepthIdx;
        vLineDepthIdx.reserve(mLastFrame.NL);
        int nLines = 0;
        for(int i=0; i<mLastFrame.NL;i++)
        {
            float z = mLastFrame.mvDepthLine[i];
            if(z==1)
            {
                bool bCreateNew = false;
                vLineDepthIdx.push_back(make_pair(z,i));
                MapLine *pML = mLastFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1) {
                    bCreateNew = true;
                }
                if (bCreateNew) {
                    Vector6d line3D= mLastFrame.obtain3DLine(i);//mvLines3D[i];
                    MapLine *pNewML=new MapLine(line3D,mpMap,&mLastFrame,i);
                    pNewML->SetCreationOrigin(
                        MapLine::ORIGIN_LEGACY,
                        static_cast<long>(mLastFrame.mnId),
                        -1,
                        mLastFrame.mTimeStamp,
                        "Tracking::CreateTemporalLine");
                    //Vector6d x3D = mLastFrame.mvLines3D(i);
                    //MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                    mLastFrame.mvpMapLines[i] = pNewML;

                    mlpTemporalLines.push_back(pNewML);
                    nLines++;
                } else {
                    nLines++;
                }

                if ( nLines> 30)
                    break;

            }
        }


}

bool Tracking::TrackWithMotionModel()
{
        cout<<"Tracking: with motion model"<<endl;
        // --step1: 建立ORB特征点的匹配
        ORBmatcher matcher(0.9,true);
        LSDmatcher lmatcher;

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        // --step2: 更新上一帧的位姿
        UpdateLastFrame();

        // --step3:根据Const Velocity Model，估计当前帧的位姿
        mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        // Project points seen in previous frame
        int th;
        if(mSensor!=System::STEREO)
            th=15;
        else
            th=7;
        // --step4：根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
        int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);
        //int lmatches = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame);
    int lmatches2=0;
    const bool useLegacyLinePath =
        mCurrentFrame.dealWithLine &&
        !mCurrentFrame.HasKb8PointObservations();

    if(useLegacyLinePath)
    {
        fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(), static_cast<MapLine*>(NULL));
        vector<MapLine*> vpMapLineMatches;
        lmatches2 = lmatcher.SearchByProjection(mpReferenceKF,mCurrentFrame,vpMapLineMatches);
        mCurrentFrame.mvpMapLines=vpMapLineMatches;
    }

        // If few matches, uses a wider window search
        // 如果跟踪的点少，则扩大搜索半径再来一次
        if(nmatches<20)
        {
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
        }

        if(useLegacyLinePath)
        {
            if(nmatches<20 &&lmatches2<6) return false;
        }
        else
        {
            if(nmatches<20 )
                return false;
        }

        // Optimize frame pose with all matches
        // --step5: 优化位姿
        cout<<"tracking motion model,pose before opti"<<mCurrentFrame.mTcw<<endl;
        if((mPointLineMode == PHYSICAL_ANGULAR_PL ||
            mPointLineMode == PHYSICAL_PIXEL_PL ||
            mPointLineMode == PHYSICAL_PIXEL_COV_PL ||
            mPointLineMode == PHYSICAL_PL) &&
           mbEnableNativeFisheyeLines && mbEnablePointLineTracking &&
           !mCurrentFrame.HasKb8PointObservations())
            Optimizer::PoseOptimization(&mCurrentFrame);
        else if(mCurrentFrame.HasKb8PointObservations())
            Optimizer::PoseOptimization(&mCurrentFrame);
        else
            Optimizer::PoseOptimization(&mCurrentFrame, true);
        mLastNativeLoopFrameDiagnostics.pose_point_edges =
            mCurrentFrame.mPoseOptimizationPointEdges;
        mLastNativeLoopFrameDiagnostics.pose_initial_inliers =
            mCurrentFrame.mPoseOptimizationInitialInliers;
        mLastNativeLoopFrameDiagnostics.pose_final_inliers =
            mCurrentFrame.mPoseOptimizationFinalInliers;
        mLastNativeLoopFrameDiagnostics.pose_outliers =
            mCurrentFrame.mPoseOptimizationOutliers;
        mLastNativeLoopFrameDiagnostics.reprojection_error_before =
            mCurrentFrame.mPoseOptimizationReprojectionBefore;
        mLastNativeLoopFrameDiagnostics.reprojection_error_after =
            mCurrentFrame.mPoseOptimizationReprojectionAfter;
        mLastNativeLoopFrameDiagnostics.pose_optimization_success =
            mCurrentFrame.mbPoseOptimizationSuccess;
        mLastNativeLoopFrameDiagnostics.pose_source = "POINT_ONLY";
        cout<<"tracking motion model,pose after opti"<<mCurrentFrame.mTcw<<endl;
        // Discard outliers
        // --step6：优化位姿后剔除outlier的mvpMapPoints
        int nmatchesMap = 0;
        int nmatchesLineMap=0;

        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }

        for(int i=0; i<mCurrentFrame.NL;i++)
        {
            if(mCurrentFrame.mvpMapLines[i])
            {
                if(mCurrentFrame.mvbLineOutlier[i])
                {
                    MapLine* pML=mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
                    mCurrentFrame.mvbLineOutlier[i]=false;
                    pML->mbTrackInView=false;
                    pML->mnLastFrameSeen=mCurrentFrame.mnId;
                    lmatches2--;
                }
                else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                    nmatchesLineMap++;
            }

        }

        if(mbOnlyTracking)
        {
            mbVO = nmatchesMap<10;
            return nmatches>20;
        }

        if(useLegacyLinePath)
        {
            return (nmatchesMap >= 10 || nmatchesLineMap >= 6);
        }
        else
        {
            return nmatchesMap>=10;
        }
}

bool Tracking::TrackLocalMap()
{
        UpdateLocalMap();
        const bool runSerialLocalSearchDiagnostic =
            UseSerialLocalSearchDiagnostic();
        const bool runLineSearch =
            mPointLineMode != POINT_ONLY &&
            (!mbUseKb8PointObservations ||
             (mbEnableNativeFisheyeLines && mbEnablePointLineTracking));
        const std::string localMappingState =
            mpLocalMapper
                ? (mpLocalMapper->isStopped()
                       ? "STOPPED"
                       : (mpLocalMapper->isFinished() ? "FINISHED"
                                                      : "RUNNING"))
                : "NULL";
        const bool loopClosingRunning =
            mpLoopClosing ? !mpLoopClosing->isFinished() : false;
        auto logLocalSearchEvent =
            [&](const std::string& event,
                const std::string& workerType,
                const std::thread::id& threadId,
                const bool exceptionCaught,
                const std::string& exceptionMessage)
            {
                AppendTrackingLocalSearchThreadEvent(
                    event,
                    mCurrentFrame.mnId,
                    threadId,
                    workerType,
                    &mCurrentFrame,
                    mpMap,
                    mvpLocalMapPoints.size(),
                    mvpLocalMapLines.size(),
                    mCurrentFrame.mvpMapPoints.size(),
                    mCurrentFrame.mvpMapLines.size(),
                    loopClosingRunning,
                    localMappingState,
                    exceptionCaught,
                    exceptionMessage);
            };

        std::exception_ptr pointException;
        std::exception_ptr lineException;

        if(runSerialLocalSearchDiagnostic)
        {
            logLocalSearchEvent("POINT_THREAD_ENTER",
                                "POINT_SERIAL_DIAGNOSTIC",
                                std::this_thread::get_id(),
                                false,
                                "NA");
            try
            {
                SearchLocalPoints();
            }
            catch(const std::exception& e)
            {
                pointException = std::current_exception();
                logLocalSearchEvent("POINT_THREAD_EXIT",
                                    "POINT_SERIAL_DIAGNOSTIC",
                                    std::this_thread::get_id(),
                                    true,
                                    e.what());
            }
            catch(...)
            {
                pointException = std::current_exception();
                logLocalSearchEvent("POINT_THREAD_EXIT",
                                    "POINT_SERIAL_DIAGNOSTIC",
                                    std::this_thread::get_id(),
                                    true,
                                    "unknown_exception");
            }
            if(!pointException)
                logLocalSearchEvent("POINT_THREAD_EXIT",
                                    "POINT_SERIAL_DIAGNOSTIC",
                                    std::this_thread::get_id(),
                                    false,
                                    "NA");

            if(runLineSearch)
            {
                logLocalSearchEvent("LINE_THREAD_ENTER",
                                    "LINE_SERIAL_DIAGNOSTIC",
                                    std::this_thread::get_id(),
                                    false,
                                    "NA");
                try
                {
                    SearchLocalLines();
                }
                catch(const std::exception& e)
                {
                    lineException = std::current_exception();
                    logLocalSearchEvent("LINE_THREAD_EXIT",
                                        "LINE_SERIAL_DIAGNOSTIC",
                                        std::this_thread::get_id(),
                                        true,
                                        e.what());
                }
                catch(...)
                {
                    lineException = std::current_exception();
                    logLocalSearchEvent("LINE_THREAD_EXIT",
                                        "LINE_SERIAL_DIAGNOSTIC",
                                        std::this_thread::get_id(),
                                        true,
                                        "unknown_exception");
                }
                if(!lineException)
                    logLocalSearchEvent("LINE_THREAD_EXIT",
                                        "LINE_SERIAL_DIAGNOSTIC",
                                        std::this_thread::get_id(),
                                        false,
                                        "NA");
            }
        }
        else
        {
            logLocalSearchEvent("POINT_THREAD_CREATED",
                                "POINT",
                                std::thread::id(),
                                false,
                                "NA");
            thread threadPoints([&]()
            {
                logLocalSearchEvent("POINT_THREAD_ENTER",
                                    "POINT",
                                    std::this_thread::get_id(),
                                    false,
                                    "NA");
                try
                {
                    SearchLocalPoints();
                }
                catch(const std::exception& e)
                {
                    pointException = std::current_exception();
                    logLocalSearchEvent("POINT_THREAD_EXIT",
                                        "POINT",
                                        std::this_thread::get_id(),
                                        true,
                                        e.what());
                    return;
                }
                catch(...)
                {
                    pointException = std::current_exception();
                    logLocalSearchEvent("POINT_THREAD_EXIT",
                                        "POINT",
                                        std::this_thread::get_id(),
                                        true,
                                        "unknown_exception");
                    return;
                }
                logLocalSearchEvent("POINT_THREAD_EXIT",
                                    "POINT",
                                    std::this_thread::get_id(),
                                    false,
                                    "NA");
            });

            std::unique_ptr<thread> threadLines;
            if(runLineSearch)
            {
                logLocalSearchEvent("LINE_THREAD_CREATED",
                                    "LINE",
                                    std::thread::id(),
                                    false,
                                    "NA");
                threadLines.reset(new thread([&]()
                {
                    logLocalSearchEvent("LINE_THREAD_ENTER",
                                        "LINE",
                                        std::this_thread::get_id(),
                                        false,
                                        "NA");
                    try
                    {
                        SearchLocalLines();
                    }
                    catch(const std::exception& e)
                    {
                        lineException = std::current_exception();
                        logLocalSearchEvent("LINE_THREAD_EXIT",
                                            "LINE",
                                            std::this_thread::get_id(),
                                            true,
                                            e.what());
                        return;
                    }
                    catch(...)
                    {
                        lineException = std::current_exception();
                        logLocalSearchEvent("LINE_THREAD_EXIT",
                                            "LINE",
                                            std::this_thread::get_id(),
                                            true,
                                            "unknown_exception");
                        return;
                    }
                    logLocalSearchEvent("LINE_THREAD_EXIT",
                                        "LINE",
                                        std::this_thread::get_id(),
                                        false,
                                        "NA");
                }));
            }

            logLocalSearchEvent("POINT_JOIN_BEGIN",
                                "POINT",
                                std::this_thread::get_id(),
                                false,
                                "NA");
            if(threadPoints.joinable())
                threadPoints.join();
            logLocalSearchEvent("POINT_JOIN_END",
                                "POINT",
                                std::this_thread::get_id(),
                                false,
                                "NA");
            if(threadLines)
            {
                logLocalSearchEvent("LINE_JOIN_BEGIN",
                                    "LINE",
                                    std::this_thread::get_id(),
                                    false,
                                    "NA");
                if(threadLines->joinable())
                    threadLines->join();
                logLocalSearchEvent("LINE_JOIN_END",
                                    "LINE",
                                    std::this_thread::get_id(),
                                    false,
                                    "NA");
            }
        }

        if(pointException)
            std::rethrow_exception(pointException);
        if(lineException)
            std::rethrow_exception(lineException);

        int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
        mLastNativeLoopFrameDiagnostics.pose_point_edges =
            mCurrentFrame.mPoseOptimizationPointEdges;
        mLastNativeLoopFrameDiagnostics.pose_initial_inliers =
            mCurrentFrame.mPoseOptimizationInitialInliers;
        mLastNativeLoopFrameDiagnostics.pose_final_inliers =
            mCurrentFrame.mPoseOptimizationFinalInliers;
        mLastNativeLoopFrameDiagnostics.pose_outliers =
            mCurrentFrame.mPoseOptimizationOutliers;
        mLastNativeLoopFrameDiagnostics.reprojection_error_before =
            mCurrentFrame.mPoseOptimizationReprojectionBefore;
        mLastNativeLoopFrameDiagnostics.reprojection_error_after =
            mCurrentFrame.mPoseOptimizationReprojectionAfter;
        mLastNativeLoopFrameDiagnostics.pose_optimization_success =
            mCurrentFrame.mbPoseOptimizationSuccess;
        mLastNativeLoopFrameDiagnostics.point_inliers = nGood;
        mLastNativeLoopFrameDiagnostics.pose_source = "POINT_ONLY";
        mLastNativeLoopFrameDiagnostics.fallback_reason =
            "POINT_ONLY";
        if((mPointLineMode == PHYSICAL_ANGULAR_PL ||
            mPointLineMode == PHYSICAL_PIXEL_PL ||
            mPointLineMode == PHYSICAL_PIXEL_COV_PL ||
            mPointLineMode == PHYSICAL_PL) &&
           mbEnableNativeFisheyeLines && mbEnablePointLineTracking &&
           nGood > 0)
        {
            Optimizer::NativeFisheyePoseOptimizationResult native_result;
            if(mPointLineMode == PHYSICAL_ANGULAR_PL &&
               mCurrentFrame.HasKb8PointObservations())
            {
                native_result =
                    Optimizer::PoseOptimizationWithKb8PointLine(
                        &mCurrentFrame);
            }
            else if(mCurrentFrame.HasKb8PointObservations())
            {
                native_result =
                    Optimizer::PoseOptimizationWithPhysicalPixelLines(
                        &mCurrentFrame, 2,
                        mPointLineMode == PHYSICAL_PIXEL_COV_PL);
            }
            else
            {
                native_result =
                    Optimizer::PoseOptimizationWithNativeFisheyeLines(
                        &mCurrentFrame);
            }
            mLastNativeLoopFrameDiagnostics.point_edges =
                native_result.point_edge_count;
            mLastNativeLoopFrameDiagnostics.line_edges =
                native_result.line_edge_count;
            mLastNativeLoopFrameDiagnostics.physical_observations =
                native_result.physical_observation_count;
            mLastNativeLoopFrameDiagnostics.physical_matches =
                native_result.physical_match_count;
            mLastNativeLoopFrameDiagnostics.physical_residual_created =
                native_result.line_edge_count;
            mLastNativeLoopFrameDiagnostics.physical_residual_active =
                native_result.line_edge_count;
            mLastNativeLoopFrameDiagnostics.physical_residual_accepted =
                native_result.line_inliers;
            mLastNativeLoopFrameDiagnostics.physical_residual_rejected =
                native_result.line_bad;
            mLastNativeLoopFrameDiagnostics
                .duplicate_physical_residual_violations =
                native_result.duplicate_physical_residual_violations;
            mLastNativeLoopFrameDiagnostics.point_inliers =
                native_result.point_inliers;
            mLastNativeLoopFrameDiagnostics.line_inliers =
                native_result.line_inliers;
            mLastNativeLoopFrameDiagnostics.point_cost_before =
                native_result.initial_point_cost;
            mLastNativeLoopFrameDiagnostics.point_cost_after =
                native_result.final_point_cost;
            mLastNativeLoopFrameDiagnostics.line_cost_before =
                native_result.initial_line_cost;
            mLastNativeLoopFrameDiagnostics.line_cost_after =
                native_result.final_line_cost;
            mLastNativeLoopFrameDiagnostics.pose_delta_translation =
                native_result.pose_delta_translation;
            mLastNativeLoopFrameDiagnostics.pose_delta_rotation =
                native_result.pose_delta_rotation_deg;
            mLastNativeLoopFrameDiagnostics.joint_accepted =
                native_result.success;
            mLastNativeLoopFrameDiagnostics.pose_source =
                native_result.success ? "JOINT_ACCEPTED"
                                      : "POINT_ONLY_FALLBACK";
            mLastNativeLoopFrameDiagnostics.fallback_reason =
                native_result.status;
            if(native_result.success)
                nGood = native_result.point_inliers;
        }

        mnMatchesInliers = 0;
        mnLineMatchesInliers = 0;

        for(int i=0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if(!mbOnlyTracking)
                    {
                        if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                            mnMatchesInliers++;
                    }
                    else
                        mnMatchesInliers++;
                }
                else if(mSensor==System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] =
                        static_cast<MapPoint*>(NULL);
            }
        }

        for(size_t i = 0; i < mCurrentFrame.mvpFisheyeMapLines.size() &&
             i < mCurrentFrame.mvbFisheyeLineOutlier.size(); ++i)
        {
            MapLine* pML = mCurrentFrame.mvpFisheyeMapLines[i];
            if(!pML)
                continue;
            if(!mCurrentFrame.mvbFisheyeLineOutlier[i])
            {
                pML->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(pML->Observations() > 0)
                        mnLineMatchesInliers++;
                }
                else
                    mnLineMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
            {
                mCurrentFrame.mvpFisheyeMapLines[i] =
                    static_cast<MapLine*>(NULL);
            }
        }

        const int early_local_map_min_inliers =
            mbUseKb8PointObservations
                ? ReadEnvIntClamped(
                      "CAM_B_TRACK_LOCAL_MAP_EARLY_MIN_INLIERS", 50, 8)
                : 50;
        const int local_map_min_inliers =
            mbUseKb8PointObservations
                ? ReadEnvIntClamped(
                      "CAM_B_TRACK_LOCAL_MAP_MIN_INLIERS", 30, 8)
                : 30;

        if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames &&
           mnMatchesInliers<early_local_map_min_inliers)
            return false;

        return mnMatchesInliers >= local_map_min_inliers;
}

bool Tracking::TrackLocalMapWithLines()
{
        UpdateLocalMap();
        thread threadPoints(&Tracking::SearchLocalPoints, this);
        thread threadLines(&Tracking::SearchLocalLines, this);
        threadPoints.join();
        threadLines.join();

        Optimizer::PoseOptimization(&mCurrentFrame, true);
        mnMatchesInliers = 0;
        mnLineMatchesInliers = 0;

        for(int i=0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if(!mbOnlyTracking)
                    {
                        if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                            mnMatchesInliers++;
                    }
                    else
                        mnMatchesInliers++;
                }
                else if(mSensor==System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }
        }

        for(int i=0; i<mCurrentFrame.NL; i++)
        {
            if(mCurrentFrame.mvpMapLines[i])
            {
                if(!mCurrentFrame.mvbLineOutlier[i])
                {
                    mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                    if(!mbOnlyTracking)
                    {
                        if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                            mnLineMatchesInliers++;
                    }
                    else
                        mnLineMatchesInliers++;
                }
                else if(mSensor==System::STEREO)
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
            }
        }
        if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
            return false;

        if(mnMatchesInliers<30)
            return false;
        else
            return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
    // "total matches = matches to map + visual odometry matches"
    // Visual odometry matches will become MapPoints if we insert a keyframe.
    // This ratio measures how many MapPoints we could create if we insert a keyframe.
    int nMap = 0; //nTrackedClose
    int nTotal= 0;
    int nNonTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                nTotal++;
                if(mCurrentFrame.mvpMapPoints[i]&&!mCurrentFrame.mvbOutlier[i])
                        nMap++;
                else
                    nNonTrackedClose++;
            }
        }
    }
    else
    {
        // There are no visual odometry matches in the monocular case
        nMap=1;
        nTotal=1;
    }

    const float ratioMap = (float)nMap/fmax(1.0f,nTotal);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    float thMapRatio = 0.35f;
    if(mnMatchesInliers>300)
        thMapRatio = 0.20f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || ratioMap<0.3f) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| ratioMap<thMapRatio) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;
    mLastNativeLoopFrameDiagnostics.new_keyframe = true;
    if(!mCurrentFrame.dealWithLine && mPointLineMode != POINT_ONLY)
    {
        if(mCurrentFrame.NL==0)
        {
            if(!mCurrentFrame.mpLineSegment)
                throw std::logic_error(
                    "POINT_LINE keyframe creation requires LineSegment extractor");
            mCurrentFrame.ExtractLSD(this->mImGray);
            mCurrentFrame.mvDepthLine = vector<float>(mCurrentFrame.mvKeylinesUn.size(),-1.0f);
            mCurrentFrame.mvLines3D=vector<Vector6d>(mCurrentFrame.mvKeylinesUn.size(), static_cast<Vector6d>(NULL));
            for(int i=0; i<mCurrentFrame.mvKeylinesUn.size(); ++i)	{ // each line
                double len = cv::norm(mCurrentFrame.mvKeylinesUn[i].getStartPoint() - mCurrentFrame.mvKeylinesUn[i].getEndPoint());
                vector<cv::Point3d> pts3d;
                double numSmp = (double) min((int)len, 100); //number of line points sampled
                pts3d.reserve(numSmp);//预留数量

                for(int j=0; j<=numSmp; ++j) {
                    // use nearest neighbor to querry depth value
                    // assuming position (0,0) is the top-left corner of image, then the
                    // top-left pixel's center would be (0.5,0.5)
                    cv::Point2d pt = mCurrentFrame.mvKeylinesUn[i].getStartPoint() * (1-j/numSmp) + mCurrentFrame.mvKeylinesUn[i].getEndPoint()* (j/numSmp);
                    if(pt.x<0 || pt.y<0 || pt.x >= this->mImDepth.cols || pt.y >= this->mImDepth.rows ) continue;
                    int row, col; // nearest pixel for pt
                    if((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) {// boundary issue
                        col = max(int(pt.x-1),0);
                        row = max(int(pt.y-1),0);
                    } else {
                        col = int(pt.x);
                        row = int(pt.y);
                    }
                    float d=-1;
                    if(this->mImGray.at<float>(row,col) <=0.01) { // no depth info
                        //cout<<"no depth info"<<endl;
                        continue;
                    }
                    else {
                        d = this->mImGray.ptr<float>(row)[col];
                    }
                    cv::Point3d p;

                    // 计算这个点的空间坐标
                    p.z = d;
                    p.x = (col - mCurrentFrame.cx) * p.z *mCurrentFrame.invfx;//K.at<float>(0,0);//invfx;
                    p.y = (row - mCurrentFrame.cy) * p.z *mCurrentFrame.invfy;//K.at<float>(1,1);//invfy;

                    pts3d.push_back(p);

                }
                //如果 点数量少于 10或者 ，就抛弃该线
                if (pts3d.size() < 10.0)//sysPara.ratio_of_collinear_pts
                    continue;

                RandomLine3d tmpLine;
#if 1
                vector<RandomPoint3d> rndpts3d;
                rndpts3d.reserve(pts3d.size());
                // compute uncertainty of 3d points
                for(int j=0; j<pts3d.size();++j) {
                    rndpts3d.push_back(compPt3dCov(pts3d[j], mCurrentFrame.mK, 1));
                }
                // using ransac to extract a 3d line from 3d pts
                tmpLine = extract3dline_mahdist(rndpts3d);
#else
                //计算这个线的参数
        tmpLine = extract3dline(pts3d,origPoints,optiPoints);
#endif
                if(tmpLine.pts.size()/len > 0.4/*sysPara.ratio_of_collinear_pts*/	&&
                   cv::norm(tmpLine.A - tmpLine.B) >0.02) { // sysPara.line3d_length_thresh
                    //this line is reliable
                    mCurrentFrame.mvDepthLine[i]= 1.0f;
                    cout<<"gaibian line 3d"<<endl;
                    mCurrentFrame.mvLines3D[i]<<tmpLine.A.x,tmpLine.A.y,tmpLine.A.z,tmpLine.B.x,tmpLine.B.y,tmpLine.B.z;
                }
            }

            mCurrentFrame.NL=mCurrentFrame.mvKeylinesUn.size();
            mCurrentFrame.mvpMapLines=vector<MapLine *>(mCurrentFrame.NL,static_cast<MapLine*>(NULL));
            mCurrentFrame.mvbLineOutlier=vector<bool>(mCurrentFrame.NL,false);
        }
    }

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        cout<<"Tracking: newKeyframe: RGBD"<<endl;
        mCurrentFrame.UpdatePoseMatrices();
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vector<pair<float, int>> vLineDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        for(int i=0;i<mCurrentFrame.NL;i++)
        {
            float isDepthGood= mCurrentFrame.mvDepthLine[i];
            //cout<<"Tracking: StereoInitialization:线是否存在:"<<isDepthGood<<endl;
            //如果存在， 执行程序
            if(isDepthGood==1)
            {
                vLineDepthIdx.push_back(make_pair(isDepthGood,i));
            }
        }

        if(!vLineDepthIdx.empty())
        {
            //sort(vLineDepthIdx.begin(),vLineDepthIdx.end());
            int nLines = 0;
            for(size_t j=0; j<vLineDepthIdx.size();j++)
            {
                int i = vLineDepthIdx[j].second;

                bool bCreateNew = false;

                MapLine* pMP = mCurrentFrame.mvpMapLines[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
                }

                if(bCreateNew)
                {
                    Vector6d line3D= mCurrentFrame.obtain3DLine(i);//mvLines3D[i];
                    MapLine *pNewML=new MapLine(line3D,pKF,mpMap);
                    pNewML->SetCreationOrigin(
                        MapLine::ORIGIN_LEGACY,
                        pKF ? static_cast<long>(pKF->mnFrameId) : -1,
                        pKF ? static_cast<long>(pKF->mnId) : -1,
                        pKF ? pKF->mTimeStamp : -1.0,
                        "Tracking::CreateNewKeyFrame");
                    pNewML->AddObservation(pKF,i);
                    pKF->AddMapLine(pNewML,i);
                    pNewML->ComputeDistinctiveDescriptors();
                    pNewML->UpdateAverageDir();
                    mpMap->AddMapLine(pNewML);
                    mCurrentFrame.mvpMapLines[i]=pNewML;
                    nLines++;
                }
                else
                {
                    nLines++;
                }

                if(nLines>20)
                    break;
            }
        }
        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }
    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    mLastNativeLoopFrameDiagnostics.mappoint_projection_input = 0;
    mLastNativeLoopFrameDiagnostics.valid_camera_points = 0;
    mLastNativeLoopFrameDiagnostics.valid_kb8_projections = 0;
    mLastNativeLoopFrameDiagnostics.inside_image = 0;
    mLastNativeLoopFrameDiagnostics.track_in_view = 0;
    mLastNativeLoopFrameDiagnostics.search_by_projection_attempts = 0;
    mLastNativeLoopFrameDiagnostics.accepted_2d3d_matches = 0;

    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;
    const bool useKb8ProjectionDiagnostics =
        mCurrentFrame.HasKb8PointObservations() &&
        !mCurrentFrame.mTcw.empty();
    std::unique_ptr<KannalaBrandt8Camera> diagnosticCamera;
    if(useKb8ProjectionDiagnostics)
        diagnosticCamera.reset(new KannalaBrandt8Camera(
            mCurrentFrame.mKb8PointCameraParameters));
    cv::Mat Rcw;
    cv::Mat tcw;
    if(useKb8ProjectionDiagnostics)
    {
        Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
        tcw = mCurrentFrame.mTcw.rowRange(0,3).col(3);
    }

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        mLastNativeLoopFrameDiagnostics.mappoint_projection_input++;
        if(useKb8ProjectionDiagnostics)
        {
            const cv::Mat Pw = pMP->GetWorldPos();
            const cv::Mat Pc = Rcw * Pw + tcw;
            if(Pc.at<float>(2) > 0.0f)
            {
                mLastNativeLoopFrameDiagnostics.valid_camera_points++;
                Eigen::Vector2d uv;
                if(diagnosticCamera && diagnosticCamera->valid() &&
                   diagnosticCamera->project(
                       Eigen::Vector3d(Pc.at<float>(0),
                                       Pc.at<float>(1),
                                       Pc.at<float>(2)),
                       uv))
                {
                    mLastNativeLoopFrameDiagnostics.valid_kb8_projections++;
                    if(diagnosticCamera->isInImage(uv))
                        mLastNativeLoopFrameDiagnostics.inside_image++;
                }
            }
        }
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        mLastNativeLoopFrameDiagnostics.track_in_view = nToMatch;
        mLastNativeLoopFrameDiagnostics.search_by_projection_attempts =
            nToMatch;
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        mLastNativeLoopFrameDiagnostics.accepted_2d3d_matches =
            matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::SearchLocalLines()
{
        if(mPointLineMode == POINT_ONLY)
            return;
        if(mbEnableNativeFisheyeLines && mpMap &&
           !mCurrentFrame.mvFisheyeLineObservations.empty())
        {
            KannalaBrandt8Camera::Parameters native_camera_params;
            if(mpMap->GetNativeFisheyeCameraParameters(
                   native_camera_params))
            {
                NativeFisheyeMapLineProjector::Options projector_options;
                if(mpFisheyeLineFrontend)
                    projector_options.virtual_yaw_deg =
                        mpFisheyeLineFrontend->GetOptions().
                            virtual_yaw_deg;
                NativeFisheyeMapLineProjector projector(
                    native_camera_params,
                    projector_options);
                const bool use_physical_association =
                    (mPointLineMode == PHYSICAL_ANGULAR_PL ||
                     mPointLineMode == PHYSICAL_PIXEL_PL ||
                     mPointLineMode == PHYSICAL_PIXEL_COV_PL ||
                     mPointLineMode == PHYSICAL_PL) &&
                    !mCurrentFrame.mvPhysicalLineObservations.empty();
                const NativeFisheyeMapLineProjector::Result result =
                    use_physical_association
                        ? projector.AssociatePhysical(mCurrentFrame,
                                                      mvpLocalMapLines)
                        : projector.Associate(mCurrentFrame,
                                              mvpLocalMapLines);
                mLastNativeLoopFrameDiagnostics.line_projection_candidates =
                    result.candidate_count;
                mLastNativeLoopFrameDiagnostics.line_rejected_bad =
                    result.rejected_bad;
                mLastNativeLoopFrameDiagnostics.line_rejected_geometry =
                    result.rejected_geometry;
                mLastNativeLoopFrameDiagnostics.line_rejected_temporal =
                    result.rejected_temporal;
                mLastNativeLoopFrameDiagnostics.line_rejected_descriptor =
                    result.rejected_descriptor;
                mLastNativeLoopFrameDiagnostics.line_rejected_duplicate =
                    result.rejected_duplicate;
                mLastNativeLoopFrameDiagnostics.line_gate_input_pairs =
                    result.gate_input_pairs;
                mLastNativeLoopFrameDiagnostics.line_fail_input_invalid =
                    result.fail_input_invalid;
                mLastNativeLoopFrameDiagnostics.line_fail_invalid_yaw_or_pose =
                    result.fail_invalid_yaw_or_pose;
                mLastNativeLoopFrameDiagnostics.line_fail_invalid_world_segment =
                    result.fail_invalid_world_segment;
                mLastNativeLoopFrameDiagnostics.line_fail_invalid_camera_segment =
                    result.fail_invalid_camera_segment;
                mLastNativeLoopFrameDiagnostics.line_fail_depth =
                    result.fail_depth;
                mLastNativeLoopFrameDiagnostics.line_fail_range =
                    result.fail_range;
                mLastNativeLoopFrameDiagnostics.line_fail_distance_invariance =
                    result.fail_distance_invariance;
                mLastNativeLoopFrameDiagnostics.line_fail_fisheye_projection =
                    result.fail_fisheye_projection;
                mLastNativeLoopFrameDiagnostics.line_fail_fisheye_fov =
                    result.fail_fisheye_fov;
                mLastNativeLoopFrameDiagnostics.line_fail_virtual_depth =
                    result.fail_virtual_depth;
                mLastNativeLoopFrameDiagnostics.line_fail_virtual_fov =
                    result.fail_virtual_fov;
                mLastNativeLoopFrameDiagnostics.line_fail_angle =
                    result.fail_angle;
                mLastNativeLoopFrameDiagnostics.line_fail_overlap =
                    result.fail_overlap;
                mLastNativeLoopFrameDiagnostics.line_fail_temporal =
                    result.fail_temporal;
                mLastNativeLoopFrameDiagnostics.line_fail_descriptor =
                    result.fail_descriptor;
                mLastNativeLoopFrameDiagnostics.line_fail_duplicate =
                    result.fail_duplicate;
                const auto view_value =
                    [](const std::vector<long long>& values,
                       const size_t index) -> long long
                    {
                        return index < values.size() ? values[index] : 0;
                    };
                mLastNativeLoopFrameDiagnostics.line_view0_input =
                    view_value(result.view_input_pairs, 0);
                mLastNativeLoopFrameDiagnostics.line_view1_input =
                    view_value(result.view_input_pairs, 1);
                mLastNativeLoopFrameDiagnostics.line_view2_input =
                    view_value(result.view_input_pairs, 2);
                mLastNativeLoopFrameDiagnostics.line_view0_projection_pass =
                    view_value(result.view_projection_pass, 0);
                mLastNativeLoopFrameDiagnostics.line_view1_projection_pass =
                    view_value(result.view_projection_pass, 1);
                mLastNativeLoopFrameDiagnostics.line_view2_projection_pass =
                    view_value(result.view_projection_pass, 2);
                mLastNativeLoopFrameDiagnostics.line_view0_angle_pass =
                    view_value(result.view_angle_pass, 0);
                mLastNativeLoopFrameDiagnostics.line_view1_angle_pass =
                    view_value(result.view_angle_pass, 1);
                mLastNativeLoopFrameDiagnostics.line_view2_angle_pass =
                    view_value(result.view_angle_pass, 2);
                mLastNativeLoopFrameDiagnostics.line_view0_overlap_pass =
                    view_value(result.view_overlap_pass, 0);
                mLastNativeLoopFrameDiagnostics.line_view1_overlap_pass =
                    view_value(result.view_overlap_pass, 1);
                mLastNativeLoopFrameDiagnostics.line_view2_overlap_pass =
                    view_value(result.view_overlap_pass, 2);
                mLastNativeLoopFrameDiagnostics.line_view0_descriptor_pass =
                    view_value(result.view_descriptor_pass, 0);
                mLastNativeLoopFrameDiagnostics.line_view1_descriptor_pass =
                    view_value(result.view_descriptor_pass, 1);
                mLastNativeLoopFrameDiagnostics.line_view2_descriptor_pass =
                    view_value(result.view_descriptor_pass, 2);
                mLastNativeLoopFrameDiagnostics.line_view0_association =
                    view_value(result.view_association, 0);
                mLastNativeLoopFrameDiagnostics.line_view1_association =
                    view_value(result.view_association, 1);
                mLastNativeLoopFrameDiagnostics.line_view2_association =
                    view_value(result.view_association, 2);
                mLastNativeLoopFrameDiagnostics.line_angle_fail_count =
                    result.angle_fail_count;
                mLastNativeLoopFrameDiagnostics.line_angle_pass_count =
                    result.angle_pass_count;
                mLastNativeLoopFrameDiagnostics.line_angle_fail_mean =
                    result.angle_fail_count > 0
                        ? result.angle_fail_sum /
                              static_cast<double>(result.angle_fail_count)
                        : 0.0;
                mLastNativeLoopFrameDiagnostics.line_angle_pass_mean =
                    result.angle_pass_count > 0
                        ? result.angle_pass_sum /
                              static_cast<double>(result.angle_pass_count)
                        : 0.0;
                mLastNativeLoopFrameDiagnostics.line_angle_fail_within_5pct =
                    result.angle_fail_within_5pct;
                mLastNativeLoopFrameDiagnostics.line_angle_fail_within_10pct =
                    result.angle_fail_within_10pct;
                mLastNativeLoopFrameDiagnostics.line_angle_fail_within_20pct =
                    result.angle_fail_within_20pct;
                mLastNativeLoopFrameDiagnostics.line_overlap_fail_count =
                    result.overlap_fail_count;
                mLastNativeLoopFrameDiagnostics.line_overlap_pass_count =
                    result.overlap_pass_count;
                mLastNativeLoopFrameDiagnostics.line_overlap_fail_mean =
                    result.overlap_fail_count > 0
                        ? result.overlap_fail_sum /
                              static_cast<double>(result.overlap_fail_count)
                        : 0.0;
                mLastNativeLoopFrameDiagnostics.line_overlap_pass_mean =
                    result.overlap_pass_count > 0
                        ? result.overlap_pass_sum /
                              static_cast<double>(result.overlap_pass_count)
                        : 0.0;
                mLastNativeLoopFrameDiagnostics.line_overlap_fail_within_5pct =
                    result.overlap_fail_within_5pct;
                mLastNativeLoopFrameDiagnostics.line_overlap_fail_within_10pct =
                    result.overlap_fail_within_10pct;
                mLastNativeLoopFrameDiagnostics.line_overlap_fail_within_20pct =
                    result.overlap_fail_within_20pct;
            }
            if(mPointLineMode == PHYSICAL_ANGULAR_PL ||
               mPointLineMode == PHYSICAL_PIXEL_PL ||
               mPointLineMode == PHYSICAL_PIXEL_COV_PL ||
               mPointLineMode == PHYSICAL_PL)
                return;
        }

        for(vector<MapLine*>::iterator vit=mCurrentFrame.mvpMapLines.begin(), vend=mCurrentFrame.mvpMapLines.end(); vit!=vend; vit++)
        {
            MapLine* pML = *vit;
            if(pML)
            {
                if(pML->isBad())
                {
                    *vit = static_cast<MapLine*>(NULL);
                } else{
                    pML->IncreaseVisible();
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    pML->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        for(vector<MapLine*>::iterator vit=mvpLocalMapLines.begin(), vend=mvpLocalMapLines.end(); vit!=vend; vit++)
        {
            MapLine* pML = *vit;

            if(pML->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if(pML->isBad())
                continue;

            if(mCurrentFrame.isInFrustum(pML, 0.6))
            {
                pML->IncreaseVisible();
                nToMatch++;
            }
        }
        if(nToMatch>0)
        {
            LSDmatcher matcher;
            int th = 1;

            if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
                th=5;

            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
        }
    }

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
    UpdateLocalLines();
}

void Tracking::UpdateLocalLines()
{
        // step1：清空局部MapLines
        mvpLocalMapLines.clear();

        // step2：遍历局部关键帧mvpLocalKeyFrames
        for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
        {
            KeyFrame* pKF = *itKF;
            const vector<MapLine*> vpMLs = pKF->GetMapLineMatches();
            //step3：将局部关键帧的MapLines添加到mvpLocalMapLines
            for(vector<MapLine*>::const_iterator itML=vpMLs.begin(), itEndML=vpMLs.end(); itML!=itEndML; itML++)
            {
                MapLine* pML = *itML;
                if(!pML)
                    continue;
                if(pML->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                    continue;
                if(!pML->isBad())
                {
                    mvpLocalMapLines.push_back(pML);
                    pML->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                }
            }

            if(mbEnableNativeFisheyeLines)
            {
                const vector<MapLine*> vpNativeMLs =
                    pKF->GetNativeFisheyeMapLineMatches();
                for(vector<MapLine*>::const_iterator
                        itML = vpNativeMLs.begin(),
                        itEndML = vpNativeMLs.end();
                    itML != itEndML; ++itML)
                {
                    MapLine* pML = *itML;
                    if(!pML)
                        continue;
                    MapLine* pLive = pML;
                    if(pLive->isBad())
                        pLive = pLive->GetReplaced();
                    if(!pLive || pLive->isBad())
                        continue;
                    if(pLive->mnTrackReferenceForFrame ==
                       mCurrentFrame.mnId)
                        continue;
                    mvpLocalMapLines.push_back(pLive);
                    pLive->mnTrackReferenceForFrame =
                        mCurrentFrame.mnId;
                }
            }
        }
    }

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if(vpCandidateKFs.empty())
    {
        AppendRelocalizationCsv(
            "relocalization_failure_recovery.csv",
            "scenario,frame,result,reason,map_corruption,thread_error",
            "no_candidate," + std::to_string(mCurrentFrame.mnId) +
                ",PASS,DetectRelocalizationCandidates_empty,false,false");
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);
    vector<std::unique_ptr<Kb8PnPSolver> > vpKb8PnPsolvers;
    vpKb8PnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                if(mbUseKb8PointObservations)
                {
                    std::unique_ptr<Kb8PnPSolver> pSolver(
                        new Kb8PnPSolver(
                            mCurrentFrame,
                            vvpMapPointMatches[i],
                            mKb8PointCameraParameters));
                    pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                    vpKb8PnPsolvers[i] = std::move(pSolver);
                }
                else
                {
                    PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                    vpPnPsolvers[i] = pSolver;
                }
                AppendRelocalizationCsv(
                    "relocalization_solver_dispatch.csv",
                    "frame,state_before,solver_selected,candidate_kf,success",
                    std::to_string(mCurrentFrame.mnId) + ",LOST," +
                        (mbUseKb8PointObservations ?
                         "Kb8PnPSolver" : "PnPsolver") + "," +
                        std::to_string(pKF->mnId) + ",created");
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            cv::Mat Tcw;
            const bool useKb8RelocalizationSolver =
                mbUseKb8PointObservations && vpKb8PnPsolvers[i];
            if(useKb8RelocalizationSolver)
            {
                const Kb8PnPSolver::Result result =
                    vpKb8PnPsolvers[i]->find();
                Tcw = result.Tcw;
                vbInliers = result.inliers;
                nInliers = result.inlier_count;
                bNoMore = Tcw.empty();
                AppendRelocalizationCsv(
                    "relocalization_projection_runtime.csv",
                    "frame,module,kb8_projection_calls,legacy_pinhole_calls,notes",
                    std::to_string(mCurrentFrame.mnId) +
                        ",Kb8PnPSolver," +
                        std::to_string(vpKb8PnPsolvers[i]->correspondenceCount()) +
                        ",0,bearing_unproject_ransac");
            }
            else
            {
                PnPsolver* pSolver = vpPnPsolvers[i];
                Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);
                AppendRelocalizationCsv(
                    "relocalization_projection_runtime.csv",
                    "frame,module,kb8_projection_calls,legacy_pinhole_calls,notes",
                    std::to_string(mCurrentFrame.mnId) +
                        ",PnPsolver,0,1,legacy_pinhole");
            }

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
                AppendRelocalizationCsv(
                    "relocalization_failure_recovery.csv",
                    "scenario,frame,result,reason,map_corruption,thread_error",
                    std::string(useKb8RelocalizationSolver ?
                                "kb8_pnp_failed," : "legacy_pnp_failed,") +
                        std::to_string(mCurrentFrame.mnId) +
                        ",PASS,candidate_discarded_no_pose,false,false");
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                AppendRelocalizationCsv(
                    "relocalization_pose_pipeline.csv",
                    "frame,solver,t_cw_available,pose_optimization_called,pose_optimization_success,inliers_before,inliers_after,kb8_edge",
                    std::to_string(mCurrentFrame.mnId) + "," +
                        (useKb8RelocalizationSolver ?
                         "Kb8PnPSolver" : "PnPsolver") +
                        ",true,true,pending," +
                        std::to_string(nInliers) + ",0,false");

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                AppendRelocalizationCsv(
                    "relocalization_pose_pipeline.csv",
                    "frame,solver,t_cw_available,pose_optimization_called,pose_optimization_success,inliers_before,inliers_after,kb8_edge",
                    std::to_string(mCurrentFrame.mnId) + "," +
                        (useKb8RelocalizationSolver ?
                         "Kb8PnPSolver" : "PnPsolver") +
                        ",true,true," +
                        (mCurrentFrame.mbPoseOptimizationSuccess ?
                         "true" : "false") + "," +
                        std::to_string(nInliers) + "," +
                        std::to_string(nGood) + "," +
                        (mCurrentFrame.mbPoseOptimizationUsedKb8PointEdge ?
                         "true" : "false"));

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    AppendRelocalizationCsv(
                        "formal_lost_recovery.csv",
                        "frame,lost_reason,relocalization_called,solver,recovered",
                        std::to_string(mCurrentFrame.mnId) +
                            ",TRACKING_STATE_LOST,true," +
                            (useKb8RelocalizationSolver ?
                             "Kb8PnPSolver" : "PnPsolver") +
                            ",true");
                    AppendRelocalizationCsv(
                        "relocalization_solver_dispatch.csv",
                        "frame,state_before,solver_selected,candidate_kf,success",
                        std::to_string(mCurrentFrame.mnId) + ",LOST," +
                            (useKb8RelocalizationSolver ?
                             "Kb8PnPSolver" : "PnPsolver") + "," +
                            std::to_string(vpCandidateKFs[i]->mnId) +
                            ",true");
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        AppendRelocalizationCsv(
            "formal_lost_recovery.csv",
            "frame,lost_reason,relocalization_called,solver,recovered",
            std::to_string(mCurrentFrame.mnId) +
                ",TRACKING_STATE_LOST,true," +
                (mbUseKb8PointObservations ?
                 "Kb8PnPSolver" : "PnPsolver") +
                ",false");
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{
    if(mpViewer)
        mpViewer->RequestStop();

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    //mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} //namespace StructureSLAM
