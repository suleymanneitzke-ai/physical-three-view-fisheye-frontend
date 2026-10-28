#ifndef FISHEYE_VIRTUAL_CAMERA_GEOMETRY_H
#define FISHEYE_VIRTUAL_CAMERA_GEOMETRY_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <limits>

namespace StructureSLAM
{

inline double NativeFisheyeVirtualYawRad(int view_id)
{
    constexpr double kPi = 3.14159265358979323846;
    if(view_id == 0)
        return -60.0 * kPi / 180.0;
    if(view_id == 1)
        return -20.0 * kPi / 180.0;
    if(view_id == 2)
        return 20.0 * kPi / 180.0;
    return std::numeric_limits<double>::quiet_NaN();
}

inline int NativeFisheyeVirtualWidth()
{
    return 512;
}

inline int NativeFisheyeVirtualHeight()
{
    return 512;
}

inline double NativeFisheyeVirtualHorizontalFovDeg()
{
    return 100.0;
}

inline double NativeFisheyeVirtualVerticalFovDeg()
{
    return 100.0;
}

inline double NativeFisheyeVirtualFx()
{
    constexpr double kPi = 3.14159265358979323846;
    return (0.5 * NativeFisheyeVirtualWidth()) /
           std::tan(0.5 * NativeFisheyeVirtualHorizontalFovDeg() *
                    kPi / 180.0);
}

inline double NativeFisheyeVirtualFy()
{
    constexpr double kPi = 3.14159265358979323846;
    return (0.5 * NativeFisheyeVirtualHeight()) /
           std::tan(0.5 * NativeFisheyeVirtualVerticalFovDeg() *
                    kPi / 180.0);
}

inline double NativeFisheyeVirtualCx()
{
    return 0.5 * (NativeFisheyeVirtualWidth() - 1);
}

inline double NativeFisheyeVirtualCy()
{
    return 0.5 * (NativeFisheyeVirtualHeight() - 1);
}

struct FisheyeVirtualPose
{
    Eigen::Matrix3d Rcw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tcw = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rwc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d twc = Eigen::Vector3d::Zero();
};

inline Eigen::Matrix3d RotationCvFromYaw(double yaw_rad)
{
    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    Eigen::Matrix3d R;
    R << c, 0.0, s,
         0.0, 1.0, 0.0,
        -s, 0.0, c;
    return R;
}

inline Eigen::Matrix3d RotationVcFromYaw(double yaw_rad)
{
    return RotationCvFromYaw(yaw_rad).transpose();
}

inline FisheyeVirtualPose PoseFromTcw(const Eigen::Matrix3d& Rcw,
                                      const Eigen::Vector3d& tcw)
{
    FisheyeVirtualPose pose;
    pose.Rcw = Rcw;
    pose.tcw = tcw;
    pose.Rwc = Rcw.transpose();
    pose.twc = -pose.Rwc * tcw;
    return pose;
}

inline FisheyeVirtualPose VirtualPoseFromPhysicalTcw(
    const FisheyeVirtualPose& physical,
    double yaw_rad)
{
    const Eigen::Matrix3d Rvc = RotationVcFromYaw(yaw_rad);
    return PoseFromTcw(Rvc * physical.Rcw, Rvc * physical.tcw);
}

inline Eigen::Vector3d CameraCenterFromTcw(const Eigen::Matrix3d& Rcw,
                                          const Eigen::Vector3d& tcw)
{
    return -Rcw.transpose() * tcw;
}

inline Eigen::Vector3d PlaneNormalWorldFromCamera(
    const Eigen::Matrix3d& Rcw,
    const Eigen::Vector3d& nc)
{
    return Rcw.transpose() * nc;
}

inline double PlaneOffsetWorldFromCamera(const Eigen::Vector3d& nc,
                                         const Eigen::Vector3d& tcw)
{
    return nc.dot(tcw);
}

inline Eigen::Vector3d VirtualRayFromPixel(double u,
                                           double v,
                                           double fx,
                                           double fy,
                                           double cx,
                                           double cy)
{
    Eigen::Vector3d ray((u - cx) / fx, (v - cy) / fy, 1.0);
    return ray.normalized();
}

inline Eigen::Vector3d VirtualPlaneNormalFromEndpoints(
    double u1,
    double v1,
    double u2,
    double v2,
    double fx,
    double fy,
    double cx,
    double cy)
{
    const Eigen::Vector3d r1 = VirtualRayFromPixel(u1, v1, fx, fy, cx, cy);
    const Eigen::Vector3d r2 = VirtualRayFromPixel(u2, v2, fx, fy, cx, cy);
    Eigen::Vector3d n = r1.cross(r2);
    const double norm = n.norm();
    if(norm > 1e-12)
        n /= norm;
    return n;
}

} // namespace StructureSLAM

#endif
