#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>
#include <geometry_msgs/Point.h>
#include <Eigen/Eigen>

#include <algorithm>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include "bspline_opt/uniform_bspline.h"
#include "ego_planner/Bspline.h"

using namespace ego_planner;

namespace
{
constexpr int NUM_UAV = 3;

struct Box
{
  Eigen::Vector3d min_corner;
  Eigen::Vector3d max_corner;
};

ros::Subscriber python_traj_subs[NUM_UAV];
ros::Publisher bspline_pub[NUM_UAV];
ros::Publisher safety_fallback_pub[NUM_UAV];

std::vector<Box> obstacle_boxes;
Eigen::Vector3d map_min(0.0, 0.0, 0.0);
Eigen::Vector3d map_max(20.0, 20.0, 8.0);

double knot_interval = 1.0;
double voxel_resolution = 1.0;
int traj_seq[NUM_UAV] = {0, 0, 0};

std::string readFile(const std::string &path)
{
  std::ifstream input(path.c_str());
  if (!input.good())
    return "";
  return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

void loadObstacleBoxes(const std::string &episode_json)
{
  obstacle_boxes.clear();
  const std::string text = readFile(episode_json);
  if (text.empty())
  {
    ROS_WARN("[traj_bridge] episode_json is empty or unreadable, obstacle safety checks use bounds only.");
    return;
  }

  std::regex box_re(
      "\\\"min_corner\\\"\\s*:\\s*\\[\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\]\\s*,\\s*"
      "\\\"max_corner\\\"\\s*:\\s*\\[\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\]");
  auto begin = std::sregex_iterator(text.begin(), text.end(), box_re);
  auto end = std::sregex_iterator();
  for (auto iter = begin; iter != end; ++iter)
  {
    Box box;
    box.min_corner = Eigen::Vector3d(
        std::stod((*iter)[1].str()),
        std::stod((*iter)[2].str()),
        std::stod((*iter)[3].str())) *
                     voxel_resolution;
    box.max_corner = (Eigen::Vector3d(
                          std::stod((*iter)[4].str()),
                          std::stod((*iter)[5].str()),
                          std::stod((*iter)[6].str())) +
                      Eigen::Vector3d::Ones()) *
                     voxel_resolution;
    obstacle_boxes.push_back(box);
  }
  ROS_INFO("[traj_bridge] loaded %zu obstacle boxes from %s", obstacle_boxes.size(), episode_json.c_str());
}

bool pointInBounds(const Eigen::Vector3d &point)
{
  return (point.array() >= map_min.array()).all() && (point.array() <= map_max.array()).all();
}

bool pointInObstacle(const Eigen::Vector3d &point)
{
  for (const auto &box : obstacle_boxes)
  {
    if ((point.array() >= box.min_corner.array()).all() && (point.array() <= box.max_corner.array()).all())
      return true;
  }
  return false;
}

bool checkTrajSafety(const std::vector<Eigen::Vector3d> &points)
{
  for (const auto &point : points)
  {
    if (!pointInBounds(point))
    {
      ROS_WARN("[traj_bridge] trajectory point outside bounds: (%.2f, %.2f, %.2f)",
               point.x(), point.y(), point.z());
      return false;
    }
    if (pointInObstacle(point))
    {
      ROS_WARN("[traj_bridge] trajectory point inside exported obstacle: (%.2f, %.2f, %.2f)",
               point.x(), point.y(), point.z());
      return false;
    }
  }
  return true;
}

void ensureMinimumPointCount(std::vector<Eigen::Vector3d> &points)
{
  if (points.empty())
    return;
  while (points.size() < 4)
    points.push_back(points.back());
}

void publishBsplineTraj(int uav_id, std::vector<Eigen::Vector3d> points)
{
  ensureMinimumPointCount(points);
  if (points.size() < 4)
    return;

  if (!checkTrajSafety(points))
  {
    ROS_WARN("[traj_bridge] UAV %d trajectory failed safety check", uav_id + 1);
    safety_fallback_pub[uav_id].publish(std_msgs::Empty());
    return;
  }

  std::vector<Eigen::Vector3d> start_end_deriv(4, Eigen::Vector3d::Zero());
  Eigen::MatrixXd ctrl_pts;
  UniformBspline::parameterizeToBspline(knot_interval, points, start_end_deriv, ctrl_pts);
  if (ctrl_pts.cols() == 0)
  {
    ROS_WARN("[traj_bridge] UAV %d failed to parameterize B-spline", uav_id + 1);
    safety_fallback_pub[uav_id].publish(std_msgs::Empty());
    return;
  }

  UniformBspline bspline(ctrl_pts, 3, knot_interval);
  ego_planner::Bspline bspline_msg;
  bspline_msg.order = 3;
  bspline_msg.traj_id = ++traj_seq[uav_id];
  bspline_msg.start_time = ros::Time::now();

  Eigen::MatrixXd pos_pts = bspline.getControlPoint();
  for (int i = 0; i < pos_pts.cols(); ++i)
  {
    geometry_msgs::Point point;
    point.x = pos_pts(0, i);
    point.y = pos_pts(1, i);
    point.z = pos_pts(2, i);
    bspline_msg.pos_pts.push_back(point);
  }

  Eigen::VectorXd knots = bspline.getKnot();
  for (int i = 0; i < knots.rows(); ++i)
    bspline_msg.knots.push_back(knots(i));

  bspline_msg.yaw_pts.push_back(0.0);
  bspline_msg.yaw_dt = knot_interval;
  bspline_pub[uav_id].publish(bspline_msg);

  ROS_INFO("[traj_bridge] UAV %d published B-spline traj_id=%ld with %ld control points",
           uav_id + 1, static_cast<long>(bspline_msg.traj_id), static_cast<long>(pos_pts.cols()));
}

void pythonTrajCallback(const nav_msgs::Path::ConstPtr &msg, int uav_id)
{
  if (msg->poses.empty())
  {
    ROS_WARN("[traj_bridge] received empty trajectory for UAV %d", uav_id + 1);
    return;
  }

  std::vector<Eigen::Vector3d> points;
  points.reserve(msg->poses.size());
  for (const auto &pose : msg->poses)
  {
    points.emplace_back(
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z);
  }

  ROS_INFO("[traj_bridge] UAV %d received ROS-meter trajectory with %zu points", uav_id + 1, points.size());
  publishBsplineTraj(uav_id, points);
}
} // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "traj_bridge");
  ros::NodeHandle nh("~");

  std::string episode_json;
  nh.param("episode_json", episode_json, std::string(""));
  nh.param("knot_interval", knot_interval, 1.0);
  nh.param("voxel_resolution", voxel_resolution, 1.0);
  nh.param("map/min_x", map_min.x(), 0.0);
  nh.param("map/min_y", map_min.y(), 0.0);
  nh.param("map/min_z", map_min.z(), 0.0);
  nh.param("map/max_x", map_max.x(), 20.0);
  nh.param("map/max_y", map_max.y(), 20.0);
  nh.param("map/max_z", map_max.z(), 8.0);

  if (!episode_json.empty())
    loadObstacleBoxes(episode_json);

  ROS_INFO("[traj_bridge] starting trajectory bridge node with 1 voxel = %.2fm", voxel_resolution);
  for (int i = 0; i < NUM_UAV; ++i)
  {
    std::string uav_ns = "/uav" + std::to_string(i + 1);
    python_traj_subs[i] = nh.subscribe<nav_msgs::Path>(
        uav_ns + "/python_traj", 1, boost::bind(&pythonTrajCallback, _1, i));
    bspline_pub[i] = nh.advertise<ego_planner::Bspline>(uav_ns + "/planning/bspline", 1, true);
    safety_fallback_pub[i] = nh.advertise<std_msgs::Empty>(uav_ns + "/traj_bridge/fallback", 1);
    ROS_INFO("[traj_bridge] UAV %d: %s -> %s",
             i + 1,
             (uav_ns + "/python_traj").c_str(),
             (uav_ns + "/planning/bspline").c_str());
  }

  ros::spin();
  return 0;
}
