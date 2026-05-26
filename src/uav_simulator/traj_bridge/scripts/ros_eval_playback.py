#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


COLORS = [
    (1.0, 0.1, 0.1, 1.0),
    (0.1, 0.8, 0.1, 1.0),
    (0.1, 0.25, 1.0, 1.0),
    (0.9, 0.6, 0.1, 1.0),
]


class RosEvalPlayback:
    def __init__(self) -> None:
        self.episode_json = Path(rospy.get_param("~episode_json"))
        self.metrics_dir = Path(rospy.get_param("~metrics_dir", str(self.episode_json.parent)))
        self.frame_id = str(rospy.get_param("~frame_id", "world"))
        self.start_delay = float(rospy.get_param("~start_delay", 2.0))
        self.rate_hz = float(rospy.get_param("~rate", 1.0))

        with self.episode_json.open("r", encoding="utf-8") as handle:
            self.episode = json.load(handle)

        self.start_time = rospy.Time.now() + rospy.Duration(self.start_delay)
        self.num_uavs = int(self.episode["metadata"]["num_uavs"])
        self.time_step_sec = float(self.episode["metadata"].get("time_step_sec", 1.0))
        self.coverage_curve = [float(value) for value in self.episode.get("coverage_curve", [])]
        self.odom_samples: list[list[tuple[float, np.ndarray]]] = [[] for _ in range(self.num_uavs)]

        self.path_pubs = [
            rospy.Publisher(f"/uav{i + 1}/python_traj", RosPath, queue_size=1, latch=True)
            for i in range(self.num_uavs)
        ]
        self.marker_pubs = [
            rospy.Publisher(f"/safe_ctde_mace/uav{i + 1}/python_path_marker", Marker, queue_size=1, latch=True)
            for i in range(self.num_uavs)
        ]
        self.coverage_pub = rospy.Publisher("/safe_ctde_mace/coverage_ratio", Float32, queue_size=1, latch=True)
        self.odom_subs = [
            rospy.Subscriber(f"/uav{i + 1}/sim/odom", Odometry, self._odom_callback, callback_args=i)
            for i in range(self.num_uavs)
        ]

        rospy.on_shutdown(self._write_execution_metrics)
        rospy.Timer(rospy.Duration(max(self.rate_hz, 1e-3) ** -1), self._coverage_timer)
        rospy.Timer(rospy.Duration(self.start_delay), self._publish_once, oneshot=True)

    def _publish_once(self, _event: Any) -> None:
        stamp = rospy.Time.now()
        for index, uav in enumerate(self.episode["uavs"]):
            points = [[float(value) for value in point] for point in uav["trajectory_meters"]]
            path_msg = self._path_message(points, stamp)
            marker_msg = self._marker_message(points, stamp, index)
            self.path_pubs[index].publish(path_msg)
            self.marker_pubs[index].publish(marker_msg)
            rospy.loginfo(
                "[ros_eval_playback] published /uav%d/python_traj with %d points",
                index + 1,
                len(points),
            )
        if self.coverage_curve:
            self.coverage_pub.publish(Float32(data=self.coverage_curve[0]))

    def _path_message(self, points: list[list[float]], stamp: rospy.Time) -> RosPath:
        msg = RosPath()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = stamp
        for point in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def _marker_message(self, points: list[list[float]], stamp: rospy.Time, index: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "safe_ctde_mace_python_paths"
        marker.id = index + 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        color = COLORS[index % len(COLORS)]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        for point in points:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            marker.points.append(pose.pose.position)
        return marker

    def _coverage_timer(self, _event: Any) -> None:
        if not self.coverage_curve:
            return
        elapsed = max((rospy.Time.now() - self.start_time).to_sec(), 0.0)
        index = min(int(elapsed / max(self.time_step_sec, 1e-6)), len(self.coverage_curve) - 1)
        self.coverage_pub.publish(Float32(data=self.coverage_curve[index]))

    def _odom_callback(self, msg: Odometry, index: int) -> None:
        position = np.asarray(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        self.odom_samples[index].append((msg.header.stamp.to_sec(), position))

    def _write_execution_metrics(self) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        destination = self.metrics_dir / "ros_execution_metrics.csv"
        rows = [self._metrics_row(index, samples) for index, samples in enumerate(self.odom_samples)]
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        rospy.loginfo("[ros_eval_playback] wrote %s", destination)

    def _metrics_row(self, index: int, samples: list[tuple[float, np.ndarray]]) -> dict[str, float | int]:
        if len(samples) < 2:
            return {
                "uav_id": index + 1,
                "sample_count": len(samples),
                "path_length": 0.0,
                "mean_speed": 0.0,
                "max_speed": 0.0,
                "mean_acceleration": 0.0,
                "max_acceleration": 0.0,
                "mean_jerk": 0.0,
                "max_jerk": 0.0,
            }
        times = np.asarray([item[0] for item in samples], dtype=float)
        positions = np.stack([item[1] for item in samples])
        dt = np.maximum(np.diff(times), 1e-6)
        displacements = np.diff(positions, axis=0)
        speeds = np.linalg.norm(displacements, axis=1) / dt
        velocities = displacements / dt[:, None]
        accels = np.diff(velocities, axis=0) / np.maximum(dt[1:], 1e-6)[:, None] if len(velocities) > 1 else np.zeros((0, 3))
        jerks = np.diff(accels, axis=0) / np.maximum(dt[2:], 1e-6)[:, None] if len(accels) > 1 else np.zeros((0, 3))
        accel_norms = np.linalg.norm(accels, axis=1) if len(accels) else np.zeros(1)
        jerk_norms = np.linalg.norm(jerks, axis=1) if len(jerks) else np.zeros(1)
        return {
            "uav_id": index + 1,
            "sample_count": len(samples),
            "path_length": float(np.sum(np.linalg.norm(displacements, axis=1))),
            "mean_speed": float(np.mean(speeds)),
            "max_speed": float(np.max(speeds)),
            "mean_acceleration": float(np.mean(accel_norms)),
            "max_acceleration": float(np.max(accel_norms)),
            "mean_jerk": float(np.mean(jerk_norms)),
            "max_jerk": float(np.max(jerk_norms)),
        }


def main() -> None:
    rospy.init_node("ros_eval_playback")
    RosEvalPlayback()
    rospy.spin()


if __name__ == "__main__":
    main()
