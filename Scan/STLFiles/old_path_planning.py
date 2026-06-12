#!/usr/bin/env python3
"""
path_planning.py
Connects to the already-running MoveIt instance and executes a coverage
(painting) path over two faces of a box object on the turntable.

Run AFTER your normal startup (UR driver + MoveIt already running):

    # Visualise waypoints in RViz only (no robot movement):
    source /opt/ros/jazzy/setup.bash
    python3 path_planning.py --visualise

    # Plan and execute on the real robot:
    source /opt/ros/jazzy/setup.bash
    python3 path_planning.py
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, WorkspaceParameters,
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, RobotState, MoveItErrorCodes
)
from moveit_msgs.srv import GetCartesianPath
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

MOVE_GROUP     = "ur_manipulator"
BASE_FRAME     = "base_link"
EEF_LINK       = "tool0"

# Box centre — matches corrected STL offset in environment_setup.py
BOX_CENTRE = np.array([0.89, -0.18, 0.255 + 0.075])

# Box dimensions
BOX_W = 0.10   # width  (X axis)
BOX_D = 0.10   # depth  (Y axis)
BOX_H = 0.15   # height (Z axis)

# Coverage settings
STANDOFF    = 0.15   # 15cm standoff from face surface
STRIPE_STEP = 0.03   # 3cm between stripes (paintbrush width)

# Cartesian path settings
EEF_STEP        = 0.005   # 5mm interpolation resolution
JUMP_THRESHOLD  = 5.0   # was 0.0 — allows larger joint jumps between waypoints
VELOCITY_SCALE  = 0.1     # 10% speed — safe for first run
ACCEL_SCALE     = 0.1
FRACTION_MIN    = 1     # Minimum acceptable path coverage


# ─────────────────────────────────────────────
#  WAYPOINT GENERATORS
# ─────────────────────────────────────────────

def make_pose(position: np.ndarray, quaternion: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x    = float(position[0])
    pose.position.y    = float(position[1])
    pose.position.z    = float(position[2])
    pose.orientation.x = float(quaternion[0])
    pose.orientation.y = float(quaternion[1])
    pose.orientation.z = float(quaternion[2])
    pose.orientation.w = float(quaternion[3])
    return pose

def front_face_waypoints():
    """
    Coverage path for the front face (facing robot, -X direction).
    Tool angled 45 degrees downward toward the face.
    Stripes sweep left-right (Y axis), stepping down from top in Z.
    """
    waypoints = []

    # 45 degree downward tilt toward the front face
    # Rotate around Y to point toward box (-X), then tilt down 45 degrees
    rot  = R.from_euler('YX', [90, -45], degrees=True)
    quat = rot.as_quat()

    face_x     = BOX_CENTRE[0] - BOX_W / 2
    approach_x = face_x - STANDOFF

    z_bottom = BOX_CENTRE[2] - BOX_H / 2
    z_top    = BOX_CENTRE[2] + BOX_H / 2
    y_left   = BOX_CENTRE[1] - BOX_D / 2
    y_right  = BOX_CENTRE[1] + BOX_D / 2

    # Start from top and work downward
    z_values = np.arange(z_top, z_bottom - STRIPE_STEP, -STRIPE_STEP)

    for i, z in enumerate(z_values):
        y_start, y_end = (y_left, y_right) if i % 2 == 0 else (y_right, y_left)
        waypoints.append((make_pose(np.array([approach_x, y_start, z]), quat), 'front'))
        waypoints.append((make_pose(np.array([approach_x, y_end,   z]), quat), 'front'))

    return waypoints


def top_face_waypoints():
    """
    Coverage path for the top face (+Z direction).
    10cm (X) x 15cm (Y). Tool points downward.
    Stripes sweep front-back, stepping across 3cm each time.
    """
    waypoints = []
    rot  = R.from_euler('x', 180, degrees=True)
    quat = rot.as_quat()

    face_z     = BOX_CENTRE[2] + BOX_H / 2
    approach_z = face_z + STANDOFF

    x_left  = BOX_CENTRE[0] - BOX_W / 2
    x_right = BOX_CENTRE[0] + BOX_W / 2
    y_front = BOX_CENTRE[1] - BOX_D / 2
    y_back  = BOX_CENTRE[1] + BOX_D / 2

    x_values = np.arange(x_left, x_right + STRIPE_STEP, STRIPE_STEP)

    for i, x in enumerate(x_values):
        y_start, y_end = (y_front, y_back) if i % 2 == 0 else (y_back, y_front)
        waypoints.append((make_pose(np.array([x, y_start, approach_z]), quat), 'top'))
        waypoints.append((make_pose(np.array([x, y_end,   approach_z]), quat), 'top'))

    return waypoints


# ─────────────────────────────────────────────
#  RVIZ MARKER VISUALISATION
# ─────────────────────────────────────────────

def build_marker_array(waypoints_with_labels: list) -> MarkerArray:
    marker_array = MarkerArray()

    # Arrow markers
    for i, (pose, label) in enumerate(waypoints_with_labels):
        marker        = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.ns     = "waypoints"
        marker.id     = i
        marker.type   = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose   = pose
        marker.scale.x = 0.04
        marker.scale.y = 0.006
        marker.scale.z = 0.010
        marker.color  = (
            ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9) if label == 'front'
            else ColorRGBA(r=0.2, g=1.0, b=0.4, a=0.9)
        )
        marker_array.markers.append(marker)

    # White path line
    line              = Marker()
    line.header.frame_id = BASE_FRAME
    line.ns           = "path_line"
    line.id           = 9000
    line.type         = Marker.LINE_STRIP
    line.action       = Marker.ADD
    line.scale.x      = 0.003
    line.color        = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.5)
    for pose, _ in waypoints_with_labels:
        line.points.append(pose.position)
    marker_array.markers.append(line)

    # Numbered text labels every 4th waypoint
    for i, (pose, _) in enumerate(waypoints_with_labels):
        if i % 4 != 0:
            continue
        text              = Marker()
        text.header.frame_id = BASE_FRAME
        text.ns           = "waypoint_labels"
        text.id           = 10000 + i
        text.type         = Marker.TEXT_VIEW_FACING
        text.action       = Marker.ADD
        text.pose         = pose
        text.pose.position.z += 0.03
        text.scale.z      = 0.02
        text.color        = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text         = str(i)
        marker_array.markers.append(text)

    return marker_array


# ─────────────────────────────────────────────
#  MAIN NODE
# ─────────────────────────────────────────────

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        self.marker_pub = self.create_publisher(
            MarkerArray, '/waypoint_markers', 10)
        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path')
        self.move_action_client = ActionClient(
            self, MoveGroup, '/move_action')
        self.get_logger().info("Path Planner node started")

    def visualise_only(self, waypoints_with_labels: list):
        marker_array = build_marker_array(waypoints_with_labels)
        self.get_logger().info(
            f"Publishing {len(waypoints_with_labels)} waypoint markers to /waypoint_markers")
        self.get_logger().info("  Blue  = front face waypoints")
        self.get_logger().info("  Green = top face waypoints")
        self.get_logger().info("  White line = path order")
        self.get_logger().info(
            "In RViz: Add → By topic → /waypoint_markers → MarkerArray")
        self.get_logger().info("Press Ctrl+C when done viewing.")

        for _ in range(5):
            self.marker_pub.publish(marker_array)
            time.sleep(0.5)

        while rclpy.ok():
            self.marker_pub.publish(marker_array)
            time.sleep(2.0)

    def run(self, waypoints_with_labels: list):
        # Publish markers so you can watch in RViz while the robot moves
        marker_array = build_marker_array(waypoints_with_labels)
        self.marker_pub.publish(marker_array)

        # Strip labels — service just needs Pose objects
        poses = [pose for pose, _ in waypoints_with_labels]

        # ── Wait for cartesian path service ──────────────────
        self.get_logger().info("Waiting for /compute_cartesian_path service...")
        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "Service /compute_cartesian_path not available. "
                "Is MoveIt running?")
            return

        # ── Build cartesian path request ──────────────────────
        self.get_logger().info(
            f"Requesting Cartesian path for {len(poses)} waypoints...")

        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.header.stamp     = self.get_clock().now().to_msg()
        req.group_name       = MOVE_GROUP
        req.link_name        = EEF_LINK
        req.max_step         = EEF_STEP
        req.jump_threshold   = JUMP_THRESHOLD
        req.avoid_collisions = True

        # Wrap each Pose in a PoseStamped
        for pose in poses:
            ps = PoseStamped()
            ps.header.frame_id = BASE_FRAME
            ps.pose = pose
            req.waypoints.append(ps.pose)

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is None:
            self.get_logger().error("Cartesian path service call failed.")
            return

        response = future.result()
        fraction = response.fraction
        self.get_logger().info(
            f"Cartesian path coverage: {fraction * 100:.1f}%")

        if fraction < FRACTION_MIN:
            self.get_logger().warn(
                f"Only {fraction*100:.1f}% of path is reachable. "
                "Check waypoints in RViz — some may be out of reach or in collision.")
            return

        # ── Execute the planned trajectory directly ───────────
        from moveit_msgs.action import ExecuteTrajectory

        execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.get_logger().info("Waiting for /execute_trajectory server...")
        if not execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("/execute_trajectory server not available.")
            return

        self.get_logger().info("Fraction achieved: checking first waypoint reachability...")
        self.get_logger().info(f"First waypoint: {poses[0]}")
        self.get_logger().info(f"Last reachable waypoint index: {int(response.fraction * len(poses))}")

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution

        self.get_logger().info("Executing path on robot...")
        send_goal_future = execute_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=30.0)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

        result = result_future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("✔ Path execution complete!")
        else:
            self.get_logger().error(
                f"Execution failed with error code: {result.error_code.val}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--visualise', action='store_true',
        help="Publish waypoint markers to RViz without moving the robot")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PathPlanner()

    front_wps = front_face_waypoints()
    top_wps     = top_face_waypoints()
    all_waypoints = front_wps + top_wps

    node.get_logger().info(f"  Front face : {len(front_wps)} waypoints")
    node.get_logger().info(f"  Top face   : {len(top_wps)} waypoints")
    node.get_logger().info(f"  Total      : {len(all_waypoints)} waypoints")

    try:
        if args.visualise:
            node.visualise_only(all_waypoints)
        else:
            node.run(all_waypoints)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
