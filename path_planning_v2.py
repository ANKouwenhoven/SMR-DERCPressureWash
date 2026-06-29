#!/usr/bin/env python3
"""
path_planning.py

Connects to the already-running MoveIt instance and executes a coverage
(painting) path over faces extracted automatically from clean_surfaces.stl.

Run AFTER your normal startup (UR driver + MoveIt already running):

    # Visualise waypoints in RViz only (no robot movement):
    source /opt/ros/jazzy/setup.bash
    python3 path_planning.py --visualise --stl ~/Downloads/clean_surfaces.stl

    # Plan and execute on the real robot:
    source /opt/ros/jazzy/setup.bash
    python3 path_planning.py --stl ~/Downloads/clean_surfaces.stl
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints, MoveItErrorCodes
)
from moveit_msgs.srv import GetCartesianPath
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time
import sys
import os

# Face analyser must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_analyser import extract_faces, FaceRegion

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

MOVE_GROUP  = "ur_manipulator"
BASE_FRAME  = "base_link"
EEF_LINK    = "tool0"

# World position of the object origin (turntable centre top surface)
# This offsets the STL coordinates into the robot's world frame
OBJECT_ORIGIN = np.array([0.89, -0.18, 0.255])  # metres

# STL settings
STL_SCALE   = 0.001   # mm to metres

# Coverage settings
STANDOFF    = 0.15    # 15cm standoff from face surface
STRIPE_STEP = 0.03    # 3cm between stripes (paintbrush width)

# Cartesian path settings
EEF_STEP        = 0.005
JUMP_THRESHOLD  = 0.0
VELOCITY_SCALE  = 0.1
ACCEL_SCALE     = 0.1
FRACTION_MIN    = 0.5   # Accept 50% — some faces may be partially unreachable

# Colours per face index for RViz markers
FACE_COLOURS = [
    ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9),   # blue
    ColorRGBA(r=0.2, g=1.0, b=0.4, a=0.9),   # green
    ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9),   # orange
    ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9),   # red
    ColorRGBA(r=0.8, g=0.2, b=1.0, a=0.9),   # purple
]


# ─────────────────────────────────────────────
#  WAYPOINT GENERATION
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


def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    """
    Convert a surface normal into a tool quaternion so that the tool Z axis
    points AGAINST the normal (i.e. toward the surface).

    The tool approaches from the normal direction and points inward.
    """
    # We want the tool Z to point in the -normal direction (toward surface)
    tool_z = -normal / np.linalg.norm(normal)

    # Build a rotation matrix where Z = tool_z
    # Choose an arbitrary up vector not parallel to tool_z
    if abs(tool_z[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])

    tool_x = np.cross(up, tool_z)
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)

    # Build rotation matrix (columns = x, y, z axes of tool frame)
    rot_matrix = np.column_stack([tool_x, tool_y, tool_z])
    rot = R.from_matrix(rot_matrix)
    return rot.as_quat()  # [x, y, z, w]


def generate_coverage_waypoints(face: FaceRegion,
                                  standoff: float = STANDOFF,
                                  stripe_step: float = STRIPE_STEP,
                                  face_idx: int = 0):
    """
    Generate lawnmower coverage waypoints for a single face region.

    The tool sweeps across the face in parallel stripes, maintaining
    a fixed standoff distance and pointing toward the surface.

    Args:
        face        : FaceRegion from face_analyser
        standoff    : Distance from face surface to tool tip (metres)
        stripe_step : Distance between stripes (metres)
        face_idx    : Index for labelling

    Returns:
        List of (Pose, label) tuples
    """
    waypoints = []

    # Tool quaternion — pointing toward face from standoff direction
    quat = normal_to_quaternion(face.normal)

    approach_origin = (face.centroid + OBJECT_ORIGIN) + face.normal * standoff

    # Generate stripe positions along v_axis
    half_height = face.height / 2.0
    half_width  = face.width  / 2.0

    v_values = np.arange(-half_height, half_height + stripe_step, stripe_step)
    label    = f"face_{face_idx}_{face.label}"

    for i, v in enumerate(v_values):
        # Step along v_axis (stripe position)
        stripe_centre = approach_origin + face.v_axis * v

        # Stripe endpoints along u_axis
        p_start = stripe_centre - face.u_axis * half_width
        p_end   = stripe_centre + face.u_axis * half_width

        # Alternate direction (lawnmower)
        if i % 2 != 0:
            p_start, p_end = p_end, p_start

        waypoints.append((make_pose(p_start, quat), label))
        waypoints.append((make_pose(p_end,   quat), label))

    return waypoints


# ─────────────────────────────────────────────
#  RVIZ MARKER VISUALISATION
# ─────────────────────────────────────────────

def build_marker_array(waypoints_with_labels: list,
                        face_regions: list) -> MarkerArray:
    marker_array = MarkerArray()

    # Build label → colour mapping
    colour_map = {}
    for i, face in enumerate(face_regions):
        colour_map[f"face_{i}_{face.label}"] = FACE_COLOURS[i % len(FACE_COLOURS)]

    # Arrow markers
    for i, (pose, label) in enumerate(waypoints_with_labels):
        marker             = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.ns          = "waypoints"
        marker.id          = i
        marker.type        = Marker.ARROW
        marker.action      = Marker.ADD
        marker.pose        = pose
        marker.scale.x     = 0.04
        marker.scale.y     = 0.006
        marker.scale.z     = 0.010
        marker.color       = colour_map.get(label, FACE_COLOURS[0])
        marker_array.markers.append(marker)

    # White path line
    line                   = Marker()
    line.header.frame_id   = BASE_FRAME
    line.ns                = "path_line"
    line.id                = 9000
    line.type              = Marker.LINE_STRIP
    line.action            = Marker.ADD
    line.scale.x           = 0.003
    line.color             = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.5)
    for pose, _ in waypoints_with_labels:
        line.points.append(pose.position)
    marker_array.markers.append(line)

    # Numbered text labels every 4th waypoint
    for i, (pose, _) in enumerate(waypoints_with_labels):
        if i % 4 != 0:
            continue
        text                   = Marker()
        text.header.frame_id   = BASE_FRAME
        text.ns                = "waypoint_labels"
        text.id                = 10000 + i
        text.type              = Marker.TEXT_VIEW_FACING
        text.action            = Marker.ADD
        text.pose              = pose
        text.pose.position.z  += 0.03
        text.scale.z           = 0.02
        text.color             = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text              = str(i)
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
        self.get_logger().info("Path Planner node started")

    def visualise_only(self, waypoints_with_labels, face_regions):
        marker_array = build_marker_array(waypoints_with_labels, face_regions)

        self.get_logger().info(
            f"Publishing {len(waypoints_with_labels)} waypoint markers")
        for i, face in enumerate(face_regions):
            self.get_logger().info(f"  Face {i}: {face.label}")
        self.get_logger().info(
            "In RViz: Add → By topic → /waypoint_markers → MarkerArray")
        self.get_logger().info("Press Ctrl+C when done viewing.")

        for _ in range(5):
            self.marker_pub.publish(marker_array)
            time.sleep(0.5)

        while rclpy.ok():
            self.marker_pub.publish(marker_array)
            time.sleep(2.0)

    def run(self, waypoints_with_labels, face_regions):
        # Publish markers
        marker_array = build_marker_array(waypoints_with_labels, face_regions)
        self.marker_pub.publish(marker_array)

        poses = [pose for pose, _ in waypoints_with_labels]

        # Wait for cartesian path service
        self.get_logger().info("Waiting for /compute_cartesian_path service...")
        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Service not available. Is MoveIt running?")
            return

        self.get_logger().info(
            f"Requesting Cartesian path for {len(poses)} waypoints...")

        req                  = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.header.stamp     = self.get_clock().now().to_msg()
        req.group_name       = MOVE_GROUP
        req.link_name        = EEF_LINK
        req.max_step         = EEF_STEP
        req.jump_threshold   = JUMP_THRESHOLD
        req.avoid_collisions = True
        for pose in poses:
            req.waypoints.append(pose)

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
                f"Only {fraction*100:.1f}% reachable. "
                "Check waypoints in RViz.")
            return

        # Execute trajectory
        execute_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory')
        self.get_logger().info("Waiting for /execute_trajectory server...")
        if not execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("/execute_trajectory not available.")
            return

        exec_goal            = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution

        self.get_logger().info("Executing path on robot...")
        send_goal_future = execute_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(
            self, send_goal_future, timeout_sec=30.0)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=60.0)

        result = result_future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("✔ Path execution complete!")
        else:
            self.get_logger().error(
                f"Execution failed with error code: {result.error_code.val}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stl',       type=str, required=True,
                        help="Path to clean_surfaces.stl")
    parser.add_argument('--visualise', action='store_true',
                        help="Publish markers to RViz without moving robot")
    parser.add_argument('--scale',     type=float, default=STL_SCALE,
                        help="STL scale factor (default 0.001 = mm to metres)")
    parser.add_argument('--standoff',  type=float, default=STANDOFF,
                        help="Tool standoff distance in metres (default 0.15)")
    parser.add_argument('--stripe',    type=float, default=STRIPE_STEP,
                        help="Stripe step in metres (default 0.03)")
    args, _ = parser.parse_known_args()

    # ── Extract faces from STL ────────────────────────────────
    print("\nAnalysing STL file...")
    face_regions = extract_faces(
        stl_filepath        = args.stl,
        scale               = args.scale,
        angle_threshold_deg = 10.0,
        min_area_cm2        = 0.5,
    )

    if not face_regions:
        print("[ERROR] No faces extracted from STL. Check --scale and --min-area.")
        return

    # ── Generate waypoints for all faces ─────────────────────
    all_waypoints = []
    for i, face in enumerate(face_regions):
        wps = generate_coverage_waypoints(
            face, standoff=args.standoff,
            stripe_step=args.stripe, face_idx=i)
        all_waypoints += wps
        print(f"  Face {i} ({face.label}): {len(wps)} waypoints")

    print(f"  Total: {len(all_waypoints)} waypoints across "
          f"{len(face_regions)} faces")

    # ── ROS2 node ─────────────────────────────────────────────
    rclpy.init()
    node = PathPlanner()

    try:
        if args.visualise:
            node.visualise_only(all_waypoints, face_regions)
        else:
            node.run(all_waypoints, face_regions)
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
