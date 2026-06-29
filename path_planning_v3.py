#!/usr/bin/env python3

# path_planning.py
#
# Connects to the already-running MoveIt instance and executes a coverage
# path over faces extracted automatically from clean_surfaces.stl.
#
# Usage:
#     # Visualise only:
#     python3 path_planning.py --stl ~/Downloads/clean_surfaces.stl --visualise
#
#     # Execute on robot:
#     python3 path_planning.py --stl ~/Downloads/clean_surfaces.stl

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_analyser import extract_faces, FaceRegion

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

MOVE_GROUP = "ur_manipulator"
BASE_FRAME = "base_link"
EEF_LINK   = "tool0"
STL_SCALE  = 0.001   # mm to metres

# Turntable — object centred in XY, base on top surface
TURNTABLE_CENTRE_X = 0.65
TURNTABLE_CENTRE_Y = 0.0
TURNTABLE_TOP_Z    = 0.034

# 5cm clearance above turntable for side faces
SIDE_CLEARANCE_Z   = TURNTABLE_TOP_Z + 0.05   # 0.305m

# Coverage
STANDOFF = 0.19   # was 0.15 — reduced by 4cm for cone tip TCP
STRIPE_STEP = 0.03

# Cartesian path
EEF_STEP       = 0.001   # 1mm interpolation — finer to avoid collision false positives
JUMP_THRESHOLD = 0.0
VELOCITY_SCALE = 0.1
ACCEL_SCALE    = 0.1
FRACTION_MIN   = 0.1

APPROACH_VELOCITY_SCALE = 0.9   # between position moves — faster
APPROACH_ACCEL_SCALE    = 0.9

HOME_JOINTS = {
    'elbow_joint':         -1.968860928212301,
    'shoulder_lift_joint': -0.7263553778277796,
    'shoulder_pan_joint':  -3.0284958521472376,
    'wrist_1_joint':       -1.0970047155963343,
    'wrist_2_joint':        1.3243067264556885,
    'wrist_3_joint':       -2.3142405192004603,
}

FACE_COLOURS = [
    ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9),
    ColorRGBA(r=0.2, g=1.0, b=0.4, a=0.9),
    ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9),
    ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9),
    ColorRGBA(r=0.8, g=0.2, b=1.0, a=0.9),
]

# ─────────────────────────────────────────────
#  KNOWN APPROACH JOINT POSITIONS
#  Measured by manually jogging robot to a good
#  position for each side face and reading joint_states
# ─────────────────────────────────────────────

KNOWN_APPROACH_JOINTS = {
    'right': {
        'elbow_joint':         -1.2880061308490198,
        'shoulder_lift_joint': -2.0423267523394983,
        'shoulder_pan_joint':  -2.5952096621142786,
        'wrist_1_joint':       -2.940122429524557,
        'wrist_2_joint':        0.5996955037117004,
        'wrist_3_joint':        0.28577208518981934,
    },
    'left': {
        'elbow_joint':         -0.7745645681964319,
        'shoulder_lift_joint': -2.3418105284320276,
        'shoulder_pan_joint':  -3.6955350081073206,
        'wrist_1_joint':       -3.006813351308004,
        'wrist_2_joint':        2.5477938652038574,
        'wrist_3_joint':        0.2857601046562195,
    },
}

# ─────────────────────────────────────────────
#  WORLD OFFSET
# ─────────────────────────────────────────────

def compute_world_offset(bbox_size: np.ndarray) -> np.ndarray:
    """
    Offset to place object centred in XY on turntable, base on top surface.
    face_analyser recentres mesh to bbox centre, so:
      XY: turntable centre
      Z:  turntable top + half object height  (puts bbox centre at mid-height)
    """
    return np.array([
        TURNTABLE_CENTRE_X,
        TURNTABLE_CENTRE_Y,
        TURNTABLE_TOP_Z + bbox_size[2] / 2.0,
    ])


# ─────────────────────────────────────────────
#  ORIENTATION
# ─────────────────────────────────────────────

# Measured quaternions — robot manually jogged to face each side
# Tool Z axis points toward the face in each case
SIDE_FACE_QUATS = {
    '-X': np.array([ 0.054436,  0.699910, -0.030011,  0.711521]),  # front
    '-Y': np.array([-0.517208,  0.465416,  0.523521,  0.491742]),  # left
    '+X': np.array([-0.699910,  0.054436,  0.711521,  0.030011]),  # back
    '+Y': np.array([ 0.493914, -0.507041,  0.528914,  0.468197]),  # right
}

# Reference tool Z directions for each entry
SIDE_FACE_REFS = {
    '-X': np.array([ 1.0,  0.0, 0.0]),
    '-Y': np.array([ 0.0,  1.0, 0.0]),
    '+X': np.array([-1.0,  0.0, 0.0]),
    '+Y': np.array([ 0.0, -1.0, 0.0]),
}


def normal_to_quaternion(normal: np.ndarray, position: np.ndarray = None) -> np.ndarray:
    """
    Top face  : tool points straight down (perpendicular to table).
    Side faces: tool points horizontally in the direction of the face normal.

    To adjust side face tilt, change the tool_z Z component:
      0.0  = perfectly horizontal (current)
      -0.1 = slight downward tilt
    """
    n = normal / np.linalg.norm(normal)

    if n[2] > 0.9:
        # Top face — straight down
        return R.from_euler('xyz', [0.0, 180.0, 0.0], degrees=True).as_quat()

    # Side face — point horizontally in direction of face normal
    tool_z = -n.copy()
    tool_z[2] = 0.0
    norm = np.linalg.norm(tool_z)
    if norm < 1e-6:
        tool_z = np.array([1.0, 0.0, 0.0])
    else:
        tool_z /= norm

    tool_x = np.array([0.0, 0.0, 1.0])
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    tool_x = np.cross(tool_y, tool_z)
    return R.from_matrix(np.column_stack([tool_x, tool_y, tool_z])).as_quat()

# def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
#     """Convert face normal to tool quaternion."""
#     normal = normal / np.linalg.norm(normal)
#
#     if abs(normal[2]) > 0.9:
#         # Top or bottom face — tool points straight down
#         tool_z = -normal
#         up     = np.array([1.0, 0.0, 0.0])
#         tool_x = np.cross(up, tool_z);     tool_x /= np.linalg.norm(tool_x)
#         tool_y = np.cross(tool_z, tool_x); tool_y /= np.linalg.norm(tool_y)
#         return R.from_matrix(np.column_stack([tool_x, tool_y, tool_z])).as_quat()
#
#     # Side face — tool should point in -normal direction (toward face)
#     target = -normal
#
#     # Find best match in lookup table
#     best_key = max(SIDE_FACE_REFS,
#                    key=lambda k: np.dot(SIDE_FACE_REFS[k], target))
#     best_dot = np.dot(SIDE_FACE_REFS[best_key], target)
#
#     if best_dot > 0.99:
#         return SIDE_FACE_QUATS[best_key].copy()
#
#     # Diagonal — slerp between two closest
#     sorted_keys = sorted(SIDE_FACE_REFS,
#                           key=lambda k: np.dot(SIDE_FACE_REFS[k], target),
#                           reverse=True)
#     k1, k2 = sorted_keys[0], sorted_keys[1]
#     d1 = np.dot(SIDE_FACE_REFS[k1], target)
#     d2 = np.dot(SIDE_FACE_REFS[k2], target)
#     t  = 1.0 - d1 / (d1 + d2) if (d1 + d2) > 1e-6 else 0.5
#     r1 = R.from_quat(SIDE_FACE_QUATS[k1])
#     r2 = R.from_quat(SIDE_FACE_QUATS[k2])
#     return (r1 * R.from_rotvec((r1.inv() * r2).as_rotvec() * t)).as_quat()


# ─────────────────────────────────────────────
#  FACE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_face(normal: np.ndarray) -> str:
    """
    Classify face for cleaning strategy.
    top    : +Z normal  → full coverage
    left   : -Y normal  → top 50% (or clearance limit)
    right  : +Y normal  → top 50% (or clearance limit)
    skip   : -X (front), +X (back), -Z (bottom)
    """
    n = normal / np.linalg.norm(normal)
    if n[2] > 0.9:   return 'top'
    if n[2] < -0.9:  return 'skip'
    if abs(n[0]) > abs(n[1]):
        return 'skip'   # front or back
    return 'left' if n[1] < 0 else 'right'


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


def compute_side_z_range(world_offset: np.ndarray) -> tuple:
    """
    Z range for side face cleaning.
    Start from top of object, go down to max(50% height, SIDE_CLEARANCE_Z).
    """
    object_height  = (world_offset[2] - TURNTABLE_TOP_Z) * 2.0
    z_top          = TURNTABLE_TOP_Z + object_height
    z_bottom_50pct = z_top - object_height * 0.5
    z_bottom       = max(z_bottom_50pct, SIDE_CLEARANCE_Z)
    print(f"    Side face Z range: {z_bottom:.3f}m to {z_top:.3f}m "
          f"({(z_top-z_bottom)*100:.1f}cm of {object_height*100:.1f}cm total)")
    return z_top, z_bottom


def generate_coverage_waypoints(face: FaceRegion,
                                  world_offset: np.ndarray,
                                  bbox_size: np.ndarray,
                                  standoff: float = STANDOFF,
                                  stripe_step: float = STRIPE_STEP,
                                  face_idx: int = 0):
    """Generate boustrophedon (lawnmower) coverage waypoints for a single face."""
    face_type = classify_face(face.normal)
    label     = f"face_{face_idx}_{face.label}"

    if face_type == 'skip':
        print(f"    Skipping {face.label} (front/back/bottom)")
        return []

    waypoints = []
    world_centroid  = face.centroid + world_offset
    approach_origin = world_centroid + face.normal * standoff

    if face_type == 'top':
        # Sweep along Y axis, step along X axis
        # Stripes run left-right, stepping front-to-back
        sweep_axis = np.array([0.0, 1.0, 0.0])   # left to right
        step_axis  = np.array([1.0, 0.0, 0.0])   # front to back

        half_sweep = face.width  / 2.0
        half_step  = face.height / 2.0

        step_vals     = np.arange(-half_step, half_step + stripe_step, stripe_step)
        left_to_right = True

        for v in step_vals:
            sc      = approach_origin + step_axis * v
            p_start = sc - sweep_axis * half_sweep
            p_end   = sc + sweep_axis * half_sweep
            if not left_to_right:
                p_start, p_end = p_end, p_start
            waypoints.append((make_pose(p_start, normal_to_quaternion(face.normal, p_start)), label))
            waypoints.append((make_pose(p_end,   normal_to_quaternion(face.normal, p_end)),   label))
            left_to_right = not left_to_right

    else:
        # Side faces — sweep horizontally, step downward
        # Stripes run left-right, stepping top-to-bottom
        z_top, z_bottom = compute_side_z_range(world_offset)
        half_w          = face.width / 2.0

        # Horizontal sweep axis — flatten to XY plane
        sweep_axis = face.u_axis.copy()
        sweep_axis[2] = 0.0
        if np.linalg.norm(sweep_axis) < 1e-6:
            sweep_axis = np.array([1.0, 0.0, 0.0])
        else:
            sweep_axis /= np.linalg.norm(sweep_axis)

        z_vals        = np.arange(z_top, z_bottom - stripe_step, -stripe_step)
        left_to_right = True

        for z in z_vals:
            sc      = approach_origin.copy()
            sc[2]   = z
            p_start = sc - sweep_axis * half_w
            p_end   = sc + sweep_axis * half_w
            if not left_to_right:
                p_start, p_end = p_end, p_start
            waypoints.append((make_pose(p_start, normal_to_quaternion(face.normal, p_start)), label))
            waypoints.append((make_pose(p_end,   normal_to_quaternion(face.normal, p_end)),   label))
            left_to_right = not left_to_right

    return waypoints


# ─────────────────────────────────────────────
#  RVIZ MARKERS
# ─────────────────────────────────────────────

def build_marker_array(waypoints_with_labels, face_regions) -> MarkerArray:
    marker_array = MarkerArray()
    colour_map   = {
        f"face_{i}_{face.label}": FACE_COLOURS[i % len(FACE_COLOURS)]
        for i, face in enumerate(face_regions)
    }
    for i, (pose, label) in enumerate(waypoints_with_labels):
        m                 = Marker()
        m.header.frame_id = BASE_FRAME
        m.ns              = "waypoints"
        m.id              = i
        m.type            = Marker.ARROW
        m.action          = Marker.ADD
        m.pose            = pose
        m.scale.x         = 0.04
        m.scale.y         = 0.006
        m.scale.z         = 0.010
        m.color           = colour_map.get(label, FACE_COLOURS[0])
        marker_array.markers.append(m)

    line                  = Marker()
    line.header.frame_id  = BASE_FRAME
    line.ns               = "path_line"
    line.id               = 9000
    line.type             = Marker.LINE_STRIP
    line.action           = Marker.ADD
    line.scale.x          = 0.003
    line.color            = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.5)
    for pose, _ in waypoints_with_labels:
        line.points.append(pose.position)
    marker_array.markers.append(line)

    for i, (pose, _) in enumerate(waypoints_with_labels):
        if i % 4 != 0:
            continue
        t                  = Marker()
        t.header.frame_id  = BASE_FRAME
        t.ns               = "labels"
        t.id               = 10000 + i
        t.type             = Marker.TEXT_VIEW_FACING
        t.action           = Marker.ADD
        t.pose             = pose
        t.pose.position.z += 0.03
        t.scale.z          = 0.02
        t.color            = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        t.text             = str(i)
        marker_array.markers.append(t)

    return marker_array


# ─────────────────────────────────────────────
#  PLANNING SCENE OBJECT
# ─────────────────────────────────────────────

def add_object_to_scene(node, stl_filepath, bbox_centre,
                         world_offset, scale=0.001):
    """
    Add STL as collision object to MoveIt planning scene.
    Vertices are recentred (subtract bbox_centre) then placed at world_offset.
    """
    import pyassimp
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import Mesh, MeshTriangle
    from geometry_msgs.msg import Point
    from std_msgs.msg import Header

    collision_pub = node.create_publisher(
        CollisionObject, '/collision_object', 10)

    obj                    = CollisionObject()
    obj.header             = Header()
    obj.header.frame_id    = BASE_FRAME
    obj.header.stamp       = node.get_clock().now().to_msg()
    obj.id                 = "scan_object"
    obj.operation          = CollisionObject.ADD

    with pyassimp.load(os.path.expanduser(stl_filepath)) as scene:
        raw_verts = np.array(scene.meshes[0].vertices, dtype=float) * scale
        # Recentre vertices around bbox centre (same as face_analyser does)
        centred_verts = raw_verts - bbox_centre

        mesh = Mesh()
        for face in scene.meshes[0].faces:
            tri                   = MeshTriangle()
            tri.vertex_indices    = [face[0], face[1], face[2]]
            mesh.triangles.append(tri)
        for v in centred_verts:
            pt   = Point()
            pt.x = float(v[0])
            pt.y = float(v[1])
            pt.z = float(v[2])
            mesh.vertices.append(pt)

    obj.meshes.append(mesh)

    # Place at world offset (bbox centre in world frame)
    pose               = Pose()
    pose.position.x    = float(world_offset[0])
    pose.position.y    = float(world_offset[1])
    pose.position.z    = float(world_offset[2])
    pose.orientation.w = 1.0
    obj.mesh_poses.append(pose)

    node.get_logger().info("Adding scan object to planning scene...")
    for _ in range(5):
        collision_pub.publish(obj)
        time.sleep(0.3)
    node.get_logger().info("✔ Scan object added")

def sort_faces(waypoints_by_face, face_regions):
    """Sort faces into execution order: top → right → left → skip."""
    FACE_ORDER = {'top': 0, 'right': 1, 'left': 2, 'skip': 3}
    paired = list(zip(waypoints_by_face, face_regions))
    paired.sort(key=lambda x: FACE_ORDER.get(classify_face(x[1].normal), 3))
    return paired

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

    def visualise_only(self, waypoints_with_labels, face_regions,
                        stl_filepath, bbox_centre, world_offset, bbox_size):
        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset)
        ma = build_marker_array(waypoints_with_labels, face_regions)
        self.get_logger().info(
            f"Publishing {len(waypoints_with_labels)} waypoints "
            f"for {len(face_regions)} faces")
        for i, face in enumerate(face_regions):
            self.get_logger().info(f"  Face {i}: {face.label}")
        self.get_logger().info(
            "In RViz: Add → By topic → /waypoint_markers → MarkerArray")
        self.get_logger().info("Press Ctrl+C when done.")
        for _ in range(5):
            self.marker_pub.publish(ma)
            time.sleep(0.5)
        while rclpy.ok():
            self.marker_pub.publish(ma)
            time.sleep(2.0)

    def preview_path(self, waypoints_by_face, face_regions,
                     stl_filepath, bbox_centre, world_offset, bbox_size):
        """Plan all trajectories and preview in RViz face by face — no robot movement."""
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
        from trajectory_msgs.msg import JointTrajectory
        from sensor_msgs.msg import JointState

        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset)

        display_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)
        paired = sort_faces(waypoints_by_face, face_regions)
        sorted_wps = [wp for wps, _ in paired for wp in wps]
        sorted_faces = [face for _, face in paired]

        # Publish full path markers to RViz
        ma = build_marker_array(sorted_wps, sorted_faces)
        for _ in range(5):
            self.marker_pub.publish(ma)
            time.sleep(0.3)

        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Service not available. Is MoveIt running?")
            return

        print("\n" + "─" * 60)
        print("PREVIEW MODE — no robot movement")
        print("Each face is animated in RViz before you confirm.")
        print("Press ENTER to preview next face, Ctrl+C to abort.")
        print("─" * 60)

        last_state = None

        for face_wps, face in paired:
            if not face_wps:
                continue

            face_type = classify_face(face.normal)
            poses = [pose for pose, _ in face_wps]
            stripes = [(poses[i], poses[i + 1]) for i in range(0, len(poses) - 1, 2)]

            print(f"\n── Planning: {face.label} ({face_type.upper()}) "
                  f"— {len(stripes)} stripes ──")

            combined_points = []
            joint_names = None


            for si, (p_start, p_end) in enumerate(stripes):
                req = GetCartesianPath.Request()
                req.header.frame_id = BASE_FRAME
                req.header.stamp = self.get_clock().now().to_msg()
                req.group_name = MOVE_GROUP
                req.link_name = EEF_LINK
                req.max_step = EEF_STEP
                req.jump_threshold = JUMP_THRESHOLD
                req.avoid_collisions = True
                req.waypoints = [p_start, p_end]

                # Chain from end of previous stripe
                if last_state is not None:
                    req.start_state = last_state

                future = self.cartesian_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

                if future.result() is None or future.result().fraction < FRACTION_MIN:
                    print(f"  Stripe {si}: could not plan — skipping")
                    continue

                result = future.result()
                new_points = list(result.solution.joint_trajectory.points)
                joint_names = result.solution.joint_trajectory.joint_names

                # Offset timestamps so stripes chain sequentially
                if combined_points:
                    last_time = combined_points[-1].time_from_start
                    for pt in new_points:
                        pt.time_from_start.sec += last_time.sec
                        pt.time_from_start.nanosec += last_time.nanosec
                        if pt.time_from_start.nanosec >= 1_000_000_000:
                            pt.time_from_start.sec += 1
                            pt.time_from_start.nanosec -= 1_000_000_000

                combined_points += new_points

                # Save end state for next stripe
                last_js = JointState()
                last_js.name = list(joint_names)
                last_js.position = list(new_points[-1].positions)
                last_state = RobotState()
                last_state.joint_state = last_js

            if not combined_points or joint_names is None:
                print(f"  No trajectories planned for {face.label} — skipping")
                continue

            # Build combined trajectory for this face
            traj = RobotTrajectory()
            traj.joint_trajectory = JointTrajectory()
            traj.joint_trajectory.joint_names = list(joint_names)
            traj.joint_trajectory.points = combined_points

            # Publish to RViz display
            display_msg = DisplayTrajectory()
            display_msg.model_id = "ur10"
            display_msg.trajectory = [traj]
            display_msg.trajectory_start.is_diff = True

            print(f"  Publishing to RViz — watch the robot model...")
            for _ in range(3):
                display_pub.publish(display_msg)
                time.sleep(0.5)

            try:
                input("  Press ENTER to preview next face > ")
            except KeyboardInterrupt:
                print("\nPreview aborted.")
                return

        # Final confirmation
        print("\n" + "─" * 60)
        print("Preview complete.")
        print("─" * 60)
        try:
            confirm = input("Execute for real on the robot? (y/n) > ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted.")
            return

        if confirm == 'y':
            print("Starting execution...")
            self.run(waypoints_by_face, face_regions,
                     stl_filepath, bbox_centre, world_offset, bbox_size)
        else:
            print("Execution cancelled. Run again when ready.")

    def sanitise_trajectory(self, trajectory):
        """Fix CB3 TypeError: recursively flatten any nested lists to floats."""

        def to_float(val):
            while isinstance(val, (list, tuple)):
                val = val[0]
            return float(val)

        for point in trajectory.joint_trajectory.points:
            point.positions = [to_float(p) for p in point.positions]
            point.velocities = [to_float(v) for v in point.velocities]
            point.accelerations = [to_float(a) for a in point.accelerations]
            if point.effort:
                point.effort = [to_float(e) for e in point.effort]
        return trajectory

    def move_to_home(self) -> bool:
        """Return robot to home position after cleaning is complete."""
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import JointConstraint, Constraints

        self.get_logger().info("Returning to home position...")
        move_client = ActionClient(self, MoveGroup, '/move_action')
        if not move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("  /move_action not available")
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = MOVE_GROUP
        goal.request.allowed_planning_time = 10.0
        goal.request.num_planning_attempts = 5
        goal.request.max_velocity_scaling_factor = APPROACH_VELOCITY_SCALE
        goal.request.max_acceleration_scaling_factor = APPROACH_ACCEL_SCALE
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True

        constraints = Constraints()
        for name, value in HOME_JOINTS.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        future = move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None or not future.result().accepted:
            self.get_logger().warn("Home goal rejected")
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        self.get_logger().info("✔ Home position reached")
        return True

    def plan_and_execute_face(self, waypoints_with_labels, face_label, execute_client):
        """Plan and execute a face by planning each stripe separately."""
        if not waypoints_with_labels:
            return False

        poses = [pose for pose, _ in waypoints_with_labels]

        # Split into stripes — each pair of waypoints is one stripe
        stripes = [(poses[i], poses[i + 1]) for i in range(0, len(poses) - 1, 2)]

        execute_client_local = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory')
        if not execute_client_local.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("/execute_trajectory not available.")
            return False

        success_count = 0
        for si, (p_start, p_end) in enumerate(stripes):
            req = GetCartesianPath.Request()
            req.header.frame_id = BASE_FRAME
            req.header.stamp = self.get_clock().now().to_msg()
            req.group_name = MOVE_GROUP
            req.link_name = EEF_LINK
            req.max_step = EEF_STEP
            req.jump_threshold = JUMP_THRESHOLD
            req.avoid_collisions = True
            req.waypoints = [p_start, p_end]

            future = self.cartesian_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

            if future.result() is None:
                self.get_logger().warn(f"  {face_label} stripe {si}: service call failed")
                continue

            response = future.result()
            if response.fraction < 0.1:
                self.get_logger().warn(
                    f"  {face_label} stripe {si}: only {response.fraction*100:.0f}% — skipping")
                continue

            MAX_RETRIES = 10
            for attempt in range(MAX_RETRIES):
                exec_goal            = ExecuteTrajectory.Goal()
                exec_goal.trajectory = self.sanitise_trajectory(response.solution)
                send_future          = execute_client_local.send_goal_async(exec_goal)
                rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)

                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    self.get_logger().warn(f"  {face_label} stripe {si} attempt {attempt+1}: rejected")
                    break

                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

                result = result_future.result().result
                if result.error_code.val == MoveItErrorCodes.SUCCESS:
                    success_count += 1
                    break
                else:
                    self.get_logger().warn(
                        f"  {face_label} stripe {si} attempt {attempt+1}: error {result.error_code.val}")
                    if result.error_code.val == -4:
                        if attempt < MAX_RETRIES - 1:
                            self.get_logger().info(f"  Retrying stripe {si} in 2 seconds...")
                            time.sleep(2.0)
                        else:
                            self.get_logger().warn(
                                f"  {face_label} stripe {si}: failed after {MAX_RETRIES} attempts — "
                                "resolve on pendant then press Enter...")
                            input("  Press Enter to continue > ")
                    else:
                        break

        self.get_logger().info(
            f"  ✔ {face_label}: {success_count}/{len(stripes)} stripes complete")
        return success_count > 0

    def move_to_approach_pose(self, face_type: str) -> bool:
        """
        Move to the known-good joint position for a side face using
        joint-space planning. Called before planning each side face's
        Cartesian stripes to ensure the robot is in a good configuration.
        """
        if face_type not in KNOWN_APPROACH_JOINTS:
            return True   # no known pose for this face type — skip

        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import JointConstraint, Constraints

        joints = KNOWN_APPROACH_JOINTS[face_type]
        self.get_logger().info(
            f"  Moving to {face_type} approach position...")

        move_client = ActionClient(self, MoveGroup, '/move_action')
        if not move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("  /move_action not available — skipping")
            return False

        goal                                         = MoveGroup.Goal()
        goal.request.group_name                      = MOVE_GROUP
        goal.request.allowed_planning_time           = 10.0
        goal.request.num_planning_attempts           = 20
        goal.request.max_velocity_scaling_factor     = APPROACH_VELOCITY_SCALE
        goal.request.max_acceleration_scaling_factor = APPROACH_ACCEL_SCALE
        goal.planning_options.plan_only              = False
        goal.planning_options.replan                 = True

        constraints = Constraints()
        for name, value in joints.items():
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = value
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        future = move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None or not future.result().accepted:
            self.get_logger().warn(f"  {face_type} approach goal rejected")
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        self.get_logger().info(f"  ✔ {face_type} approach position reached")
        return True

    def run(self, waypoints_by_face, face_regions,
             stl_filepath, bbox_centre, world_offset, bbox_size):
        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset)

        # Sort faces: right first, then left, then top
        FACE_ORDER = {'top': 0, 'right': 1, 'left': 2, 'skip': 3}
        paired = list(zip(waypoints_by_face, face_regions))
        paired.sort(key=lambda x: FACE_ORDER.get(classify_face(x[1].normal), 3))

        # Use sorted paired order for marker array too
        paired_wps   = [wps for wps, _ in paired]
        paired_faces = [face for _, face in paired]
        ma = build_marker_array(
            [wp for wps in paired_wps for wp in wps], paired_faces)
        self.marker_pub.publish(ma)

        self.get_logger().info("Waiting for /compute_cartesian_path...")
        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Service not available. Is MoveIt running?")
            return

        visited_sides = set()
        for i, (face_wps, face) in enumerate(paired):
            if not face_wps:
                continue
            label     = f"face_{i}_{face.label}"
            face_type = classify_face(face.normal)
            self.get_logger().info(f"\nFace {i}: {face.label} ({face_type})")

            # Only move to approach position the first time we visit each side
            if face_type in ('left', 'right') and face_type not in visited_sides:
                self.move_to_approach_pose(face_type)
                visited_sides.add(face_type)
                time.sleep(0.5)

            self.plan_and_execute_face(face_wps, label, None)
            time.sleep(1.0)

        self.get_logger().info("\n✔ All faces complete!")
        self.move_to_home()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stl',       type=str,   required=True)
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--preview', action='store_true', help="Preview planned trajectories in RViz without executing")
    parser.add_argument('--scale',     type=float, default=STL_SCALE)
    parser.add_argument('--standoff',  type=float, default=STANDOFF)
    parser.add_argument('--stripe',    type=float, default=STRIPE_STEP)
    args, _ = parser.parse_known_args()

    face_regions, bbox_centre, bbox_size = extract_faces(
        stl_filepath=args.stl,
        scale=args.scale,
        angle_threshold_deg=30.0,
        min_area_cm2=5.0,
    )

    if not face_regions:
        print("[ERROR] No faces extracted. Check --scale.")
        return


    world_offset = compute_world_offset(bbox_size)
    print(f"\nObject placement:")
    print(f"  Turntable centre   : ({TURNTABLE_CENTRE_X}, {TURNTABLE_CENTRE_Y})")
    print(f"  Object half-height : {bbox_size[2]/2*100:.1f}cm")
    print(f"  World offset       : {np.round(world_offset, 3)}")
    print(f"  SIDE_CLEARANCE_Z   : {SIDE_CLEARANCE_Z:.3f}m")

    waypoints_by_face = []
    all_waypoints     = []
    for i, face in enumerate(face_regions):
        wps = generate_coverage_waypoints(
            face, world_offset, bbox_size,
            standoff    = args.standoff,
            stripe_step = args.stripe,
            face_idx    = i,
        )
        waypoints_by_face.append(wps)
        all_waypoints += wps
        print(f"  Face {i} ({face.label}): {len(wps)} waypoints")

    print(f"  Total: {len(all_waypoints)} waypoints")

    # Sort faces: right first, then left, then top
    FACE_ORDER = {'top': 0, 'right': 1, 'left': 2, 'skip': 3}
    paired = list(zip(waypoints_by_face, face_regions))
    paired.sort(key=lambda x: FACE_ORDER.get(classify_face(x[1].normal), 3))
    sorted_wps = [wp for wps, _ in paired for wp in wps]
    sorted_faces = [face for _, face in paired]

    rclpy.init()
    node = PathPlanner()

    try:
        if args.visualise:
            node.visualise_only(sorted_wps, sorted_faces,
                                args.stl, bbox_centre, world_offset, bbox_size)
        elif args.preview:
            node.preview_path(waypoints_by_face, face_regions,
                              args.stl, bbox_centre, world_offset, bbox_size)
        else:
            node.run(waypoints_by_face, face_regions,
                     args.stl, bbox_centre, world_offset, bbox_size)
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