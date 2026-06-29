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
from cleaning_visualizer import CleaningVisualizer

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
TURNTABLE_TOP_Z    = 0.038

# The scan bbox floor is clipped ~3cm above the real turntable surface
SCAN_FLOOR_CLIP_M  = 0.03

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
        TURNTABLE_TOP_Z + SCAN_FLOOR_CLIP_M + bbox_size[2] / 2.0,
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
    """
    Generate simple paired (start, end) waypoints per stripe.
    Each stripe is two points — MoveIt plans a straight Cartesian line between them.
    The Z/orientation lock is enforced via path_constraints in the planner request,
    not by pre-stuffing extra waypoints.
    """
    face_type = classify_face(face.normal)
    label     = f"face_{face_idx}_{face.label}"

    if face_type == 'skip':
        print(f"    Skipping {face.label} (front/back/bottom)")
        return []

    waypoints       = []
    world_centroid  = face.centroid + world_offset
    approach_origin = world_centroid + face.normal * standoff

    if face_type == 'top':
        sweep_axis = np.array([0.0, 1.0, 0.0])   # sweep left-right in Y
        step_axis  = np.array([1.0, 0.0, 0.0])   # step front-back in X
        half_sweep = bbox_size[1] / 2.0
        half_step  = bbox_size[0] / 2.0
        step_vals     = np.arange(-half_step, half_step + stripe_step, stripe_step)
        left_to_right = True
        q             = normal_to_quaternion(face.normal)

        for i, v in enumerate(step_vals):
            sc      = approach_origin + step_axis * v
            p_start = sc - sweep_axis * half_sweep
            p_end   = sc + sweep_axis * half_sweep
            if not left_to_right:
                p_start, p_end = p_end, p_start

            waypoints.append((make_pose(p_start, q), label))
            waypoints.append((make_pose(p_end,   q), label))

            # Insert an L-shaped corner between this stripe end and next stripe start.
            # Corner = (next_stripe_x, current_stripe_y, same_z)
            # This splits the diagonal into: step X → step Y, two clean straight moves.
            if i < len(step_vals) - 1:
                v_next     = step_vals[i + 1]
                sc_next    = approach_origin + step_axis * v_next
                # p_end is a numpy position array at this point
                corner_pos = np.array([sc_next[0],    # X of next stripe
                                       p_end[1],       # Y stays at current stripe end
                                       approach_origin[2]])  # Z constant
                waypoints.append((make_pose(corner_pos, q), label + '_corner'))

            left_to_right = not left_to_right

    else:
        z_top, z_bottom = compute_side_z_range(world_offset)

        # Sweep axis = world axis perpendicular to the face normal in XY plane.
        # For a +Y/-Y face (right/left), sweep along world X.
        # For a +X/-X face (front/back), sweep along world Y.
        # This guarantees perfectly horizontal stripes along a clean world axis —
        # no diagonal drift from face.u_axis which can be arbitrary.
        n = face.normal / np.linalg.norm(face.normal)
        if abs(n[1]) > abs(n[0]):
            sweep_axis = np.array([1.0, 0.0, 0.0])   # +Y/-Y face → sweep in X
        else:
            sweep_axis = np.array([0.0, 1.0, 0.0])   # +X/-X face → sweep in Y

        # Use the object bbox projected onto the sweep axis as the sweep width,
        # not face.width which can be noisy from the STL mesh.
        half_w = (bbox_size[0] if abs(n[1]) > abs(n[0]) else bbox_size[1]) / 2.0

        # Keep approach_origin XY fixed — only vary Z per stripe
        approach_xy = approach_origin[:2].copy()

        z_vals        = np.arange(z_top, z_bottom - stripe_step, -stripe_step)
        left_to_right = True
        q             = normal_to_quaternion(face.normal)   # same for every stripe

        for z in z_vals:
            sc        = np.array([approach_xy[0], approach_xy[1], z])
            p_start   = sc - sweep_axis * half_w
            p_end     = sc + sweep_axis * half_w
            if not left_to_right:
                p_start, p_end = p_end, p_start
            waypoints.append((make_pose(p_start, q), label))
            waypoints.append((make_pose(p_end,   q), label))
            left_to_right = not left_to_right

    print(f"    {label}: {len(waypoints)} waypoints ({len(waypoints)//2} stripes)")
    return waypoints


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
                         world_offset, scale=0.001, scene_stl=None):
    """
    Add STL as collision object to MoveIt planning scene.
    Vertices are recentred (subtract bbox_centre) then placed at world_offset.
    If scene_stl is provided it is used for the visual mesh (e.g. hull/primitive)
    while stl_filepath drives face extraction / path planning as normal.
    A 90° rotation around Z is applied to align the scanner world frame
    (Y = toward camera) with the robot world frame (Y = forward).
    """
    import pyassimp
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import Mesh, MeshTriangle
    from geometry_msgs.msg import Point
    from std_msgs.msg import Header

    collision_pub = node.create_publisher(
        CollisionObject, '/collision_object', 10)

    # Use the dedicated scene STL if provided, otherwise fall back to the planning STL
    mesh_file = os.path.expanduser(scene_stl if scene_stl else stl_filepath)

    # 90° rotation around Z: scanner Y-toward-camera → robot Y-forward
    angle = np.pi / 2.0
    R_z90 = np.array([
        [ np.cos(angle), -np.sin(angle), 0.0],
        [ np.sin(angle),  np.cos(angle), 0.0],
        [ 0.0,            0.0,           1.0],
    ])

    obj                    = CollisionObject()
    obj.header             = Header()
    obj.header.frame_id    = BASE_FRAME
    obj.header.stamp       = node.get_clock().now().to_msg()
    obj.id                 = "scan_object"
    obj.operation          = CollisionObject.ADD

    with pyassimp.load(mesh_file) as scene:
        raw_verts = np.array(scene.meshes[0].vertices, dtype=float) * scale
        # Recentre vertices around bbox centre (same as face_analyser does)
        centred_verts = raw_verts - bbox_centre
        # Apply 90° Z rotation to align frames
        rotated_verts = (R_z90 @ centred_verts.T).T

        mesh = Mesh()
        for face in scene.meshes[0].faces:
            tri                   = MeshTriangle()
            tri.vertex_indices    = [face[0], face[1], face[2]]
            mesh.triangles.append(tri)
        for v in rotated_verts:
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
        self._visualizer: "CleaningVisualizer | None" = None
        self.get_logger().info("Path Planner node started")

    def visualise_only(self, waypoints_with_labels, face_regions,
                        stl_filepath, bbox_centre, world_offset, bbox_size,
                        scene_stl=None):
        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset,
                            scene_stl=scene_stl)
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
                     stl_filepath, bbox_centre, world_offset, bbox_size,
                     scene_stl=None, viz_stl=None):
        """
        Plan all trajectories and preview in RViz face by face — no robot movement.

        Communication protocol (all via stdout → app.py SocketIO):
          PREVIEW_FACE:<index>:<label>:<face_type>:<n_stripes>
              — UI should show a "Next face" confirm button
          PREVIEW_FACE_DONE:<index>
              — face trajectory published to RViz, waiting for stdin NEXT\\n
          PREVIEW_ALL_DONE
              — all faces previewed, waiting for stdin EXECUTE\\n or CANCEL\\n
          PREVIEW_ABORTED
              — cancelled by user or error

        The UI writes a single line to stdin to unblock each wait:
          NEXT    — show next face
          EXECUTE — proceed to real execution
          CANCEL  — abort
        """
        import sys
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
        from trajectory_msgs.msg import JointTrajectory
        from sensor_msgs.msg import JointState

        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset,
                            scene_stl=scene_stl)

        display_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)

        # Give RViz time to discover the new publisher before we send anything.
        # Without this the first message is dropped because no subscriber exists yet.
        print("[PREVIEW] Waiting for RViz /display_planned_path subscriber…", flush=True)
        deadline = time.time() + 5.0
        while display_pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if display_pub.get_subscription_count() == 0:
            print("[PREVIEW] Warning: no subscriber on /display_planned_path — "
                  "is RViz open with MotionPlanning display?", flush=True)
        else:
            print("[PREVIEW] RViz subscriber found.", flush=True)

        paired = sort_faces(waypoints_by_face, face_regions)
        sorted_wps = [wp for wps, _ in paired for wp in wps]
        sorted_faces = [face for _, face in paired]

        # Publish full path waypoint markers to RViz
        ma = build_marker_array(sorted_wps, sorted_faces)
        for _ in range(5):
            self.marker_pub.publish(ma)
            time.sleep(0.2)

        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Service not available. Is MoveIt running?")
            print("PREVIEW_ABORTED", flush=True)
            return

        def wait_for_signal():
            """Block until a line arrives on stdin. Returns the stripped line."""
            try:
                line = sys.stdin.readline().strip()
                return line
            except (KeyboardInterrupt, EOFError):
                return "CANCEL"

        last_state = None
        total_faces = sum(1 for wps, _ in paired if wps)
        face_idx = 0

        for face_wps, face in paired:
            if not face_wps:
                continue

            face_type = classify_face(face.normal)
            poses = [pose for pose, _ in face_wps]
            # Only count actual stripe pairs (not corner waypoints)
            stripe_poses = [(poses[i], poses[i+1])
                            for i in range(0, len(poses)-1, 2)
                            if not face_wps[i][1].endswith('_corner')]
            n_stripes = len(stripe_poses)

            print(f"PREVIEW_FACE:{face_idx}:{face.label}:{face_type}:{n_stripes}", flush=True)
            print(f"[PREVIEW] Planning face {face_idx+1}/{total_faces}: "
                  f"{face.label} ({face_type}) — {n_stripes} stripes", flush=True)

            combined_points = []
            joint_names = None

            for si, (p_start, p_end) in enumerate(stripe_poses):
                req = GetCartesianPath.Request()
                req.header.frame_id = BASE_FRAME
                req.header.stamp = self.get_clock().now().to_msg()
                req.group_name = MOVE_GROUP
                req.link_name = EEF_LINK
                req.max_step = EEF_STEP
                req.jump_threshold = JUMP_THRESHOLD
                req.avoid_collisions = True
                req.waypoints = [p_start, p_end]

                if last_state is not None:
                    req.start_state = last_state

                future = self.cartesian_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

                if future.result() is None or future.result().fraction < FRACTION_MIN:
                    print(f"[PREVIEW]   Stripe {si}: could not plan — skipping", flush=True)
                    continue

                result = future.result()
                new_points = list(result.solution.joint_trajectory.points)
                joint_names = result.solution.joint_trajectory.joint_names

                if combined_points:
                    last_time = combined_points[-1].time_from_start
                    for pt in new_points:
                        pt.time_from_start.sec += last_time.sec
                        pt.time_from_start.nanosec += last_time.nanosec
                        if pt.time_from_start.nanosec >= 1_000_000_000:
                            pt.time_from_start.sec += 1
                            pt.time_from_start.nanosec -= 1_000_000_000

                combined_points += new_points

                last_js = JointState()
                last_js.name = list(joint_names)
                last_js.position = list(new_points[-1].positions)
                last_state = RobotState()
                last_state.joint_state = last_js

            if not combined_points or joint_names is None:
                print(f"[PREVIEW]   No trajectories planned for {face.label} — skipping", flush=True)
                face_idx += 1
                continue

            traj = RobotTrajectory()
            traj.joint_trajectory = JointTrajectory()
            traj.joint_trajectory.joint_names = list(joint_names)
            traj.joint_trajectory.points = combined_points

            display_msg = DisplayTrajectory()
            display_msg.model_id = "ur10"
            display_msg.trajectory = [traj]
            # is_diff=False: use the trajectory's own start state, not the
            # current robot state. This makes it work even if the robot hasn't
            # moved yet and ensures RViz animates from the correct position.
            display_msg.trajectory_start.is_diff = False

            # Publish immediately so RViz starts animating
            for _ in range(5):
                display_pub.publish(display_msg)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.15)

            print(f"PREVIEW_FACE_DONE:{face_idx}", flush=True)
            print(f"[PREVIEW]   ✔ Face {face_idx+1} published to RViz — waiting for UI confirm", flush=True)

            # Keep re-publishing every 2s while waiting so RViz doesn't lose it.
            # stdin.readline() blocks, so we do it in a thread and keep spinning.
            import threading as _threading
            _stop_republish = _threading.Event()

            def _republish_loop():
                while not _stop_republish.is_set():
                    display_pub.publish(display_msg)
                    _stop_republish.wait(2.0)

            _t = _threading.Thread(target=_republish_loop, daemon=True)
            _t.start()

            signal = wait_for_signal()

            _stop_republish.set()
            _t.join(timeout=3.0)

            if signal == "CANCEL":
                print("PREVIEW_ABORTED", flush=True)
                return

            face_idx += 1

        # All faces previewed — ask for final execute/cancel
        print("PREVIEW_ALL_DONE", flush=True)
        print("[PREVIEW] All faces previewed. Waiting for execute/cancel...", flush=True)

        signal = wait_for_signal()
        if signal == "EXECUTE":
            print("[PREVIEW] ✔ Execute confirmed — starting real run", flush=True)
            self.run(waypoints_by_face, face_regions,
                     stl_filepath, bbox_centre, world_offset, bbox_size,
                     viz_stl=viz_stl, scene_stl=scene_stl)
        else:
            print("[PREVIEW] Cancelled.", flush=True)
            print("PREVIEW_ABORTED", flush=True)

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

    def plan_and_execute_face(self, waypoints_with_labels, face_label,
                              execute_client, face_type: str = 'top'):
        """
        Execute each stripe as its own Cartesian request, then execute the
        step to the next stripe start as a SEPARATE Cartesian request.

        Keeping stripe and step as separate requests is the key — it forces
        MoveIt to execute each segment independently so it cannot blend a
        diagonal arc across both. Each segment is a clean straight line.

        Segment order per stripe:
          1. Execute stripe  (p_start → p_end)
          2. Execute step    (p_end → next_p_start)  ← separate request
          3. Execute stripe  (next_p_start → next_p_end)
          ...
        """
        if not waypoints_with_labels:
            return False

        execute_client_local = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory')
        if not execute_client_local.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("/execute_trajectory not available.")
            return False

        # Separate stripe waypoints from corner (step) waypoints.
        # Waypoints alternate: start, end, corner, start, end, corner, ...
        # We execute each consecutive pair as its own Cartesian request.
        all_poses  = [(pose, label) for pose, label in waypoints_with_labels]

        def cartesian_and_execute(seg_poses, seg_label):
            """Plan and execute a single Cartesian segment. Returns True on success."""
            req                  = GetCartesianPath.Request()
            req.header.frame_id  = BASE_FRAME
            req.header.stamp     = self.get_clock().now().to_msg()
            req.group_name       = MOVE_GROUP
            req.link_name        = EEF_LINK
            req.max_step         = EEF_STEP
            req.jump_threshold   = JUMP_THRESHOLD
            req.avoid_collisions = True
            req.waypoints        = seg_poses

            future = self.cartesian_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.result() is None:
                self.get_logger().warn(f"  {seg_label}: service call failed")
                return False
            response = future.result()
            if response.fraction < 0.5:
                self.get_logger().warn(
                    f"  {seg_label}: only {response.fraction*100:.0f}% planned — skipping")
                return False

            MAX_RETRIES = 10
            for attempt in range(MAX_RETRIES):
                exec_goal            = ExecuteTrajectory.Goal()
                exec_goal.trajectory = self.sanitise_trajectory(response.solution)
                send_future          = execute_client_local.send_goal_async(exec_goal)
                rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)
                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    self.get_logger().warn(f"  {seg_label} attempt {attempt+1}: rejected")
                    break
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
                result = result_future.result().result
                if result.error_code.val == MoveItErrorCodes.SUCCESS:
                    return True
                self.get_logger().warn(
                    f"  {seg_label} attempt {attempt+1}: error {result.error_code.val}")
                if result.error_code.val == -4 and attempt < MAX_RETRIES - 1:
                    self.get_logger().info("  Retrying in 2s...")
                    time.sleep(2.0)
                else:
                    break
            return False

        # Execute each consecutive pair of waypoints as its own Cartesian segment.
        # Stripe segments: label does NOT end with _corner
        # Step/corner segments: label ends with _corner
        success_count = 0
        stripe_count  = 0

        for i in range(len(all_poses) - 1):
            p_from, lbl_from = all_poses[i]
            p_to,   lbl_to   = all_poses[i + 1]
            is_stripe = not lbl_from.endswith('_corner') and not lbl_to.endswith('_corner')
            seg_type  = "stripe" if is_stripe else "step"

            ok = cartesian_and_execute(
                [p_from, p_to],
                f"{face_label} {seg_type} {i}")

            if ok and is_stripe:
                stripe_count += 1
                success_count += 1
                if self._visualizer is not None:
                    p_s = np.array([p_from.position.x, p_from.position.y, p_from.position.z])
                    p_e = np.array([p_to.position.x,   p_to.position.y,   p_to.position.z])
                    q_s = np.array([p_from.orientation.x, p_from.orientation.y,
                                    p_from.orientation.z, p_from.orientation.w])
                    q_e = np.array([p_to.orientation.x,   p_to.orientation.y,
                                    p_to.orientation.z,   p_to.orientation.w])
                    self._visualizer.commit_stripe(p_s, p_e, q_s, q_e)

        total_stripes = sum(1 for _, lbl in all_poses if not lbl.endswith('_corner')) - 1
        self.get_logger().info(
            f"  ✔ {face_label}: {success_count}/{total_stripes} stripes complete")
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
             stl_filepath, bbox_centre, world_offset, bbox_size,
             cone_half_angle_deg: float = 10.0, use_viz: bool = True,
             viz_stl: str = None, scene_stl: str = None):
        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset,
                            scene_stl=scene_stl)

        # ── Cleaning visualizer ──────────────────────────────────────────────
        if use_viz:
            # Clear any patches from the previous run before starting fresh
            if self._visualizer is not None:
                self._visualizer.clear_all()
            self._visualizer = CleaningVisualizer(
                node                = self,
                stl_path            = viz_stl if viz_stl else stl_filepath,
                world_offset        = world_offset,
                bbox_size           = bbox_size,
                standoff            = STANDOFF,
                cone_half_angle_deg = cone_half_angle_deg,
                zone_type           = 'clean',
            )
            self._visualizer.start_live_cone()
            self._visualizer.publish_bbox()
            self._visualizer.publish_mesh_marker()   # solid grey object in RViz
        # ────────────────────────────────────────────────────────────────────

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

        try:
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

                self.plan_and_execute_face(face_wps, label, None, face_type=face_type)
                time.sleep(1.0)

            self.get_logger().info("\n✔ All faces complete!")
        finally:
            # Always stop the live cone, even on error / KeyboardInterrupt
            if self._visualizer is not None:
                self._visualizer.stop()

        self.move_to_home()

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stl',        type=str,   required=True)
    parser.add_argument('--visualise',  action='store_true')
    parser.add_argument('--preview',    action='store_true',
                        help='Preview planned trajectories in RViz without executing')
    parser.add_argument('--scale',      type=float, default=STL_SCALE)
    parser.add_argument('--standoff',   type=float, default=STANDOFF)
    parser.add_argument('--stripe',     type=float, default=STRIPE_STEP)
    parser.add_argument('--cone-angle', type=float, default=10.0,
                        help='Spray cone half-angle in degrees (default 10)')
    parser.add_argument('--no-viz',     action='store_true',
                        help='Disable cleaning visualizer (no /cleaning_coverage topic)')
    parser.add_argument('--viz-stl',    type=str, default=None,
                        help='STL to paint on in the visualizer (e.g. ~/scan_clean.stl). '
                             'Defaults to --stl if not set.')
    parser.add_argument('--scene-stl',  type=str, default=None,
                        help='STL to show in the MoveIt planning scene (e.g. hull/primitive). '
                             'Defaults to --stl if not set.')
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

    # Clear leftover markers from any previous run.
    # Wait for RViz to subscribe before sending DELETEALL — if we fire too early
    # (RViz still loading) the messages are dropped and old patches stay visible.
    rclpy.init()
    _clear_node = rclpy.create_node('viz_clearer')
    try:
        from visualization_msgs.msg import Marker, MarkerArray
        import time as _time

        _pubs = {}
        for _topic in ['/cleaning_coverage', '/waypoint_markers']:
            _pubs[_topic] = _clear_node.create_publisher(MarkerArray, _topic, 10)

        # Wait up to 8 s for RViz to subscribe (at least one subscriber on either topic)
        print("[VIZ] Waiting for RViz subscriber before clearing…", flush=True)
        _deadline = _time.time() + 8.0
        while _time.time() < _deadline:
            if any(p.get_subscription_count() > 0 for p in _pubs.values()):
                break
            rclpy.spin_once(_clear_node, timeout_sec=0.1)
        if any(p.get_subscription_count() > 0 for p in _pubs.values()):
            print("[VIZ] RViz subscriber found — sending DELETEALL", flush=True)
        else:
            print("[VIZ] No subscriber yet — sending DELETEALL anyway", flush=True)

        # Send DELETEALL repeatedly over 1.5 s so any late-subscribing display
        # also receives the clear (RViz MarkerArray displays re-subscribe on restart)
        _ma = MarkerArray()
        _m  = Marker(); _m.action = Marker.DELETEALL; _m.ns = ''; _m.id = 0
        _ma.markers.append(_m)
        for _ in range(8):
            for _pub in _pubs.values():
                _pub.publish(_ma)
            rclpy.spin_once(_clear_node, timeout_sec=0.0)
            _time.sleep(0.2)
        print("[VIZ] Cleared previous markers", flush=True)
    finally:
        _clear_node.destroy_node()

    node = PathPlanner()

    try:
        if args.visualise:
            node.visualise_only(sorted_wps, sorted_faces,
                                args.stl, bbox_centre, world_offset, bbox_size,
                                scene_stl=args.scene_stl)
        elif args.preview:
            node.preview_path(waypoints_by_face, face_regions,
                              args.stl, bbox_centre, world_offset, bbox_size,
                              scene_stl=args.scene_stl, viz_stl=args.viz_stl)
        else:
            viz_stl = args.viz_stl if args.viz_stl else args.stl
            node.run(waypoints_by_face, face_regions,
                     args.stl, bbox_centre, world_offset, bbox_size,
                     cone_half_angle_deg = args.cone_angle,
                     use_viz             = not args.no_viz,
                     viz_stl             = viz_stl,
                     scene_stl           = args.scene_stl)
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