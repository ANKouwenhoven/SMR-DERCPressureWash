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
from scipy.spatial import cKDTree
import argparse
import time
import sys
import os
import struct
import tempfile
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_analyser import extract_faces, FaceRegion, load_mesh
from cleaning_visualizer import CleaningVisualizer

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

MOVE_GROUP = "ur_manipulator"
BASE_FRAME = "base_link"
EEF_LINK   = "tool0"
STL_SCALE  = 0.001   # mm to metres

# Fixed yaw calibration from scanner/STL coordinates into the robot/base frame.
# The scanner mesh arrives 90° away from the real turntable/object orientation
# in base_link. This offset must be applied consistently to face regions,
# generated waypoints, collision mesh, clearance mesh, and cleaning visualizer.
# --table-rotation is added on top for intentional second-pass rotations.
SCANNER_TO_ROBOT_YAW_DEG = 90.0

# Turntable — object centred in XY, base on top surface
TURNTABLE_CENTRE_X = 0.65
TURNTABLE_CENTRE_Y = 0.0
TURNTABLE_TOP_Z    = 0.038

# The scan bbox floor is clipped ~3cm above the real turntable surface.
# This is used to place the scanned mesh correctly; it is NOT the side-cleaning
# no-go height.
SCAN_FLOOR_CLIP_M  = 0.0

# Permanent no-go zone near the turntable.
# Side cleaning is never allowed below TURNTABLE_TOP_Z + SIDE_NO_GO_ZONE_M.
SIDE_NO_GO_ZONE_M = 0.05
SIDE_CLEARANCE_Z  = TURNTABLE_TOP_Z + SIDE_NO_GO_ZONE_M

# Coverage / nozzle model
STANDOFF = 0.25   # default tool-to-surface standoff (m). Overridden by --standoff,
                  # which the web UI sets from the "Cleaning Standoff" setting.
                  # NOTE: this is the distance from EEF_LINK ("tool0", the wrist
                  # flange) to the surface — if the physical nozzle/tip extends
                  # past tool0, the real-world gap will be shorter than this by
                  # that extension length.
STRIPE_STEP = 0.03                  # fallback/manual stripe spacing in metres
DEFAULT_NOZZLE_ANGLE_DEG = 25.0     # common washing nozzles: 15°, 25°, 40°
STRIPE_OVERLAP = 0.25               # 25% overlap between adjacent spray passes
MIN_STRIPE_STEP = 0.005             # guard against accidental near-zero spacing

# Keep left/right side stripes slightly inside the front/back bbox edges.
# The actual inset is recomputed as 0.5 * effective stripe spacing at runtime.
SIDE_EDGE_INSET = 0.5 * STRIPE_STEP

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

# Top face quaternion — tool Z points down (-Z)
TOP_QUAT = np.array([-0.838795, 0.544327, 0.000368, -0.011938])


def normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    """Return a calibrated quaternion for the face normal."""
    n = normal / np.linalg.norm(normal)

    # Top face: keep old fixed orientation
    if n[2] > 0.9:
        return TOP_QUAT

    # Side faces: choose closest calibrated side orientation
    horizontal = n.copy()
    horizontal[2] = 0.0
    if np.linalg.norm(horizontal) < 1e-6:
        return TOP_QUAT
    horizontal /= np.linalg.norm(horizontal)

    # The nozzle should point opposite the face normal, i.e. toward the surface
    target = -horizontal

    best_key = max(
        SIDE_FACE_REFS.keys(),
        key=lambda k: np.dot(SIDE_FACE_REFS[k], target)
    )
    return SIDE_FACE_QUATS[best_key]

# ─────────────────────────────────────────────
#  FACE CLASSIFICATION
# ─────────────────────────────────────────────

def classify_face(normal: np.ndarray) -> str:
    """
    Classify face for cleaning strategy.

    top    : near +Z normal  → full coverage
    left   : near -Y normal  → side coverage
    right  : near +Y normal  → side coverage
    skip   : front/back, bottom, and diagonal/chamfer faces

    This intentionally accepts only true +/-Y side faces in the robot frame.
    The face regions are rotated into the robot frame before this function is
    used, so a second pass with --table-rotation 90 will naturally clean the
    physical front/back sides as the new left/right sides.
    """
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return 'skip'

    n = normal / norm

    # Horizontal top face.
    if n[2] > 0.9:
        return 'top'

    # Bottom or steeply tilted faces are not cleaned as side faces.
    if n[2] < -0.9:
        return 'skip'

    # Only true left/right faces should be accepted as side faces.
    # These thresholds intentionally reject front/back and diagonal/chamfer faces.
    y_aligned = abs(n[1]) >= 0.85
    x_small   = abs(n[0]) <= 0.3
    z_small   = abs(n[2]) <= 0.35

    if y_aligned and x_small and z_small:
        return 'right' if n[1] > 0 else 'left'

    return 'skip'


def classify_region(face: FaceRegion) -> str:
    """
    Region-aware classifier.

    face_analyser labels large planar regions as e.g. '+Z (top)', '+Y (right)',
    '-Y (left)', '+X (back)', '-X (front)'.  Use that label when available so
    the waypoint generator cannot accidentally treat front/back (+/-X) faces as
    cleanable side faces.
    """
    label = getattr(face, 'label', '') or ''

    if '+Z' in label and 'top' in label:
        return 'top'
    if '+Y' in label and 'right' in label:
        return 'right'
    if '-Y' in label and 'left' in label:
        return 'left'

    if '+X' in label or '-X' in label or '-Z' in label:
        return 'skip'

    return classify_face(face.normal)


def get_region_area(face: FaceRegion) -> float:
    """Best-effort face area for representative face selection."""
    for attr in ('area', 'surface_area'):
        value = getattr(face, attr, None)
        if value is not None:
            try:
                area = float(value)
                if area > 0:
                    return area
            except Exception:
                pass

    width = getattr(face, 'width', None)
    height = getattr(face, 'height', None)
    try:
        if width is not None and height is not None:
            area = abs(float(width) * float(height))
            if area > 0:
                return area
    except Exception:
        pass

    bmin = getattr(face, 'bounds_min', None)
    bmax = getattr(face, 'bounds_max', None)
    if bmin is not None and bmax is not None:
        try:
            ext = np.maximum(np.asarray(bmax, dtype=float) - np.asarray(bmin, dtype=float), 0.0)
            n = np.asarray(getattr(face, 'normal', np.array([0.0, 0.0, 1.0])), dtype=float)
            axis = int(np.argmax(np.abs(n)))
            dims = [0, 1, 2]
            dims.remove(axis)
            area = float(ext[dims[0]] * ext[dims[1]])
            if area > 0:
                return area
        except Exception:
            pass

    return 0.0


def select_representative_cleaning_faces(face_regions):
    """
    Keep the path compact and aligned with the intended workflow:
      1) one main top face,
      2) one main right face,
      3) one main left face.

    Without this filter, noisy STL segmentation can produce many small top/side
    regions; each side region may generate its own bbox-spanning path, which
    creates the 'waypoints everywhere' failure mode.
    """
    selected = {'top': None, 'right': None, 'left': None}

    for face in face_regions:
        kind = classify_region(face)
        if kind not in selected:
            continue
        if selected[kind] is None or get_region_area(face) > get_region_area(selected[kind]):
            selected[kind] = face

    selected_ids = {id(face) for face in selected.values() if face is not None}

    print("\nRepresentative cleaning-face selection:", flush=True)
    for kind in ('top', 'right', 'left'):
        face = selected[kind]
        if face is None:
            print(f"  {kind:5s}: none found", flush=True)
        else:
            print(
                f"  {kind:5s}: {getattr(face, 'label', 'unnamed')} "
                f"(area≈{get_region_area(face)*1e4:.1f} cm²)",
                flush=True,
            )

    return selected_ids


# ─────────────────────────────────────────────
#  NOZZLE / STRIPE SPACING
# ─────────────────────────────────────────────

def compute_stripe_step_from_nozzle(standoff: float,
                                    nozzle_angle_deg: float = DEFAULT_NOZZLE_ANGLE_DEG,
                                    overlap: float = STRIPE_OVERLAP):
    """
    Compute effective stripe spacing from spray angle and standoff.

    spray_width = 2 * standoff * tan(angle / 2)
    stripe_step = spray_width * (1 - overlap)
    """
    angle = max(1.0, min(120.0, float(nozzle_angle_deg)))
    overlap = max(0.0, min(0.9, float(overlap)))
    spray_width = 2.0 * float(standoff) * np.tan(np.deg2rad(angle) / 2.0)
    stripe_step = max(MIN_STRIPE_STEP, spray_width * (1.0 - overlap))
    return stripe_step, spray_width


def print_nozzle_model(standoff: float, nozzle_angle_deg: float,
                       overlap: float, stripe_step: float,
                       spray_width: float):
    print("[NOZZLE] Coverage model:", flush=True)
    print(f"  Standoff       : {standoff*100:.1f}cm", flush=True)
    print(f"  Nozzle angle   : {nozzle_angle_deg:.1f}° full spray angle", flush=True)
    print(f"  Spray width    : {spray_width*100:.1f}cm at the object", flush=True)
    print(f"  Overlap        : {overlap*100:.0f}%", flush=True)
    print(f"  Stripe spacing : {stripe_step*100:.1f}cm", flush=True)


# ─────────────────────────────────────────────
#  GEOMETRY TRANSFORMS
# ─────────────────────────────────────────────

def rotation_matrix_z(degrees: float) -> np.ndarray:
    """Rotation matrix about world/object Z."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotate_bbox_size_about_z(bbox_size: np.ndarray, degrees: float) -> np.ndarray:
    """
    Return the axis-aligned bbox size after rotating a centred bbox about Z.

    This keeps side-plane generation honest when the fixed scanner->robot yaw is
    90 degrees: the X/Y extents swap in the robot frame.
    """
    bbox_size = np.asarray(bbox_size, dtype=float)
    if abs(degrees) < 1e-9:
        return bbox_size.copy()

    hx, hy, hz = bbox_size[0] / 2.0, bbox_size[1] / 2.0, bbox_size[2] / 2.0
    corners = np.array([
        [sx * hx, sy * hy, sz * hz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])
    rot = rotation_matrix_z(degrees)
    rc = (rot @ corners.T).T
    return rc.max(axis=0) - rc.min(axis=0)


def label_from_normal(normal: np.ndarray, old_label: str = '') -> str:
    """Best-effort analyser-style label after rotating a face normal."""
    n = np.asarray(normal, dtype=float)
    if np.linalg.norm(n) < 1e-9:
        return old_label
    n = n / np.linalg.norm(n)
    axis = int(np.argmax(np.abs(n)))
    sign = '+' if n[axis] >= 0 else '-'
    axis_name = 'XYZ'[axis]
    name = {
        '+Z': 'top', '-Z': 'bottom',
        '+Y': 'right', '-Y': 'left',
        '+X': 'back', '-X': 'front',
    }.get(f'{sign}{axis_name}', 'unknown')
    prefix = ''
    if old_label:
        prefix = old_label.split('_')[0]
        if prefix.startswith('face'):
            prefix += '_'
        else:
            prefix = ''
    return f"{prefix}{sign}{axis_name} ({name})"


def rotate_face_regions_about_z(face_regions, degrees: float):
    """
    Return face regions rotated about the object origin.

    This function is intentionally defensive: the deployed FaceRegion class does
    not expose `.points`, and different analyser versions expose either
    centroid/centre/center.  We therefore shallow-copy the object and rotate only
    the attributes that actually exist.
    """
    if abs(degrees) < 1e-9:
        return face_regions

    rot = rotation_matrix_z(degrees)
    rotated = []

    for face in face_regions:
        new_face = copy.copy(face)

        n = np.asarray(getattr(face, 'normal', np.zeros(3)), dtype=float)
        if n.shape == (3,):
            new_n = rot @ n
            setattr(new_face, 'normal', new_n)
            setattr(new_face, 'label', label_from_normal(new_n, getattr(face, 'label', '')))

        for attr in ('centroid', 'centre', 'center'):
            value = getattr(face, attr, None)
            if value is not None:
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.shape == (3,):
                        setattr(new_face, attr, rot @ arr)
                except Exception:
                    pass

        for attr in ('u_axis', 'v_axis'):
            value = getattr(face, attr, None)
            if value is not None:
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.shape == (3,):
                        setattr(new_face, attr, rot @ arr)
                except Exception:
                    pass

        pts = getattr(face, 'points', None)
        if pts is not None:
            try:
                arr = np.asarray(pts, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    setattr(new_face, 'points', (rot @ arr.T).T)
            except Exception:
                pass

        # Bounds are only approximate after rotation unless points exist, but
        # they are used only as fallback dimensions. Keep them safe if present.
        bmin = getattr(face, 'bounds_min', None)
        bmax = getattr(face, 'bounds_max', None)
        if bmin is not None and bmax is not None:
            try:
                bmin = np.asarray(bmin, dtype=float)
                bmax = np.asarray(bmax, dtype=float)
                if bmin.shape == (3,) and bmax.shape == (3,):
                    corners = np.array([
                        [bmin[0], bmin[1], bmin[2]],
                        [bmin[0], bmin[1], bmax[2]],
                        [bmin[0], bmax[1], bmin[2]],
                        [bmin[0], bmax[1], bmax[2]],
                        [bmax[0], bmin[1], bmin[2]],
                        [bmax[0], bmin[1], bmax[2]],
                        [bmax[0], bmax[1], bmin[2]],
                        [bmax[0], bmax[1], bmax[2]],
                    ])
                    rc = (rot @ corners.T).T
                    setattr(new_face, 'bounds_min', rc.min(axis=0))
                    setattr(new_face, 'bounds_max', rc.max(axis=0))
            except Exception:
                pass

        rotated.append(new_face)

    return rotated


def write_z_rotated_stl(input_path: str, degrees: float, output_path: str,
                        scale: float = STL_SCALE):
    """
    Write an ASCII STL rotated around local Z for visualization-only consumers
    that do not have a separate transform field.  Keeps the same units convention
    as the path planner.
    """
    vertices, faces, bbox_centre, _bbox_size, _origin = load_mesh(input_path)
    rot = rotation_matrix_z(degrees)
    rotated_m = (rot @ vertices.T).T
    rotated_mm = rotated_m / scale

    with open(output_path, 'w') as f:
        f.write('solid rotated_for_visualizer\n')
        for tri in faces:
            pts = rotated_mm[tri]
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0])
            f.write(f'  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}\n')
            f.write('    outer loop\n')
            for p in pts:
                f.write(f'      vertex {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}\n')
            f.write('    endloop\n')
            f.write('  endfacet\n')
        f.write('endsolid rotated_for_visualizer\n')


# ─────────────────────────────────────────────
#  WAYPOINT UTILITIES
# ─────────────────────────────────────────────

def make_pose(position: np.ndarray, quat: np.ndarray) -> Pose:
    p = Pose()
    p.position.x = float(position[0])
    p.position.y = float(position[1])
    p.position.z = float(position[2])
    p.orientation.x = float(quat[0])
    p.orientation.y = float(quat[1])
    p.orientation.z = float(quat[2])
    p.orientation.w = float(quat[3])
    return p


def get_pose_position(pose: Pose) -> np.ndarray:
    return np.array([
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ], dtype=float)


def get_pose_orientation(pose: Pose) -> np.ndarray:
    """Return quaternion as [x, y, z, w], matching make_pose()'s convention."""
    return np.array([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ], dtype=float)


def set_pose_position(pose: Pose, position: np.ndarray):
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])


def get_face_centre(face: FaceRegion, bbox_size: np.ndarray) -> np.ndarray:
    """Return face centre in local, recentred object coordinates."""
    for attr in ('centroid', 'centre', 'center'):
        value = getattr(face, attr, None)
        if value is not None:
            try:
                arr = np.asarray(value, dtype=float)
                if arr.shape == (3,) and np.all(np.isfinite(arr)):
                    return arr
            except Exception:
                pass

    bmin = getattr(face, 'bounds_min', None)
    bmax = getattr(face, 'bounds_max', None)
    if bmin is not None and bmax is not None:
        try:
            bmin = np.asarray(bmin, dtype=float)
            bmax = np.asarray(bmax, dtype=float)
            if bmin.shape == (3,) and bmax.shape == (3,):
                centre = 0.5 * (bmin + bmax)
                if np.all(np.isfinite(centre)):
                    return centre
        except Exception:
            pass

    n = np.asarray(getattr(face, 'normal', np.array([0.0, 0.0, 1.0])), dtype=float)
    if np.linalg.norm(n) < 1e-9:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / np.linalg.norm(n)

    # Fallback: centre of the corresponding side/top bbox plane.
    centre = np.zeros(3)
    axis = int(np.argmax(np.abs(n)))
    centre[axis] = np.sign(n[axis]) * bbox_size[axis] / 2.0
    return centre


def face_bounds_from_region(face: FaceRegion, bbox_size: np.ndarray,
                            default_axes: tuple[np.ndarray, np.ndarray]):
    """
    Return centre, sweep_axis, step_axis, width, height using only safe attrs.
    """
    centre = get_face_centre(face, bbox_size)

    u = getattr(face, 'u_axis', None)
    v = getattr(face, 'v_axis', None)
    if u is not None and v is not None:
        try:
            u = np.asarray(u, dtype=float)
            v = np.asarray(v, dtype=float)
            if np.linalg.norm(u) > 1e-6 and np.linalg.norm(v) > 1e-6:
                u = u / np.linalg.norm(u)
                v = v / np.linalg.norm(v)
            else:
                u, v = default_axes
        except Exception:
            u, v = default_axes
    else:
        u, v = default_axes

    width = getattr(face, 'width', None)
    height = getattr(face, 'height', None)
    try:
        width = float(width)
        height = float(height)
    except Exception:
        width = height = None

    if width is None or height is None or width <= 0 or height <= 0:
        bmin = getattr(face, 'bounds_min', None)
        bmax = getattr(face, 'bounds_max', None)
        if bmin is not None and bmax is not None:
            try:
                ext = np.maximum(np.asarray(bmax, dtype=float) - np.asarray(bmin, dtype=float), 0.0)
                n = np.asarray(getattr(face, 'normal', np.array([0.0, 0.0, 1.0])), dtype=float)
                axis = int(np.argmax(np.abs(n)))
                dims = [0, 1, 2]
                dims.remove(axis)
                width = float(ext[dims[0]])
                height = float(ext[dims[1]])
                u = np.eye(3)[dims[0]]
                v = np.eye(3)[dims[1]]
            except Exception:
                width = height = None

    if width is None or height is None or width <= 0 or height <= 0:
        n = np.asarray(getattr(face, 'normal', np.array([0.0, 0.0, 1.0])), dtype=float)
        axis = int(np.argmax(np.abs(n)))
        dims = [0, 1, 2]
        dims.remove(axis)
        width = float(bbox_size[dims[0]])
        height = float(bbox_size[dims[1]])
        u = np.eye(3)[dims[0]]
        v = np.eye(3)[dims[1]]

    return centre, u, v, width, height


# ─────────────────────────────────────────────
#  WAYPOINT GENERATION
# ─────────────────────────────────────────────

def generate_top_waypoints(face: FaceRegion, world_offset: np.ndarray,
                           standoff: float, stripe_step: float,
                           bbox_size: np.ndarray):
    """
    Generate boustrophedon stripes across the selected top face.

    This deliberately avoids FaceRegion.points/triangles because the deployed
    face_analyser.py does not expose them.
    """
    n = np.asarray(face.normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return []
    n = n / n_norm

    centre, sweep_axis, step_axis, width, height = face_bounds_from_region(
        face,
        bbox_size,
        default_axes=(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    )

    # Keep top stripes in the horizontal XY plane.
    sweep_axis = np.asarray(sweep_axis, dtype=float)
    step_axis = np.asarray(step_axis, dtype=float)
    sweep_axis[2] = 0.0
    step_axis[2] = 0.0

    if np.linalg.norm(sweep_axis) < 1e-6:
        sweep_axis = np.array([1.0, 0.0, 0.0])
    else:
        sweep_axis /= np.linalg.norm(sweep_axis)

    if np.linalg.norm(step_axis) < 1e-6:
        step_axis = np.array([0.0, 1.0, 0.0])
    else:
        step_axis /= np.linalg.norm(step_axis)

    surface_z = float(centre[2]) if np.isfinite(float(centre[2])) else float(bbox_size[2] / 2.0)

    quat = normal_to_quaternion(n)
    poses = []

    half_sweep = max(0.001, float(width) / 2.0)
    half_step = max(0.001, float(height) / 2.0)
    n_stripes = max(1, int(np.ceil((2.0 * half_step) / max(stripe_step, MIN_STRIPE_STEP))) + 1)
    step_vals = np.linspace(-half_step, half_step, n_stripes)

    for stripe_idx, step_val in enumerate(step_vals):
        line_centre_local = np.array([
            centre[0] + step_axis[0] * step_val,
            centre[1] + step_axis[1] * step_val,
            surface_z,
        ])

        p0_local = line_centre_local - sweep_axis * half_sweep
        p1_local = line_centre_local + sweep_axis * half_sweep
        p0_local[2] = surface_z
        p1_local[2] = surface_z

        if stripe_idx % 2:
            p0_local, p1_local = p1_local, p0_local

        for surface_local in (p0_local, p1_local):
            pos = surface_local + world_offset + n * standoff
            poses.append(make_pose(pos, quat))

    print(f"    {face.label}: {len(poses)} waypoints ({n_stripes} stripes)", flush=True)
    return poses


def generate_side_waypoints(face: FaceRegion, world_offset: np.ndarray,
                            standoff: float, stripe_step: float,
                            bbox_size: np.ndarray,
                            side_no_go_zone_m: float = SIDE_NO_GO_ZONE_M):
    """
    Generate vertical stripe coverage for one representative left/right face.

    This also avoids FaceRegion.points. The selected representative side face
    determines the side (+Y or -Y); coverage uses the object's bbox side with
    a front/back edge inset.
    """
    n = np.asarray(face.normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return []
    n = n / n_norm

    face_type = classify_region(face)
    if face_type not in ('left', 'right'):
        return []

    quat = normal_to_quaternion(n)

    # Side planes are +/-Y for this workflow.
    y_sign = 1.0 if face_type == 'right' else -1.0
    side_y = y_sign * bbox_size[1] / 2.0

    # Clean across X and Z, inset from front/back edges to avoid wraparound.
    x_min = -bbox_size[0] / 2.0
    x_max =  bbox_size[0] / 2.0
    edge_inset = max(0.5 * stripe_step, 0.01)
    x_min += edge_inset
    x_max -= edge_inset
    if x_max <= x_min:
        print(f"    Skipping {face.label}: side width too small after edge inset", flush=True)
        return []

    z_max_object = bbox_size[2] / 2.0

    no_go_world_z = TURNTABLE_TOP_Z + side_no_go_zone_m
    clean_min_world_z = no_go_world_z
    clean_max_world_z = world_offset[2] + z_max_object

    clean_height = clean_max_world_z - clean_min_world_z
    if clean_height <= 0:
        print(
            f"    Skipping {face.label}: side no-go plane is above this object "
            f"({no_go_world_z:.3f}m >= {clean_max_world_z:.3f}m)",
            flush=True,
        )
        return []

    local_z_min = clean_min_world_z - world_offset[2]
    local_z_max = z_max_object

    print(
        f"    Side face Z range: {clean_min_world_z:.3f}m to {clean_max_world_z:.3f}m "
        f"({clean_height*100:.1f}cm cleanable, no-go below {no_go_world_z:.3f}m / "
        f"{side_no_go_zone_m*100:.1f}cm above turntable)",
        flush=True,
    )

    poses = []
    stripe_idx = 0
    x = x_min

    while x <= x_max + 1e-9:
        z_pair = (local_z_min, local_z_max) if stripe_idx % 2 == 0 else (local_z_max, local_z_min)
        for z in z_pair:
            surface_local = np.array([x, side_y, z])
            pos = surface_local + world_offset + n * standoff
            # Extra hard clamp: no side waypoint below the no-go plane.
            pos[2] = max(pos[2], clean_min_world_z)
            poses.append(make_pose(pos, quat))

        stripe_idx += 1
        x += max(stripe_step, MIN_STRIPE_STEP)

    print(f"    {face.label}: {len(poses)} waypoints ({stripe_idx} stripes)", flush=True)
    return poses


def generate_waypoints(face: FaceRegion, face_idx: int,
                       world_offset: np.ndarray,
                       standoff: float,
                       bbox_size: np.ndarray,
                       stripe_step: float = STRIPE_STEP,
                       side_no_go_zone_m: float = SIDE_NO_GO_ZONE_M):
    """Generate surface-offset poses for one face."""
    face_type = classify_region(face)

    if face_type == 'top':
        poses = generate_top_waypoints(face, world_offset, standoff, stripe_step, bbox_size)
    elif face_type in ('left', 'right'):
        poses = generate_side_waypoints(face, world_offset, standoff, stripe_step, bbox_size, side_no_go_zone_m)
    else:
        print(f"    Skipping {face.label} (front/back/bottom/diagonal)", flush=True)
        poses = []

    return [(p, face_idx) for p in poses]


# ─────────────────────────────────────────────
#  COLLISION CHECKING
# ─────────────────────────────────────────────

def enforce_min_clearance(waypoints_by_face, face_regions, mesh_vertices_world,
                          min_clearance: float):
    """
    Ensure no waypoint is closer than min_clearance to the STL mesh.

    This corrects cases where a simplified bbox-based side path or a noisy STL
    patch would place the flange/tool0 slightly inside the object.  The push
    direction is based on the classified face normal, so side waypoints move
    horizontally outward and top waypoints move upward.
    """
    if not waypoints_by_face or len(mesh_vertices_world) == 0:
        return waypoints_by_face

    tree = cKDTree(mesh_vertices_world)
    adjusted = 0

    new_waypoints_by_face = []
    for face_wps, face in zip(waypoints_by_face, face_regions):
        if not face_wps:
            new_waypoints_by_face.append(face_wps)
            continue

        n = np.asarray(getattr(face, 'normal', np.array([0.0, 0.0, 1.0])), dtype=float)
        if np.linalg.norm(n) < 1e-9:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / np.linalg.norm(n)

        corrected_face = []
        for pose, face_idx in face_wps:
            pos = get_pose_position(pose)
            dist, _ = tree.query(pos, k=1)
            if dist < min_clearance:
                pos = pos + n * (min_clearance - dist)
                set_pose_position(pose, pos)
                adjusted += 1
            corrected_face.append((pose, face_idx))
        new_waypoints_by_face.append(corrected_face)

    if adjusted:
        print(f"[CLEARANCE] Pushed {adjusted} waypoint(s) outward to maintain {min_clearance*100:.1f}cm minimum standoff", flush=True)

    return new_waypoints_by_face


# ─────────────────────────────────────────────
#  COLLISION SCENE CLEARING
# ─────────────────────────────────────────────

def clear_planning_scene_objects(node, object_ids=("scan_object", "avoid_zone")):
    """Remove stale collision objects so an old rotated mesh cannot remain in RViz."""
    from moveit_msgs.msg import CollisionObject
    from std_msgs.msg import Header

    pub = node.create_publisher(CollisionObject, '/collision_object', 10)
    time.sleep(0.2)

    for obj_id in object_ids:
        msg = CollisionObject()
        msg.header = Header()
        msg.header.frame_id = BASE_FRAME
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.id = obj_id
        msg.operation = CollisionObject.REMOVE
        for _ in range(5):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.05)


# ─────────────────────────────────────────────
#  RVIZ MARKERS
# ─────────────────────────────────────────────

def build_marker_array(all_waypoints, face_regions) -> MarkerArray:
    ma = MarkerArray()
    marker_id = 0

    # Clean old markers
    delete = Marker()
    delete.action = Marker.DELETEALL
    ma.markers.append(delete)

    # Waypoint arrows
    for i, (pose, face_idx) in enumerate(all_waypoints):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = rclpy.time.Time().to_msg()
        m.ns     = "waypoints"
        m.id     = marker_id
        marker_id += 1
        m.type   = Marker.SPHERE
        m.action = Marker.ADD
        m.pose   = pose
        m.scale.x = 0.012
        m.scale.y = 0.012
        m.scale.z = 0.012
        m.color = FACE_COLOURS[face_idx % len(FACE_COLOURS)]
        ma.markers.append(m)

        # Label
        t = Marker()
        t.header.frame_id = BASE_FRAME
        t.header.stamp = rclpy.time.Time().to_msg()
        t.ns = "labels"
        t.id = marker_id
        marker_id += 1
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose = copy.deepcopy(pose)
        t.pose.position.z += 0.03
        t.scale.z = 0.025
        t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        t.text = str(i)
        ma.markers.append(t)

    # Draw line strips per face
    by_face = {}
    for pose, face_idx in all_waypoints:
        by_face.setdefault(face_idx, []).append(pose)

    for face_idx, poses in by_face.items():
        if len(poses) < 2:
            continue
        line = Marker()
        line.header.frame_id = BASE_FRAME
        line.header.stamp = rclpy.time.Time().to_msg()
        line.ns = "stripes"
        line.id = marker_id
        marker_id += 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.003
        line.color = FACE_COLOURS[face_idx % len(FACE_COLOURS)]
        for p in poses:
            line.points.append(p.position)
        ma.markers.append(line)

    return ma


# ─────────────────────────────────────────────
#  ADD STL TO MOVEIT PLANNING SCENE
# ─────────────────────────────────────────────

def add_object_to_scene(node, stl_filepath, bbox_centre,
                         world_offset, scale=0.001, scene_stl=None,
                         table_rotation_deg=0.0, object_id="scan_object"):
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

    remove_obj = CollisionObject()
    remove_obj.header = Header()
    remove_obj.header.frame_id = BASE_FRAME
    remove_obj.header.stamp = node.get_clock().now().to_msg()
    remove_obj.id = object_id
    remove_obj.operation = CollisionObject.REMOVE
    for _ in range(3):
        collision_pub.publish(remove_obj)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.05)

    mesh_file = os.path.expanduser(scene_stl if scene_stl else stl_filepath)

    # table_rotation_deg is passed in as the FULL effective yaw:
    # SCANNER_TO_ROBOT_YAW_DEG + user --table-rotation.  Do not add the fixed
    # 90° again here; main() computes it once and passes the same value to every
    # consumer so mesh, path, clearance, and visualizer remain aligned.
    angle = np.deg2rad(table_rotation_deg)
    R_z = np.array([
        [ np.cos(angle), -np.sin(angle), 0.0],
        [ np.sin(angle),  np.cos(angle), 0.0],
        [ 0.0,            0.0,           1.0],
    ])

    obj                    = CollisionObject()
    obj.header             = Header()
    obj.header.frame_id    = BASE_FRAME
    obj.header.stamp       = node.get_clock().now().to_msg()
    obj.id                 = object_id
    obj.operation          = CollisionObject.ADD

    with pyassimp.load(mesh_file) as scene:
        raw_verts = np.array(scene.meshes[0].vertices, dtype=float) * scale
        centred_verts = raw_verts - bbox_centre
        rotated_verts = (R_z @ centred_verts.T).T

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

    pose               = Pose()
    pose.position.x    = float(world_offset[0])
    pose.position.y    = float(world_offset[1])
    pose.position.z    = float(world_offset[2])
    pose.orientation.w = 1.0
    obj.mesh_poses.append(pose)

    node.get_logger().info(f"Adding '{object_id}' to planning scene...")
    for _ in range(5):
        collision_pub.publish(obj)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.25)
    node.get_logger().info(f"✔ '{object_id}' added")


def add_avoid_zone_to_scene(node, avoid_stl_path, bbox_centre, world_offset,
                            scale=0.001, table_rotation_deg=0.0):
    """Add a separate avoid-zone STL as a collision-only object, if provided."""
    if not avoid_stl_path:
        return

    avoid_path = os.path.expanduser(avoid_stl_path)
    if not os.path.exists(avoid_path):
        node.get_logger().warn(f"Avoid STL not found, skipping avoid zone: {avoid_path}")
        return

    add_object_to_scene(node, avoid_path, bbox_centre, world_offset,
                        scale=scale, table_rotation_deg=table_rotation_deg,
                        object_id="avoid_zone")


def wait_for_signal():
    """Wait for web UI/app.py preview approval over stdin."""
    while True:
        line = sys.stdin.readline()
        if line == '':
            return "CANCEL"
        signal = line.strip().upper()
        if signal in ("NEXT", "EXECUTE", "CANCEL"):
            return signal
        if signal:
            print(f"[PREVIEW] Ignoring unknown UI signal: {signal}", flush=True)


# ─────────────────────────────────────────────
#  PLANNER NODE
# ─────────────────────────────────────────────

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        self.cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.marker_pub  = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        self._visualizer = None

    def _call_visualizer_if_available(self, method_name, *args, **kwargs):
        """Call an optional CleaningVisualizer method without breaking execution."""
        if self._visualizer is None:
            return False
        method = getattr(self._visualizer, method_name, None)
        if method is None:
            self.get_logger().warn(
                f"CleaningVisualizer has no optional method '{method_name}' — continuing without it")
            return False
        try:
            method(*args, **kwargs)
            return True
        except Exception as exc:
            self.get_logger().warn(
                f"CleaningVisualizer.{method_name} failed: {exc} — continuing")
            return False

    # ─────────────────────────────────────────
    #  LOW-LEVEL MOVEIT CALLS
    # ─────────────────────────────────────────

    def compute_cartesian(self, poses):
        while not self.cart_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_cartesian_path...')

        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.group_name       = MOVE_GROUP
        req.link_name        = EEF_LINK
        req.waypoints        = poses
        req.max_step         = EEF_STEP
        req.jump_threshold   = JUMP_THRESHOLD
        req.avoid_collisions = True

        future = self.cart_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)
        return future.result()

    def execute_trajectory(self, trajectory):
        if trajectory is None:
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        self.exec_client.wait_for_server()
        future = self.exec_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("ExecuteTrajectory goal rejected")
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error("No ExecuteTrajectory result")
            return False

        code = result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"ExecuteTrajectory failed with MoveIt code {code}")
            return False

        return True

    # ─────────────────────────────────────────
    #  VISUALISE ONLY
    # ─────────────────────────────────────────

    def visualise_only(self, all_waypoints, face_regions,
                        stl_filepath, bbox_centre, world_offset, bbox_size,
                        scene_stl=None, table_rotation_deg=0.0, avoid_stl_path=None):
        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset,
                            scene_stl=scene_stl, table_rotation_deg=table_rotation_deg)
        add_avoid_zone_to_scene(self, avoid_stl_path, bbox_centre, world_offset,
                                table_rotation_deg=table_rotation_deg)

        ma = build_marker_array(all_waypoints, face_regions)
        self.marker_pub.publish(ma)
        self.get_logger().info("Published waypoint markers. Ctrl-C to exit.")

        try:
            while rclpy.ok():
                self.marker_pub.publish(ma)
                rclpy.spin_once(self, timeout_sec=0.5)
        except KeyboardInterrupt:
            pass

    def preview_path(self, waypoints_by_face, face_regions,
                     stl_filepath, bbox_centre, world_offset, bbox_size,
                     scene_stl=None, table_rotation_deg=0.0,
                     avoid_stl_path=None, cone_half_angle_deg: float = 10.0,
                     viz_stl: str = None, standoff: float = STANDOFF):
        from moveit_msgs.msg import DisplayTrajectory
        from moveit_msgs.msg import RobotTrajectory
        from builtin_interfaces.msg import Duration

        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset,
                            scene_stl=scene_stl, table_rotation_deg=table_rotation_deg)
        add_avoid_zone_to_scene(self, avoid_stl_path, bbox_centre, world_offset,
                                table_rotation_deg=table_rotation_deg)

        display_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)

        FACE_ORDER = {'top': 0, 'right': 1, 'left': 2, 'skip': 3}
        paired = list(zip(waypoints_by_face, face_regions))
        paired.sort(key=lambda x: FACE_ORDER.get(classify_region(x[1]), 99))

        for display_idx, (wps, face) in enumerate(paired, start=1):
            if not wps:
                continue

            face_type = classify_region(face)
            face_name = getattr(face, 'label', f'face_{display_idx}')
            poses = [p for p, _ in wps]
            face_markers = build_marker_array(wps, face_regions)

            print(f"PREVIEW_FACE_START:{display_idx}:{face_name}:{face_type}", flush=True)
            print(f"[PREVIEW] Planning face {display_idx}: {face_name} ({face_type})", flush=True)
            res = self.compute_cartesian(poses)
            if res is None:
                print(f"PREVIEW_FACE_FAILED:{display_idx}:{face_name}:no_result", flush=True)
                print(f"[PREVIEW]   ✖ No result from Cartesian planner", flush=True)
                continue

            print(f"[PREVIEW]   Cartesian fraction: {res.fraction*100:.1f}%", flush=True)
            if res.fraction < FRACTION_MIN:
                print(f"PREVIEW_FACE_FAILED:{display_idx}:{face_name}:fraction_{res.fraction:.3f}", flush=True)
                print(f"[PREVIEW]   ✖ Fraction too low — skipping face", flush=True)
                continue

            display = DisplayTrajectory()
            display.model_id = MOVE_GROUP
            display.trajectory_start = res.start_state
            display.trajectory.append(res.solution)

            for _ in range(5):
                self.marker_pub.publish(face_markers)
                display_pub.publish(display)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.1)

            print(f"PREVIEW_FACE_DONE:{display_idx}:{face_name}:{face_type}:{res.fraction:.3f}", flush=True)
            print(f"[PREVIEW]   ✔ Face {display_idx} published to RViz — waiting for UI confirm", flush=True)

            while rclpy.ok():
                # Keep markers alive while the UI waits for the operator.
                self.marker_pub.publish(face_markers)
                display_pub.publish(display)
                rclpy.spin_once(self, timeout_sec=0.05)

                signal = wait_for_signal()
                if signal == "NEXT":
                    break
                if signal == "EXECUTE":
                    print("[PREVIEW] ✔ Execute confirmed — starting real run", flush=True)
                    self.run(waypoints_by_face, face_regions,
                             stl_filepath, bbox_centre, world_offset, bbox_size,
                             cone_half_angle_deg=cone_half_angle_deg,
                             use_viz=True, viz_stl=viz_stl, scene_stl=scene_stl,
                             standoff=standoff, table_rotation_deg=table_rotation_deg,
                             avoid_stl_path=avoid_stl_path)
                    return
                if signal == "CANCEL":
                    print("PREVIEW_ABORTED", flush=True)
                    print("[PREVIEW] Cancelled by UI", flush=True)
                    return

        print("PREVIEW_ALL_DONE", flush=True)
        print("[PREVIEW] All faces previewed — waiting for execute/cancel", flush=True)
        signal = wait_for_signal()
        if signal == "EXECUTE":
            print("[PREVIEW] ✔ Execute confirmed — starting real run", flush=True)
            self.run(waypoints_by_face, face_regions,
                     stl_filepath, bbox_centre, world_offset, bbox_size,
                     cone_half_angle_deg=cone_half_angle_deg,
                     use_viz=True, viz_stl=viz_stl, scene_stl=scene_stl,
                     standoff=standoff, table_rotation_deg=table_rotation_deg,
                     avoid_stl_path=avoid_stl_path)

    # ─────────────────────────────────────────
    #  MOVE HELPERS
    # ─────────────────────────────────────────

    def go_home(self):
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint

        self.get_logger().info("Moving to HOME joint position...")

        client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /plan_kinematic_path...')

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = MOVE_GROUP
        req.motion_plan_request.num_planning_attempts = 10
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.max_velocity_scaling_factor = APPROACH_VELOCITY_SCALE
        req.motion_plan_request.max_acceleration_scaling_factor = APPROACH_ACCEL_SCALE

        constraints = Constraints()
        for name, value in HOME_JOINTS.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.motion_plan_request.goal_constraints.append(constraints)

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None or res.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error("HOME planning failed")
            return False

        return self.execute_trajectory(res.motion_plan_response.trajectory)

    def move_to_pose(self, pose: Pose):
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import PointStamped

        client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /plan_kinematic_path...')

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = MOVE_GROUP
        req.motion_plan_request.num_planning_attempts = 10
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.max_velocity_scaling_factor = APPROACH_VELOCITY_SCALE
        req.motion_plan_request.max_acceleration_scaling_factor = APPROACH_ACCEL_SCALE

        constraints = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = BASE_FRAME
        pc.link_name = EEF_LINK
        pc.weight = 1.0
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.01]
        pc.constraint_region.primitives.append(prim)
        pc.constraint_region.primitive_poses.append(pose)
        constraints.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EEF_LINK
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.2
        oc.absolute_y_axis_tolerance = 0.2
        oc.absolute_z_axis_tolerance = 0.2
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)

        req.motion_plan_request.goal_constraints.append(constraints)

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None or res.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error("Pose move planning failed")
            return False
        return self.execute_trajectory(res.motion_plan_response.trajectory)

    def move_to_joint_positions(self, joints: dict, label: str = "joint target"):
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint

        client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /plan_kinematic_path...')

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = MOVE_GROUP
        req.motion_plan_request.num_planning_attempts = 10
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.max_velocity_scaling_factor = APPROACH_VELOCITY_SCALE
        req.motion_plan_request.max_acceleration_scaling_factor = APPROACH_ACCEL_SCALE

        constraints = Constraints()
        for name, value in joints.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.motion_plan_request.goal_constraints.append(constraints)

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None or res.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"Planning to {label} failed")
            return False
        return self.execute_trajectory(res.motion_plan_response.trajectory)

    def approach_pose_for_face(self, face_type: str, first_pose: Pose) -> Pose:
        """Return a safe approach pose offset outward from the first cleaning pose."""
        approach = copy.deepcopy(first_pose)

        if face_type == 'top':
            # Lift well above top pass before descending.
            approach.position.z += 0.12
        elif face_type == 'right':
            approach.position.y += 0.12
        elif face_type == 'left':
            approach.position.y -= 0.12
        else:
            approach.position.z += 0.12

        return approach

    def run(self, waypoints_by_face, face_regions,
            stl_filepath, bbox_centre, world_offset, bbox_size,
            cone_half_angle_deg: float = 10.0,
            use_viz: bool = True,
            viz_stl: str = None, scene_stl: str = None,
            standoff: float = STANDOFF, table_rotation_deg: float = 0.0,
            avoid_stl_path: str = None):
        add_object_to_scene(self, stl_filepath, bbox_centre, world_offset,
                            scene_stl=scene_stl, table_rotation_deg=table_rotation_deg)
        add_avoid_zone_to_scene(self, avoid_stl_path, bbox_centre, world_offset,
                                table_rotation_deg=table_rotation_deg)

        if use_viz:
            if self._visualizer is not None:
                self._call_visualizer_if_available('clear_all')

            viz_source = viz_stl if viz_stl else stl_filepath
            if table_rotation_deg:
                rotated_path = os.path.join(
                    tempfile.gettempdir(),
                    f"_viz_rot{int(round(table_rotation_deg))}_{os.path.basename(viz_source)}")
                write_z_rotated_stl(viz_source, table_rotation_deg, rotated_path)
                viz_source = rotated_path

            self._visualizer = CleaningVisualizer(
                node                = self,
                stl_path            = viz_source,
                world_offset        = world_offset,
                bbox_size           = bbox_size,
                standoff            = standoff,
                cone_half_angle_deg = cone_half_angle_deg,
                zone_type           = 'clean',
            )
            self._call_visualizer_if_available('start_live_cone')
            self._call_visualizer_if_available('publish_bbox')

        self.go_home()

        FACE_ORDER = {'top': 0, 'right': 1, 'left': 2, 'skip': 3}
        paired = list(zip(waypoints_by_face, face_regions))
        paired.sort(key=lambda x: FACE_ORDER.get(classify_region(x[1]), 99))

        # Keep the waypoint preview visible during real execution as well.
        run_markers = build_marker_array([wp for wps, _ in paired for wp in wps], face_regions)
        for _ in range(5):
            self.marker_pub.publish(run_markers)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.1)

        for face_wps, face in paired:
            if not face_wps:
                continue

            face_type = classify_region(face)
            face_name = getattr(face, 'label', 'face')
            wps = face_wps
            poses = [p for p, _ in wps]

            self.get_logger().info(
                f"Planning face {face_name} ({face_type}) with {len(poses)} waypoints")

            # Fast move to known side approach if available; otherwise use pose approach.
            known = KNOWN_APPROACH_JOINTS.get(face_type)
            if known:
                self.move_to_joint_positions(known, f"known {face_type} approach")
            else:
                approach = self.approach_pose_for_face(face_type, poses[0])
                self.move_to_pose(approach)

            result = self.compute_cartesian(poses)
            if result is None:
                self.get_logger().error("Cartesian planner returned no result")
                continue

            self.get_logger().info(f"Cartesian path fraction: {result.fraction*100:.1f}%")
            if result.fraction < FRACTION_MIN:
                self.get_logger().warn("Skipping face due to low Cartesian fraction")
                continue

            if self.execute_trajectory(result.solution):
                self.get_logger().info("✔ Face cleaned")
                # Commit each stripe to the CleaningVisualizer so the persistent
                # "paint" patches get published. start_live_cone() only draws the
                # transient spray cone following the TCP (marker id 0) — it does
                # NOT create the coverage patches. Those come exclusively from
                # commit_stripe(), which raycasts each stripe against the mesh.
                # Waypoints are emitted in (start, end) pairs per stripe by
                # generate_top_waypoints()/generate_side_waypoints().
                for i in range(0, len(poses) - 1, 2):
                    self._call_visualizer_if_available(
                        'commit_stripe',
                        get_pose_position(poses[i]),
                        get_pose_position(poses[i + 1]),
                        get_pose_orientation(poses[i]),
                        get_pose_orientation(poses[i + 1]),
                    )
            else:
                self.get_logger().error("✖ Execution failed")

        self.go_home()
        if self._visualizer is not None:
            # Older CleaningVisualizer versions expose stop(); newer/alternate
            # ones may expose stop_live_cone(). Try both safely.
            if not self._call_visualizer_if_available('stop'):
                self._call_visualizer_if_available('stop_live_cone')
        self.get_logger().info("✔ Cleaning routine complete")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='STL coverage path planner for UR10')
    parser.add_argument('--stl', required=True, help='Input STL file')
    parser.add_argument('--avoid-stl', default=None,
                        help='Optional STL file that should be added as collision-only no-go geometry.')
    parser.add_argument('--visualise', action='store_true', help='Only publish RViz markers')
    parser.add_argument('--preview', action='store_true', help='Preview each face and wait for UI confirmation')
    parser.add_argument('--standoff', type=float, default=STANDOFF,
                        help='Tool0/flange standoff from surface, in metres')
    parser.add_argument('--stripe-step', type=float, default=None,
                        help='Manual stripe spacing in metres. If omitted, computed from nozzle angle and standoff.')
    parser.add_argument('--nozzle-angle', type=float, default=DEFAULT_NOZZLE_ANGLE_DEG,
                        help='Full spray angle of nozzle in degrees, e.g. 15, 25, 40.')
    parser.add_argument('--overlap', type=float, default=STRIPE_OVERLAP,
                        help='Fractional stripe overlap, e.g. 0.25 for 25%% overlap.')
    parser.add_argument('--skip-top', action='store_true',
                        help='Skip top cleaning and only generate side paths.')
    parser.add_argument('--side-no-go-zone', type=float, default=SIDE_NO_GO_ZONE_M,
                        help='Permanent side-cleaning no-go height above turntable, in metres.')
    parser.add_argument('--table-rotation', type=float, default=0.0,
                        help='Extra physical turntable/object rotation in degrees; added on top of fixed scanner-to-robot yaw.')
    parser.add_argument('--viz-stl', type=str, default=None,
                        help='Optional STL used only by the cleaning visualizer. Defaults to --stl.')

    args = parser.parse_args()

    stl_path = os.path.expanduser(args.stl)

    if args.stripe_step is None:
        stripe_step, spray_width = compute_stripe_step_from_nozzle(
            args.standoff, args.nozzle_angle, args.overlap)
    else:
        stripe_step = max(MIN_STRIPE_STEP, float(args.stripe_step))
        spray_width = stripe_step / max(1e-6, (1.0 - max(0.0, min(0.9, args.overlap))))

    print_nozzle_model(args.standoff, args.nozzle_angle, args.overlap,
                       stripe_step, spray_width)

    print(f"Loading STL: {stl_path}", flush=True)
    face_regions, bbox_centre, bbox_size_stl = extract_faces(stl_path)

    effective_yaw_deg = SCANNER_TO_ROBOT_YAW_DEG + float(args.table_rotation)
    bbox_size = rotate_bbox_size_about_z(bbox_size_stl, effective_yaw_deg)
    face_regions = rotate_face_regions_about_z(face_regions, effective_yaw_deg)

    print(f"  Scanner->robot yaw : {SCANNER_TO_ROBOT_YAW_DEG:.1f}°", flush=True)
    print(f"  Table rotation     : {args.table_rotation:.1f}°", flush=True)
    print(f"  Effective yaw      : {effective_yaw_deg:.1f}°", flush=True)

    world_offset = compute_world_offset(bbox_size)
    print("\nObject placement:", flush=True)
    print(f"  Turntable centre   : ({TURNTABLE_CENTRE_X}, {TURNTABLE_CENTRE_Y})", flush=True)
    print(f"  Object half-height : {bbox_size[2]/2*100:.1f}cm", flush=True)
    print(f"  World offset       : {world_offset}", flush=True)
    print(f"  Side no-go plane   : {TURNTABLE_TOP_Z + args.side_no_go_zone:.3f}m "
          f"({args.side_no_go_zone*100:.1f}cm above turntable)", flush=True)

    selected_face_ids = select_representative_cleaning_faces(face_regions)

    waypoints_by_face = []
    for idx, face in enumerate(face_regions):
        if args.skip_top and classify_region(face) == 'top':
            waypoints_by_face.append([])
            continue

        if id(face) not in selected_face_ids:
            print(f"    Skipping {face.label} (not selected representative face)", flush=True)
            waypoints_by_face.append([])
            continue

        wps = generate_waypoints(face, idx, world_offset, args.standoff, bbox_size,
                                 stripe_step, args.side_no_go_zone)
        print(f"  Face {idx} ({face.label}): {len(wps)} waypoints", flush=True)
        waypoints_by_face.append(wps)

    all_waypoints = [wp for wps in waypoints_by_face for wp in wps]
    print(f"  Total: {len(all_waypoints)} waypoints", flush=True)

    # Use mesh vertices for clearance checking. load_mesh returns recentred vertices.
    vertices, faces, _, _, _ = load_mesh(stl_path)
    if abs(effective_yaw_deg) > 1e-9:
        rot = rotation_matrix_z(effective_yaw_deg)
        vertices = (rot @ vertices.T).T
    vertices_world = vertices + world_offset

    waypoints_by_face = enforce_min_clearance(
        waypoints_by_face, face_regions, vertices_world, args.standoff)
    all_waypoints = [wp for wps in waypoints_by_face for wp in wps]

    rclpy.init()
    node = PathPlanner()

    try:
        clear_planning_scene_objects(node)

        if args.visualise:
            node.visualise_only(all_waypoints, face_regions,
                                stl_path, bbox_centre, world_offset, bbox_size,
                                table_rotation_deg=effective_yaw_deg,
                                avoid_stl_path=args.avoid_stl)
        elif args.preview:
            node.preview_path(waypoints_by_face, face_regions,
                              stl_path, bbox_centre, world_offset, bbox_size,
                              table_rotation_deg=effective_yaw_deg,
                              avoid_stl_path=args.avoid_stl,
                              viz_stl=args.viz_stl,
                              standoff=args.standoff,
                              cone_half_angle_deg=args.nozzle_angle / 2.0)
        else:
            node.run(waypoints_by_face, face_regions,
                     stl_path, bbox_centre, world_offset, bbox_size,
                     viz_stl=args.viz_stl,
                     standoff=args.standoff,
                     table_rotation_deg=effective_yaw_deg,
                     avoid_stl_path=args.avoid_stl,
                     cone_half_angle_deg=args.nozzle_angle / 2.0)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()