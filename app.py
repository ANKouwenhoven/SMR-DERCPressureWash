#!/usr/bin/env python3
"""
3D Scanner Web Application — Backend
Flask server handling camera, pointcloud, STL export and ROS2/MoveIt integration.
"""

import os
import sys
import json
import time
import struct
import threading
import numpy as np
from flask import Flask, render_template, jsonify, request, Response, send_file
from flask_socketio import SocketIO, emit
import base64
import cv2
import serial

# ─────────────────────────────────────────────
#  CONFIGURATION (also editable via Settings UI)
# ─────────────────────────────────────────────
CONFIG = {
    "depth_max_m": 1.00,  # ignore anything further than this from camera
    "camera_height_m":    0.255,
    "camera_distance_m":  0.67,
    "camera_tilt_deg":    25.0,
    "camera_x_offset_m":  0.0,
    "camera_y_offset_m":  0.0,
    "turntable_x_offset_m": 0.0,
    "turntable_y_offset_m": 0.0,
    "degrees_per_step":   8,
    "camera_width":       1280,
    "camera_height":      720,
    "camera_fps":         30,
    "arduino_port":       "COM3",
    "arduino_enabled":    True,
    "robot_ip":           "192.168.0.43",
    "robot_port":         50002,
    "clean_stl_path":     os.path.expanduser("~/scan_clean.stl"),
    "avoid_stl_path":     os.path.expanduser("~/scan_avoid.stl"),
    "path_planning_script": os.path.expanduser(
        "~/Documents/SMR/Scan/STLFiles/path_planning.py"),
    "environment_setup_script": os.path.expanduser(
        "~/Documents/SMR/Scan/STLFiles/environment_setup.py"),
    "ur_network_iface":   "enp5s0",
    "ur_host_ip":         "192.168.0.100/24",
    "ur_type":            "ur10",
    # RealSense depth sensor settings
    "rs_laser_power":     150,   # IR projector brightness 0-360 mW
    "rs_confidence":      1,     # depth confidence threshold 0-3
    "rs_depth_units":     0.001, # metres per depth unit (0.001 = 1mm res)
    "rs_inter_cam_sync":  1,     # 0=off 1=master (tighter RGB/depth sync)
}

CONFIG_PATH = os.path.expanduser("~/scanner_config.json")
GENERATED_MODELS_DIR = os.path.expanduser("~/scan_models")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scanner3d'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

state = {
    "scanning":        False,
    "preview_running": False,
    "current_angle":   0.0,
    "scan_complete":   False,
    "pointcloud":      None,
    "last_frame":      None,
    "snapshot":        None,
    "clean_stl":       None,
    "avoid_stl":       None,
    "primitive_stl":   None,   # last exported primitive/hull STL for viz
    "colored_pcd":     None,
}

# ── Arduino ───────────────────────────────────────────────────

arduino_serial = None
arduino_lock   = threading.Lock()

def arduino_connect():
    global arduino_serial
    if not CONFIG["arduino_enabled"]:
        return False
    try:
        with arduino_lock:
            arduino_serial = serial.Serial(CONFIG["arduino_port"], 9600, timeout=5)
            time.sleep(2)
            arduino_serial.readline()  # Read READY
        print(f"[Arduino] Connected on {CONFIG['arduino_port']}")
        return True
    except Exception as e:
        print(f"[Arduino] Connection failed: {e}")
        arduino_serial = None
        return False

def arduino_rotate():
    global arduino_serial
    if not CONFIG["arduino_enabled"] or arduino_serial is None:
        return True
    try:
        with arduino_lock:
            degrees = CONFIG["degrees_per_step"]
            cmd = f"DEG:{degrees}\n".encode()
            arduino_serial.write(cmd)
            arduino_serial.flush()
            while True:
                line = arduino_serial.readline().decode().strip()
                print(f"[Arduino] {line}")
                if line == "DONE":
                    return True

    except Exception as e:
        print(f"[Arduino] Rotate error: {e}")
        return False

def arduino_disconnect():
    global arduino_serial
    if arduino_serial:
        try:
            arduino_serial.close()
        except:
            pass
        arduino_serial = None
    print("[Arduino] Disconnected")

# ── Lazy imports ──────────────────────────────────────────────

def get_realsense():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError:
        print("[WARN] pyrealsense2 not available")
        return None

def get_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        print("[WARN] open3d not available")
        return None

def get_rclpy():
    try:
        import rclpy
        return rclpy
    except ImportError:
        print("[WARN] rclpy not available")
        return None

# ── Config ────────────────────────────────────────────────────

def load_config():
    global CONFIG
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            CONFIG.update(json.load(f))
    # Clamp depth_units to valid values to prevent zeroing all depth data
    if CONFIG.get("rs_depth_units", 0.001) not in (0.001, 0.0001):
        CONFIG["rs_depth_units"] = 0.001
        print("[CONFIG] Reset invalid rs_depth_units to 0.001")

def save_config():
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIG, f, indent=2)

# ── Camera pipeline ───────────────────────────────────────────

pipeline  = None
align     = None
pipe_lock = threading.Lock()

def start_camera():
    global pipeline, align
    rs = get_realsense()
    if rs is None:
        return False
    try:
        with pipe_lock:
            pipeline = rs.pipeline()
            cfg      = rs.config()
            cfg.enable_stream(rs.stream.color,
                              CONFIG["camera_width"], CONFIG["camera_height"],
                              rs.format.any, CONFIG["camera_fps"])
            cfg.enable_stream(rs.stream.depth,
                              CONFIG["camera_width"], CONFIG["camera_height"],
                              rs.format.any, CONFIG["camera_fps"])
            profile = pipeline.start(cfg)
            align = rs.align(rs.stream.color)
            # Apply RealSense depth sensor settings from CONFIG
            depth_sensor = profile.get_device().first_depth_sensor()
            try:
                depth_sensor.set_option(rs.option.laser_power,
                    CONFIG["rs_laser_power"])
            except Exception: pass
            try:
                depth_sensor.set_option(rs.option.confidence_threshold,
                    CONFIG["rs_confidence"])
            except Exception: pass
            try:
                depth_sensor.set_option(rs.option.depth_units,
                    CONFIG["rs_depth_units"])
            except Exception: pass
            try:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode,
                    CONFIG["rs_inter_cam_sync"])
            except Exception: pass
        state["preview_running"] = True
        threading.Thread(target=preview_loop, daemon=True).start()
        return True
    except Exception as e:
        print(f"[ERROR] Camera start failed: {e}")
        return False

def stop_camera():
    global pipeline
    state["preview_running"] = False
    time.sleep(0.1)
    with pipe_lock:
        if pipeline:
            try:
                pipeline.stop()
            except:
                pass
            pipeline = None

def preview_loop():
    global scan_intrinsic
    o3d_mod = get_open3d()
    while state["preview_running"]:
        try:
            with pipe_lock:
                if pipeline is None:
                    break
                frameset  = pipeline.wait_for_frames()
                frameset  = align.process(frameset)
                color_f   = frameset.get_color_frame()
                if not color_f:
                    continue
                color_img = np.asanyarray(color_f.get_data())

                # Build intrinsic from the live frame as soon as possible so
                # the bbox overlay is available immediately in the preview,
                # without needing to start a scan first.
                if scan_intrinsic is None and o3d_mod is not None:
                    profile = frameset.get_profile()
                    intr    = profile.as_video_stream_profile().get_intrinsics()
                    scan_intrinsic = o3d_mod.camera.PinholeCameraIntrinsic(
                        intr.width, intr.height,
                        intr.fx, intr.fy, intr.ppx, intr.ppy)

            state["last_frame"] = color_img

            # Draw bbox overlay so the operator can frame the object before scanning
            if scan_intrinsic is not None:
                display_img = draw_bbox_on_image(color_img, scan_intrinsic, 0)
            else:
                display_img = color_img

            _, buf = cv2.imencode('.jpg',
                                  cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode('utf-8')
            socketio.emit('camera_frame', {'image': b64})
        except Exception as e:
            pass
        time.sleep(1.0 / CONFIG["camera_fps"])

# ── Pointcloud processing ─────────────────────────────────────

def get_camera_extrinsics():
    dtr  = np.pi / 180
    tilt = CONFIG["camera_tilt_deg"] * dtr  # positive = nose down

    d            = CONFIG["camera_distance_m"]
    cam_x_offset = CONFIG.get("camera_x_offset_m", 0.0)
    cam_y_offset = CONFIG.get("camera_y_offset_m", 0.0)

    # Camera position in world space:
    #   X = lateral offset (0 = centred on turntable axis)
    #   Y = -distance      (camera sits at -Y, looking toward +Y / origin)
    #   Z = height above turntable surface
    t = np.array([
        cam_x_offset,
        -d + cam_y_offset,
        CONFIG["camera_height_m"]
    ])

    # RealSense / Open3D camera convention: +X right, +Y down, +Z forward
    # World convention: +X right, +Y toward camera, +Z up
    #
    # R_base columns = world-space directions of cam +X, +Y, +Z:
    #   cam+X → world +X  : (1, 0,  0)
    #   cam+Y → world -Z  : (0, 0, -1)   (camera down  = world -Z)
    #   cam+Z → world +Y  : (0, 1,  0)   (camera fwd   = toward turntable = world +Y)
    R_base = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0],
    ])

    # Tilt: nose-down rotation around world X axis (applied after R_base)
    R_tilt = np.array([
        [1,            0,             0],
        [0,  np.cos(tilt),  np.sin(tilt)],
        [0, -np.sin(tilt),  np.cos(tilt)],
    ])

    # Combined: align axes first, then tilt
    R = R_tilt @ R_base

    return R, t

def process_frame_to_pcd(color_img, depth_img, angle_deg, intrinsic):
    o3d = get_open3d()
    if o3d is None:
        return None

    # Depth cutoff -- convert max distance to raw depth units.
    # Raw value = metres / depth_units, so 0.75m / 0.001 = 750,
    # or 0.75m / 0.0001 = 7500. Must match rs_depth_units setting.
    depth_filtered = depth_img.copy()
    depth_units = CONFIG.get("rs_depth_units", 0.001)
    depth_max_raw = int(CONFIG.get("depth_max_m", 0.75) / depth_units)
    depth_filtered[depth_filtered > depth_max_raw] = 0

    # Decimation shrinks the depth frame (e.g. 720p -> 360p); resize it back
    # to match the color image so Open3D's RGBD pairing doesn't fail
    color_h, color_w = color_img.shape[:2]
    if depth_filtered.shape[0] != color_h or depth_filtered.shape[1] != color_w:
        depth_filtered = cv2.resize(
            depth_filtered, (color_w, color_h),
            interpolation=cv2.INTER_NEAREST  # nearest-neighbour preserves depth values
        )

    # Erode the valid depth mask by 1 pixel to discard mixed/flying pixels
    # at sharp edges where depth bleeds into the background color
    valid_mask = (depth_filtered > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(valid_mask, kernel, iterations=4)
    depth_filtered[eroded_mask == 0] = 0

    depth_o3d = o3d.geometry.Image(depth_filtered)
    color_o3d = o3d.geometry.Image(color_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0 / CONFIG.get("rs_depth_units", 0.001),
        depth_trunc=CONFIG.get("depth_max_m", 0.75),
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)  # slightly larger merges nearby duplicates

    if len(np.asarray(pcd.points)) == 0:
        return pcd

    # Step 1: transform from camera space to world space (fixed camera)
    R, t = get_camera_extrinsics()
    pcd.rotate(R, center=(0, 0, 0))
    pcd.translate(t)

    # Step 2: counter-rotate by turntable angle around world Z axis
    # This "unspins" each frame so all frames align in a common world space.
    # Sign convention: positive angle_deg = turntable rotates counter-clockwise
    # when viewed from above (standard right-hand rule around +Z).
    # If scans come out as a ring instead of stacking, flip the sign here.
    dtr = np.pi / 180
    a   = -angle_deg * dtr  # positive = counter-clockwise unspin
    R_unspin = np.array([
        [ np.cos(a), -np.sin(a), 0],
        [ np.sin(a),  np.cos(a), 0],
        [         0,          0, 1],
    ])
    # Rotate around turntable centre (tunable offset if table isn't at world origin)
    tt_x = CONFIG.get("turntable_x_offset_m", 0.0)
    tt_y = CONFIG.get("turntable_y_offset_m", 0.0)
    pcd.rotate(R_unspin, center=(tt_x, tt_y, 0))

    # Crop to bbox
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        (-0.40, -0.40, 0),
        ( 0.40,  0.40,  0.80)
    )
    pcd = pcd.crop(bbox)

    if len(np.asarray(pcd.points)) == 0:
        return pcd

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.8)
    # Radius outlier removal: strips isolated floating points (e.g. leftover edge artifacts)
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.005)
    return pcd


def draw_bbox_on_image(color_img, intrinsic, angle_deg):
    """
    Project the fixed world-space bbox into the camera image.
    Since the camera never moves, this projection is identical every frame.
    The angle_deg parameter is kept for API compatibility but not used.
    """
    x_min, x_max = -0.40, 0.40
    y_min, y_max = -0.40, 0.40
    z_min, z_max =  0.00, 0.65

    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ])

    R, t = get_camera_extrinsics()
    R_inv = R.T  # inverse rotation = transpose

    fx = intrinsic.intrinsic_matrix[0][0]
    fy = intrinsic.intrinsic_matrix[1][1]
    cx = intrinsic.intrinsic_matrix[0][2]
    cy = intrinsic.intrinsic_matrix[1][2]

    img = cv2.cvtColor(color_img.copy(), cv2.COLOR_RGB2BGR)

    projected = []
    for corner in corners:
        # World → camera space
        p = R_inv @ (corner - t)
        if p[2] > 0.01:
            u = int(fx * p[0] / p[2] + cx)
            v = int(fy * p[1] / p[2] + cy)
            projected.append((u, v))
        else:
            projected.append(None)

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

    for i, j in edges:
        if projected[i] and projected[j]:
            cv2.line(img, projected[i], projected[j], (0, 255, 0), 2)

    # Turntable centre cross at world origin
    p = R_inv @ (np.array([0.0, 0.0, 0.0]) - t)
    if p[2] > 0.01:
        u = int(fx * p[0] / p[2] + cx)
        v = int(fy * p[1] / p[2] + cy)
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pcd_to_json(pcd):
    if pcd is None:
        return {"points": [], "colors": []}
    pts  = np.asarray(pcd.points).tolist()
    cols = np.asarray(pcd.colors).tolist() if pcd.has_colors() else []
    return {"points": pts, "colors": cols}


def _open_boundary_edge_count(mesh):
    """
    Count edges that belong to only one triangle — i.e. an actual hole
    boundary, as opposed to a normal interior edge (shared by 2 triangles)
    or a non-manifold edge (shared by 3+). A fully closed/watertight mesh
    has zero of these. Used as a diagnostic at each pipeline stage so we
    can see exactly which step opens a hole and whether a later step
    actually closes it, instead of inferring it from a screenshot.
    """
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return 0
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def _drop_invalid_vertices(mesh, max_extent_m=5.0):
    """
    Defensive cleanup: remove any vertex with a non-finite (NaN/Inf)
    coordinate, or one so large it's obviously corrupted rather than real
    geometry, along with whatever triangles reference it.

    simplify_quadric_decimation occasionally hits a poorly-conditioned
    region (e.g. sliver triangles left over from aggressive cropping) and
    solves a near-singular system for the optimal collapse point, landing
    a vertex somewhere absurd. One such vertex is enough to make any
    boundary loop touching it look astronomically large to fill_holes,
    which then correctly (by design) refuses to touch what looks like a
    giant hole — so the corruption silently defeats hole-filling too.
    max_extent_m is in the mesh's current units (metres, pre mm-scaling);
    5 m comfortably covers anything realistic on this turntable setup.
    """
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh
    bad = ~np.all(np.isfinite(verts), axis=1) | (np.abs(verts).max(axis=1) > max_extent_m)
    n_bad = int(np.sum(bad))
    if n_bad:
        mesh.remove_vertices_by_mask(bad)
        mesh.remove_unreferenced_vertices()
        print(f"[STL] Removed {n_bad} vertices with invalid/out-of-range coordinates")
    return mesh


def _boundary_loop_sizes(mesh):
    """
    Group open boundary edges into connected loops (one loop per distinct
    hole) and report each loop's bounding diameter, in the mesh's current
    units (metres, pre mm-scaling). Lets us see exactly how big each gap
    actually is before fill_holes runs, instead of guessing a hole_size
    threshold and finding out indirectly from a screenshot whether it was
    too big (capping real features) or too small (missing artefacts).
    """
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    if len(tris) == 0:
        return []
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if len(boundary_edges) == 0:
        return []

    # Union-find to group boundary edges into loops by shared vertices.
    parent = {}
    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent.get(x, x)
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in boundary_edges:
        parent.setdefault(int(a), int(a))
        parent.setdefault(int(b), int(b))
        union(int(a), int(b))

    groups = {}
    for v in parent:
        groups.setdefault(find(v), []).append(v)

    sizes = []
    for vidx_list in groups.values():
        pts = verts[vidx_list]
        diam = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        sizes.append(diam)
    return sorted(sizes)


def make_stl_from_pcd(pcd, label="clean", output_path=None):
    """
    Convert a point cloud to an STL for robot path planning.
    Pipeline:
      1. Downsample + outlier removal
      2. Normal estimation, oriented toward camera
      3. Poisson reconstruction (depth 9)
      4. Density trim (5th percentile floaters only)
      5. Largest connected component — drop disconnected blobs
      6. Convex hull crop (10 mm tolerance) — remove hallucinated caps/skirts
      7. Smooth (Taubin 50) + decimate + clean
      7b. Patch small artefact holes left by cropping/cleanup
      8. 7 mm outward offset + final smooth (Taubin 15)
      9. Scale to mm and write STL
    """
    o3d = get_open3d()
    if o3d is None or pcd is None:
        return None
    try:

        # ── 1. Downsample & remove outliers ───────────────────
        pcd_down = pcd.voxel_down_sample(voxel_size=0.004)
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        n_pts = len(np.asarray(pcd_down.points))
        print(f"[STL] After downsample: {n_pts} points")
        if n_pts < 100:
            print(f"[STL] Skipping '{label}' — not enough points")
            return None

        # ── 2. Normal estimation, oriented toward camera ──────
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        cam = np.array([0.0, -CONFIG["camera_distance_m"], CONFIG["camera_height_m"]])
        pcd_down.orient_normals_towards_camera_location(camera_location=cam)

        # ── 3. Poisson reconstruction ──────────────────────────
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_down, depth=9, width=0, scale=1.1, linear_fit=False)
        print(f"[STL] Poisson raw: {len(mesh.triangles)} triangles, "
              f"{_open_boundary_edge_count(mesh)} open boundary edges")

        # ── 4. Density trim (floaters only) ───────────────────
        dens = np.asarray(densities)
        mesh.remove_vertices_by_mask(dens < np.percentile(dens, 1))
        print(f"[STL] After density trim: {len(mesh.triangles)} triangles, "
              f"{_open_boundary_edge_count(mesh)} open boundary edges")

        # ── 5. Largest connected component ────────────────────
        # Drops disconnected blobs (noisy point clusters Poisson
        # wraps into small closed bubbles).
        tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        tri_clusters    = np.asarray(tri_clusters)
        cluster_n_tris  = np.asarray(cluster_n_tris)
        largest         = cluster_n_tris.argmax()
        mesh.remove_triangles_by_mask(tri_clusters != largest)
        mesh.remove_unreferenced_vertices()
        print(f"[STL] After component filter: {len(mesh.triangles)} triangles, "
              f"{_open_boundary_edge_count(mesh)} open boundary edges")
        if len(mesh.triangles) == 0:
            print("[STL] Empty mesh after component filter — aborting")
            return None

        # ── 6. Convex hull crop ────────────────────────────────
        # Remove Poisson geometry that lies outside the actual scan
        # extent (hallucinated caps, skirts, side blobs).
        # 10 mm tolerance is generous enough to keep legitimate surface
        # vertices on thin objects while still catching hallucinated fill.
        #
        # IMPORTANT: the hull is built from a freshly, lightly downsampled
        # copy of the ORIGINAL cloud — not pcd_down. pcd_down has already
        # been through remove_statistical_outlier(), which is tuned to
        # clean Poisson's input but can't tell "isolated noise point" apart
        # from "real point in a sparsely-covered concave corner / grazing
        # angle". When those legitimate points get dropped, the hull
        # shrinks exactly there, and the crop below then deletes correctly
        # -reconstructed surface in that spot, opening a hole. Using the
        # un-filtered (only voxel-downsampled) cloud keeps the hull at the
        # true scan extent; the 20 mm tolerance already added below easily
        # absorbs the handful of stray points this lets through.
        pcd_for_hull = pcd.voxel_down_sample(voxel_size=0.004)
        hull, _ = pcd_for_hull.compute_convex_hull()
        hull_scene = o3d.t.geometry.RaycastingScene()
        hull_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hull))
        verts = np.asarray(mesh.vertices).astype(np.float32)
        query = o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)
        signed_dist = hull_scene.compute_signed_distance(query).numpy()
        mesh.remove_vertices_by_mask(signed_dist > 0.020)
        print(f"[STL] After hull crop: {len(mesh.triangles)} triangles, "
              f"{_open_boundary_edge_count(mesh)} open boundary edges")
        if len(mesh.triangles) == 0:
            print("[STL] Empty mesh after hull crop — aborting")
            return None

        # ── 7. Smooth + decimate + clean ───────────────────────
        # Heavy Taubin pass first to flatten face noise before decimation.
        mesh = mesh.filter_smooth_taubin(number_of_iterations=50)
        n_tris = len(mesh.triangles)
        if n_tris > 10_000:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10_000)
            print(f"[STL] Decimated {n_tris} -> {len(mesh.triangles)} triangles")
        mesh = _drop_invalid_vertices(mesh)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        print(f"[STL] After clean: {len(mesh.triangles)} triangles, "
              f"{_open_boundary_edge_count(mesh)} open boundary edges")

        # ── 7b. Patch artefact holes ────────────────────────────
        # Everything from step 4 onward has only ever removed geometry —
        # the component filter, the hull crop, and remove_non_manifold_edges
        # (which deletes triangles to resolve bad edges) all leave whatever
        # boundary they create right where it is. This closes those gaps.
        #
        # hole_size dropped further (20mm -> 10mm): even the 20mm pass was
        # still capping real round cutouts and producing sliver-triangle
        # artefacts on elongated/non-planar boundary loops (naive boundary
        # triangulation fans a long thin triangle straight across a loop
        # that isn't a nice simple shape). The loop-diameter print below
        # shows the real size of every gap before any filling happens, so
        # the next threshold choice can be based on actual numbers instead
        # of another guess.
        print(f"[STL] boundary loop diameters (m) before hole fill: "
              f"{[round(s, 4) for s in _boundary_loop_sizes(mesh)]}")
        try:
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            mesh_t = mesh_t.fill_holes(hole_size=0.01)
            mesh = mesh_t.to_legacy()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            open_edges = _open_boundary_edge_count(mesh)
            print(f"[STL] After hole fill (10mm pass): {len(mesh.triangles)} triangles, "
                  f"{open_edges} open boundary edges")
        except Exception as fill_err:
            print(f"[WARN] Hole fill failed, continuing without it: {fill_err}")

        # ── 8. Outward offset + final smooth ──────────────────
        # Offset applied AFTER hull crop so offset geometry is not
        # clipped as "outside". Final Taubin pass cleans up the corner
        # fan artefacts that vertex offsetting introduces.
        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        norms = np.asarray(mesh.vertex_normals)
        mesh.vertices = o3d.utility.Vector3dVector(verts + norms * 0.007)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=15)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        print(f"[STL] Applied 7 mm outward offset, {len(mesh.triangles)} triangles, "
              f"{_open_boundary_edge_count(mesh)} open boundary edges")

        # ── 9. Scale to mm, write STL ──────────────────────────
        mesh = _drop_invalid_vertices(mesh)
        mesh.scale(1000, center=(0, 0, 0))
        mesh.compute_vertex_normals()
        if output_path is None:
            output_path = CONFIG["clean_stl_path"] if label == "clean" else CONFIG["avoid_stl_path"]
            output_path = os.path.expanduser(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"[STL] Saved '{label}' -> {output_path}  ({len(mesh.triangles)} triangles)")
        return output_path

    except Exception as e:
        print(f"[ERROR] STL generation failed: {e}")
        import traceback; traceback.print_exc()
        return None

def fit_primitive_stl(pcd, primitive="box", label="clean", output_path=None, hull_detail=50, lowpoly_tris=200, save_mm=True):
    """
    Fit a geometric primitive to the point cloud and export as STL.
    Produces a very clean mesh with under 100 triangles, ideal for
    robot collision/path planning where exact shape detail is not needed.

    primitive: "box"      — oriented bounding box  (12 triangles)
               "sphere"   — bounding sphere        (~80 triangles)
               "cylinder" — upright bounding cyl.  (~64 triangles)
    """
    o3d = get_open3d()
    if o3d is None or pcd is None:
        return None
    try:
        # Clean up the point cloud first so outliers don't inflate the fit
        pcd_clean = pcd.voxel_down_sample(voxel_size=0.005)
        pcd_clean, _ = pcd_clean.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pts = np.asarray(pcd_clean.points)
        if len(pts) < 10:
            print(f"[STL] Not enough points to fit primitive")
            return None

        if primitive == "box":
            # Use a world-axis-aligned box. The object sits on a flat
            # turntable so Y is always up and the horizontal orientation
            # of the scan is arbitrary anyway. Axis-aligned is stable,
            # predictable, and avoids all OBB/PCA rotation ambiguity.
            # Trim 2-98th percentile on each axis to exclude outlier points.
            lo = np.percentile(pts,  2, axis=0)
            hi = np.percentile(pts, 98, axis=0)
            center_world = (lo + hi) / 2
            h = (hi - lo) / 2
            local_corners = np.array([
                [-h[0],-h[1],-h[2]], [-h[0],-h[1], h[2]],
                [-h[0], h[1],-h[2]], [-h[0], h[1], h[2]],
                [ h[0],-h[1],-h[2]], [ h[0],-h[1], h[2]],
                [ h[0], h[1],-h[2]], [ h[0], h[1], h[2]],
            ])
            corners = local_corners + center_world
            triangles = np.array([
                [0,1,2],[2,1,3],  # -X face
                [4,6,5],[5,6,7],  # +X face
                [0,4,1],[1,4,5],  # -Y face
                [2,3,6],[3,7,6],  # +Y face
                [0,2,4],[4,2,6],  # -Z face
                [1,5,3],[3,5,7],  # +Z face
            ])
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices  = o3d.utility.Vector3dVector(corners)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            print(f"[STL] Box fit (axis-aligned): "
                  f"extent {np.round((hi-lo)*1000).astype(int)} mm")

        elif primitive == "sphere":
            # Minimum bounding sphere via centre = mean, radius = max dist
            centre = pts.mean(axis=0)
            radius = np.linalg.norm(pts - centre, axis=1).max()
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=5)
            mesh.translate(centre)
            print(f"[STL] Sphere fit: centre {np.round(centre*1000).astype(int)} mm, "
                  f"radius {round(radius*1000)} mm")

        elif primitive == "cylinder":
            # Upright cylinder: radius from XZ spread, height from Y extent
            centre_xz = np.array([pts[:, 0].mean(), pts[:, 2].mean()])
            radius = np.sqrt(
                ((pts[:, 0] - centre_xz[0]) ** 2 +
                 (pts[:, 2] - centre_xz[1]) ** 2)
            ).max()
            y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
            height = y_max - y_min
            centre = np.array([centre_xz[0], (y_min + y_max) / 2, centre_xz[1]])
            mesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, height=height, resolution=8, split=1)
            mesh.translate(centre)
            print(f"[STL] Cylinder fit: radius {round(radius*1000)} mm, "
                  f"height {round(height*1000)} mm")

        elif primitive == "hull":
            # Convex hull with controllable detail via hull_detail (0-100).
            # Low detail: downsample points heavily before computing hull
            #             -> fewer, blockier faces.
            # High detail: compute hull on the full cleaned cloud
            #              -> more faces, tighter fit.
            # hull_detail=50 is the default (full resolution hull).
            if hull_detail < 50:
                # Coarser: voxel-downsample to reduce point count before hull
                # Map 0->0.05m voxel (very coarse) to 49->0.006m (near-full)
                voxel = 0.005 + (1.0 - hull_detail / 49.0) * 0.045
                pcd_hull = pcd_clean.voxel_down_sample(voxel_size=voxel)
            else:
                pcd_hull = pcd_clean
            mesh, _ = pcd_hull.compute_convex_hull()
            mesh.orient_triangles()
            if hull_detail > 50:
                # Finer: subdivide each triangle once to add midpoint vertices,
                # giving the hull more facets to follow curved edges.
                # Map 51->1 subdivision to 100->3 subdivisions.
                subdivisions = int(1 + round((hull_detail - 51) / 49 * 2))
                mesh = mesh.subdivide_midpoint(number_of_iterations=subdivisions)
                # Project subdivided vertices back onto the convex hull surface
                # so the mesh stays tight (subdivision pulls verts inward).
                verts = np.asarray(mesh.vertices)
                centre = verts.mean(axis=0)
                dirs   = verts - centre
                norms  = np.linalg.norm(dirs, axis=1, keepdims=True)
                dirs   = dirs / np.clip(norms, 1e-12, None)
                # Push each vertex outward to the original hull surface
                hull_orig, _ = pcd_hull.compute_convex_hull()
                hull_scene2  = o3d.t.geometry.RaycastingScene()
                hull_scene2.add_triangles(
                    o3d.t.geometry.TriangleMesh.from_legacy(hull_orig))
                # Cast rays outward from centre through each vertex
                rays = np.hstack([np.tile(centre, (len(verts), 1)),
                                  dirs]).astype(np.float32)
                ray_tensor = o3d.core.Tensor(rays,
                    dtype=o3d.core.Dtype.Float32)
                hits = hull_scene2.cast_rays(ray_tensor)
                t_hit = hits['t_hit'].numpy()
                valid = np.isfinite(t_hit)
                verts[valid] = (centre + dirs[valid] *
                                t_hit[valid, np.newaxis])
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.orient_triangles()
            n_tris = len(mesh.triangles)
            print(f"[STL] Convex hull: {n_tris} triangles "
                  f"(detail={hull_detail:.0f})")

        elif primitive == "lowpoly":
            # Low-poly surface using the same proven Poisson pipeline as
            # make_stl_from_pcd, then decimated to the triangle target.
            if output_path is None:
                lp_path = os.path.expanduser(
                    CONFIG["clean_stl_path"] if label == "clean" else CONFIG["avoid_stl_path"])
            else:
                lp_path = output_path

            # ── 1. Downsample & outlier removal ───────────────────
            pcd_lp = pcd_clean.voxel_down_sample(voxel_size=0.004)
            pcd_lp, _ = pcd_lp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            n_before = len(np.asarray(pcd_lp.points))
            print(f"[STL] lowpoly: {n_before} points after cleanup")
            if n_before < 100:
                print("[STL] lowpoly: not enough points")
                return None

            # ── 2. Normal estimation ───────────────────────────────
            pcd_lp.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
            cam = np.array([0.0, -CONFIG["camera_distance_m"], CONFIG["camera_height_m"]])
            pcd_lp.orient_normals_towards_camera_location(camera_location=cam)

            # ── 3. Poisson reconstruction ──────────────────────────
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_lp, depth=8, width=0, scale=1.1, linear_fit=False)
            print(f"[STL] lowpoly Poisson raw: {len(mesh.triangles)} triangles, "
                  f"{_open_boundary_edge_count(mesh)} open boundary edges")

            # ── 4. Density trim ────────────────────────────────────
            # Was percentile=25, which trims the lowest QUARTER of density
            # values mesh-wide. That's not just "floaters" — it's also
            # exactly the legitimately-real-but-sparsely-supported surface
            # in concave corners / grazing-angle faces, which is what was
            # carving large chunks out of the lowpoly mesh. Dropped to 2,
            # matching the conservative trim used elsewhere in this file.
            dens = np.asarray(densities)
            mesh.remove_vertices_by_mask(dens < np.percentile(dens, 2))
            print(f"[STL] lowpoly after density trim: {len(mesh.triangles)} triangles, "
                  f"{_open_boundary_edge_count(mesh)} open boundary edges")

            # ── 5. Largest connected component ─────────────────────
            tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
            tri_clusters   = np.asarray(tri_clusters)
            cluster_n_tris = np.asarray(cluster_n_tris)
            mesh.remove_triangles_by_mask(tri_clusters != cluster_n_tris.argmax())
            mesh.remove_unreferenced_vertices()
            print(f"[STL] lowpoly after component filter: {len(mesh.triangles)} triangles, "
                  f"{_open_boundary_edge_count(mesh)} open boundary edges")
            if len(mesh.triangles) == 0:
                print("[STL] lowpoly: empty after component filter")
                return None

            # ── 6. Convex hull crop ────────────────────────────────
            # Same fix as make_stl_from_pcd: build the hull from a freshly,
            # lightly downsampled copy of the RAW cloud, not pcd_lp. pcd_lp
            # descends from pcd_clean, which has already been through
            # remove_statistical_outlier() — that filter can't distinguish
            # a noise point from a real point in a sparsely-covered corner,
            # so cropping against a hull built from it shrinks the hull
            # exactly in those spots and deletes real surface there.
            pcd_for_hull = pcd.voxel_down_sample(voxel_size=0.004)
            hull, _ = pcd_for_hull.compute_convex_hull()
            hull_scene = o3d.t.geometry.RaycastingScene()
            hull_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hull))
            verts = np.asarray(mesh.vertices).astype(np.float32)
            signed_dist = hull_scene.compute_signed_distance(
                o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)).numpy()
            mesh.remove_vertices_by_mask(signed_dist > 0.020)
            print(f"[STL] lowpoly after hull crop: {len(mesh.triangles)} triangles, "
                  f"{_open_boundary_edge_count(mesh)} open boundary edges")
            if len(mesh.triangles) == 0:
                print("[STL] lowpoly: empty after hull crop")
                return None

            # ── 7. Decimate to target triangle count ───────────────
            mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
            if len(mesh.triangles) > lowpoly_tris:
                mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=lowpoly_tris)
            mesh = _drop_invalid_vertices(mesh)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            print(f"[STL] lowpoly: {len(mesh.triangles)} tris (target {lowpoly_tris}), "
                  f"{_open_boundary_edge_count(mesh)} open boundary edges")

            # ── 7b. Patch artefact holes ────────────────────────────
            # hole_size dropped further (20mm -> 10mm): even 20mm was still
            # capping the real round cutouts and leaving sliver-triangle
            # artefacts on elongated/non-planar boundary loops. The loop-
            # diameter print below shows the real size of every gap before
            # filling, so we're tuning from actual numbers, not a guess.
            print(f"[STL] lowpoly boundary loop diameters (m) before hole fill: "
                  f"{[round(s, 4) for s in _boundary_loop_sizes(mesh)]}")
            try:
                mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                mesh_t = mesh_t.fill_holes(hole_size=0.01)
                mesh = mesh_t.to_legacy()
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_vertices()
                open_edges = _open_boundary_edge_count(mesh)
                print(f"[STL] lowpoly after hole fill (10mm pass): {len(mesh.triangles)} triangles, "
                      f"{open_edges} open boundary edges")
            except Exception as fill_err:
                print(f"[WARN] lowpoly hole fill failed, continuing without it: {fill_err}")

            # ── 8. Normal smoothing ────────────────────────────────
            # Recompute normals with a large search radius so flat faces
            # get a consistent perpendicular normal and cylinders get a
            # smoothly varying one — without changing the geometry at all.
            tris  = np.asarray(mesh.triangles)
            verts = np.asarray(mesh.vertices)

            # Compute per-face normals and areas
            v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0)
            face_areas   = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_normals = face_normals / np.clip(face_areas, 1e-12, None)

            # Build vertex → face adjacency
            n_verts = len(verts)
            vert_normal_acc = np.zeros((n_verts, 3))
            vert_area_acc   = np.zeros(n_verts)
            for fi, (i0, i1, i2) in enumerate(tris):
                w = face_areas[fi, 0]
                for vi in (i0, i1, i2):
                    vert_normal_acc[vi] += face_normals[fi] * w
                    vert_area_acc[vi]   += w

            # Normalise accumulated normals
            lengths = np.linalg.norm(vert_normal_acc, axis=1, keepdims=True)
            smooth_normals = vert_normal_acc / np.clip(lengths, 1e-12, None)

            # One pass of neighbour averaging for extra smoothness
            smooth2 = np.zeros_like(smooth_normals)
            counts  = np.zeros(n_verts)
            for i0, i1, i2 in tris:
                for vi, others in ((i0,(i1,i2)),(i1,(i0,i2)),(i2,(i0,i1))):
                    smooth2[vi] += smooth_normals[vi]
                    for vj in others:
                        smooth2[vi] += smooth_normals[vj]
                    counts[vi] += 3
            counts = np.clip(counts, 1, None)
            smooth2 = smooth2 / counts[:, None]
            lengths2 = np.linalg.norm(smooth2, axis=1, keepdims=True)
            smooth2  = smooth2 / np.clip(lengths2, 1e-12, None)

            mesh.vertex_normals = o3d.utility.Vector3dVector(smooth2)
            print(f"[STL] Normal smoothing applied")

            # ── 9. Standoff offset ─────────────────────────────────
            verts = np.asarray(mesh.vertices)
            mesh.vertices = o3d.utility.Vector3dVector(verts + smooth2 * 0.002)
            mesh = _drop_invalid_vertices(mesh)

            # save_mm=False from api_mesh_build; browser x1000 gives mm display
            if save_mm:
                mesh.scale(1000, center=(0, 0, 0))
            mesh.compute_vertex_normals()
            os.makedirs(os.path.dirname(lp_path), exist_ok=True)
            o3d.io.write_triangle_mesh(lp_path, mesh)
            print(f"[STL] lowpoly saved: {len(mesh.triangles)} tris -> {lp_path}")
            return lp_path

        else:
            print(f"[STL] Unknown primitive '{primitive}'")
            return None

        # Apply 7 mm outward offset for robot standoff clearance
        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        norms = np.asarray(mesh.vertex_normals)
        mesh.vertices = o3d.utility.Vector3dVector(verts + norms * 0.007)

        # Scale to mm for MoveIt/RViz compatibility (unless caller wants metres)
        if save_mm:
            mesh.scale(1000, center=(0, 0, 0))
        mesh.compute_vertex_normals()
        if output_path is None:
            output_path = CONFIG["clean_stl_path"] if label == "clean" else CONFIG["avoid_stl_path"]
            output_path = os.path.expanduser(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"[STL] Saved '{primitive}' primitive -> {output_path} "
              f"({len(mesh.triangles)} triangles)")
        return output_path

    except Exception as e:
        print(f"[ERROR] Primitive fit failed: {e}")
        import traceback; traceback.print_exc()
        return None

# ── ROS2 / MoveIt ─────────────────────────────────────────────

def send_stl_to_moveit(stl_path, object_name, operation="add", x=None, y=0.0, z=None):
    """
    Publish an STL as a MoveIt collision object, correctly centred on the turntable.
    Vertices are recentred to bbox centre, a 90° Z rotation is applied to align
    scanner frame (Y=toward camera) with robot frame (Y=forward), then placed at
    TURNTABLE_CENTRE_X, TURNTABLE_CENTRE_Y, TURNTABLE_TOP_Z + SCAN_FLOOR_CLIP_M + half_height.
    """
    TURNTABLE_CENTRE_X = 0.65
    TURNTABLE_CENTRE_Y = 0.0
    TURNTABLE_TOP_Z    = 0.038
    SCAN_FLOOR_CLIP_M  = 0.03

    rclpy_mod = get_rclpy()
    if rclpy_mod is None:
        print("[WARN] ROS2 not available")
        return False
    try:
        from moveit_msgs.msg   import CollisionObject
        from shape_msgs.msg    import Mesh, MeshTriangle
        from geometry_msgs.msg import Pose, Point
        from std_msgs.msg      import Header
        import rclpy as rclpy_lib
        from rclpy.node import Node

        raw_verts, triangles = read_stl_binary(stl_path)
        verts_m = np.array(raw_verts, dtype=float) / 1000.0
        bbox_min = verts_m.min(axis=0)
        bbox_max = verts_m.max(axis=0)
        bbox_centre = (bbox_min + bbox_max) / 2.0
        bbox_size   = bbox_max - bbox_min
        centred = verts_m - bbox_centre

        angle = np.pi / 2.0
        R_z90 = np.array([
            [ np.cos(angle), -np.sin(angle), 0.0],
            [ np.sin(angle),  np.cos(angle), 0.0],
            [ 0.0,            0.0,           1.0],
        ])
        rotated = (R_z90 @ centred.T).T

        pose_x = TURNTABLE_CENTRE_X
        pose_y = TURNTABLE_CENTRE_Y
        pose_z = TURNTABLE_TOP_Z + SCAN_FLOOR_CLIP_M + bbox_size[2] / 2.0
        print(f"[ROS2] STL placement: x={pose_x:.3f} y={pose_y:.3f} z={pose_z:.3f}  "
              f"(bbox {np.round(bbox_size*1000).astype(int)} mm)")

        mesh = Mesh()
        for v in rotated:
            p = Point()
            p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
            mesh.vertices.append(p)
        for tri in triangles:
            t = MeshTriangle()
            t.vertex_indices = [tri[0], tri[1], tri[2]]
            mesh.triangles.append(t)

        if not rclpy_lib.ok():
            rclpy_lib.init()

        node = Node('scanner_moveit_publisher')
        pub  = node.create_publisher(CollisionObject, '/collision_object', 10)
        time.sleep(0.5)

        obj = CollisionObject()
        obj.header = Header()
        obj.header.frame_id = "base_link"
        obj.header.stamp = node.get_clock().now().to_msg()
        obj.id = object_name
        obj.operation = CollisionObject.ADD if operation == "add" else CollisionObject.REMOVE
        obj.meshes.append(mesh)
        pose = Pose()
        pose.position.x = pose_x
        pose.position.y = pose_y
        pose.position.z = pose_z
        pose.orientation.w = 1.0
        obj.mesh_poses.append(pose)
        for _ in range(5):
            pub.publish(obj)
            time.sleep(0.3)
        node.destroy_node()
        print(f"[ROS2] {operation.upper()} '{object_name}' in MoveIt scene")
        return True
    except Exception as e:
        print(f"[ERROR] MoveIt publish failed: {e}")
        import traceback; traceback.print_exc()
        return False


def read_stl_binary(filepath):
    with open(filepath, 'rb') as f:
        f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        vertices, triangles, vertex_map = [], [], {}

        def get_idx(v):
            key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(v)
            return vertex_map[key]

        for _ in range(num_triangles):
            f.read(12)
            tri = [get_idx(struct.unpack('<fff', f.read(12))) for _ in range(3)]
            triangles.append(tri)
            f.read(2)
    return vertices, triangles

# ── Scan state machine ────────────────────────────────────────

scan_intrinsic = None

def start_scan():
    global scan_intrinsic
    o3d = get_open3d()
    if o3d is None:
        return False
    state["scanning"]      = True
    state["current_angle"] = 0.0
    state["scan_complete"] = False
    state["pointcloud"]    = o3d.geometry.PointCloud()
    state["colored_pcd"]   = None
    scan_intrinsic         = None
    stop_camera()
    time.sleep(0.5)
    start_camera()
    if CONFIG["arduino_enabled"]:
        arduino_connect()
    return True


def capture_at_angle(angle):
    """Capture one frame, process pointcloud, trigger Arduino rotation."""
    global scan_intrinsic
    rs  = get_realsense()
    o3d = get_open3d()
    if rs is None or o3d is None or pipeline is None:
        return False
    try:
        with pipe_lock:
            sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, 1500)

        # Warm up auto exposure
        for _ in range(15):
            with pipe_lock:
                pipeline.wait_for_frames()

        with pipe_lock:
            frameset = pipeline.wait_for_frames()
            frameset = align.process(frameset)
            color_f  = frameset.get_color_frame()
            depth_f  = frameset.get_depth_frame()
            if not color_f or not depth_f:
                return False
            profile  = frameset.get_profile()
            intr     = profile.as_video_stream_profile().get_intrinsics()
            scan_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
            # ── Post-processing filters (reduce edge artifacts) ──
            # Decimation: merges neighbouring depth pixels, reducing noise
            # and mixed pixels at edges before any other filter runs
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, 2)
            # Spatial filter: smooths depth while preserving edges
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            spatial.set_option(rs.option.holes_fill, 0)
            # Temporal filter: reduces per-frame noise
            temporal = rs.temporal_filter()
            # hole_filling_filter deliberately omitted: it fills depth
            # gaps at edges with interpolated values that pick up the
            # background color, making corner artifacts worse.
            depth_f = decimation.process(depth_f)
            depth_f = spatial.process(depth_f)
            depth_f = temporal.process(depth_f)

            color_img = np.asanyarray(color_f.get_data())
            depth_img = np.asanyarray(depth_f.get_data())

        state["snapshot"] = color_img

        # Send snapshot with bbox overlay to browser
        debug_img = draw_bbox_on_image(color_img, scan_intrinsic, angle)
        _, buf = cv2.imencode('.jpg',
                              cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        socketio.emit('snapshot', {
            'image': base64.b64encode(buf).decode('utf-8'),
            'angle': angle
        })

        # Process pointcloud
        pcd = process_frame_to_pcd(color_img, depth_img, angle, scan_intrinsic)
        pts = len(np.asarray(pcd.points)) if pcd is not None else 0
        print(f"[SCAN] {angle:.1f}° → {pts} points")

        if pcd is not None and pts > 0:
            state["pointcloud"] += pcd

        # Rotate table
        if CONFIG["arduino_enabled"]:
            socketio.emit('scan_status', {
                'message': f'Rotating to {angle + CONFIG["degrees_per_step"]:.1f}°...'
            })
            arduino_rotate()

        return True
    except Exception as e:
        print(f"[ERROR] Capture failed at {angle}°: {e}")
        import traceback
        traceback.print_exc()
        return False


def finish_scan():
    state["scanning"]      = False
    state["scan_complete"] = True
    if CONFIG["arduino_enabled"]:
        arduino_disconnect()
    o3d = get_open3d()
    total = len(np.asarray(state["pointcloud"].points)) if state["pointcloud"] else 0
    print(f"[SCAN] Complete — {total} total points")
    pcd_data = pcd_to_json(state["pointcloud"])
    socketio.emit('pointcloud_update', pcd_data)
    socketio.emit('scan_complete', {'points': total})


def auto_scan_loop():
    time.sleep(1.0)  # Let camera settle
    while state["scanning"]:
        angle = state["current_angle"]
        if angle >= 360.0:
            finish_scan()
            return
        socketio.emit('scan_status', {'message': f'Capturing at {angle:.1f}°...'})
        ok = capture_at_angle(angle)
        if ok:
            state["current_angle"] += CONFIG["degrees_per_step"]
            socketio.emit('scan_progress', {
                'angle': state["current_angle"],
                'next_angle': state["current_angle"]
            })
            # Emit a live pointcloud update every 5 steps if enabled
            if state.get("live_pcd_update", True):
                step = round(state["current_angle"] / CONFIG["degrees_per_step"])
                if step % 5 == 0 and state["pointcloud"] is not None:
                    socketio.emit('pointcloud_update', pcd_to_json(state["pointcloud"]))
        else:
            print(f"[ERROR] Capture failed at {angle}°")
            break
    if state["scanning"]:
        finish_scan()

# ── Flask routes ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', config=CONFIG)

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(CONFIG)

@app.route('/api/config', methods=['POST'])
def update_config():
    CONFIG.update(request.json)
    save_config()
    return jsonify({'status': 'ok'})

@app.route('/api/camera/start', methods=['POST'])
def api_start_camera():
    return jsonify({'status': 'ok' if start_camera() else 'error'})

@app.route('/api/camera/stop', methods=['POST'])
def api_stop_camera():
    stop_camera()
    return jsonify({'status': 'ok'})

@app.route('/api/scan/start', methods=['POST'])
def api_start_scan():
    ok = start_scan()
    if ok and CONFIG["arduino_enabled"]:
        threading.Thread(target=auto_scan_loop, daemon=True).start()
        return jsonify({'status': 'ok', 'mode': 'auto'})
    return jsonify({'status': 'ok' if ok else 'error', 'mode': 'manual'})

@app.route('/api/scan/capture', methods=['POST'])
def api_capture():
    if not state["scanning"]:
        return jsonify({'status': 'error', 'message': 'Not scanning'})
    if CONFIG["arduino_enabled"]:
        return jsonify({'status': 'error', 'message': 'Arduino mode — auto only'})
    angle = state["current_angle"]
    ok    = capture_at_angle(angle)
    if ok:
        state["current_angle"] += CONFIG["degrees_per_step"]
        if state["current_angle"] >= 360.0:
            finish_scan()
            return jsonify({'status': 'complete', 'angle': angle})
        return jsonify({'status': 'ok', 'angle': angle,
                        'next_angle': state["current_angle"]})
    return jsonify({'status': 'error'})

@app.route('/api/scan/stop', methods=['POST'])
def api_stop_scan():
    finish_scan()
    return jsonify({'status': 'ok'})

@app.route('/api/pointcloud', methods=['GET'])
def api_get_pointcloud():
    return jsonify(pcd_to_json(state["pointcloud"]))

@app.route('/api/scan/live_update', methods=['POST'])
def api_set_live_update():
    state["live_pcd_update"] = bool(request.json.get("enabled", True))
    return jsonify({'status': 'ok', 'live_pcd_update': state["live_pcd_update"]})

@app.route('/api/export/stl', methods=['POST'])
def api_export_stl():
    data      = request.json
    clean_idx = data.get("clean_indices", [])
    avoid_idx = data.get("avoid_indices", [])
    filename  = data.get("filename", "").strip()
    o3d       = get_open3d()
    if o3d is None or state["pointcloud"] is None:
        return jsonify({'status': 'error'})
    pcd = state["pointcloud"]

    # Determine output paths
    if filename:
        # Sanitise: strip extension, keep only safe characters
        safe_name = "".join(c for c in os.path.splitext(filename)[0] if c.isalnum() or c in "-_ ")
        safe_name = safe_name.strip() or "scan"
        os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)
        clean_path = os.path.join(GENERATED_MODELS_DIR, f"{safe_name}_clean.stl")
        avoid_path = os.path.join(GENERATED_MODELS_DIR, f"{safe_name}_avoid.stl")
    else:
        clean_path = None  # make_stl_from_pcd will fall back to CONFIG paths
        avoid_path = None

    clean_result = make_stl_from_pcd(pcd.select_by_index(clean_idx), "clean", clean_path if clean_idx else None)
    avoid_result = make_stl_from_pcd(pcd.select_by_index(avoid_idx), "avoid", avoid_path if avoid_idx else None)

    results = {}
    if clean_result:
        results['clean'] = send_stl_to_moveit(clean_result, "clean_zone")
    if avoid_result:
        results['avoid'] = send_stl_to_moveit(avoid_result, "avoid_zone")
    return jsonify({'status': 'ok', 'results': results,
                    'clean_path': clean_result, 'avoid_path': avoid_result,
                    'filename': safe_name if filename else None})

@app.route('/api/export/stl/download', methods=['GET'])
def api_download_stl():
    """Stream an STL file back to the browser as a download."""
    label    = request.args.get('label', 'clean')
    filename = request.args.get('filename', '').strip()

    if filename:
        safe_name = "".join(c for c in os.path.splitext(filename)[0] if c.isalnum() or c in "-_ ").strip()
        dl_path = os.path.join(GENERATED_MODELS_DIR, f"{safe_name}_{label}.stl")
    else:
        dl_path = CONFIG["clean_stl_path"] if label == 'clean' else CONFIG["avoid_stl_path"]
        dl_path = os.path.expanduser(dl_path)

    if not os.path.exists(dl_path):
        return jsonify({'status': 'error', 'message': 'File not found — export first'}), 404

    dl_name = os.path.basename(dl_path)
    return send_file(dl_path, mimetype='application/octet-stream',
                     as_attachment=True, download_name=dl_name)


@app.route('/api/export/primitive', methods=['POST'])
def api_export_primitive():
    """Export the point cloud fitted to a simple geometric primitive."""
    data      = request.json
    primitive    = data.get("primitive", "box")   # box | sphere | cylinder | hull | lowpoly
    hull_detail  = float(data.get("hull_detail", 50))
    lowpoly_tris = int(data.get("lowpoly_tris", 200))
    clean_idx = data.get("clean_indices", [])
    avoid_idx = data.get("avoid_indices", [])
    filename  = data.get("filename", "").strip()
    o3d       = get_open3d()
    if o3d is None or state["pointcloud"] is None:
        return jsonify({'status': 'error', 'message': 'No point cloud available'})
    pcd = state["pointcloud"]
    if filename:
        safe_name = "".join(c for c in os.path.splitext(filename)[0]
                            if c.isalnum() or c in "-_ ").strip() or "scan"
        os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)
        clean_path = os.path.join(GENERATED_MODELS_DIR, f"{safe_name}_{primitive}_clean.stl")
        avoid_path = os.path.join(GENERATED_MODELS_DIR, f"{safe_name}_{primitive}_avoid.stl")
    else:
        # Always use a named file so the viewer can fetch it reliably
        safe_name = 'scan'
        os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)
        clean_path = os.path.join(GENERATED_MODELS_DIR, f"{safe_name}_{primitive}_clean.stl")
        avoid_path = os.path.join(GENERATED_MODELS_DIR, f"{safe_name}_{primitive}_avoid.stl")
    results = {}
    exported_clean = None
    exported_avoid = None
    if clean_idx:
        r = fit_primitive_stl(pcd.select_by_index(clean_idx), primitive, "clean", clean_path, hull_detail, lowpoly_tris)
        if r:
            exported_clean = r
            results['clean'] = send_stl_to_moveit(r, "clean_zone")
    if avoid_idx:
        r = fit_primitive_stl(pcd.select_by_index(avoid_idx), primitive, "avoid", avoid_path, hull_detail, lowpoly_tris)
        if r:
            exported_avoid = r
            results['avoid'] = send_stl_to_moveit(r, "avoid_zone")
    if exported_clean:
        state['primitive_stl'] = exported_clean
        # Use the primitive as the primary STL for path planning — it's cleaner
        # than the full surface and path_planning.py works correctly with it.
        CONFIG['clean_stl_path'] = exported_clean
    return jsonify({'status': 'ok', 'results': results, 'filename': safe_name,
                    'clean_path': exported_clean, 'avoid_path': exported_avoid})

@app.route('/api/mesh/build', methods=['POST'])
def api_mesh_build():
    """
    Fit a primitive to the ENTIRE point cloud (no zone indices).
    Saves the STL, stores its path in state, returns a download URL.
    """
    data         = request.json or {}
    primitive    = data.get('primitive',    'hull')
    hull_detail  = float(data.get('hull_detail',  50))
    lowpoly_tris = int(data.get('lowpoly_tris',  200))
    filename     = data.get('filename', '').strip()
    o3d          = get_open3d()
    if o3d is None or state['pointcloud'] is None:
        return jsonify({'status': 'error', 'message': 'No point cloud — complete a scan first'})

    safe_name = ("".join(c for c in os.path.splitext(filename)[0]
                         if c.isalnum() or c in '-_ ').strip() or 'scan') if filename else 'scan'
    os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)
    out_path = os.path.join(GENERATED_MODELS_DIR, f'{safe_name}_{primitive}_full.stl')

    pcd = state['pointcloud']
    result = fit_primitive_stl(pcd, primitive, 'clean', out_path, hull_detail, lowpoly_tris, save_mm=False)
    if not result:
        return jsonify({'status': 'error', 'message': 'Mesh build failed — check server log'})

    # Count triangles
    try:
        mesh = o3d.io.read_triangle_mesh(result)
        tri_count = len(mesh.triangles)
    except Exception:
        tri_count = 0

    state['built_mesh_path'] = result
    state['primitive_stl']   = result
    CONFIG['clean_stl_path'] = result

    return jsonify({
        'status':       'ok',
        'stl_path':     result,
        'download_url': f'/api/mesh/download?path={result}',
        'tri_count':    tri_count,
    })


@app.route('/api/mesh/download', methods=['GET'])
def api_mesh_download():
    """Serve a mesh STL file for the browser viewer."""
    path = request.args.get('path', '')
    path = os.path.expanduser(path)
    if not path or not os.path.exists(path):
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    return send_file(path, mimetype='application/octet-stream',
                     as_attachment=False, download_name=os.path.basename(path))


@app.route('/api/mesh/split', methods=['POST'])
def api_mesh_split():
    """
    Split the already-built mesh STL into clean/avoid STLs by triangle index.
    clean_tris / avoid_tris: lists of triangle indices in the full mesh.
    Saves <name>_clean.stl and <name>_avoid.stl, sends them to MoveIt.
    """
    data       = request.json or {}
    stl_path   = data.get('stl_path', state.get('built_mesh_path', ''))
    clean_tris = data.get('clean_tris', [])
    avoid_tris = data.get('avoid_tris', [])
    filename   = data.get('filename', '').strip()
    o3d        = get_open3d()

    if not stl_path or not os.path.exists(stl_path):
        return jsonify({'status': 'error', 'message': f'Built mesh not found: {stl_path}'})
    if not clean_tris and not avoid_tris:
        return jsonify({'status': 'error', 'message': 'No triangles selected'})

    try:
        mesh = o3d.io.read_triangle_mesh(stl_path)
        tris = np.asarray(mesh.triangles)
        verts = np.asarray(mesh.vertices)

        safe_name = ("".join(c for c in os.path.splitext(filename)[0]
                             if c.isalnum() or c in '-_ ').strip() or 'scan') if filename else 'scan'
        os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)

        results = {}
        exported_clean = None

        for label, idx_list in [('clean', clean_tris), ('avoid', avoid_tris)]:
            if not idx_list:
                continue
            idx_arr = np.array(idx_list, dtype=int)
            idx_arr = idx_arr[idx_arr < len(tris)]  # bounds check
            if not len(idx_arr):
                continue

            sub = o3d.geometry.TriangleMesh()
            sub.vertices  = o3d.utility.Vector3dVector(verts)
            sub.triangles = o3d.utility.Vector3iVector(tris[idx_arr])
            sub.remove_unreferenced_vertices()
            sub.compute_vertex_normals()
            # Scale to mm for MoveIt/path_planning compatibility
            sub.scale(1000, center=(0, 0, 0))

            out = os.path.join(GENERATED_MODELS_DIR, f'{safe_name}_{label}.stl')
            o3d.io.write_triangle_mesh(out, sub)
            print(f'[MESH] Split {label}: {len(idx_arr)} tris → {out}')

            results[label] = send_stl_to_moveit(out, f'{label}_zone')
            if label == 'clean':
                exported_clean = out

        if exported_clean:
            state['primitive_stl']   = exported_clean
            state['built_mesh_path'] = exported_clean
            CONFIG['clean_stl_path'] = exported_clean

        return jsonify({'status': 'ok', 'results': results})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/moveit/send', methods=['POST'])
def api_send_to_moveit():
    data    = request.json
    ok      = send_stl_to_moveit(data.get('path'), data.get('name', 'scanned_object'))
    return jsonify({'status': 'ok' if ok else 'error'})

# ── Robot / path-planning subprocess control ──────────────────

_robot_proc      = None
_robot_proc_lock = threading.Lock()

# ── UR10 Startup sequence ─────────────────────────────────────

_startup_running = False
_startup_lock    = threading.Lock()

def _run_startup_sequence():
    """Run the UR10 startup steps in order, streaming log lines via SocketIO."""
    global _startup_running
    import subprocess as _sp

    def _log(msg):
        print(f'[STARTUP] {msg}')
        socketio.emit('startup_log', {'line': msg})

    _log(f'DISPLAY={os.environ.get("DISPLAY", "NOT SET")}')
    _log(f'XAUTHORITY={os.environ.get("XAUTHORITY", "NOT SET")}')

    def _run(cmd, shell=False, env=None):
        """Run a command, stream its output, return exit code."""
        try:
            proc = _sp.Popen(
                cmd, shell=shell,
                stdout=_sp.PIPE, stderr=_sp.STDOUT,
                env=env,
            )
            for raw in iter(proc.stdout.readline, b''):
                _log(raw.decode('utf-8', errors='replace').rstrip())
            proc.wait()
            return proc.returncode
        except Exception as e:
            _log(f'ERROR: {e}')
            return -1

    ros_setup = '/opt/ros/jazzy/setup.bash'
    if not os.path.exists(ros_setup):
        for d in ('iron', 'humble', 'galactic'):
            c = f'/opt/ros/{d}/setup.bash'
            if os.path.exists(c):
                ros_setup = c
                break

    try:
        # ── Step 0: Kill any stale ROS2 processes ─────────────────
        _log('=== Step 0: Clearing stale ROS2 processes ===')
        for pattern in ['ur_robot_driver', 'ur_moveit', 'rviz2', 'ros2']:
            _sp.run(['pkill', '-f', pattern], capture_output=True)
        _log('Stale processes cleared. Waiting 2s…')
        time.sleep(2)

        # ── Step 1: Network ───────────────────────────────────
        _log('=== Step 1: Configuring network interface ===')
        iface = CONFIG.get('ur_network_iface', 'enp5s0')
        host_ip = CONFIG.get('ur_host_ip', '192.168.0.100/24')
        for cmd in [
            ['sudo', 'ip', 'addr', 'flush', 'dev', iface],
            ['sudo', 'ip', 'addr', 'add', host_ip, 'dev', iface],
            ['sudo', 'ip', 'link', 'set', iface, 'up'],
        ]:
            rc = _run(cmd)
            if rc != 0:
                _log(f'WARNING: command returned {rc}: {" ".join(cmd)}')
        _log('Network configured. Waiting 2s…')
        time.sleep(2)

        # ── Step 2: UR Driver ─────────────────────────────────
        _log('=== Step 2: Launching UR Driver ===')
        robot_ip = CONFIG.get('robot_ip', '192.168.0.43')
        ur_type  = CONFIG.get('ur_type', 'ur10')
        _sp.Popen(
            ['bash', '-c',
             f'source {ros_setup} && '
             f'ros2 launch ur_robot_driver ur_control.launch.py '
             f'ur_type:={ur_type} robot_ip:={robot_ip}'],
            stdout=_sp.PIPE, stderr=_sp.STDOUT,
        )
        _log('UR Driver launched. Waiting 10s for initialisation…')
        _log('(Press Play on the pendant now if you haven\'t already)')
        time.sleep(10)

        # ── Step 3: MoveIt ────────────────────────────────────
        _log('=== Step 3: Launching MoveIt ===')
        display_env = dict(os.environ)
        display_env['DISPLAY'] = os.environ.get('DISPLAY', ':1')
        display_env['XAUTHORITY'] = os.environ.get('XAUTHORITY',
                                                   os.path.expanduser('~/.Xauthority'))
        moveit_proc = _sp.Popen(
            ['bash', '-c',
             f'source {ros_setup} && '
             f'ros2 launch ur_moveit_config ur_moveit.launch.py '
             f'ur_type:={ur_type} launch_rviz:=false'],
            stdout=_sp.PIPE, stderr=_sp.STDOUT,
            env=display_env,
        )
        threading.Thread(
            target=lambda: [_log(l.decode('utf-8', errors='replace').rstrip())
                            for l in iter(moveit_proc.stdout.readline, b'')],
            daemon=True
        ).start()
        _log('MoveIt launched. Waiting 5s…')
        time.sleep(5)
        _log('=== Step 3b: Launching RViz ===')
        launch_rviz_for_execution()
        _log('RViz launched.')

        # ── Step 4: Check joint states ────────────────────────
        _log('=== Step 4: Checking joint states ===')
        _run(['bash', '-c',
              f'source {ros_setup} && '
              f'ros2 topic echo /joint_states --once'])

        # ── Step 5: Load environment ──────────────────────────
        _log('=== Step 5: Loading Environment ===')
        env_script = os.path.expanduser(
            CONFIG.get('environment_setup_script',
                       '~/Documents/SMR/Scan/STLFiles/environment_setup.py'))
        if os.path.isfile(env_script):
            _run(['bash', '-c',
                  f'source {ros_setup} && python3 {env_script}'],
                 env=dict(os.environ))
        else:
            _log(f'WARNING: environment_setup.py not found at {env_script} — skipping')

        _log('=== All done! System ready. ===')
        socketio.emit('startup_done', {'code': 0})

    except Exception as e:
        _log(f'FATAL: {e}')
        socketio.emit('startup_done', {'code': -1})
    finally:
        with _startup_lock:
            _startup_running = False


@app.route('/api/startup', methods=['POST'])
def api_startup():
    """Begin the UR10 startup sequence in a background thread."""
    global _startup_running
    with _startup_lock:
        if _startup_running:
            return jsonify({'status': 'error', 'message': 'Already running'})
        _startup_running = True
    threading.Thread(target=_run_startup_sequence, daemon=True).start()
    return jsonify({'status': 'ok'})


@app.route('/api/startup/status', methods=['GET'])
def api_startup_status():
    with _startup_lock:
        running = _startup_running
    return jsonify({'running': running})

def _stream_robot_output(proc):
    """Stream path_planning.py stdout → SocketIO.

    Protocol lines emitted as structured events:
      PREVIEW_FACE:<idx>:<label>:<type>:<n_stripes>
      PREVIEW_FACE_DONE:<idx>
      PREVIEW_ALL_DONE
      PREVIEW_ABORTED
    All lines also go to robot_log for the console.
    """
    try:
        for raw in iter(proc.stdout.readline, b''):
            line = raw.decode('utf-8', errors='replace').rstrip()
            if line.startswith('PREVIEW_FACE:') and not line.startswith('PREVIEW_FACE_DONE'):
                parts = line.split(':')
                socketio.emit('preview_face', {
                    'index': int(parts[1]), 'label': parts[2],
                    'type': parts[3],       'stripes': int(parts[4]),
                })
            elif line.startswith('PREVIEW_FACE_DONE:'):
                socketio.emit('preview_face_done', {'index': int(line.split(':')[1])})
            elif line == 'PREVIEW_ALL_DONE':
                socketio.emit('preview_all_done', {})
            elif line == 'PREVIEW_ABORTED':
                socketio.emit('preview_aborted', {})
            elif line.startswith('COVERAGE_UPDATE:'):
                # COVERAGE_UPDATE:<pct>:<stripe>:<covered_cm2>:<total_cm2>
                parts = line.split(':')
                socketio.emit('coverage_update', {
                    'pct':        float(parts[1]),
                    'stripe':     int(parts[2]),
                    'covered_cm2': float(parts[3]),
                    'total_cm2':  float(parts[4]),
                })
            socketio.emit('robot_log', {'line': line})
        proc.wait()
        code = proc.returncode
        socketio.emit('robot_done', {
            'code': code,
            'message': '✔ Complete.' if code == 0 else f'⚠ Exited with code {code}.'
        })
    except Exception as e:
        socketio.emit('robot_done', {'code': -1, 'message': f'Stream error: {e}'})
    finally:
        with _robot_proc_lock:
            global _robot_proc
            _robot_proc = None


def _launch_path_planner(extra_args):
    """Start path_planning.py subprocess. Returns (proc, error_str)."""
    global _robot_proc
    with _robot_proc_lock:
        if _robot_proc is not None and _robot_proc.poll() is None:
            return None, 'Already running'

    clean_stl = os.path.expanduser(CONFIG['clean_stl_path'])
    if not os.path.exists(clean_stl):
        return None, f'Clean STL not found: {clean_stl} — export faces first'

    default_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path_planning.py')
    script = os.path.expanduser(CONFIG.get('path_planning_script', '').strip() or default_script)

    if os.path.isdir(script):
        return None, (
            f'path_planning_script points to a directory: {script}\n'
            f'It must be the full path to the .py file, e.g.:\n'
            f'  {os.path.join(script, "path_planning.py")}'
        )
    if not os.path.isfile(script):
        return None, (
            f'path_planning.py not found at: {script}\n'
            f'Set the correct path in ⚙ Settings → Path Planning Script.'
        )

    cmd = [sys.executable, script, '--stl', clean_stl] + extra_args

    # Use the primitive/hull STL for the visualizer mesh if one was exported —
    # it's a much cleaner shape than the full surface reconstruction
    viz_stl = state.get('primitive_stl')
    if viz_stl and os.path.exists(viz_stl):
        cmd += ['--viz-stl', viz_stl]
        print(f"[ROBOT] Using primitive STL for visualizer: {viz_stl}")
    else:
        print(f"[ROBOT] No primitive STL found — visualizer will use full surface mesh")

    print(f"[ROBOT] Launching: {' '.join(cmd)}")

    import subprocess
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(script),
        )
        with _robot_proc_lock:
            _robot_proc = proc
        threading.Thread(target=_stream_robot_output, args=(proc,), daemon=True).start()
        return proc, None
    except Exception as e:
        return None, str(e)


@app.route('/api/robot/preview', methods=['POST'])
def api_robot_preview():
    """Launch path_planning.py --preview. UI drives it via /api/robot/confirm."""
    proc, err = _launch_path_planner(['--preview'])
    if err:
        return jsonify({'status': 'error', 'message': err})
    return jsonify({'status': 'ok', 'pid': proc.pid})


@app.route('/api/robot/confirm', methods=['POST'])
def api_robot_confirm():
    """Send NEXT / EXECUTE / CANCEL to the running preview process stdin."""
    global _robot_proc
    with _robot_proc_lock:
        proc = _robot_proc
    if proc is None or proc.poll() is not None:
        return jsonify({'status': 'error', 'message': 'No preview running'})
    signal = request.json.get('signal', 'NEXT')
    try:
        proc.stdin.write((signal + '\n').encode())
        proc.stdin.flush()
        return jsonify({'status': 'ok', 'signal': signal})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


def generate_rviz_config(config_path: str):
    """
    Write a minimal RViz2 config that shows:
      - Grid (ground reference)
      - RobotModel (live UR10 from /robot_description)
      - TF (so arm links move)
      - MarkerArray on /cleaning_coverage  (cone + red patches + HUD)
      - MarkerArray on /waypoint_markers   (planned path lines)
      - CollisionObject mesh via PlanningScene display
    """
    cfg = """\
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Cell Size: 0.5
      Color: 160; 160; 164
      Line Style:
        Line Width: 0.03
        Value: Lines
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Enabled: false
      Description Topic:
        Depth: 5
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Visual Enabled: true
      Collision Enabled: false
      Update Interval: 0
    - Class: moveit_rviz_plugin/MotionPlanning
      Name: MotionPlanning
      Enabled: true
      Move Group Namespace: ""
      Robot Description: robot_description
      Planning Scene Topic:
        Value: /monitored_planning_scene
      Scene Geometry:
        Scene Alpha: 0.9
        Scene Color: 50; 230; 50
      Scene Robot:
        Show Robot Visual: true
        Show Robot Collision: true
        Robot Alpha: 0.5
      Planned Path:
        Show Robot Visual: true
        Show Robot Collision: false
        State Display Time: 0.05 s
        Loop Animation: true
      Alpha: 1
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true
      Show Arrows: false
      Show Axes: false
      Show Names: false
      Marker Scale: 0.3
      Update Interval: 0
      Frame Timeout: 15
      Frames:
        All Enabled: false
    - Class: rviz_default_plugins/MarkerArray
      Name: CleaningCoverage
      Enabled: true
      Topic:
        Depth: 50
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /cleaning_coverage
      Namespaces: {}
    - Class: rviz_default_plugins/MarkerArray
      Name: WaypointPath
      Enabled: true
      Topic:
        Depth: 10
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /waypoint_markers
      Namespaces: {}
  Enabled: true
  Global Options:
    Background Color: 13; 17; 23
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.5
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.06
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0.65
        Y: 0
        Z: 0.4
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.45
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 3.8
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 900
  Hide Left Dock: false
  Hide Right Dock: true
  Views:
    collapsed: false
  Width: 1400
  X: 50
  Y: 50
"""
    with open(config_path, 'w') as f:
        f.write(cfg)
    print(f"[RViz] Config written to {config_path}")


_rviz_proc = None

def launch_rviz_for_execution():
    """Launch a fresh RViz2 window configured for the cleaning run."""
    global _rviz_proc

    # Kill any previous RViz we launched
    if _rviz_proc is not None and _rviz_proc.poll() is None:
        _rviz_proc.terminate()
        _rviz_proc = None

    config_path = os.path.expanduser('~/scanner_cleaning_rviz.rviz')
    generate_rviz_config(config_path)

    import subprocess
    try:
        # Detect ROS2 distro from environment, fall back to humble
        ros_distro = os.environ.get('ROS_DISTRO', 'humble')
        ros_setup  = f'/opt/ros/{ros_distro}/setup.bash'
        if not os.path.exists(ros_setup):
            # Try to find any installed distro
            for distro in ('jazzy', 'iron', 'humble', 'galactic'):
                candidate = f'/opt/ros/{distro}/setup.bash'
                if os.path.exists(candidate):
                    ros_setup = candidate
                    break

        cmd = ['bash', '-c',
               f'source {ros_setup} && rviz2 -d {config_path}']

        rviz_env = dict(os.environ)
        rviz_env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        rviz_env.pop('QT_PLUGIN_PATH', None)

        _rviz_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=rviz_env,
        )
        threading.Thread(
            target=lambda: [print(f'[RViz] {l.decode("utf-8", errors="replace").rstrip()}')
                            for l in iter(_rviz_proc.stdout.readline, b'')],
            daemon=True
        ).start()
        print(f"[RViz] Launched (pid {_rviz_proc.pid}) with config {config_path}")
        return True
    except Exception as e:
        print(f"[RViz] Failed to launch: {e}")
        return False


@app.route('/api/robot/execute', methods=['POST'])
def api_robot_execute():
    """Launch a fresh RViz window then start path_planning.py in execute mode."""
    launch_rviz_for_execution()

    def _delayed_launch():
        # Give RViz time to start subscribing before path_planning's clearer fires
        time.sleep(1.5)
        proc, err = _launch_path_planner([])
        if err:
            socketio.emit('robot_log', {'line': f'✗ {err}'})
            socketio.emit('robot_done', {'code': -1, 'message': err})

    threading.Thread(target=_delayed_launch, daemon=True).start()
    return jsonify({'status': 'ok'})


@app.route('/api/robot/stop', methods=['POST'])
def api_robot_stop():
    global _robot_proc, _rviz_proc
    with _robot_proc_lock:
        proc = _robot_proc
    if proc is None or proc.poll() is not None:
        return jsonify({'status': 'ok', 'message': 'Not running'})
    try:
        proc.terminate()
        socketio.emit('robot_log', {'line': '⚠ Stopped by user.'})
        socketio.emit('robot_done', {'code': -1, 'message': 'Stopped by user.'})
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'scanning':          state["scanning"],
        'scan_complete':     state["scan_complete"],
        'current_angle':     state["current_angle"],
        'has_pointcloud':    state["pointcloud"] is not None,
        'camera_running':    state["preview_running"],
        'arduino_enabled':   CONFIG["arduino_enabled"],
        'arduino_connected': arduino_serial is not None,
    })

# ── SocketIO ──────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    print("[WS] Client connected")
    emit('config', CONFIG)
    emit('status', {
        'scanning':        state["scanning"],
        'scan_complete':   state["scan_complete"],
        'current_angle':   state["current_angle"],
        'arduino_enabled': CONFIG["arduino_enabled"],
    })

@socketio.on('request_pointcloud')
def on_request_pointcloud():
    emit('pointcloud_update', pcd_to_json(state["pointcloud"]))

# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    load_config()
    os.makedirs(GENERATED_MODELS_DIR, exist_ok=True)
    print("=" * 50)
    print("  3D Scanner Web App")
    print("  Open http://localhost:5000 in your browser")
    print(f"  Arduino: {'ENABLED' if CONFIG['arduino_enabled'] else 'DISABLED'}")
    print(f"  Models dir: {GENERATED_MODELS_DIR}")
    print(f"  Camera tilt: {CONFIG['camera_tilt_deg']}°")
    print(f"  Camera distance: {CONFIG['camera_distance_m']}m")
    print(f"  Camera height: {CONFIG['camera_height_m']}m")
    print("=" * 50)
    start_camera()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)