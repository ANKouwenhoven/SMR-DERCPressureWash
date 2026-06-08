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
    "depth_max_m": 0.75,  # ignore anything further than this from camera
    "camera_height_m":    0.24,
    "camera_distance_m":  0.52,
    "camera_tilt_deg":    22.5,
    "camera_x_offset_m":  0.0,
    "camera_y_offset_m":  0.0,
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
    # RealSense depth sensor settings
    "rs_laser_power":     150,   # IR projector brightness 0-360 mW
    "rs_confidence":      1,     # depth confidence threshold 0-3
    "rs_depth_units":     0.001, # metres per depth unit (0.001 = 1mm res)
    "rs_inter_cam_sync":  1,     # 0=off 1=master (tighter RGB/depth sync)
}

CONFIG_PATH = os.path.expanduser("~/scanner_config.json")
GENERATED_MODELS_DIR = r"C:\Users\Arnoud\Documents\HHS\MINOR SMR\SMR-DERCPressureWash\generated_models"

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
            arduino_serial.write(b"ROTATE\n")
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
            state["last_frame"] = color_img
            _, buf = cv2.imencode('.jpg',
                                  cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode('utf-8')
            socketio.emit('camera_frame', {'image': b64})
        except Exception as e:
            pass
        time.sleep(1.0 / CONFIG["camera_fps"])

# ── Pointcloud processing ─────────────────────────────────────

def get_camera_extrinsics():
    dtr  = np.pi / 180
    # 25° below horizontal = rotate camera downward by 25°
    tilt = CONFIG["camera_tilt_deg"] * dtr  # positive = downward tilt

    d = CONFIG["camera_distance_m"]
    cam_x_offset = CONFIG.get("camera_x_offset_m", -0.15)
    cam_y_offset = CONFIG.get("camera_y_offset_m", 0.0)

    t = np.array([
        cam_x_offset,
        -d + cam_y_offset,
        CONFIG["camera_height_m"]
    ])

    # Camera faces +Y (toward turntable), tilted downward around X axis
    # tilt > 0 means nose down
    R_tilt = np.array([
        [1,            0,           0],
        [0,  np.cos(tilt), np.sin(tilt)],
        [0, -np.sin(tilt), np.cos(tilt)],
    ])

    return R_tilt, t

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
    # This "unspins" each frame so all frames align in a common world space
    dtr = np.pi / 180
    a   = -angle_deg * dtr  # negative = counter-rotate
    R_unspin = np.array([
        [ np.cos(a), -np.sin(a), 0],
        [ np.sin(a),  np.cos(a), 0],
        [         0,          0, 1],
    ])
    # Rotate around turntable centre (world origin in XY)
    pcd.rotate(R_unspin, center=(0, 0, 0))

    # Crop to bbox
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        (-0.40, -0.40, -0.005),
        ( 0.40,  0.40, 0.80)
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
    z_min, z_max =  0.01, 0.60

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
        print(f"[STL] Poisson raw: {len(mesh.triangles)} triangles")

        # ── 4. Density trim (floaters only) ───────────────────
        dens = np.asarray(densities)
        mesh.remove_vertices_by_mask(dens < np.percentile(dens, 5))
        print(f"[STL] After density trim: {len(mesh.triangles)} triangles")

        # ── 5. Largest connected component ────────────────────
        # Drops disconnected blobs (noisy point clusters Poisson
        # wraps into small closed bubbles).
        tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        tri_clusters    = np.asarray(tri_clusters)
        cluster_n_tris  = np.asarray(cluster_n_tris)
        largest         = cluster_n_tris.argmax()
        mesh.remove_triangles_by_mask(tri_clusters != largest)
        mesh.remove_unreferenced_vertices()
        print(f"[STL] After component filter: {len(mesh.triangles)} triangles")
        if len(mesh.triangles) == 0:
            print("[STL] Empty mesh after component filter — aborting")
            return None

        # ── 6. Convex hull crop ────────────────────────────────
        # Remove Poisson geometry that lies outside the actual scan
        # extent (hallucinated caps, skirts, side blobs).
        # 10 mm tolerance is generous enough to keep legitimate surface
        # vertices on thin objects while still catching hallucinated fill.
        hull, _ = pcd_down.compute_convex_hull()
        hull_scene = o3d.t.geometry.RaycastingScene()
        hull_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hull))
        verts = np.asarray(mesh.vertices).astype(np.float32)
        query = o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)
        signed_dist = hull_scene.compute_signed_distance(query).numpy()
        mesh.remove_vertices_by_mask(signed_dist > 0.010)
        print(f"[STL] After hull crop: {len(mesh.triangles)} triangles")
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
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

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
        print(f"[STL] Applied 7 mm outward offset")

        # ── 9. Scale to mm, write STL ──────────────────────────
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

def fit_primitive_stl(pcd, primitive="box", label="clean", output_path=None, hull_detail=50, lowpoly_tris=200):
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
            # Load the already-exported full surface STL and decimate it.
            # The source file is already clean, cropped, and in mm, so
            # we just collapse triangles to the target count and write out.
            if output_path is None:
                src = CONFIG["clean_stl_path"] if label == "clean" else CONFIG["avoid_stl_path"]
                src = os.path.expanduser(src)
                lp_path = src
            else:
                src = output_path.replace(f"_lowpoly_{label}.stl", f"_{label}.stl")
                lp_path = output_path
            if not os.path.exists(src):
                raise RuntimeError(
                    f"Low-poly requires the full surface STL to exist first.\n"
                    f"Use 'BOTH + PREVIEW' to export both together.\n"
                    f"Expected: {src}")
            mesh = o3d.io.read_triangle_mesh(src)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            n_before = len(mesh.triangles)
            if n_before > lowpoly_tris:
                mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=lowpoly_tris)
                mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_vertices()
            # Push vertices outward along normals to ensure the low-poly
            # mesh sits slightly outside the real surface after decimation
            # (decimation can pull vertices inward as it collapses edges).
            # 5 mm in mm-space = 5.0 units.
            mesh.compute_vertex_normals()
            verts = np.asarray(mesh.vertices)
            norms = np.asarray(mesh.vertex_normals)
            mesh.vertices = o3d.utility.Vector3dVector(verts + norms * 5.0)
            mesh.compute_vertex_normals()
            os.makedirs(os.path.dirname(lp_path), exist_ok=True)
            o3d.io.write_triangle_mesh(lp_path, mesh)
            print(f"[STL] Low-poly: {n_before} -> {len(mesh.triangles)} tris "
                  f"(+5 mm offset) -> {lp_path}")
            return lp_path

        else:
            print(f"[STL] Unknown primitive '{primitive}'")
            return None

        # Apply 7 mm outward offset for robot standoff clearance
        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        norms = np.asarray(mesh.vertex_normals)
        mesh.vertices = o3d.utility.Vector3dVector(verts + norms * 0.007)

        # Scale to mm and write
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

def send_stl_to_moveit(stl_path, object_name, operation="add", x=0.5, y=0.0, z=0.2):
    rclpy_mod = get_rclpy()
    if rclpy_mod is None:
        print("[WARN] ROS2 not available")
        return False
    try:
        from moveit_msgs.msg   import CollisionObject, PlanningScene
        from shape_msgs.msg    import Mesh, MeshTriangle
        from geometry_msgs.msg import Pose, Point
        from std_msgs.msg      import Header
        import rclpy as rclpy_lib
        from rclpy.node import Node

        vertices, triangles = read_stl_binary(stl_path)
        mesh = Mesh()
        for v in vertices:
            p = Point()
            p.x, p.y, p.z = float(v[0])/1000.0, float(v[1])/1000.0, float(v[2])/1000.0
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
        obj.header.frame_id = "world"
        obj.header.stamp = node.get_clock().now().to_msg()
        obj.id = object_name
        obj.operation = CollisionObject.ADD if operation == "add" else CollisionObject.REMOVE
        obj.meshes.append(mesh)
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.w = 1.0
        obj.mesh_poses.append(pose)
        pub.publish(obj)
        time.sleep(0.5)
        node.destroy_node()
        print(f"[ROS2] {operation.upper()} '{object_name}' in MoveIt scene")
        return True
    except Exception as e:
        print(f"[ERROR] MoveIt publish failed: {e}")
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
            # Emit a live pointcloud update every 5 steps so the
            # viewer builds up progressively during the scan
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
        results['clean'] = send_stl_to_moveit(clean_result, "clean_zone", x=0.5, y=0.0, z=0.2)
    if avoid_result:
        results['avoid'] = send_stl_to_moveit(avoid_result, "avoid_zone", x=0.5, y=0.0, z=0.2)
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
    if clean_idx:
        r = fit_primitive_stl(pcd.select_by_index(clean_idx), primitive, "clean", clean_path, hull_detail, lowpoly_tris)
        if r: results['clean'] = send_stl_to_moveit(r, "clean_zone", x=0.5, y=0.0, z=0.2)
    if avoid_idx:
        r = fit_primitive_stl(pcd.select_by_index(avoid_idx), primitive, "avoid", avoid_path, hull_detail, lowpoly_tris)
        if r: results['avoid'] = send_stl_to_moveit(r, "avoid_zone", x=0.5, y=0.0, z=0.2)
    return jsonify({'status': 'ok', 'results': results, 'filename': safe_name})

@app.route('/api/moveit/send', methods=['POST'])
def api_send_to_moveit():
    data    = request.json
    ok      = send_stl_to_moveit(data.get('path'), data.get('name', 'scanned_object'),
                                  x=data.get('x', 0.5), y=data.get('y', 0.0), z=data.get('z', 0.2))
    return jsonify({'status': 'ok' if ok else 'error'})

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
    print("=" * 50)
    print("  3D Scanner Web App")
    print("  Open http://localhost:5000 in your browser")
    print(f"  Arduino: {'ENABLED' if CONFIG['arduino_enabled'] else 'DISABLED'}")
    print(f"  Camera tilt: {CONFIG['camera_tilt_deg']}°")
    print(f"  Camera distance: {CONFIG['camera_distance_m']}m")
    print(f"  Camera height: {CONFIG['camera_height_m']}m")
    print("=" * 50)
    start_camera()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)