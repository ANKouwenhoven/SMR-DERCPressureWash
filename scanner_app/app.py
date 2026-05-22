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
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import base64
import cv2

# ─────────────────────────────────────────────
#  CONFIGURATION (also editable via Settings UI)
# ─────────────────────────────────────────────
CONFIG = {
    "camera_height_m":    0.165,
    "camera_distance_m":  0.258,
    "camera_tilt_deg":    112.5,
    "degrees_per_step":   7.5,
    "camera_width":       848,
    "camera_height":      480,
    "camera_fps":         30,
    "arduino_port":       "/dev/ttyACM0",
    "arduino_enabled":    False,
    "robot_ip":           "192.168.0.43",
    "robot_port":         50002,
    "clean_stl_path":     os.path.expanduser("~/scan_clean.stl"),
    "avoid_stl_path":     os.path.expanduser("~/scan_avoid.stl"),
}

CONFIG_PATH = os.path.expanduser("~/scanner_config.json")

# ─────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'scanner3d'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
state = {
    "scanning":        False,
    "preview_running": False,
    "current_angle":   0.0,
    "scan_complete":   False,
    "pointcloud":      None,   # open3d PointCloud
    "last_frame":      None,   # numpy array
    "snapshot":        None,   # numpy array
    "clean_stl":       None,
    "avoid_stl":       None,
    "colored_pcd":     None,   # pointcloud with paint colours
}

# ── Lazy imports (graceful if hardware not present) ──────────

def get_realsense():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError:
        print("[WARN] pyrealsense2 not available — using dummy camera")
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
        print("[WARN] rclpy not available — ROS2 features disabled")
        return None

# ── Config persistence ────────────────────────────────────────

def load_config():
    global CONFIG
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            CONFIG.update(json.load(f))

def save_config():
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIG, f, indent=2)

# ── Camera pipeline ───────────────────────────────────────────

pipeline = None
align    = None
pipe_lock = threading.Lock()

def start_camera():
    global pipeline, align
    rs = get_realsense()
    if rs is None:
        return False
    try:
        with pipe_lock:
            pipeline = rs.pipeline()
            config   = rs.config()
            config.enable_stream(rs.stream.color,
                                 CONFIG["camera_width"],
                                 CONFIG["camera_height"],
                                 rs.format.any,
                                 CONFIG["camera_fps"])
            config.enable_stream(rs.stream.depth,
                                 CONFIG["camera_width"],
                                 CONFIG["camera_height"],
                                 rs.format.any,
                                 CONFIG["camera_fps"])
            pipeline.start(config)
            align = rs.align(rs.stream.color)
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

def get_frame():
    rs = get_realsense()
    if rs is None or pipeline is None:
        return None, None
    with pipe_lock:
        frameset = pipeline.wait_for_frames()
        frameset = align.process(frameset)
        color = frameset.get_color_frame()
        depth = frameset.get_depth_frame()
        if not color or not depth:
            return None, None
        color_img = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())
        return color_img, depth_img

def preview_loop():
    while state["preview_running"]:
        try:
            color, _ = get_frame()
            if color is not None:
                state["last_frame"] = color
                # Encode and emit to browser
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode('utf-8')
                socketio.emit('camera_frame', {'image': b64})
        except Exception as e:
            pass
        time.sleep(1.0 / CONFIG["camera_fps"])

# ── Pointcloud processing ─────────────────────────────────────

def process_frame_to_pcd(color_img, depth_img, angle_deg, intrinsic):
    o3d = get_open3d()
    if o3d is None:
        return None
    dtr = np.pi / 180
    depth_o3d = o3d.geometry.Image(depth_img)
    color_o3d = o3d.geometry.Image(color_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    a    = angle_deg * dtr
    dist = CONFIG["camera_distance_m"]
    x    = np.sin(a) * dist - np.cos(a) * 0.035
    y    = -np.cos(a) * dist - np.sin(a) * 0.035
    z    = CONFIG["camera_height_m"]

    o_r  = angle_deg * dtr
    tilt = (-CONFIG["camera_tilt_deg"]) * dtr
    t    = 0.0
    R = [
        [np.cos(o_r)*np.cos(t) - np.cos(tilt)*np.sin(o_r)*np.sin(t),
         -np.cos(o_r)*np.sin(t) - np.cos(tilt)*np.cos(t)*np.sin(o_r),
         np.sin(o_r)*np.sin(tilt)],
        [np.cos(t)*np.sin(o_r) + np.cos(o_r)*np.cos(tilt)*np.sin(t),
         np.cos(o_r)*np.cos(tilt)*np.cos(t) - np.sin(o_r)*np.sin(t),
         -np.cos(o_r)*np.sin(tilt)],
        [np.sin(tilt)*np.sin(t), np.cos(t)*np.sin(tilt), np.cos(tilt)]
    ]
    pcd.rotate(R, (0, 0, 0))
    pcd.translate((x, y, z))
    bbox = o3d.geometry.AxisAlignedBoundingBox((-0.13, -0.13, 0), (0.13, 0.13, 0.2))
    pcd  = pcd.crop(bbox)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)
    return pcd

def pcd_to_json(pcd):
    """Convert open3d pointcloud to JSON for Three.js rendering."""
    if pcd is None:
        return {"points": [], "colors": []}
    pts  = np.asarray(pcd.points).tolist()
    cols = np.asarray(pcd.colors).tolist() if pcd.has_colors() else []
    return {"points": pts, "colors": cols}

def make_stl_from_pcd(pcd, label="clean"):
    """Generate STL mesh from a pointcloud."""
    o3d = get_open3d()
    if o3d is None or pcd is None:
        return None
    try:
        pcd_down = pcd.uniform_down_sample(every_k_points=10)
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=7)
        mesh = mesh.filter_smooth_simple(number_of_iterations=8)
        mesh.scale(1000, center=(0, 0, 0))
        mesh.compute_vertex_normals()
        path = CONFIG["clean_stl_path"] if label == "clean" else CONFIG["avoid_stl_path"]
        o3d.io.write_triangle_mesh(path, mesh)
        return path
    except Exception as e:
        print(f"[ERROR] STL generation failed: {e}")
        return None

# ── ROS2 / MoveIt integration ─────────────────────────────────

def send_stl_to_moveit(stl_path, object_name, operation="add",
                        x=0.5, y=0.0, z=0.2):
    """Publish an STL file to the MoveIt planning scene."""
    rclpy = get_rclpy()
    if rclpy is None:
        print("[WARN] ROS2 not available")
        return False
    try:
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from shape_msgs.msg  import Mesh, MeshTriangle
        from geometry_msgs.msg import Pose, Point
        from std_msgs.msg    import Header
        import rclpy as rclpy_mod
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

        if not rclpy_mod.ok():
            rclpy_mod.init()

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
    stop_camera()
    time.sleep(0.5)
    start_camera()
    return True

def capture_at_angle(angle):
    """Capture and process a frame at the given angle."""
    global scan_intrinsic
    rs  = get_realsense()
    o3d = get_open3d()
    if rs is None or o3d is None or pipeline is None:
        return False
    try:
        # Warm up auto exposure
        for _ in range(5):
            pipeline.wait_for_frames()

        with pipe_lock:
            frameset = pipeline.wait_for_frames()
            frameset = align.process(frameset)
            color_f  = frameset.get_color_frame()
            depth_f  = frameset.get_depth_frame()
            if not color_f or not depth_f:
                return False

            profile = frameset.get_profile()
            intr    = profile.as_video_stream_profile().get_intrinsics()
            scan_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

            color_img = np.asanyarray(color_f.get_data())
            depth_img = np.asanyarray(depth_f.get_data())

        state["snapshot"] = color_img
        # Encode snapshot
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode('utf-8')
        socketio.emit('snapshot', {'image': b64, 'angle': angle})

        pcd = process_frame_to_pcd(color_img, depth_img, angle, scan_intrinsic)
        if pcd is not None:
            state["pointcloud"] += pcd
            # Stream pointcloud update
            pcd_data = pcd_to_json(state["pointcloud"])
            socketio.emit('pointcloud_update', pcd_data)
        return True
    except Exception as e:
        print(f"[ERROR] Capture failed: {e}")
        return False

def finish_scan():
    state["scanning"]      = False
    state["scan_complete"] = True
    socketio.emit('scan_complete', {
        'points': len(np.asarray(state["pointcloud"].points))
    })

# ── Flask routes ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', config=CONFIG)

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(CONFIG)

@app.route('/api/config', methods=['POST'])
def update_config():
    data = request.json
    CONFIG.update(data)
    save_config()
    return jsonify({'status': 'ok'})

@app.route('/api/camera/start', methods=['POST'])
def api_start_camera():
    ok = start_camera()
    return jsonify({'status': 'ok' if ok else 'error'})

@app.route('/api/camera/stop', methods=['POST'])
def api_stop_camera():
    stop_camera()
    return jsonify({'status': 'ok'})

@app.route('/api/scan/start', methods=['POST'])
def api_start_scan():
    ok = start_scan()
    return jsonify({'status': 'ok' if ok else 'error'})

@app.route('/api/scan/capture', methods=['POST'])
def api_capture():
    """Take a photo at the current angle (called after user rotates table)."""
    if not state["scanning"]:
        return jsonify({'status': 'error', 'message': 'Not scanning'})
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
    if state["pointcloud"] is None:
        return jsonify({"points": [], "colors": []})
    return jsonify(pcd_to_json(state["pointcloud"]))

@app.route('/api/export/stl', methods=['POST'])
def api_export_stl():
    """
    Export clean and avoid STLs from painted pointcloud.
    Expects JSON: { "clean_indices": [...], "avoid_indices": [...] }
    """
    data         = request.json
    clean_idx    = data.get("clean_indices", [])
    avoid_idx    = data.get("avoid_indices", [])
    o3d          = get_open3d()
    if o3d is None or state["pointcloud"] is None:
        return jsonify({'status': 'error'})

    pcd = state["pointcloud"]
    clean_pcd = pcd.select_by_index(clean_idx)
    avoid_pcd = pcd.select_by_index(avoid_idx)

    clean_path = make_stl_from_pcd(clean_pcd, "clean")
    avoid_path = make_stl_from_pcd(avoid_pcd, "avoid")

    # Auto send to MoveIt
    results = {}
    if clean_path:
        results['clean'] = send_stl_to_moveit(
            clean_path, "clean_zone", x=0.5, y=0.0, z=0.2)
    if avoid_path:
        results['avoid'] = send_stl_to_moveit(
            avoid_path, "avoid_zone", x=0.5, y=0.0, z=0.2)

    return jsonify({'status': 'ok', 'results': results,
                    'clean_path': clean_path, 'avoid_path': avoid_path})

@app.route('/api/moveit/send', methods=['POST'])
def api_send_to_moveit():
    data   = request.json
    path   = data.get('path')
    name   = data.get('name', 'scanned_object')
    x, y, z = data.get('x', 0.5), data.get('y', 0.0), data.get('z', 0.2)
    ok     = send_stl_to_moveit(path, name, x=x, y=y, z=z)
    return jsonify({'status': 'ok' if ok else 'error'})

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'scanning':      state["scanning"],
        'scan_complete': state["scan_complete"],
        'current_angle': state["current_angle"],
        'has_pointcloud': state["pointcloud"] is not None,
        'camera_running': state["preview_running"],
    })

# ── SocketIO events ───────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    print("[WS] Client connected")
    emit('config', CONFIG)
    emit('status', {
        'scanning':      state["scanning"],
        'scan_complete': state["scan_complete"],
        'current_angle': state["current_angle"],
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
    print("=" * 50)
    start_camera()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
