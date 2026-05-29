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
import serial

# ─────────────────────────────────────────────
#  CONFIGURATION (also editable via Settings UI)
# ─────────────────────────────────────────────
CONFIG = {
    "depth_max_m": 0.75,  # ignore anything further than this from camera
    "camera_height_m":    0.24,
    "camera_distance_m":  0.52,
    "camera_tilt_deg":    25,
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
}

CONFIG_PATH = os.path.expanduser("~/scanner_config.json")

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
            for i in range(2):
                arduino_serial.write(b"ROTATE\n")
                arduino_serial.flush()
                while True:
                    line = arduino_serial.readline().decode().strip()
                    print(f"[Arduino] {line}")
                    if line == "DONE":
                        #return True
                        break
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
            pipeline.start(cfg)
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

    # Depth cutoff
    depth_filtered = depth_img.copy()
    depth_max_mm = CONFIG.get("depth_max_m", 0.75) * 1000
    depth_filtered[depth_filtered > depth_max_mm] = 0

    depth_o3d = o3d.geometry.Image(depth_filtered)
    color_o3d = o3d.geometry.Image(color_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,
        depth_trunc=CONFIG.get("depth_max_m", 0.75),
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.voxel_down_sample(voxel_size=0.006)  # slightly larger merges nearby duplicates

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


def make_stl_from_pcd(pcd, label="clean"):
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
    o3d       = get_open3d()
    if o3d is None or state["pointcloud"] is None:
        return jsonify({'status': 'error'})
    pcd        = state["pointcloud"]
    clean_path = make_stl_from_pcd(pcd.select_by_index(clean_idx), "clean")
    avoid_path = make_stl_from_pcd(pcd.select_by_index(avoid_idx), "avoid")
    results    = {}
    if clean_path:
        results['clean'] = send_stl_to_moveit(clean_path, "clean_zone", x=0.5, y=0.0, z=0.2)
    if avoid_path:
        results['avoid'] = send_stl_to_moveit(avoid_path, "avoid_zone", x=0.5, y=0.0, z=0.2)
    return jsonify({'status': 'ok', 'results': results,
                    'clean_path': clean_path, 'avoid_path': avoid_path})

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