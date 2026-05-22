# -*- coding: utf-8 -*-
"""
3D Scanner Application
Scans an object using Intel RealSense D435, displays live camera feed and pointcloud,
exports STL for use in ROS2/MoveIt.

Author: Modified for UR10 + NEMA17 stepper setup
"""

import os
os.environ["XDG_SESSION_TYPE"] = "x11"

# ─────────────────────────────────────────────
#  CONFIGURATION — change these to match your setup
# ─────────────────────────────────────────────
ARDUINO_PORT        = "/dev/ttyACM0"   # Change to your Arduino port
ARDUINO_BAUDRATE    = 9600
ARDUINO_ENABLED     = False            # Set to True when Arduino is connected

STEP_PIN            = 3               # Arduino STEP pin
DIR_PIN             = 4               # Arduino DIR pin
EN_PIN              = 5               # Arduino EN pin

DEGREES_PER_STEP    = 7.5             # Degrees to rotate per photo
CAMERA_WIDTH        = 848
CAMERA_HEIGHT       = 480
CAMERA_FPS          = 30
CAMERA_DISTANCE_M   = 0.258           # Distance from camera to turntable centre (metres)
CAMERA_HEIGHT_M     = 0.165           # Height of camera above turntable (metres)
CAMERA_TILT_DEG     = 112.5           # Camera tilt angle in degrees
STL_OUTPUT_PATH     = os.path.expanduser("~/scan_output.stl")
# ─────────────────────────────────────────────

import serial
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter.ttk import Progressbar
import tkinter.filedialog
from PIL import ImageTk, Image
import ast


class ArduinoStepper:
    """Controls a NEMA17 stepper motor via Arduino."""

    def __init__(self, port, baudrate, step_pin, dir_pin, en_pin):
        self.port = port
        self.baudrate = baudrate
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.en_pin = en_pin
        self.connected = False
        self.s = None

    def connect(self):
        try:
            self.s = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.connected = True
            # Send pin config to Arduino
            self.s.write(f"INIT {self.step_pin} {self.dir_pin} {self.en_pin}\n".encode())
            print(f"Arduino connected on {self.port}")
        except Exception as e:
            print(f"Arduino connection failed: {e}")
            self.connected = False

    def rotate(self, degrees):
        """Rotate the turntable by the given degrees."""
        if self.connected and self.s:
            self.s.write(f"ROTATE {degrees}\n".encode())
            # Wait for Arduino to confirm rotation complete
            while True:
                line = self.s.readline().decode().strip()
                if line == "DONE":
                    break

    def disconnect(self):
        if self.s:
            self.s.close()
            self.connected = False


class Scanner:
    """Handles RealSense camera pipeline and pointcloud processing."""

    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.any, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.any, fps)
        self.align = None
        self.main_pcd = o3d.geometry.PointCloud()
        self.dtr = np.pi / 180
        self.distance = CAMERA_DISTANCE_M
        self.bbox = o3d.geometry.AxisAlignedBoundingBox((-0.13, -0.13, 0), (0.13, 0.13, 0.2))

    def start(self):
        self.pipe.start(self.config)
        self.align = rs.align(rs.stream.color)
        print("Camera pipeline started")

    def stop(self):
        self.pipe.stop()
        print("Camera pipeline stopped")

    def get_preview_frame(self):
        """Get a single colour frame for live preview."""
        frameset = self.pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def capture(self, auto_exposure_frames=10):
        """Capture a depth+colour frame pair."""
        for _ in range(auto_exposure_frames):
            self.pipe.wait_for_frames()
        frameset = self.pipe.wait_for_frames()
        frameset = self.align.process(frameset)
        profile = frameset.get_profile()
        intr = profile.as_video_stream_profile().get_intrinsics()
        self.w, self.h = intr.width, intr.height
        self.fx, self.fy = intr.fx, intr.fy
        self.px, self.py = intr.ppx, intr.ppy
        self.color_frame = frameset.get_color_frame()
        self.depth_frame = frameset.get_depth_frame()
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.w, self.h, self.fx, self.fy, self.px, self.py)

    def process(self, angle_deg):
        """Add current frame to the pointcloud at the given rotation angle."""
        depth_o3d = o3d.geometry.Image(self.depth_image)
        color_o3d = o3d.geometry.Image(self.color_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        # Camera position in world space
        a = angle_deg * self.dtr
        x = np.sin(a) * self.distance - np.cos(a) * 0.035
        y = -np.cos(a) * self.distance - np.sin(a) * 0.035
        z = CAMERA_HEIGHT_M

        # Rotation matrix
        o = angle_deg * self.dtr
        tilt = (-CAMERA_TILT_DEG) * self.dtr
        t = 0.0
        R = [
            [np.cos(o)*np.cos(t) - np.cos(tilt)*np.sin(o)*np.sin(t),
             -np.cos(o)*np.sin(t) - np.cos(tilt)*np.cos(t)*np.sin(o),
             np.sin(o)*np.sin(tilt)],
            [np.cos(t)*np.sin(o) + np.cos(o)*np.cos(tilt)*np.sin(t),
             np.cos(o)*np.cos(tilt)*np.cos(t) - np.sin(o)*np.sin(t),
             -np.cos(o)*np.sin(tilt)],
            [np.sin(tilt)*np.sin(t), np.cos(t)*np.sin(tilt), np.cos(tilt)]
        ]
        pcd.rotate(R, (0, 0, 0))
        pcd.translate((x, y, z))
        pcd = pcd.crop(self.bbox)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2)
        self.main_pcd += pcd

    def get_pointcloud(self):
        return self.main_pcd

    def make_stl(self, k_points=10, std_ratio=0.5, depth=7, iterations=8):
        """Generate an STL mesh from the pointcloud."""
        pcd = self.main_pcd.uniform_down_sample(every_k_points=k_points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=std_ratio)
        bbox_bottom = o3d.geometry.AxisAlignedBoundingBox((-0.13, -0.13, 0), (0.13, 0.13, 0.01))
        bottom = pcd.crop(bbox_bottom)
        try:
            hull, _ = bottom.compute_convex_hull()
            bottom = hull.sample_points_uniformly(number_of_points=10000)
            bottom.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            bottom.orient_normals_towards_camera_location(
                camera_location=np.array([0., 0., -10.]))
            bottom.paint_uniform_color([0, 0, 0])
            _, pt_map = bottom.hidden_point_removal([0, 0, -1], 1)
            bottom = bottom.select_by_index(pt_map)
            pcd = pcd + bottom
        except Exception as e:
            print(f"No bottom could be made: {e}")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)
        mesh.scale(1000, center=(0, 0, 0))
        mesh.compute_vertex_normals()
        return mesh


class App(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("3D Scanner — UR10 Cleaning System")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)

        self.title_font = tkfont.Font(family="Courier", size=13, weight="bold")
        self.label_font = tkfont.Font(family="Courier", size=9)

        self.scanner = Scanner(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        self.arduino = ArduinoStepper(
            ARDUINO_PORT, ARDUINO_BAUDRATE, STEP_PIN, DIR_PIN, EN_PIN)

        self.is_scanning = False
        self.preview_running = False
        self.current_angle = 0.0
        self.stl_mesh = None

        self._build_ui()
        self._start_preview()

    def _build_ui(self):
        # ── Title bar ──
        title_bar = tk.Frame(self, bg="#0f3460", pady=8)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="◈  3D SCANNER CONTROL SYSTEM",
                 font=self.title_font, bg="#0f3460", fg="#e94560").pack()

        # ── Main content area ──
        content = tk.Frame(self, bg="#1a1a2e")
        content.pack(padx=10, pady=10)

        # Left panel — controls
        left = tk.Frame(content, bg="#16213e", bd=1, relief="solid", padx=10, pady=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        tk.Label(left, text="CONTROLS", font=self.title_font,
                 bg="#16213e", fg="#e94560").pack(pady=(0, 10))

        btn_style = dict(width=22, font=self.label_font, relief="flat",
                         cursor="hand2", pady=6)

        self.btn_scan = tk.Button(left, text="▶  START SCAN",
                                  bg="#e94560", fg="white",
                                  command=self._start_scan, **btn_style)
        self.btn_scan.pack(pady=4)

        self.btn_show_pcd = tk.Button(left, text="◈  SHOW POINTCLOUD",
                                      bg="#0f3460", fg="#aaaaaa",
                                      state="disabled",
                                      command=self._show_pointcloud, **btn_style)
        self.btn_show_pcd.pack(pady=4)

        self.btn_make_stl = tk.Button(left, text="⬡  MAKE STL",
                                      bg="#0f3460", fg="#aaaaaa",
                                      state="disabled",
                                      command=self._make_stl, **btn_style)
        self.btn_make_stl.pack(pady=4)

        self.btn_save_stl = tk.Button(left, text="💾  SAVE STL",
                                      bg="#0f3460", fg="#aaaaaa",
                                      state="disabled",
                                      command=self._save_stl, **btn_style)
        self.btn_save_stl.pack(pady=4)

        tk.Frame(left, bg="#e94560", height=1).pack(fill="x", pady=10)

        self.btn_quit = tk.Button(left, text="✕  QUIT",
                                  bg="#333355", fg="#aaaaaa",
                                  command=self._quit, **btn_style)
        self.btn_quit.pack(pady=4)

        # Status
        tk.Label(left, text="STATUS", font=self.label_font,
                 bg="#16213e", fg="#555577").pack(pady=(15, 2))
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(left, textvariable=self.status_var,
                                     font=self.label_font, bg="#16213e",
                                     fg="#00ff88", wraplength=180, justify="left")
        self.status_label.pack()

        # Arduino status
        ard_text = f"Arduino: {'ENABLED' if ARDUINO_ENABLED else 'DISABLED'}"
        ard_color = "#00ff88" if ARDUINO_ENABLED else "#ff6644"
        tk.Label(left, text=ard_text, font=self.label_font,
                 bg="#16213e", fg=ard_color).pack(pady=(10, 0))
        tk.Label(left, text=f"Port: {ARDUINO_PORT}", font=self.label_font,
                 bg="#16213e", fg="#555577").pack()
        tk.Label(left, text=f"Step: {DEGREES_PER_STEP}° per photo",
                 font=self.label_font, bg="#16213e", fg="#555577").pack()

        # Progress
        tk.Label(left, text="PROGRESS", font=self.label_font,
                 bg="#16213e", fg="#555577").pack(pady=(15, 2))
        self.angle_var = tk.StringVar(value="0.0° / 360°")
        tk.Label(left, textvariable=self.angle_var, font=self.label_font,
                 bg="#16213e", fg="#e94560").pack()
        self.progress = Progressbar(left, orient="horizontal", length=180,
                                    mode="determinate", maximum=360)
        self.progress.pack(pady=4)

        # Centre panel — camera feed
        centre = tk.Frame(content, bg="#16213e", bd=1, relief="solid")
        centre.grid(row=0, column=1, padx=(0, 10))

        tk.Label(centre, text="CAMERA FEED", font=self.label_font,
                 bg="#16213e", fg="#555577").pack(pady=4)

        crop_w = 340
        crop_h = 460
        self.canvas = tk.Canvas(centre, width=crop_w, height=crop_h, bg="#000000",
                                highlightthickness=0)
        self.canvas.pack(padx=8, pady=(0, 8))
        self._blank = ImageTk.PhotoImage(
            Image.fromarray(np.zeros((crop_h, crop_w, 3), dtype=np.uint8), 'RGB'))
        self.canvas.create_image(0, 0, anchor='nw', image=self._blank)
        self.canvas.image = self._blank
        self._crop = (330, 20, 670, 480)

        # Right panel — rotate prompt (hidden until scan starts)
        self.right_panel = tk.Frame(content, bg="#16213e", bd=1, relief="solid",
                                    padx=10, pady=10, width=220)
        self.right_panel.grid(row=0, column=2)
        self.right_panel.grid_propagate(False)

        tk.Label(self.right_panel, text="SCAN CONTROL", font=self.title_font,
                 bg="#16213e", fg="#e94560").pack(pady=(0, 10))

        self.rotate_info = tk.Label(self.right_panel,
                                    text="Press START SCAN\nto begin.",
                                    font=self.label_font, bg="#16213e",
                                    fg="#aaaaaa", justify="center", wraplength=200)
        self.rotate_info.pack(pady=20)

        self.btn_continue = tk.Button(self.right_panel, text="✔  TABLE ROTATED\n    TAKE PHOTO",
                                      bg="#00aa55", fg="white", font=self.label_font,
                                      width=20, pady=10, relief="flat",
                                      cursor="hand2", state="disabled",
                                      command=self._on_rotated)
        self.btn_continue.pack(pady=6)

        self.btn_stop_early = tk.Button(self.right_panel, text="■  STOP SCAN",
                                        bg="#cc3333", fg="white", font=self.label_font,
                                        width=20, pady=6, relief="flat",
                                        cursor="hand2", state="disabled",
                                        command=self._stop_early)
        self.btn_stop_early.pack(pady=6)

        # Bottom bar
        bottom = tk.Frame(self, bg="#0f3460", pady=4)
        bottom.pack(fill="x", side="bottom")
        tk.Label(bottom, text=f"Camera: {CAMERA_WIDTH}×{CAMERA_HEIGHT} @ {CAMERA_FPS}fps  |  "
                 f"Step: {DEGREES_PER_STEP}°  |  "
                 f"Photos per scan: {int(360 / DEGREES_PER_STEP)}  |  "
                 f"STL output: {STL_OUTPUT_PATH}",
                 font=self.label_font, bg="#0f3460", fg="#555577").pack()

    # ── Preview ──────────────────────────────────────────────

    def _start_preview(self):
        try:
            self.scanner.start()
            self.preview_running = True
            self._update_preview()
            self._set_status("Camera live. Ready to scan.")
        except Exception as e:
            self._set_status(f"Camera error: {e}")

    def _update_preview(self):
        if self.preview_running and not self.is_scanning:
            try:
                frame = self.scanner.get_preview_frame()
                if frame is not None:
                    self._show_frame(frame)
            except Exception as e:
                print(f"Preview error: {e}")
            self.after(33, self._update_preview)

    def _stop_preview(self):
        self.preview_running = False
        try:
            self.scanner.stop()
        except:
            pass

    def _show_frame(self, array):
        l, u, r, b = self._crop
        img = Image.fromarray(array, 'RGB').crop((l, u, r, b))
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.image = photo

    # ── Scan ─────────────────────────────────────────────────

    def _start_scan(self):
        self.is_scanning = True
        self.current_angle = 0.0
        self.scanner.main_pcd = o3d.geometry.PointCloud()  # reset pointcloud
        self._stop_preview()
        self.scanner.start()

        if ARDUINO_ENABLED:
            self.arduino.connect()

        self.btn_scan.config(state="disabled", bg="#333355")
        self.btn_continue.config(state="normal")
        self.btn_stop_early.config(state="normal")
        self._set_status("Scan started.\nRotate table then click\nTABLE ROTATED.")
        self._do_scan_step()

    def _do_scan_step(self):
        if self.current_angle >= 360.0:
            self._finish_scan()
            return

        try:
            self.scanner.capture(auto_exposure_frames=5)
            self._show_frame(self.scanner.color_image)
            self.scanner.process(self.current_angle)
            self.progress["value"] = self.current_angle
            self.angle_var.set(f"{self.current_angle:.1f}° / 360°")
            self._set_status(
                f"Photo taken at {self.current_angle:.1f}°\n\n"
                f"Rotate table {DEGREES_PER_STEP}° then\nclick TABLE ROTATED."
            )
            self.rotate_info.config(
                text=f"Next angle:\n{self.current_angle + DEGREES_PER_STEP:.1f}°")

            if ARDUINO_ENABLED:
                threading.Thread(
                    target=self.arduino.rotate,
                    args=(DEGREES_PER_STEP,), daemon=True).start()

        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_rotated(self):
        """User confirms table has been rotated."""
        self.current_angle += DEGREES_PER_STEP
        self._do_scan_step()

    def _stop_early(self):
        self._finish_scan()

    def _finish_scan(self):
        self.is_scanning = False
        self.scanner.stop()
        self.btn_continue.config(state="disabled")
        self.btn_stop_early.config(state="disabled")
        self.btn_scan.config(state="normal", bg="#e94560")
        self.btn_show_pcd.config(state="normal", bg="#0f3460", fg="white")
        self.btn_make_stl.config(state="normal", bg="#0f3460", fg="white")
        self.progress["value"] = 360
        self.angle_var.set("360° / 360° ✔")
        self.rotate_info.config(text="Scan complete!\nView pointcloud\nor make STL.")
        self._set_status("Scan complete!\nPointcloud ready.")
        if ARDUINO_ENABLED:
            self.arduino.disconnect()
        self.scanner.start()
        self.preview_running = True
        self._update_preview()

    # ── Pointcloud & STL ─────────────────────────────────────

    def _show_pointcloud(self):
        self._set_status("Opening pointcloud\nviewer...")
        threading.Thread(target=self._open_pcd_viewer, daemon=True).start()

    def _open_pcd_viewer(self):
        o3d.visualization.draw_geometries(
            [self.scanner.get_pointcloud()],
            window_name="Pointcloud — 3D Scanner",
            width=800, height=600)

    def _make_stl(self):
        self._set_status("Generating STL mesh...\nThis may take a minute.")
        self.btn_make_stl.config(state="disabled")
        threading.Thread(target=self._generate_stl, daemon=True).start()

    def _generate_stl(self):
        try:
            self.stl_mesh = self.scanner.make_stl()
            o3d.io.write_triangle_mesh(STL_OUTPUT_PATH, self.stl_mesh)
            self.after(0, lambda: self._set_status(
                f"STL saved to:\n{STL_OUTPUT_PATH}\n\nReady for ROS2/MoveIt!"))
            self.after(0, lambda: self.btn_save_stl.config(
                state="normal", bg="#0f3460", fg="white"))
            self.after(0, lambda: self._open_stl_viewer())
        except Exception as e:
            self.after(0, lambda: self._set_status(f"STL error: {e}"))
        finally:
            self.after(0, lambda: self.btn_make_stl.config(state="normal"))

    def _open_stl_viewer(self):
        if self.stl_mesh:
            o3d.visualization.draw_geometries(
                [self.stl_mesh],
                window_name="STL Mesh — Ready for MoveIt",
                width=800, height=600)

    def _save_stl(self):
        path = tk.filedialog.asksaveasfilename(
            initialfile="scan_output.stl",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")])
        if path:
            o3d.io.write_triangle_mesh(path, self.stl_mesh)
            self._set_status(f"STL saved to:\n{path}")

    # ── Helpers ──────────────────────────────────────────────

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _quit(self):
        self.preview_running = False
        try:
            self.scanner.stop()
        except:
            pass
        if ARDUINO_ENABLED:
            self.arduino.disconnect()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
