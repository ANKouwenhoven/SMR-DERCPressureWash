#!/usr/bin/env python3
"""
cleaning_visualizer.py

Real-time cleaning coverage visualization for RViz.

Publishes on /cleaning_coverage (MarkerArray):
  id=0          : live semi-transparent cone following the TCP
  id=1          : coverage HUD text (% cleaned)
  id=100+       : committed surface patches (one per completed stripe)

Usage — instantiate in PathPlanner, then:

    viz = CleaningVisualizer(node, stl_path, world_offset, bbox_size)
    viz.start_live_cone()           # begins TF polling thread
    viz.commit_stripe(p_start, p_end, q_start, q_end)   # after each stripe
    viz.stop()                      # on completion
"""

import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rclpy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

# ── Open3D lazy import ────────────────────────────────────────────────────────
def _get_o3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  TUNABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CONE_HALF_ANGLE_DEG = 10.0      # spray half-angle — adjust to real nozzle
CONE_SEGMENTS       = 20        # triangle fan resolution (sides of cone)
PATCH_DISC_SEGS     = 10        # triangle fan resolution for surface discs
PATCH_SAMPLE_STEP_M = 0.003     # sample stripe every 3 mm for raycasting (tighter)
PATCH_DISC_SCALE    = 0.35      # disc radius = hit_dist * tan(angle) * this scale factor
PATCH_DISC_MIN_M    = 0.008     # minimum disc radius 8 mm (was 15 mm)
LIVE_CONE_HZ        = 10        # TF poll rate for live cone
TOPIC               = '/cleaning_coverage'
BASE_FRAME          = 'base_link'
EEF_LINK            = 'tool0'

# Colors
COLOR_CONE    = ColorRGBA(r=0.2,  g=0.7,  b=1.0,  a=0.25)   # icy blue, transparent
COLOR_CLEAN   = ColorRGBA(r=1.0,  g=0.1,  b=0.1,  a=1.0)   # red patch — fully opaque
COLOR_AVOID   = ColorRGBA(r=0.1,  g=0.3,  b=1.0,  a=1.0)   # blue patch — fully opaque
COLOR_HUD     = ColorRGBA(r=1.0,  g=1.0,  b=1.0,  a=1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cone_triangles(apex: np.ndarray,
                    axis: np.ndarray,
                    length: float,
                    half_angle_deg: float,
                    segments: int) -> list[Point]:
    """
    Build a TRIANGLE_LIST for a cone.
    apex      : tip of cone (nozzle position in world frame)
    axis      : unit vector pointing FROM apex TOWARD surface (-tool_z)
    length    : standoff distance (cone height)
    Returns list of geometry_msgs/Point, 3 per triangle.
    """
    base_radius = length * np.tan(np.deg2rad(half_angle_deg))
    base_centre = apex + axis * length

    # Build a local frame at the base
    ref    = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(axis, ref);      u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(axis, u_axis);   v_axis /= np.linalg.norm(v_axis)

    angles       = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    base_ring    = [base_centre + base_radius * (np.cos(a) * u_axis + np.sin(a) * v_axis)
                    for a in angles]

    def pt(v):
        p = Point(); p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2]); return p

    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        # Side triangle: apex → ring[i] → ring[j]
        tris += [pt(apex), pt(base_ring[i]), pt(base_ring[j])]
        # Base disc: centre → ring[j] → ring[i]  (wound inward for flat cap)
        tris += [pt(base_centre), pt(base_ring[j]), pt(base_ring[i])]

    return tris


def _disc_triangles(centre: np.ndarray,
                    normal: np.ndarray,
                    radius: float,
                    segments: int) -> list[Point]:
    """Filled disc as TRIANGLE_LIST, lying in the plane defined by normal."""
    ref    = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(normal, ref);    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis); v_axis /= np.linalg.norm(v_axis)

    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring   = [centre + radius * (np.cos(a) * u_axis + np.sin(a) * v_axis) for a in angles]

    def pt(v):
        p = Point(); p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2]); return p

    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris += [pt(centre), pt(ring[i]), pt(ring[j])]
    return tris


def _get_tool_z(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return the tool Z axis (nozzle direction) from a quaternion."""
    rot = Rot.from_quat(quaternion_xyzw)
    return rot.apply(np.array([0.0, 0.0, 1.0]))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CleaningVisualizer:
    """
    Manages the three-layer cleaning visualization:
      1. Live cone  — follows TCP in real-time
      2. Surface patches — committed after each completed stripe
      3. HUD text   — live coverage percentage
    """

    def __init__(self,
                 node,
                 stl_path:    str,
                 world_offset: np.ndarray,
                 bbox_size:    np.ndarray,
                 standoff:     float = 0.19,
                 cone_half_angle_deg: float = CONE_HALF_ANGLE_DEG,
                 zone_type:    str = 'clean'):
        """
        node          : rclpy Node (PathPlanner)
        stl_path      : path to clean_surfaces.stl (metres scale after STL_SCALE)
        world_offset  : np.array [x, y, z] — object origin in world frame
        bbox_size     : np.array [w, d, h] — object bounding box in metres
        standoff      : nominal nozzle-to-surface distance
        zone_type     : 'clean' or 'avoid' — sets patch color
        """
        self._node             = node
        self._world_offset     = world_offset
        self._bbox_size        = bbox_size
        self._standoff         = standoff
        self._half_angle       = cone_half_angle_deg
        self._zone_type        = zone_type
        self._patch_color      = COLOR_CLEAN if zone_type == 'clean' else COLOR_AVOID

        self._pub = node.create_publisher(MarkerArray, TOPIC, 10)

        # Coverage tracking
        self._covered_triangle_ids: set[int] = set()
        self._total_mesh_area:  float = 0.0
        self._covered_area:     float = 0.0
        self._next_patch_id:    int   = 100          # ids 0,1 reserved for cone/HUD
        self._stripe_count:     int   = 0

        # TF listener (lazy — only created when start_live_cone() called)
        self._tf_buffer   = None
        self._tf_listener = None
        self._cone_thread: threading.Thread | None = None
        self._running     = False

        # Load mesh + build raycast scene
        self._raycast_scene   = None
        self._mesh_tri_areas  = None
        self._mesh_verts      = None   # (N,3) float64 in centred local coords
        self._mesh_tris       = None   # (M,3) int
        self._stl_path        = stl_path
        self._load_mesh(stl_path)

        node.get_logger().info(
            f"[CleaningViz] Ready — {len(self._mesh_tri_areas) if self._mesh_tri_areas is not None else 0} "
            f"triangles, total area {self._total_mesh_area*1e4:.1f} cm²")

    # ── Mesh loading ──────────────────────────────────────────────────────────

    def _load_mesh(self, stl_path: str):
        """Load STL into Open3D RaycastingScene. Applies same centring as face_analyser.

        Auto-detects unit scale: if the raw bbox diagonal > 5 (i.e. clearly in mm
        rather than metres) the mesh is scaled by 0.001 automatically.
        """
        o3d = _get_o3d()
        if o3d is None:
            self._node.get_logger().warn("[CleaningViz] open3d not available — coverage tracking disabled")
            return

        import os
        path = os.path.expanduser(stl_path)
        if not os.path.exists(path):
            self._node.get_logger().warn(f"[CleaningViz] STL not found: {path}")
            return

        try:
            mesh = o3d.io.read_triangle_mesh(path)

            # Auto-detect units: STL files from make_stl_from_pcd are saved in mm
            # (scale * 1000 at the end of that pipeline). face_analyser loads with
            # scale=0.001 itself. We detect which case we have by checking bbox size.
            verts_raw = np.asarray(mesh.vertices)
            raw_diag  = float(np.linalg.norm(verts_raw.max(axis=0) - verts_raw.min(axis=0)))
            if raw_diag > 5.0:
                # Clearly in mm — scale to metres
                mesh.scale(0.001, center=(0, 0, 0))
                self._node.get_logger().info(
                    f"[CleaningViz] STL auto-detected as mm (diag={raw_diag:.1f}) — scaled to metres")
            else:
                self._node.get_logger().info(
                    f"[CleaningViz] STL auto-detected as metres (diag={raw_diag:.3f})")

            # Re-centre around bbox (same transform face_analyser applies)
            verts      = np.asarray(mesh.vertices)
            bbox_min   = verts.min(axis=0)
            bbox_max   = verts.max(axis=0)
            bbox_ctr   = (bbox_min + bbox_max) / 2.0
            mesh.translate(-bbox_ctr)

            # Apply 90° rotation around Z to align scanner frame (Y=toward camera)
            # with robot frame (Y=forward). Must match send_stl_to_moveit and
            # add_object_to_scene so the viz mesh lines up with the planned path.
            angle = np.pi / 2.0
            R_z90 = np.array([
                [ np.cos(angle), -np.sin(angle), 0.0],
                [ np.sin(angle),  np.cos(angle), 0.0],
                [ 0.0,            0.0,           1.0],
            ])
            verts_centred = np.asarray(mesh.vertices)
            rotated = (R_z90 @ verts_centred.T).T
            mesh.vertices = o3d.utility.Vector3dVector(rotated)

            # Pre-compute per-triangle areas for coverage metric
            mesh.compute_triangle_normals()
            verts  = np.asarray(mesh.vertices)
            tris   = np.asarray(mesh.triangles)

            # Store for mesh marker publishing
            self._mesh_verts = verts.copy()
            self._mesh_tris  = tris.copy()
            v0, v1, v2      = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
            cross           = np.cross(v1 - v0, v2 - v0)
            self._mesh_tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
            self._total_mesh_area = float(self._mesh_tri_areas.sum())

            # Build raycast scene
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            self._raycast_scene = o3d.t.geometry.RaycastingScene()
            self._raycast_scene.add_triangles(t_mesh)

            self._node.get_logger().info(
                f"[CleaningViz] Mesh loaded: {len(tris)} tris, "
                f"area={self._total_mesh_area*1e4:.1f} cm²")
        except Exception as e:
            self._node.get_logger().error(f"[CleaningViz] Mesh load failed: {e}")

    # ── TF / live cone ────────────────────────────────────────────────────────

    def start_live_cone(self):
        """Spawn background thread that polls TF and publishes the live cone."""
        try:
            import tf2_ros
            self._tf_buffer   = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        except ImportError:
            self._node.get_logger().warn("[CleaningViz] tf2_ros not available — live cone disabled")
            return

        self._running     = True
        self._cone_thread = threading.Thread(target=self._cone_loop, daemon=True)
        self._cone_thread.start()
        self._node.get_logger().info("[CleaningViz] Live cone started")

    def stop(self):
        """Stop the cone thread and publish a final clear for the live cone."""
        self._running = False
        if self._cone_thread:
            self._cone_thread.join(timeout=2.0)
        # Delete the live cone marker from RViz
        ma = MarkerArray()
        m         = Marker()
        m.action  = Marker.DELETE
        m.ns      = 'cone'
        m.id      = 0
        ma.markers.append(m)
        self._pub.publish(ma)
        self._node.get_logger().info(
            f"[CleaningViz] Done — {self._stripe_count} stripes, "
            f"coverage {self._coverage_pct():.1f}%")

    def _cone_loop(self):
        rate = 1.0 / LIVE_CONE_HZ
        while self._running and rclpy.ok():
            try:
                tf = self._tf_buffer.lookup_transform(
                    BASE_FRAME, EEF_LINK,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1))
                t   = tf.transform.translation
                q   = tf.transform.rotation
                pos = np.array([t.x, t.y, t.z])
                quat= np.array([q.x, q.y, q.z, q.w])
                self._publish_live_cone(pos, quat)
            except Exception:
                pass   # TF not ready yet — silently skip
            time.sleep(rate)

    def _publish_live_cone(self, tcp_pos: np.ndarray, tcp_quat: np.ndarray):
        """Build and publish the live cone marker at the current TCP pose."""
        tool_z = _get_tool_z(tcp_quat)   # direction nozzle points
        apex   = tcp_pos
        axis   = tool_z                   # cone opens in +tool_z direction

        pts = _cone_triangles(
            apex, axis,
            length         = self._standoff,
            half_angle_deg = self._half_angle,
            segments       = CONE_SEGMENTS)

        m              = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self._node.get_clock().now().to_msg()
        m.ns           = 'cone'
        m.id           = 0
        m.type         = Marker.TRIANGLE_LIST
        m.action       = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color        = COLOR_CONE
        m.points       = pts

        ma = MarkerArray()
        ma.markers.append(m)
        self._pub.publish(ma)

    # ── Stripe commitment (surface patches + coverage) ────────────────────────

    def commit_stripe(self,
                      p_start:  np.ndarray,
                      p_end:    np.ndarray,
                      q_start:  np.ndarray,
                      q_end:    np.ndarray):
        """
        Called after a stripe successfully executes.
        Raycasts along the stripe, builds surface patches, updates coverage.

        p_start / p_end  : TCP world positions (metres)
        q_start / q_end  : TCP quaternions xyzw at start / end
        """
        if self._raycast_scene is None:
            return

        o3d = _get_o3d()
        if o3d is None:
            return

        import open3d.core as o3c

        stripe_vec  = p_end - p_start
        stripe_len  = float(np.linalg.norm(stripe_vec))
        if stripe_len < 1e-6:
            return

        n_samples   = max(2, int(np.ceil(stripe_len / PATCH_SAMPLE_STEP_M)))
        ts          = np.linspace(0.0, 1.0, n_samples)

        all_disc_pts: list[Point] = []
        new_triangle_ids: set[int] = set()

        for t in ts:
            # Interpolate position and orientation along the stripe
            pos  = p_start + t * stripe_vec
            quat = _slerp(q_start, q_end, t)

            # Spray direction: +tool_z (nozzle points toward surface)
            tool_z     = _get_tool_z(quat)
            ray_origin = pos
            ray_dir    = tool_z   # already pointing toward surface

            # Raycast — mesh lives in centred local coords (origin = bbox centre),
            # so subtract world_offset from ray origin before casting.
            ray_origin_local = ray_origin - self._world_offset
            rays = o3c.Tensor(
                np.array([[*ray_origin_local, *ray_dir]], dtype=np.float32),
                dtype=o3c.Dtype.Float32)
            result = self._raycast_scene.cast_rays(rays)

            hit_dist = float(result['t_hit'][0].numpy())
            if not np.isfinite(hit_dist) or hit_dist > self._standoff * 2.5:
                continue   # missed the object

            # hit_pt is in local mesh coords; convert back to world for RViz
            hit_pt_local = ray_origin_local + ray_dir * hit_dist
            tri_id       = int(result['primitive_ids'][0].numpy())
            hit_normal   = result['primitive_normals'][0].numpy().astype(float)

            # Coverage accounting
            if tri_id not in self._covered_triangle_ids:
                new_triangle_ids.add(tri_id)

            # Disc radius: spread at this distance, scaled down for tighter patches.
            # PATCH_DISC_SCALE < 1.0 makes patches smaller than the full spray cone footprint.
            disc_r = max(PATCH_DISC_MIN_M,
                         hit_dist * np.tan(np.deg2rad(self._half_angle)) * PATCH_DISC_SCALE)

            # Offset 5mm above surface (in local coords), then move to world frame
            disc_centre_world = (hit_pt_local + hit_normal * 0.005) + self._world_offset

            disc_pts = _disc_triangles(
                disc_centre_world, hit_normal, disc_r, PATCH_DISC_SEGS)
            all_disc_pts.extend(disc_pts)

        if not all_disc_pts:
            self._node.get_logger().warn(
                f"[CleaningViz] Stripe got no raycast hits — check viz_stl path and world_offset")
            return

        self._node.get_logger().info(
            f"[CleaningViz] Stripe {self._stripe_count+1}: "            f"{len(all_disc_pts)//3} discs, first hit near "            f"{all_disc_pts[0].x:.3f},{all_disc_pts[0].y:.3f},{all_disc_pts[0].z:.3f}")

        # Update coverage
        for tid in new_triangle_ids:
            self._covered_triangle_ids.add(tid)
            self._covered_area += float(self._mesh_tri_areas[tid])

        self._stripe_count += 1

        # Emit coverage update on stdout — app.py picks this up and relays to UI
        pct = self._coverage_pct()
        print(f"COVERAGE_UPDATE:{pct:.1f}:{self._stripe_count}:"
              f"{self._covered_area*1e4:.1f}:{self._total_mesh_area*1e4:.1f}", flush=True)

        # Build patch marker
        patch           = Marker()
        patch.header.frame_id = BASE_FRAME
        patch.header.stamp    = self._node.get_clock().now().to_msg()
        patch.ns        = 'patches'
        patch.id        = self._next_patch_id
        patch.type      = Marker.TRIANGLE_LIST
        patch.action    = Marker.ADD
        patch.scale.x   = patch.scale.y = patch.scale.z = 1.0
        patch.color     = self._patch_color
        patch.points    = all_disc_pts
        # Keep patches forever
        patch.lifetime.sec = 0

        self._next_patch_id += 1

        # Build HUD marker
        hud = self._build_hud()

        ma = MarkerArray()
        ma.markers += [patch, hud]
        self._pub.publish(ma)

        self._node.get_logger().info(
            f"[CleaningViz] Stripe {self._stripe_count}: "
            f"+{len(new_triangle_ids)} tris → coverage {self._coverage_pct():.1f}%")

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _coverage_pct(self) -> float:
        if self._total_mesh_area < 1e-9:
            return 0.0
        return min(100.0, self._covered_area / self._total_mesh_area * 100.0)

    def _build_hud(self) -> Marker:
        pct   = self._coverage_pct()
        # Float HUD above the object
        hud_pos = self._world_offset + np.array([0.0, 0.0, self._bbox_size[2] * 0.5 + 0.12])

        hud              = Marker()
        hud.header.frame_id = BASE_FRAME
        hud.header.stamp    = self._node.get_clock().now().to_msg()
        hud.ns           = 'hud'
        hud.id           = 1
        hud.type         = Marker.TEXT_VIEW_FACING
        hud.action       = Marker.ADD
        hud.pose.position.x = float(hud_pos[0])
        hud.pose.position.y = float(hud_pos[1])
        hud.pose.position.z = float(hud_pos[2])
        hud.pose.orientation.w = 1.0
        hud.scale.z      = 0.04   # text height in metres
        hud.color        = COLOR_HUD
        hud.text         = (
            f"COVERAGE: {pct:.1f}%\n"
            f"Stripes: {self._stripe_count}\n"
            f"Area: {self._covered_area*1e4:.0f} / {self._total_mesh_area*1e4:.0f} cm²"
        )
        return hud

    # ── Convenience: publish full state (e.g. after restoring from checkpoint) -

    def republish_all(self):
        """Re-emit all committed patch markers + HUD. Useful after RViz restart."""
        # The individual patch markers are persistent (lifetime=0) so RViz
        # keeps them. This just refreshes the HUD.
        ma = MarkerArray()
        ma.markers.append(self._build_hud())
        self._pub.publish(ma)

    # ── Clear ─────────────────────────────────────────────────────────────────

    def publish_bbox(self):
        """Publish a wireframe bounding box around the object in RViz."""
        hw = self._bbox_size[0] / 2.0
        hd = self._bbox_size[1] / 2.0
        hh = self._bbox_size[2] / 2.0
        o  = self._world_offset

        corners = np.array([
            [-hw, -hd, -hh], [ hw, -hd, -hh],
            [ hw,  hd, -hh], [-hw,  hd, -hh],
            [-hw, -hd,  hh], [ hw, -hd,  hh],
            [ hw,  hd,  hh], [-hw,  hd,  hh],
        ]) + o

        edges = [
            (0,1),(1,2),(2,3),(3,0),   # bottom face
            (4,5),(5,6),(6,7),(7,4),   # top face
            (0,4),(1,5),(2,6),(3,7),   # verticals
        ]

        def pt(v):
            p = Point(); p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2]); return p

        points = []
        for i, j in edges:
            points += [pt(corners[i]), pt(corners[j])]

        m              = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self._node.get_clock().now().to_msg()
        m.ns           = 'bbox'
        m.id           = 2
        m.type         = Marker.LINE_LIST
        m.action       = Marker.ADD
        m.scale.x      = 0.003
        m.color        = ColorRGBA(r=0.0, g=1.0, b=0.4, a=0.8)
        m.points       = points
        m.lifetime.sec = 0

        ma = MarkerArray()
        ma.markers.append(m)
        self._pub.publish(ma)

    def publish_mesh_marker(self):
        """
        Publish the scanned object as a solid semi-transparent TRIANGLE_LIST marker
        on /cleaning_coverage (id=3, ns='object').

        This appears in the same MarkerArray display as the patches — no extra
        RViz displays or buttons needed. The mesh is rendered in a neutral grey
        so the red paint patches are clearly visible on top of it.
        """
        if self._mesh_verts is None or self._mesh_tris is None:
            self._node.get_logger().warn("[CleaningViz] No mesh loaded — skipping object marker")
            return

        def pt(v):
            p = Point()
            p.x = float(v[0] + self._world_offset[0])
            p.y = float(v[1] + self._world_offset[1])
            p.z = float(v[2] + self._world_offset[2])
            return p

        # Build TRIANGLE_LIST: 3 points per triangle, vertices in world frame
        points = []
        for tri in self._mesh_tris:
            points.append(pt(self._mesh_verts[tri[0]]))
            points.append(pt(self._mesh_verts[tri[1]]))
            points.append(pt(self._mesh_verts[tri[2]]))

        m              = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp    = self._node.get_clock().now().to_msg()
        m.ns           = 'object'
        m.id           = 3
        m.type         = Marker.TRIANGLE_LIST
        m.action       = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 1.0
        # Neutral grey, semi-transparent so patches show on top
        m.color        = ColorRGBA(r=0.7, g=0.7, b=0.75, a=0.55)
        m.points       = points
        m.lifetime.sec = 0   # persist until deleted

        ma = MarkerArray()
        ma.markers.append(m)
        self._pub.publish(ma)
        self._node.get_logger().info(
            f"[CleaningViz] Object mesh marker published "
            f"({len(self._mesh_tris)} triangles)")

    def clear_all(self):
        """Delete all visualization markers from RViz and reset coverage."""
        ma = MarkerArray()
        for mid in range(self._next_patch_id):
            m = Marker(); m.action = Marker.DELETE
            m.ns = 'patches'; m.id = mid; ma.markers.append(m)
        for mid, ns in [(0, 'cone'), (1, 'hud'), (2, 'bbox'), (3, 'object')]:
            m = Marker(); m.action = Marker.DELETE
            m.ns = ns; m.id = mid; ma.markers.append(m)
        self._pub.publish(ma)

        self._covered_triangle_ids.clear()
        self._covered_area   = 0.0
        self._next_patch_id  = 100
        self._stripe_count   = 0


# ─────────────────────────────────────────────────────────────────────────────
#  SLERP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (xyzw)."""
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1  = -q1
        dot = -dot
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta0 = np.arccos(dot)
    theta  = theta0 * t
    sin0   = np.sin(theta0)
    return (np.cos(theta) * q0 + np.sin(theta) * (q1 - dot * q0) / sin0)