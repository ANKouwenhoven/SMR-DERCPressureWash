#!/usr/bin/env python3
"""
stl_to_moveit.py

ROS2 node that:
1. Watches a directory for a new/updated STL file
2. Loads it into the MoveIt planning scene as a collision object
3. Can be triggered manually or automatically when the scanner saves a new STL

Usage:
    python3 stl_to_moveit.py                          # watches default path
    python3 stl_to_moveit.py --stl /path/to/file.stl  # load a specific STL

Requirements:
    ros2, moveit_msgs, shape_msgs, geometry_msgs
    pip install watchdog
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import GetPlanningScene
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Header
import os
import sys
import time
import struct
import argparse
import threading

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_STL_PATH    = os.path.expanduser("~/scan_output.stl")
OBJECT_NAME         = "scanned_object"        # Name in MoveIt planning scene
OBJECT_FRAME        = "world"                 # Reference frame
WATCH_FOR_CHANGES   = True                    # Auto-reload when STL file changes
CHECK_INTERVAL_SEC  = 2.0                     # How often to check for file changes

# Position of the scanned object relative to the robot base (metres)
OBJECT_X            = 0.5                     # Forward
OBJECT_Y            = 0.0                     # Left/right
OBJECT_Z            = 0.0                     # Height (0 = on the floor/table)
# ─────────────────────────────────────────────


def read_stl_binary(filepath):
    """Parse a binary STL file and return vertices and triangles."""
    with open(filepath, 'rb') as f:
        f.read(80)  # Skip header
        num_triangles = struct.unpack('<I', f.read(4))[0]
        vertices = []
        triangles = []
        vertex_map = {}

        def get_vertex_index(v):
            key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(v)
            return vertex_map[key]

        for _ in range(num_triangles):
            f.read(12)  # Skip normal vector
            tri_indices = []
            for _ in range(3):
                v = struct.unpack('<fff', f.read(12))
                tri_indices.append(get_vertex_index(v))
            triangles.append(tri_indices)
            f.read(2)  # Skip attribute byte count

    return vertices, triangles


def stl_to_mesh_msg(filepath):
    """Convert an STL file to a ROS2 Mesh message."""
    vertices, triangles = read_stl_binary(filepath)
    mesh = Mesh()

    for v in vertices:
        p = Point()
        p.x, p.y, p.z = float(v[0]) / 1000.0, float(v[1]) / 1000.0, float(v[2]) / 1000.0
        mesh.vertices.append(p)

    for tri in triangles:
        t = MeshTriangle()
        t.vertex_indices = [tri[0], tri[1], tri[2]]
        mesh.triangles.append(t)

    return mesh


class STLToMoveIt(Node):

    def __init__(self, stl_path):
        super().__init__('stl_to_moveit')
        self.stl_path = stl_path
        self.last_mtime = None

        # Publisher for the planning scene
        self.scene_pub = self.create_publisher(
            PlanningScene, '/monitored_planning_scene', 10)

        # Also publish to the apply_planning_scene service topic
        self.collision_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10)

        self.get_logger().info(f"STL to MoveIt node started")
        self.get_logger().info(f"Watching: {self.stl_path}")
        self.get_logger().info(f"Object name in MoveIt: '{OBJECT_NAME}'")
        self.get_logger().info(f"Object position: x={OBJECT_X}, y={OBJECT_Y}, z={OBJECT_Z}")

        # Load immediately if file exists
        if os.path.exists(self.stl_path):
            self.get_logger().info("STL file found — loading into MoveIt...")
            # Small delay to let MoveIt start up
            time.sleep(2.0)
            self.load_stl()
        else:
            self.get_logger().info(f"Waiting for STL file at: {self.stl_path}")

        # Start file watcher
        if WATCH_FOR_CHANGES:
            self.timer = self.create_timer(CHECK_INTERVAL_SEC, self.check_for_update)

    def check_for_update(self):
        """Check if the STL file has been created or updated."""
        if not os.path.exists(self.stl_path):
            return
        mtime = os.path.getmtime(self.stl_path)
        if mtime != self.last_mtime:
            self.get_logger().info("STL file updated — reloading into MoveIt...")
            self.load_stl()
            self.last_mtime = mtime

    def load_stl(self):
        """Load the STL file and publish it to the MoveIt planning scene."""
        try:
            self.get_logger().info(f"Reading STL: {self.stl_path}")
            mesh = stl_to_mesh_msg(self.stl_path)
            self.get_logger().info(
                f"Mesh loaded: {len(mesh.vertices)} vertices, "
                f"{len(mesh.triangles)} triangles")

            # First remove the old object if it exists
            self._remove_object()
            time.sleep(0.5)

            # Build collision object
            obj = CollisionObject()
            obj.header = Header()
            obj.header.frame_id = OBJECT_FRAME
            obj.header.stamp = self.get_clock().now().to_msg()
            obj.id = OBJECT_NAME
            obj.operation = CollisionObject.ADD

            # Set the mesh
            obj.meshes.append(mesh)

            # Set the pose
            pose = Pose()
            pose.position.x = float(OBJECT_X)
            pose.position.y = float(OBJECT_Y)
            pose.position.z = float(OBJECT_Z)
            pose.orientation.w = 1.0
            obj.mesh_poses.append(pose)

            # Publish
            self.collision_pub.publish(obj)
            self.get_logger().info(
                f"✔ Object '{OBJECT_NAME}' added to MoveIt planning scene at "
                f"({OBJECT_X}, {OBJECT_Y}, {OBJECT_Z})")

            # Also update planning scene directly
            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects.append(obj)
            self.scene_pub.publish(scene)

        except Exception as e:
            self.get_logger().error(f"Failed to load STL: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _remove_object(self):
        """Remove the existing collision object from the scene."""
        obj = CollisionObject()
        obj.header = Header()
        obj.header.frame_id = OBJECT_FRAME
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = OBJECT_NAME
        obj.operation = CollisionObject.REMOVE
        self.collision_pub.publish(obj)
        self.get_logger().info(f"Removed old '{OBJECT_NAME}' from planning scene")


def main():
    global OBJECT_X, OBJECT_Y, OBJECT_Z
    parser = argparse.ArgumentParser(description="Load STL into MoveIt planning scene")
    parser.add_argument('--stl', type=str, default=DEFAULT_STL_PATH,
                        help=f"Path to STL file (default: {DEFAULT_STL_PATH})")
    parser.add_argument('--x', type=float, default=OBJECT_X,
                        help="Object X position in metres")
    parser.add_argument('--y', type=float, default=OBJECT_Y,
                        help="Object Y position in metres")
    parser.add_argument('--z', type=float, default=OBJECT_Z,
                        help="Object Z position in metres")
    args, _ = parser.parse_known_args()

    # Override position if given
    OBJECT_X, OBJECT_Y, OBJECT_Z = args.x, args.y, args.z

    rclpy.init()
    node = STLToMoveIt(args.stl)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
