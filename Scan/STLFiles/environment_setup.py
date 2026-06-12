#!/usr/bin/env python3
"""
environment_setup.py

ROS2 node that loads the permanent environment into the MoveIt planning scene.
Run this once after launching the UR driver and MoveIt.

Objects added:
  - Robot mounting table (large plane below robot base)
  - Wall (80cm behind robot)
  - Turntable (75cm diameter, 89cm in front of robot, 18cm above base)
  - Camera mount (approximate position)

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 environment_setup.py

    # Or clear the scene:
    python3 environment_setup.py --clear
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import argparse
import time
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from geometry_msgs.msg import Point
import pyassimp
import os

# ─────────────────────────────────────────────
#  ENVIRONMENT CONFIGURATION
#  All measurements in metres, relative to robot base frame
# ─────────────────────────────────────────────

ENVIRONMENT = {

    # ── Robot mounting table ──────────────────────────────────
    # Large flat plane that the robot is bolted to
    "robot_table": {
        "type":    "box",
        "size":    (2.5, 2.5, 0.05),   # 2m x 2m x 5cm thick
        "pos":     (0.0, 0.0, -0.025), # Centred on robot base, just below
        "comment": "Table the robot is mounted on"
    },

    # ── Wall behind robot ─────────────────────────────────────
    # 80cm behind robot base (negative Y in UR base frame)
    "wall_behind": {
        "type": "box",
        "size": (0.05, 3.0, 2.5),  # 3m wide, 5cm thick, 2.5m tall
        "pos": (-0.8, 0.0, 1.25),  # 80cm behind, centred, full height
        "comment": "Wall 80cm behind robot"
    },

# ── Wall to the side of the robot ─────────────────────────────────────
    # 70cm to the left of robot base
    "wall_on_right": {
        "type":    "box",
        "size":    (3.0, 0.05, 2.5),   # 3m wide, 5cm thick, 2.5m tall
        "pos":     (0.0, 0.85, 1.25), # 80cm behind, centred, full height
        "comment": "Wall 80cm behind robot"
    },

    # ── Turntable ─────────────────────────────────────────────
    # 75cm diameter cylinder, 89cm in front of robot, 18cm above base
    # Modelled as a flat cylinder (disc)
    "turntable": {
        "type":    "cylinder",
        "radius":  0.400,
        "height":  0.075,
        "pos":     (0.65, 0, 0),
        "comment": "Turntable sitting on robot table"
    },
}

# ─────────────────────────────────────────────


class EnvironmentSetup(Node):

    def __init__(self):
        super().__init__('environment_setup')

        self.collision_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10)
        self.scene_pub = self.create_publisher(
            PlanningScene, '/monitored_planning_scene', 10)



        self.get_logger().info("Environment Setup Node started")
        self.get_logger().info("Waiting for MoveIt to be ready...")
        time.sleep(5.0)  # ← change from 2.0 to 5.0

    def add_box(self, name, size, pos, comment=""):
        """Add a box collision object to the scene."""
        obj = CollisionObject()
        obj.header = Header()
        obj.header.frame_id = "base_link"
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = name
        obj.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [size[0], size[1], size[2]]
        obj.primitives.append(primitive)

        pose = Pose()
        pose.position.x = float(pos[0])
        pose.position.y = float(pos[1])
        pose.position.z = float(pos[2])
        pose.orientation.w = 1.0
        obj.primitive_poses.append(pose)

        self.collision_pub.publish(obj)
        self.get_logger().info(
            f"  ✔ Added box '{name}' at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
            f"size ({size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}m)")
        if comment:
            self.get_logger().info(f"    → {comment}")

    def add_cylinder(self, name, radius, height, pos, comment=""):
        """Add a cylinder collision object to the scene."""
        obj = CollisionObject()
        obj.header = Header()
        obj.header.frame_id = "base_link"
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = name
        obj.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [height, radius]  # MoveIt: [height, radius]
        obj.primitives.append(primitive)

        pose = Pose()
        pose.position.x = float(pos[0])
        pose.position.y = float(pos[1])
        pose.position.z = float(pos[2])
        pose.orientation.w = 1.0
        obj.primitive_poses.append(pose)

        self.collision_pub.publish(obj)
        self.get_logger().info(
            f"  ✔ Added cylinder '{name}' at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
            f"r={radius:.2f}m h={height:.2f}m")
        if comment:
            self.get_logger().info(f"    → {comment}")


    def remove_object(self, name):
        """Remove a collision object from the scene."""
        obj = CollisionObject()
        obj.header = Header()
        obj.header.frame_id = "base_link"
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = name
        obj.operation = CollisionObject.REMOVE
        self.collision_pub.publish(obj)
        self.get_logger().info(f"  ✗ Removed '{name}'")

    def load_environment(self):
        """Load all permanent environment objects into MoveIt."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("  Loading permanent environment...")
        self.get_logger().info("=" * 50)

        for name, obj in ENVIRONMENT.items():
            time.sleep(0.3)  # Small delay between publishes
            if obj["type"] == "box":
                self.add_box(
                    name,
                    obj["size"],
                    obj["pos"],
                    obj.get("comment", "")
                )
            elif obj["type"] == "cylinder":
                self.add_cylinder(
                    name,
                    obj["radius"],
                    obj["height"],
                    obj["pos"],
                    obj.get("comment", "")
                )

        self.get_logger().info("=" * 50)
        self.get_logger().info("  ✔ Environment loaded successfully!")
        self.get_logger().info("  Republishing objects to ensure MoveIt received them...")
        for _ in range(3):
            time.sleep(1.0)
            for name, obj in ENVIRONMENT.items():
                if obj["type"] == "box":
                    self.add_box(name, obj["size"], obj["pos"])
                elif obj["type"] == "cylinder":
                    self.add_cylinder(name, obj["radius"], obj["height"], obj["pos"])
        self.get_logger().info("  ✔ Done republishing")
        self.get_logger().info("  Objects in scene:")
        for name in ENVIRONMENT:
            self.get_logger().info(f"    - {name}")
        self.get_logger().info("=" * 50)

    def clear_environment(self):
        """Remove all permanent environment objects from MoveIt."""
        self.get_logger().info("Clearing environment...")
        for name in ENVIRONMENT:
            self.remove_object(name)
            time.sleep(0.2)
        self.get_logger().info("✔ Environment cleared")

    def update_turntable_position(self, x, y, z):
        """Update just the turntable position — useful when adjusting setup."""
        self.get_logger().info(f"Updating turntable position to ({x}, {y}, {z})")
        self.remove_object("turntable")
        time.sleep(0.3)
        self.add_cylinder(
            "turntable",
            ENVIRONMENT["turntable"]["radius"],
            ENVIRONMENT["turntable"]["height"],
            (x, y, z),
            "Updated turntable position"
        )


def main():
    parser = argparse.ArgumentParser(description="Load environment into MoveIt")
    parser.add_argument('--clear', action='store_true',
                        help="Clear the environment instead of loading it")
    parser.add_argument('--turntable-x', type=float, default=None,
                        help="Override turntable X position (metres)")
    parser.add_argument('--turntable-y', type=float, default=None,
                        help="Override turntable Y position (metres)")
    parser.add_argument('--turntable-z', type=float, default=None,
                        help="Override turntable Z position (metres)")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = EnvironmentSetup()

    try:
        if args.clear:
            node.clear_environment()
        elif args.turntable_x is not None:
            x = args.turntable_x
            y = args.turntable_y if args.turntable_y is not None else ENVIRONMENT["turntable"]["pos"][1]
            z = args.turntable_z if args.turntable_z is not None else ENVIRONMENT["turntable"]["pos"][2]
            node.update_turntable_position(x, y, z)
        else:
            node.load_environment()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        time.sleep(1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
