#!/usr/bin/env python3
"""
clear_viz.py
Clears all visualization markers from the previous run.
Usage:
    python3 clear_viz.py
"""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import time

TOPICS = ['/cleaning_coverage', '/waypoint_markers']
MAX_ID  = 200   # covers all patch ids (100+) plus cone/hud/bbox

class Clearer(Node):
    def __init__(self):
        super().__init__('viz_clearer')
        self._pubs = {t: self.create_publisher(MarkerArray, t, 10) for t in TOPICS}

    def clear(self):
        for topic, pub in self._pubs.items():
            ma = MarkerArray()
            # DELETE_ALL action clears every marker on the topic at once
            m          = Marker()
            m.action   = Marker.DELETEALL
            m.ns       = ''
            m.id       = 0
            ma.markers.append(m)
            # Publish a few times so RViz definitely gets it
            for _ in range(5):
                pub.publish(ma)
                time.sleep(0.1)
            self.get_logger().info(f'Cleared {topic}')

def main():
    rclpy.init()
    node = Clearer()
    node.clear()
    node.destroy_node()
    rclpy.shutdown()
    print('Done — all markers cleared.')

if __name__ == '__main__':
    main()
