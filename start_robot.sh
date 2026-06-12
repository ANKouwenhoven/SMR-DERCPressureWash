#!/bin/bash
# UR10 Startup Script
# Opens three terminals in order: network setup, UR driver, MoveIt+RViz

set -e

echo "=== Step 1: Configuring network interface ==="
sudo ip addr flush dev enp5s0
sudo ip addr add 192.168.0.100/24 dev enp5s0
sudo ip link set enp5s0 up
echo "Network configured. Waiting 2s..."
sleep 2

echo "=== Step 2: Launching UR Driver ==="
gnome-terminal --title="UR Driver" -- bash -c "
  source /opt/ros/jazzy/setup.bash
  ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10 robot_ip:=192.168.0.43
  exec bash
"

echo "Waiting 10s for UR driver to initialise..."
echo "(Press Play on the pendant now if you haven't already)"
sleep 10

echo "=== Step 3: Launching MoveIt + RViz ==="
gnome-terminal --title="MoveIt + RViz" -- bash -c "
  source /opt/ros/jazzy/setup.bash
  ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10 launch_rviz:=true
  exec bash
"

echo "Waiting 5s for MoveIt to initialise..."
sleep 5

echo "=== Step 4: Checking joint states ==="
gnome-terminal --title="Joint States" -- bash -c "
  source /opt/ros/jazzy/setup.bash
  ros2 topic echo /joint_states --once
  exec bash
"
echo "=== Step 5: Loading Environment ==="
gnome-terminal --title="Environment" -- bash -c "
  cd /home/smrcomputer/Documents/SMR/Scan/STLFiles
  python3 environment_setup.py
  exec bash
"

echo "=== All done! ==="
