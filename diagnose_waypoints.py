#!/usr/bin/env python3
"""
diagnose_waypoints.py — prints all waypoints for inspection before running.
Usage:
    python3 diagnose_waypoints.py --stl ~/Downloads/Box.stl
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_analyser import extract_faces

TURNTABLE_CENTRE_X = 0.89
TURNTABLE_CENTRE_Y = -0.18
TURNTABLE_TOP_Z    = 0.255
MIN_TOOL_Z         = TURNTABLE_TOP_Z + 0.02   # 0.275m
STANDOFF           = 0.15
STRIPE_STEP        = 0.03
REACH_MAX          = 1.3
REACH_MIN          = 0.2

SIDE_FACE_BASE_QUAT = np.array([0.5214244973558314,
                                  0.4703570623044211,
                                  0.5216876284092515,
                                  0.4844819355376465])

def compute_world_offset(bbox_size):
    return np.array([TURNTABLE_CENTRE_X,
                     TURNTABLE_CENTRE_Y,
                     TURNTABLE_TOP_Z + bbox_size[2] / 2.0])

def is_side_face(normal):
    return abs(normal[2]) < 0.5

def normal_to_quaternion(normal):
    if not is_side_face(normal):
        tool_z = -normal / np.linalg.norm(normal)
        up     = np.array([1.0, 0.0, 0.0])
        tool_x = np.cross(up, tool_z);    tool_x /= np.linalg.norm(tool_x)
        tool_y = np.cross(tool_z, tool_x); tool_y /= np.linalg.norm(tool_y)
        return R.from_matrix(np.column_stack([tool_x, tool_y, tool_z])).as_quat()
    else:
        base_tool_z = np.array([1.0, 0.0, 0.0])
        target_z    = -normal / np.linalg.norm(normal)
        dot = np.dot(base_tool_z, target_z)
        if abs(dot) > 0.9999:
            if dot > 0:
                return SIDE_FACE_BASE_QUAT.copy()
            else:
                return (R.from_euler('z', 180, degrees=True) *
                        R.from_quat(SIDE_FACE_BASE_QUAT)).as_quat()
        rot_align = R.align_vectors([target_z], [base_tool_z])[0]
        return (rot_align * R.from_quat(SIDE_FACE_BASE_QUAT)).as_quat()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stl',   type=str,   required=True)
    parser.add_argument('--scale', type=float, default=0.001)
    args = parser.parse_args()

    face_regions, _, bbox_size = extract_faces(
        args.stl, scale=args.scale, min_area_cm2=0.5)

    world_offset = compute_world_offset(bbox_size)

    print(f"\n{'='*70}")
    print(f"Object bbox    : {bbox_size[0]*100:.1f} x "
          f"{bbox_size[1]*100:.1f} x {bbox_size[2]*100:.1f} cm")
    print(f"World offset   : {np.round(world_offset, 3)}")
    print(f"MIN_TOOL_Z     : {MIN_TOOL_Z:.3f}m")
    print(f"{'='*70}")

    for fi, face in enumerate(face_regions):
        side  = is_side_face(face.normal)
        quat  = normal_to_quaternion(face.normal)
        rpy   = R.from_quat(quat).as_euler('xyz', degrees=True)
        tz    = R.from_quat(quat).as_matrix()[:, 2]

        world_centroid     = face.centroid + world_offset
        verts_rel          = face.vertices - face.centroid
        face_surface_dist  = (verts_rel @ face.normal).max()
        face_surface_world = world_centroid + face.normal * face_surface_dist
        approach_origin    = face_surface_world + face.normal * STANDOFF

        half_h = face.height / 2.0
        half_w = face.width  / 2.0
        v_vals = np.arange(-half_h, half_h + STRIPE_STEP, STRIPE_STEP)

        print(f"\n{'─'*70}")
        print(f"Face {fi}: {face.label}  ({'side' if side else 'top/bottom'})")
        print(f"  Normal          : {np.round(face.normal, 3)}")
        print(f"  World centroid  : {np.round(world_centroid, 3)}")
        print(f"  Face surface    : {np.round(face_surface_world, 3)}  "
              f"(offset {face_surface_dist*100:.1f}cm from centroid)")
        print(f"  Approach origin : {np.round(approach_origin, 3)}")
        print(f"  Size            : {face.width*100:.1f}cm x {face.height*100:.1f}cm")
        print(f"  Tool RPY (deg)  : {np.round(rpy, 1)}")
        print(f"  Tool Z axis     : {np.round(tz, 3)}")
        print(f"\n  {'#':>3}  {'X':>8}  {'Y':>8}  {'Z':>8}  "
              f"{'dist':>8}  {'ok':>4}  notes")
        print(f"  {'─'*60}")

        wp = 0
        for i, v in enumerate(v_vals):
            sc = approach_origin + face.v_axis * v
            for u_sign in [-1, 1]:
                p     = sc + face.u_axis * (u_sign * half_w)
                tilt  = False
                if side and p[2] < MIN_TOOL_Z:
                    p     = p.copy()
                    p[2]  = MIN_TOOL_Z
                    tilt  = True
                dist  = np.linalg.norm(p)
                ok    = REACH_MIN < dist < REACH_MAX
                notes = []
                if tilt:            notes.append("TILT")
                if dist > REACH_MAX: notes.append(f"TOO FAR")
                if dist < REACH_MIN: notes.append(f"TOO CLOSE")
                if p[2] < 0:         notes.append("BELOW FLOOR")
                print(f"  {wp:>3}  {p[0]:>8.3f}  {p[1]:>8.3f}  {p[2]:>8.3f}  "
                      f"{dist:>8.3f}  {'✓' if ok else '✗':>4}  "
                      f"{', '.join(notes)}")
                wp += 1

    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()