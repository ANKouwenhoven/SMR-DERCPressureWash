#!/usr/bin/env python3
"""
face_analyser.py

Analyses a 'clean_surfaces.stl' file and extracts the individual faces
to be cleaned. The mesh is automatically centred around its bounding box
centre so that OBJECT_ORIGIN in path_planning.py always places the object
correctly regardless of where the STL origin was modelled.

Usage (standalone test):
    python3 face_analyser.py --stl ~/Downloads/clean_surfaces.stl

Or import in path_planning.py:
    from face_analyser import extract_faces
    faces, bbox_centre, bbox_size = extract_faces('~/Downloads/clean_surfaces.stl')
"""

import numpy as np
import pyassimp
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class FaceRegion:
    """Represents a single flat surface region to be cleaned."""
    normal:   np.ndarray
    centroid: np.ndarray   # in centred coordinates (object origin = bbox centre)
    width:    float
    height:   float
    u_axis:   np.ndarray
    v_axis:   np.ndarray
    area:     float
    vertices: np.ndarray
    label:    str = ""

    def __repr__(self):
        n = self.normal
        c = self.centroid
        return (f"FaceRegion({self.label}): "
                f"normal=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f}) "
                f"centroid=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
                f"size={self.width*100:.1f}cm x {self.height*100:.1f}cm "
                f"area={self.area*1e4:.1f}cm²")


# ─────────────────────────────────────────────
#  MESH LOADING
# ─────────────────────────────────────────────

def load_mesh(filepath: str, scale: float = 0.001):
    """
    Load STL, scale to metres, centre around bounding box centre.
    Returns vertices, faces, per-triangle normals, bbox_centre, bbox_size.
    """
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STL file not found: {filepath}")

    with pyassimp.load(filepath) as scene:
        if not scene.meshes:
            raise ValueError(f"No meshes found in {filepath}")
        mesh     = scene.meshes[0]
        vertices = np.array(mesh.vertices, dtype=float) * scale
        faces    = np.array(mesh.faces,    dtype=int)

    # Centre vertices around bounding box centre
    bbox_min    = vertices.min(axis=0)
    bbox_max    = vertices.max(axis=0)
    bbox_centre = (bbox_min + bbox_max) / 2.0
    bbox_size   = bbox_max - bbox_min
    vertices    = vertices - bbox_centre

    print(f"  Bounding box size : "
          f"{bbox_size[0]*100:.1f} x {bbox_size[1]*100:.1f} x {bbox_size[2]*100:.1f} cm")
    print(f"  Original STL origin was at "
          f"({bbox_centre[0]:.3f}, {bbox_centre[1]:.3f}, {bbox_centre[2]:.3f}) m — recentred to (0,0,0)")

    # Compute per-triangle normals from geometry
    v0      = vertices[faces[:, 0]]
    v1      = vertices[faces[:, 1]]
    v2      = vertices[faces[:, 2]]
    cross   = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-10, 1.0, lengths)
    normals = cross / lengths

    return vertices, faces, normals, bbox_centre, bbox_size


# ─────────────────────────────────────────────
#  FACE CLUSTERING
# ─────────────────────────────────────────────

def cluster_faces_by_normal(normals: np.ndarray,
                             angle_threshold_deg: float = 10.0) -> List[np.ndarray]:
    """Group triangle indices by similar normal direction."""
    threshold = np.deg2rad(angle_threshold_deg)
    n_faces   = len(normals)
    assigned  = np.full(n_faces, -1, dtype=int)
    clusters  = []

    for i in range(n_faces):
        if assigned[i] != -1:
            continue
        cluster_id  = len(clusters)
        assigned[i] = cluster_id
        ref         = normals[i]
        dots        = np.clip(np.dot(normals, ref), -1.0, 1.0)
        angles      = np.arccos(dots)
        mask        = (angles < threshold) & (assigned == -1)
        assigned[mask] = cluster_id
        clusters.append(np.where(assigned == cluster_id)[0])

    return clusters


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────

def compute_cluster_normal(vertices, faces, cluster_indices):
    tri_normals = []
    for idx in cluster_indices:
        face  = faces[idx]
        v0    = vertices[face[0]]
        v1    = vertices[face[1]]
        v2    = vertices[face[2]]
        cross = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(cross)
        if length > 1e-10:
            tri_normals.append(cross / length)
    if not tri_normals:
        return None
    ref     = tri_normals[0]
    aligned = [n if np.dot(n, ref) >= 0 else -n for n in tri_normals]
    mean    = np.mean(aligned, axis=0)
    length  = np.linalg.norm(mean)
    return mean / length if length > 1e-10 else None


def compute_cluster_area(vertices, faces, cluster_indices):
    total = 0.0
    for idx in cluster_indices:
        face = faces[idx]
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        total += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return total


def compute_local_axes(normal):
    normal = normal / np.linalg.norm(normal)
    ref    = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(normal, ref);    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis); v_axis /= np.linalg.norm(v_axis)
    return u_axis, v_axis


def compute_face_bounds(vertices, centroid, u_axis, v_axis):
    rel    = vertices - centroid
    u_proj = rel @ u_axis
    v_proj = rel @ v_axis
    return u_proj.max() - u_proj.min(), v_proj.max() - v_proj.min()


def label_face(normal):
    idx   = np.argmax(np.abs(normal))
    names = {(0, True): '+X (back)',  (0, False): '-X (front)',
             (1, True): '+Y (right)', (1, False): '-Y (left)',
             (2, True): '+Z (top)',   (2, False): '-Z (bottom)'}
    return names.get((idx, normal[idx] > 0), f"axis_{idx}")


# ─────────────────────────────────────────────
#  MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────

def extract_faces(stl_filepath: str,
                  scale: float = 0.001,
                  angle_threshold_deg: float = 10.0,
                  min_area_cm2: float = 0.5
                  ) -> Tuple[List[FaceRegion], np.ndarray, np.ndarray]:
    """
    Load an STL and extract distinct flat face regions.
    Mesh is centred around its bounding box centre automatically.

    Returns:
        face_regions : List of FaceRegion (centroid in centred coordinates)
        bbox_centre  : Original STL bbox centre before recentring (metres)
        bbox_size    : Object bounding box dimensions (metres)
    """
    print(f"Loading STL: {stl_filepath}")
    vertices, faces, normals, bbox_centre, bbox_size = load_mesh(stl_filepath, scale)
    print(f"  Loaded {len(vertices)} vertices, {len(faces)} triangles")

    print(f"  Clustering by normal (threshold: {angle_threshold_deg}°)...")
    clusters = cluster_faces_by_normal(normals, angle_threshold_deg)
    print(f"  Found {len(clusters)} clusters")

    min_area_m2  = min_area_cm2 * 1e-4
    face_regions = []

    for i, cluster_indices in enumerate(clusters):
        mean_normal = compute_cluster_normal(vertices, faces, cluster_indices)
        if mean_normal is None:
            continue

        area = compute_cluster_area(vertices, faces, cluster_indices)
        if area < min_area_m2:
            continue

        cluster_faces       = faces[cluster_indices]
        unique_vert_indices = np.unique(cluster_faces.flatten())
        cluster_vertices    = vertices[unique_vert_indices]
        centroid            = cluster_vertices.mean(axis=0)

        u_axis, v_axis = compute_local_axes(mean_normal)
        width, height  = compute_face_bounds(cluster_vertices, centroid, u_axis, v_axis)
        label          = f"face_{len(face_regions)}_{label_face(mean_normal)}"

        face_regions.append(FaceRegion(
            normal   = mean_normal,
            centroid = centroid,
            width    = width,
            height   = height,
            u_axis   = u_axis,
            v_axis   = v_axis,
            area     = area,
            vertices = cluster_vertices,
            label    = label,
        ))

    face_regions.sort(key=lambda f: f.area, reverse=True)
    print(f"\n  Extracted {len(face_regions)} face regions")
    return face_regions, bbox_centre, bbox_size


# ─────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stl',      type=str,   required=True)
    parser.add_argument('--scale',    type=float, default=0.001)
    parser.add_argument('--angle',    type=float, default=10.0)
    parser.add_argument('--min-area', type=float, default=0.5)
    args = parser.parse_args()

    faces, bbox_centre, bbox_size = extract_faces(
        stl_filepath        = args.stl,
        scale               = args.scale,
        angle_threshold_deg = args.angle,
        min_area_cm2        = args.min_area,
    )

    print(f"\nSummary: {len(faces)} faces to clean")
    print(f"Object size: {bbox_size[0]*100:.1f} x {bbox_size[1]*100:.1f} x {bbox_size[2]*100:.1f} cm")
    print("─" * 60)
    for i, face in enumerate(faces):
        print(f"\nFace {i}: {face.label}")
        print(f"  Normal   : ({face.normal[0]:.3f}, {face.normal[1]:.3f}, {face.normal[2]:.3f})")
        print(f"  Centroid : ({face.centroid[0]:.3f}, {face.centroid[1]:.3f}, {face.centroid[2]:.3f})")
        print(f"  Size     : {face.width*100:.1f}cm x {face.height*100:.1f}cm")
        print(f"  Area     : {face.area*1e4:.1f}cm²")

    print(f"  u_axis   : ({face.u_axis[0]:.3f}, {face.u_axis[1]:.3f}, {face.u_axis[2]:.3f})")
    print(f"  v_axis   : ({face.v_axis[0]:.3f}, {face.v_axis[1]:.3f}, {face.v_axis[2]:.3f})")

if __name__ == '__main__':
    main()