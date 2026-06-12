#!/usr/bin/env python3
"""
face_analyser.py

Analyses a 'clean_surfaces.stl' file and extracts the individual faces
to be cleaned, including:
  - Face normal (approach direction)
  - Face centroid (centre position)
  - Face bounds (width and height for stripe generation)
  - A local coordinate frame (for stripe direction calculation)

Usage (standalone test):
    python3 face_analyser.py --stl ~/Downloads/clean_surfaces.stl

Or import in path_planning.py:
    from face_analyser import extract_faces
    faces = extract_faces('~/Downloads/clean_surfaces.stl')
"""

import numpy as np
import pyassimp
import argparse
import os
from dataclasses import dataclass
from typing import List


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class FaceRegion:
    """Represents a single flat surface region to be cleaned."""
    normal:   np.ndarray
    centroid: np.ndarray
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
    Load STL and return vertices, faces, and per-triangle normals.
    Normals are computed from geometry, not from stored values.
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

    # Compute per-triangle normals from geometry
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross   = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-10, 1.0, lengths)
    normals = cross / lengths

    return vertices, faces, normals


# ─────────────────────────────────────────────
#  FACE CLUSTERING
# ─────────────────────────────────────────────

def cluster_faces_by_normal(normals: np.ndarray,
                             angle_threshold_deg: float = 10.0) -> List[np.ndarray]:
    """
    Group triangle indices by similar normal direction.
    Uses abs(dot) so opposite-facing normals don't split the same flat face.
    Returns list of index arrays, one per cluster.
    """
    threshold = np.deg2rad(angle_threshold_deg)
    n_faces  = len(normals)
    assigned = np.full(n_faces, -1, dtype=int)
    clusters = []

    for i in range(n_faces):
        if assigned[i] != -1:
            continue

        cluster_id  = len(clusters)
        assigned[i] = cluster_id
        ref         = normals[i]

        dots   = np.clip(np.dot(normals, ref), -1.0, 1.0)
        angles = np.arccos(np.abs(dots))
        mask   = (angles < threshold) & (assigned == -1)
        assigned[mask] = cluster_id

        clusters.append(np.where(assigned == cluster_id)[0])

    return clusters


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────

def compute_cluster_normal(vertices: np.ndarray,
                            faces: np.ndarray,
                            cluster_indices: np.ndarray) -> np.ndarray:
    """
    Compute the mean normal for a cluster of triangles.
    Flips individual normals to be consistent with the first triangle.
    """
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


def compute_cluster_area(vertices: np.ndarray,
                          faces: np.ndarray,
                          cluster_indices: np.ndarray) -> float:
    """Compute total surface area of a cluster of triangles."""
    total = 0.0
    for idx in cluster_indices:
        face = faces[idx]
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        total += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return total


def compute_local_axes(normal: np.ndarray):
    """Compute two orthogonal axes lying in the plane of the surface."""
    normal = normal / np.linalg.norm(normal)
    ref    = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(normal, ref);  u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis); v_axis /= np.linalg.norm(v_axis)
    return u_axis, v_axis


def compute_face_bounds(vertices, centroid, u_axis, v_axis):
    """Project vertices onto local axes and return (width, height)."""
    rel     = vertices - centroid
    u_proj  = rel @ u_axis
    v_proj  = rel @ v_axis
    return u_proj.max() - u_proj.min(), v_proj.max() - v_proj.min()


def label_face(normal: np.ndarray) -> str:
    """Human-readable label from dominant normal direction."""
    idx = np.argmax(np.abs(normal))
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
                  min_area_cm2: float = 1.0) -> List[FaceRegion]:
    """
    Load an STL and extract distinct flat face regions.

    Args:
        stl_filepath        : Path to the clean_surfaces.stl file
        scale               : Unit conversion (0.001 = mm to metres)
        angle_threshold_deg : Faces within this angle are grouped together
        min_area_cm2        : Ignore faces smaller than this (filters noise)

    Returns:
        List of FaceRegion objects, sorted by area (largest first)
    """
    print(f"Loading STL: {stl_filepath}")
    vertices, faces, normals = load_mesh(stl_filepath, scale)
    print(f"  Loaded {len(vertices)} vertices, {len(faces)} triangles")

    print(f"  Clustering by normal (threshold: {angle_threshold_deg}°)...")
    clusters = cluster_faces_by_normal(normals, angle_threshold_deg)
    print(f"  Found {len(clusters)} clusters")

    min_area_m2  = min_area_cm2 * 1e-4
    face_regions = []

    for i, cluster_indices in enumerate(clusters):
        # Compute normal
        mean_normal = compute_cluster_normal(vertices, faces, cluster_indices)
        if mean_normal is None:
            print(f"  Cluster {i}: skipped (could not compute normal)")
            continue

        # Compute area
        area = compute_cluster_area(vertices, faces, cluster_indices)
        print(f"  Cluster {i}: {len(cluster_indices)} triangles, "
              f"normal=({mean_normal[0]:.2f},{mean_normal[1]:.2f},{mean_normal[2]:.2f}), "
              f"area={area*1e4:.2f}cm²")

        if area < min_area_m2:
            print(f"    → skipped (area {area*1e4:.2f}cm² < min {min_area_cm2}cm²)")
            continue

        # Get unique vertices for this cluster
        cluster_faces        = faces[cluster_indices]
        unique_vert_indices  = np.unique(cluster_faces.flatten())
        cluster_vertices     = vertices[unique_vert_indices]
        centroid             = cluster_vertices.mean(axis=0)

        # Compute local axes and bounds
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
    for face in face_regions:
        print(f"    {face}")

    return face_regions


# ─────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse clean_surfaces.stl and extract face regions")
    parser.add_argument('--stl',      type=str,   required=True)
    parser.add_argument('--scale',    type=float, default=0.001,
                        help="Scale factor (default 0.001 = mm to metres)")
    parser.add_argument('--angle',    type=float, default=10.0,
                        help="Normal clustering angle threshold in degrees")
    parser.add_argument('--min-area', type=float, default=1.0,
                        help="Minimum face area in cm²")
    args = parser.parse_args()

    faces = extract_faces(
        stl_filepath        = args.stl,
        scale               = args.scale,
        angle_threshold_deg = args.angle,
        min_area_cm2        = args.min_area,
    )

    print(f"\nSummary: {len(faces)} faces to clean")
    print("─" * 60)
    for i, face in enumerate(faces):
        print(f"\nFace {i}: {face.label}")
        print(f"  Normal   : ({face.normal[0]:.3f}, {face.normal[1]:.3f}, {face.normal[2]:.3f})")
        print(f"  Centroid : ({face.centroid[0]:.3f}, {face.centroid[1]:.3f}, {face.centroid[2]:.3f})")
        print(f"  Size     : {face.width*100:.1f}cm x {face.height*100:.1f}cm")
        print(f"  Area     : {face.area*1e4:.1f}cm²")
        print(f"  U axis   : ({face.u_axis[0]:.3f}, {face.u_axis[1]:.3f}, {face.u_axis[2]:.3f})")
        print(f"  V axis   : ({face.v_axis[0]:.3f}, {face.v_axis[1]:.3f}, {face.v_axis[2]:.3f})")


if __name__ == '__main__':
    main()
