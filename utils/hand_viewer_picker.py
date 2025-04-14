"""
Hand Vertex Picker and Visualizer

This script provides an interactive 3D visualization tool for examining and selecting vertices
from hand mesh OBJ files. It's particularly useful for identifying and selecting specific 
keypoints on a hand mesh.

Key Features:
- Loads and visualizes 3D hand meshes from OBJ files
- Provides interactive point picking functionality (Shift + Left Click)
- Displays vertex indices and 3D coordinates of selected points
- Highlights pre-defined keypoints in red
- Supports scaling and rotation of the mesh for better visualization

Usage:
    python3 hand_viewer_picker.py --hand path/to/hand.obj

Controls:
    - Shift + Left Click: Select a vertex
    - Mouse Left Button: Rotate the view
    - Mouse Right Button: Pan
    - Mouse Wheel: Zoom in/out
    - Q: Exit and display picked points

Example:
    python3 hand_viewer_picker.py --hand /data/jacinto/robot-grasp/data/mask_references/carton_milk_reference_20241203/hand_frames/hand_grasping_0.obj
"""

import argparse

import numpy as np
import open3d as o3d

# example usage: python3 hand_visualizer.py --hand /data/jacinto/robot-grasp/data/mask_references/carton_milk_reference_20241203/hand_frames/hand_grasping_0.obj

# HAND_KEYPOINTS = [
#     47, 48, 66, 67, 68, 69, 70, 71, 72, 73, 74,
#     77, 94, 141, 142, 146, 147, 148, 151, 152,
#     157, 165, 166, 194, 268, 271, 275, 280, 288,
#     328, 340, 341, 342, 343, 356, 357,
#     358, 359, 370, 375, 376, 378, 379, 386, 387, 392,
#     402, 438, 440, 452, 453, 454, 455, 468,
#     469, 470, 471, 485, 486, 489, 496, 497, 513, 566, 570, 572, 573,
#     683, 690, 755, 756, 763
# ]
HAND_KEYPOINTS = [763]

def load_obj_with_indices(file_path, scale=1000.0):
    """Load OBJ file and return pointcloud with all vertices"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    points = []
    colors = []
    vertex_indices = []

    lines = lines[1:] # ignore the first line with the comment
    
    for idx, line in enumerate(lines):
        if line.startswith('v '):
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            if idx in HAND_KEYPOINTS:
                r, g, b = 1.0, 0.0, 0.0
            else:
                r, g, b = map(float, parts[4:7]) if len(parts) > 6 else (0.5, 0.5, 0.5)
            points.append([x, y, z])
            colors.append([r, g, b])
            vertex_indices.append(idx)

    points = np.array(points, dtype=np.float32) / scale
    colors = np.array(colors, dtype=np.float32)
    
    return points, colors, vertex_indices

def custom_draw_geometry_with_picking(pcd):
    """Interactive visualization that shows vertex index when clicked"""
    # Create visualization window with editing enabled
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # Set up the visualization
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    
    # Get view control
    view_control = vis.get_view_control()
    view_control.set_up([0, -1, 0])
    view_control.set_front([0, 0, -1])
    view_control.set_lookat(pcd.get_center())
    view_control.set_zoom(0.8)
    
    print("Shift + Left Click to pick points")
    print("Press 'Q' to exit and display picked points")
    
    # Run picking interface
    picked_indices = vis.run()  # Returns indices of picked points
    
    # Print picked points info
    points = np.asarray(pcd.points)
    for idx in picked_indices:
        point = points[idx]
        print(f"Vertex {idx}: position {point}")
        
    vis.destroy_window()
    return picked_indices

def main(hand_file_path):
    # Load pointcloud with indices
    points, colors, indices = load_obj_with_indices(hand_file_path)
    
    # Create main pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Apply rotation to match visualization
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    
    # Visualize with picking functionality
    custom_draw_geometry_with_picking(pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand mesh vertices with indices")
    parser.add_argument('--hand', required=True, help="Path to the OBJ file")
    
    args = parser.parse_args()
    main(args.hand)