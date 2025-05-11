"""
RGB-D Camera Data Visualization Tool

This module provides functionality to visualize RGB-D camera data stored in .npy files.
It creates visualizations of RGB images and depth maps, along with their camera matrices.

Example usage:
    python visualize_rgbdk.py path/to/your/data.npy

The input .npy file should contain a dictionary with the following keys:
    - 'rgb': RGB image array (H x W x 3)
    - 'depth': Depth map array (H x W)
    - 'K': Camera matrix (3 x 3)

The script will create a 'visualizations' directory containing:
    - rgb.png: The RGB image
    - depth.png: The depth map with a custom colormap (black for 0, viridis for other values)
    - combined.png: A side-by-side visualization of both images
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def get_black_viridis():
    """
    Creates a custom colormap where 0 values are black and other values use the viridis colormap.

    Returns:
        matplotlib.colors.ListedColormap: Custom colormap with black for zero values
    """
    viridis = plt.cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0] = [0, 0, 0, 1]  # Set the first color (for 0) to black
    black_viridis = mcolors.ListedColormap(newcolors)
    return black_viridis


def visualize_rgbdk(npy_file_path):
    """
    Visualizes RGB and depth data from a .npy file.

    Args:
        npy_file_path (str): Path to the .npy file containing RGB-D data

    The function will:
    1. Load the RGB image, depth map, and camera matrix
    2. Create a 'visualizations' directory next to the input file
    3. Save individual RGB and depth visualizations
    4. Create and save a combined visualization
    5. Display the camera matrix
    6. Show the combined visualization

    The depth map uses a custom colormap where 0 values are black and other values
    use the viridis colormap. The depth visualization is normalized using the 99th
    percentile of non-zero depth values.
    """
    # Load the .npy file
    data = np.load(npy_file_path, allow_pickle=True).item()

    # Extract RGB, depth, and camera matrix
    rgb = data['rgb']
    depth = data['depth']
    print(rgb.shape, depth.shape)
    K = data['K']

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(npy_file_path), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Custom colormap: 0 is black, rest is viridis
    black_viridis = get_black_viridis()

    # Save RGB image
    rgb_path = os.path.join(output_dir, 'rgb.png')
    plt.imsave(rgb_path, rgb)

    # Save depth image with custom colormap
    depth_path = os.path.join(output_dir, 'depth.png')
    vmax = np.percentile(depth[depth > 0], 99) if np.any(depth > 0) else 1.0
    plt.imsave(depth_path, depth, cmap=black_viridis, vmin=0, vmax=vmax)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot RGB image
    ax1.imshow(rgb)
    ax1.set_title('RGB Image')
    ax1.axis('off')

    # Plot depth image with custom colormap
    ax2.imshow(depth, cmap=black_viridis, vmin=0, vmax=vmax)
    ax2.set_title('Depth Map')
    ax2.axis('off')

    # Print camera matrix
    print("\nCamera Matrix (K):")
    print(K)

    # Save the combined visualization
    combined_path = os.path.join(output_dir, 'combined.png')
    plt.savefig(combined_path, bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.show()

    print(f"\nSaved images to:")
    print(f"RGB: {rgb_path}")
    print(f"Depth: {depth_path}")
    print(f"Combined: {combined_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize RGB, depth, and camera matrix from .npy file')
    parser.add_argument('npy_file', type=str, help='Path to the .npy file')

    args = parser.parse_args()
    visualize_rgbdk(args.npy_file)
