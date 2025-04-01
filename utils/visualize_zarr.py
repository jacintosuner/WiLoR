import argparse
import numpy as np
import zarr
import open3d as o3d
import time

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

class ZarrVisualizer:
    def __init__(self, input_folder: str):
        """Initialize the visualizer.
        
        Args:
            input_folder: Path to zarr dataset
        """
        self.input_folder = input_folder
        
        # Visualization parameters
        self.current_frame = 0
        self.current_sequence_idx = 0
        self.paused = True
        self.window_width = 1280
        self.window_height = 720
        self.point_size = 3
        self.depth_scale = 1000.0  # Convert depth to meters
        self.depth_trunc = 3.0  # Max depth in meters
        
        # Get all valid sequences
        self.root = zarr.group(self.input_folder)
        self.sequences = [seq for seq in self.root.keys() 
                        if "_rgb_image_rect" in self.root[seq].keys() and 
                           "_depth_registered_image_rect" in self.root[seq].keys() and
                           "_rgb_camera_info" in self.root[seq].keys() and
                           "gripper_pos" in self.root[seq].keys()]
        
        if not self.sequences:
            raise ValueError("No valid sequences found in the input folder")
            
        print(f"\nFound {len(self.sequences)} valid sequences")
        
        # Load data and setup visualization
        self.load_sequence()
        self.setup_visualization()
        
    def load_sequence(self):
        """Load data for the current sequence."""
        sequence = self.sequences[self.current_sequence_idx]
        demo = self.root[sequence]

        # Load data
        self.rgb = np.asarray(demo["_rgb_image_rect"]["img"])
        self.depth = np.asarray(demo["_depth_registered_image_rect"]["img"])
        self.K = np.asarray(demo["_rgb_camera_info"]["k"])[0]
        self.gripper_pos = np.asarray(demo["gripper_pos"])
        
        self.num_frames = len(self.rgb)
        self.current_frame = 0  # Reset frame counter for new sequence
        print(f"\nLoaded sequence {sequence} with {self.num_frames} frames")
        
    def create_point_cloud(self, rgb, depth):
        """Create point cloud from RGB-D data."""
        # Convert depth to meters
        depth = depth.astype(np.float32) / self.depth_scale
        
        # Create RGBD image
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0,  # Already converted to meters
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Create camera intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.intrinsic_matrix = self.K
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Remove points that are too far or have zero depth
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        mask = (points[:, 2] > 0) & (points[:, 2] < self.depth_trunc)
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        
        return pcd
        
    def setup_visualization(self):
        """Setup Open3D visualization window and interaction callbacks."""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("WiLoR Hand Visualization", 
                             width=self.window_width, 
                             height=self.window_height)
        
        # Create geometries
        self.scene_pcd = o3d.geometry.PointCloud()
        self.hand_pcd = o3d.geometry.PointCloud()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        
        # Add geometries to visualizer
        self.vis.add_geometry(self.scene_pcd)
        self.vis.add_geometry(self.hand_pcd)
        self.vis.add_geometry(self.coord_frame)
        
        # Set render options for better visibility
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray
        render_option.point_size = self.point_size
        render_option.light_on = True
        render_option.point_show_normal = False
        
        # Make hand points larger for better visibility
        self.hand_point_size = self.point_size * 2
        
        # Register key callbacks
        self.vis.register_key_callback(ord('Q'), lambda vis: exit())
        self.vis.register_key_callback(ord(' '), self.toggle_pause)
        self.vis.register_key_callback(ord(','), lambda vis: self.prev_frame())
        self.vis.register_key_callback(ord('.'), lambda vis: self.next_frame())
        self.vis.register_key_callback(ord('N'), lambda vis: self.next_sequence())
        self.vis.register_key_callback(ord('P'), lambda vis: self.prev_sequence())
        self.vis.register_key_callback(ord('+'), lambda vis: self.adjust_point_size(0.5))
        self.vis.register_key_callback(ord('-'), lambda vis: self.adjust_point_size(-0.5))
        
        # Initialize the first frame
        self.update_frame()
        
        # Set initial view
        self.reset_view()
        
    def reset_view(self):
        """Reset camera view to show all points."""
        # Get bounds of both point clouds
        scene_center = self.scene_pcd.get_center()
        
        # Set view parameters similar to viz_flow_3d_data.py and visualize_rgbdk_or_obj_or_pcd.py
        view_control = self.vis.get_view_control()
        view_control.set_lookat(scene_center)  # Look at center of scene
        view_control.set_up([0.0, -1.0, 0.0])  # Keep Y axis pointing up
        view_control.set_front(-scene_center)  # Camera direction from origin to center
        
        # Set zoom to 1 for consistent initial view
        view_control.set_zoom(4)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def adjust_point_size(self, delta):
        """Adjust point size."""
        self.point_size = max(1, self.point_size + delta)
        self.hand_point_size = self.point_size * 2
        render_option = self.vis.get_render_option()
        render_option.point_size = self.point_size
        return False
        
    def update_frame(self):
        """Update visualization for current frame."""
        # Create scene point cloud
        rgb = self.rgb[self.current_frame]
        depth = self.depth[self.current_frame]
        scene_pcd = self.create_point_cloud(rgb, depth)
        
        # Update scene point cloud
        self.scene_pcd.points = scene_pcd.points
        self.scene_pcd.colors = scene_pcd.colors
        self.vis.update_geometry(self.scene_pcd)
        
        # Update hand point cloud
        hand_vertices = self.gripper_pos[self.current_frame]
        if len(hand_vertices) > 0:  # Only update if we have vertices
            # Scale hand vertices to match scene scale (convert from millimeters to meters)
            scaled_hand_vertices = hand_vertices / self.depth_scale
            self.hand_pcd.points = o3d.utility.Vector3dVector(scaled_hand_vertices)
            self.hand_pcd.paint_uniform_color(LIGHT_PURPLE)
            self.vis.update_geometry(self.hand_pcd)
        
        # Update view
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def toggle_pause(self, vis):
        """Toggle animation pause state."""
        self.paused = not self.paused
        return False
        
    def next_frame(self):
        """Advance to next frame."""
        self.current_frame = (self.current_frame + 1) % self.num_frames
        return False
        
    def prev_frame(self):
        """Go to previous frame."""
        self.current_frame = (self.current_frame - 1) % self.num_frames
        return False
        
    def next_sequence(self):
        """Load next sequence."""
        self.current_sequence_idx = (self.current_sequence_idx + 1) % len(self.sequences)
        self.load_sequence()
        self.update_frame()
        self.reset_view()
        return False
        
    def prev_sequence(self):
        """Load previous sequence."""
        self.current_sequence_idx = (self.current_sequence_idx - 1) % len(self.sequences)
        self.load_sequence()
        self.update_frame()
        self.reset_view()
        return False
        
    def run(self):
        """Main visualization loop."""
        print("\nVisualization Controls:")
        print("  Space: Play/Pause animation")
        print("  ,/.: Previous/Next frame")
        print("  N/P: Next/Previous sequence")
        print("  +/-: Increase/Decrease point size")
        print("  Q: Quit")
        print(f"\nCurrent sequence: {self.sequences[self.current_sequence_idx]}")
        print(f"Current frame: 0/{self.num_frames-1}")
        
        last_update = time.time()
        
        while True:
            now = time.time()
            
            # Update frame if not paused and enough time has passed
            if not self.paused and (now - last_update) > 0.033:  # ~30 FPS
                self.next_frame()
                last_update = now
                print(f"\rCurrent frame: {self.current_frame}/{self.num_frames-1}", end="")
            
            self.update_frame()
            
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()
            
        self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize WiLoR hand and scene point clouds')
    parser.add_argument('input_folder', type=str,
                       help='Path to zarr dataset')
    
    args = parser.parse_args()
    
    visualizer = ZarrVisualizer(args.input_folder)
    visualizer.run()

if __name__ == '__main__':
    main()
