#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from threading import Lock
import time
import message_filters
from collections import deque


class RGBDPointCloudScanner(Node):
    """
    Enhanced ROS2 node for creating colored point clouds from RGBD data.
    
    Features:
    - Synchronized RGB and depth image processing
    - Configurable camera intrinsics
    - Point cloud filtering and optimization
    - Real-time visualization
    - Point cloud accumulation for complete object scanning
    """
    
    def __init__(self):
        super().__init__('rgbd_pointcloud_scanner')
        
        # Initialize bridge and thread safety
        self.bridge = CvBridge()
        self.processing_lock = Lock()
        
        # Point cloud accumulation
        self.accumulated_points = []
        self.max_accumulated_clouds = 100
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Camera topics
                ('depth_topic', '/depth/image_raw'),
                ('rgb_topic', '/rgb/image_raw'),
                ('camera_info_topic', '/rgb/camera_info'),
                ('output_topic', '/pointcloud'),
                
                # Depth processing
                ('depth_scale', 1000.0),  # mm to meters
                ('min_depth', 0.1),       # meters
                ('max_depth', 5.0),       # meters
                
                # Camera intrinsics (defaults for common cameras)
                ('fx', 607.7908935546875),            # focal length x
                ('fy', 607.75390625),            # focal length y  
                ('cx', 640.822509765625),            # principal point x
                ('cy', 369.03350830078125),            # principal point y
                
                # Point cloud processing
                ('downsample_factor', 2),  # Skip every N pixels
                ('use_camera_info', True), # Get intrinsics from camera_info
                ('publish_rate', 10.0),    # Hz
                ('accumulate_clouds', False), # Accumulate point clouds
                
                # Filtering
                ('enable_statistical_filter', True),
                ('statistical_neighbors', 20),
                ('statistical_std_ratio', 2.0),
                
                # Visualization
                ('display_images', True),
                ('display_fps', True),
            ]
        )
        
        self._update_parameters()
        
        # Camera intrinsics matrix
        self.camera_matrix = None
        self.got_camera_info = False
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Publishers
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, 
            self.output_topic, 
            10
        )
        
        # Subscribers with message synchronization
        if self.use_camera_info:
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                10
            )
        
        # Synchronized subscribers for RGB and Depth
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        
        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Timer for publishing accumulated point clouds
        if self.accumulate_clouds:
            self.publish_timer = self.create_timer(
                1.0 / self.publish_rate, 
                self.publish_accumulated_pointcloud
            )
        
        self.get_logger().info(
            f'🔍 RGBD Point Cloud Scanner started\n'
            f'   RGB Topic: {self.rgb_topic}\n'
            f'   Depth Topic: {self.depth_topic}\n'
            f'   Output Topic: {self.output_topic}\n'
            f'   Depth Range: {self.min_depth:.2f} - {self.max_depth:.2f}m\n'
            f'   Downsample Factor: {self.downsample_factor}\n'
            f'   Accumulate Clouds: {self.accumulate_clouds}'
        )

    def _update_parameters(self):
        """Update internal parameters from ROS parameters."""
        self.depth_topic = self.get_parameter('depth_topic').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        
        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        
        self.downsample_factor = self.get_parameter('downsample_factor').value
        self.use_camera_info = self.get_parameter('use_camera_info').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.accumulate_clouds = self.get_parameter('accumulate_clouds').value
        
        self.enable_statistical_filter = self.get_parameter('enable_statistical_filter').value
        self.statistical_neighbors = self.get_parameter('statistical_neighbors').value
        self.statistical_std_ratio = self.get_parameter('statistical_std_ratio').value
        
        self.display_images = self.get_parameter('display_images').value
        self.display_fps = self.get_parameter('display_fps').value

    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message."""
        if not self.got_camera_info:
            self.fx = msg.k[0]  # K[0,0]
            self.fy = msg.k[4]  # K[1,1]
            self.cx = msg.k[2]  # K[0,2]
            self.cy = msg.k[5]  # K[1,2]
            
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
            
            self.got_camera_info = True
            self.get_logger().info(
                f'📷 Camera intrinsics received:\n'
                f'   fx: {self.fx:.2f}, fy: {self.fy:.2f}\n'
                f'   cx: {self.cx:.2f}, cy: {self.cy:.2f}'
            )

    def synchronized_callback(self, rgb_msg, depth_msg):
        """Process synchronized RGB and depth images."""
        if not self.processing_lock.acquire(blocking=False):
            return
        
        try:
            start_time = time.time()
            
            # Convert images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            
            # Generate point cloud
            points_3d, colors = self.generate_pointcloud(rgb_image, depth_image)
            
            if points_3d is not None and len(points_3d) > 0:
                # Create and publish point cloud message
                if not self.accumulate_clouds:
                    pointcloud_msg = self.create_pointcloud_msg(
                        points_3d, colors, rgb_msg.header
                    )
                    self.pointcloud_pub.publish(pointcloud_msg)
                else:
                    # Add to accumulated points
                    self.add_to_accumulated_clouds(points_3d, colors)
                
                # Display images if enabled
                if self.display_images:
                    self.display_rgbd_images(rgb_image, depth_image, points_3d.shape[0])
                
                # Performance tracking
                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)
            
        except Exception as e:
            self.get_logger().error(f'Error in synchronized processing: {e}')
        finally:
            self.processing_lock.release()

    def generate_pointcloud(self, rgb_image, depth_image):
        """
        Generate 3D point cloud from RGB and depth images.
        
        Returns:
            tuple: (points_3d, colors) where points_3d is Nx3 and colors is Nx3
        """
        try:
            # Ensure images have same dimensions
            if rgb_image.shape[:2] != depth_image.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))
            
            h, w = depth_image.shape
            
            # Convert depth to meters
            if depth_image.dtype == np.uint16:
                depth_meters = depth_image.astype(np.float32) / self.depth_scale
            else:
                depth_meters = depth_image.astype(np.float32)
            
            # Create coordinate grids (with downsampling)
            step = self.downsample_factor
            v_coords, u_coords = np.mgrid[0:h:step, 0:w:step]
            
            # Get corresponding depth and color values
            depth_sampled = depth_meters[v_coords, u_coords]
            rgb_sampled = rgb_image[v_coords, u_coords]
            
            # Create mask for valid depths
            valid_mask = (depth_sampled >= self.min_depth) & (depth_sampled <= self.max_depth)
            
            if not np.any(valid_mask):
                self.get_logger().warn('No valid depth points found')
                return None, None
            
            # Filter valid points
            u_valid = u_coords[valid_mask]
            v_valid = v_coords[valid_mask]
            depth_valid = depth_sampled[valid_mask]
            colors_valid = rgb_sampled[valid_mask]
            
            # Convert to 3D coordinates using camera intrinsics
            x = (u_valid - self.cx) * depth_valid / self.fx
            y = (v_valid - self.cy) * depth_valid / self.fy
            z = depth_valid
            
            # Stack to create point cloud
            points_3d = np.column_stack((x, y, z))
            
            # Convert BGR to RGB for colors
            colors_rgb = colors_valid[:, [2, 1, 0]]  # BGR to RGB
            
            # Apply statistical filtering if enabled
            if self.enable_statistical_filter and len(points_3d) > self.statistical_neighbors:
                points_3d, colors_rgb = self.statistical_outlier_filter(points_3d, colors_rgb)
            
            return points_3d, colors_rgb
            
        except Exception as e:
            self.get_logger().error(f'Error generating point cloud: {e}')
            return None, None

    def statistical_outlier_filter(self, points, colors):
        """Apply statistical outlier removal filter."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Find k nearest neighbors for each point
            nbrs = NearestNeighbors(n_neighbors=self.statistical_neighbors).fit(points)
            distances, indices = nbrs.kneighbors(points)
            
            # Calculate mean distances
            mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self (index 0)
            
            # Calculate statistics
            global_mean = np.mean(mean_distances)
            global_std = np.std(mean_distances)
            
            # Filter outliers
            threshold = global_mean + self.statistical_std_ratio * global_std
            inlier_mask = mean_distances < threshold
            
            filtered_points = points[inlier_mask]
            filtered_colors = colors[inlier_mask]
            
            self.get_logger().debug(
                f'Statistical filter: {len(points)} -> {len(filtered_points)} points'
            )
            
            return filtered_points, filtered_colors
            
        except ImportError:
            self.get_logger().warn('sklearn not available, skipping statistical filter')
            return points, colors
        except Exception as e:
            self.get_logger().error(f'Error in statistical filtering: {e}')
            return points, colors

    def add_to_accumulated_clouds(self, points, colors):
        """Add points to accumulated point cloud."""
        timestamp = time.time()
        cloud_data = {
            'points': points,
            'colors': colors,
            'timestamp': timestamp
        }
        
        self.accumulated_points.append(cloud_data)
        
        # Remove old clouds if we have too many
        if len(self.accumulated_points) > self.max_accumulated_clouds:
            self.accumulated_points.pop(0)

    def publish_accumulated_pointcloud(self):
        """Publish accumulated point cloud."""
        if not self.accumulated_points:
            return
        
        try:
            # Combine all accumulated points
            all_points = []
            all_colors = []
            
            for cloud in self.accumulated_points:
                all_points.append(cloud['points'])
                all_colors.append(cloud['colors'])
            
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            
            # Create header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'camera_frame'
            
            # Create and publish point cloud
            pointcloud_msg = self.create_pointcloud_msg(
                combined_points, combined_colors, header
            )
            self.pointcloud_pub.publish(pointcloud_msg)
            
            self.get_logger().debug(
                f'Published accumulated point cloud: {len(combined_points)} points'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error publishing accumulated point cloud: {e}')

    def create_pointcloud_msg(self, points_3d, colors, header):
        """Create PointCloud2 message from points and colors."""
        try:
            # Create list of points with RGB data
            points_list = []
            
            for i in range(len(points_3d)):
                # Pack RGB values into a single uint32
                r = int(colors[i, 0]) & 0xFF
                g = int(colors[i, 1]) & 0xFF
                b = int(colors[i, 2]) & 0xFF
                rgb = (r << 16) | (g << 8) | b
                
                # Create point [x, y, z, rgb]
                point = [
                    float(points_3d[i, 0]),  # x
                    float(points_3d[i, 1]),  # y
                    float(points_3d[i, 2]),  # z
                    rgb                      # rgb as uint32
                ]
                points_list.append(point)
            
            # Define point cloud fields
            fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1),
            ]
            
            # Create PointCloud2 message using create_cloud
            pointcloud_msg = pc2.create_cloud(header, fields, points_list)
            
            return pointcloud_msg
            
        except Exception as e:
            self.get_logger().error(f'Error creating point cloud message: {e}')
            return None

    def display_rgbd_images(self, rgb_image, depth_image, num_points):
        """Display RGB and depth images with information overlay."""
        try:
            # Ensure both images have the same dimensions for display
            target_height = min(rgb_image.shape[0], depth_image.shape[0])
            target_width = min(rgb_image.shape[1], depth_image.shape[1])
            
            # Resize both images to the same size
            rgb_display = cv2.resize(rgb_image, (target_width, target_height))
            depth_display = cv2.resize(depth_image, (target_width, target_height))
            
            # Create depth colormap for visualization
            depth_meters = depth_display.astype(np.float32) / self.depth_scale
            valid_mask = (depth_meters >= self.min_depth) & (depth_meters <= self.max_depth)
            
            depth_viz = np.zeros_like(depth_meters, dtype=np.uint8)
            if np.any(valid_mask):
                depth_valid = depth_meters[valid_mask]
                if len(depth_valid) > 0:
                    depth_min = np.min(depth_valid)
                    depth_max = np.max(depth_valid)
                    if depth_max > depth_min:
                        norm_depth = ((depth_valid - depth_min) / 
                                    (depth_max - depth_min) * 255).astype(np.uint8)
                    else:
                        norm_depth = np.full_like(depth_valid, 128, dtype=np.uint8)
                    depth_viz[valid_mask] = norm_depth
            
            depth_colormap = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            depth_colormap[~valid_mask] = [0, 0, 0]
            
            # Add information overlay
            if self.display_fps and hasattr(self, 'current_fps'):
                info_text = f'FPS: {self.current_fps:.1f} | Points: {num_points}'
                cv2.putText(rgb_display, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(depth_colormap, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display images side by side
            combined = np.hstack([rgb_display, depth_colormap])
            cv2.imshow('RGBD Point Cloud Scanner', combined)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error displaying images: {e}')

    def _update_performance_stats(self, processing_time):
        """Update performance statistics."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def clear_accumulated_clouds(self):
        """Clear accumulated point clouds."""
        self.accumulated_points.clear()
        self.get_logger().info('Cleared accumulated point clouds')

    def cleanup(self):
        """Clean up resources."""
        self.get_logger().info('Cleaning up RGBD Point Cloud Scanner...')
        cv2.destroyAllWindows()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = RGBDPointCloudScanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()