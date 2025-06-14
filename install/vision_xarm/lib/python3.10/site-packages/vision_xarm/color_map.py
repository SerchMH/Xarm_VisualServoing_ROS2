#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from threading import Lock
import time


class DepthColorMap(Node):
    """
    Enhanced ROS2 node for converting depth images to colorized visualizations.
    
    Features:
    - Configurable depth range and colormap
    - Performance monitoring
    - Thread-safe processing
    - Error recovery
    - Dynamic parameter updates
    """
    
    def __init__(self):
        super().__init__('depth_colormap')
        
        # Initialize bridge and thread safety
        self.bridge = CvBridge()
        self.processing_lock = Lock()
        
        # Declare parameters with defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                ('depth_scale', 150.0),  # Depth scale factor (mm to meters)
                ('min_depth', 0.1),       # Minimum valid depth (meters)
                ('max_depth', 1.5),       # Maximum valid depth (meters)
                ('colormap', cv2.COLORMAP_JET),  # OpenCV colormap
                ('topic_name', '/depth/image_raw'),
                ('display_fps', True),
                ('resize_factor', 1.0),   # Scale factor for display
            ]
        )
        
        # Get initial parameter values
        self._update_parameters()
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processing_times = []
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.depth_callback,
            10
        )
        
        # Parameter change callback
        self.add_on_set_parameters_callback(self._on_parameter_change)
        
        self.get_logger().info(
            f'🌈 DepthColorMap node started\n'
            f'   Topic: {self.topic_name}\n'
            f'   Depth range: {self.min_depth:.2f} - {self.max_depth:.2f}m\n'
            f'   Scale factor: {self.depth_scale}'
        )

    def _update_parameters(self):
        """Update internal parameters from ROS parameters."""
        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.colormap = self.get_parameter('colormap').value
        self.topic_name = self.get_parameter('topic_name').value
        self.display_fps = self.get_parameter('display_fps').value
        self.resize_factor = self.get_parameter('resize_factor').value

    def _on_parameter_change(self, params):
        """Handle parameter changes at runtime."""
        for param in params:
            if param.name in ['depth_scale', 'min_depth', 'max_depth', 
                             'colormap', 'display_fps', 'resize_factor']:
                self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
        
        self._update_parameters()
        return rclpy.parameter.SetParametersResult(successful=True)

    def depth_callback(self, msg):
        """Process incoming depth image messages."""
        if not self.processing_lock.acquire(blocking=False):
            self.get_logger().debug('Skipping frame - previous frame still processing')
            return
        
        try:
            start_time = time.time()
            
            # Convert ROS image to OpenCV format
            try:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except CvBridgeError as e:
                self.get_logger().error(f'CV Bridge error: {e}')
                return
            
            # Process depth image
            colorized_depth = self._process_depth_image(depth_image)
            
            if colorized_depth is not None:
                self._display_image(colorized_depth)
                
                # Performance tracking
                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)
                
        except Exception as e:
            self.get_logger().error(f'Unexpected error in depth processing: {e}')
            
        finally:
            self.processing_lock.release()

    def _process_depth_image(self, depth_image):
        """
        Convert depth image to colorized visualization.
        
        Args:
            depth_image: Raw depth image from sensor
            
        Returns:
            numpy.ndarray: Colorized depth image or None if processing fails
        """
        try:
            # Handle different depth image formats
            if depth_image.dtype == np.uint16:
                # Convert from millimeters to meters
                depth_meters = depth_image.astype(np.float32) / self.depth_scale
            elif depth_image.dtype == np.float32:
                depth_meters = depth_image
            else:
                self.get_logger().warn(f'Unexpected depth image dtype: {depth_image.dtype}')
                depth_meters = depth_image.astype(np.float32) / self.depth_scale
            
            # Create mask for valid depth values
            valid_mask = (depth_meters >= self.min_depth) & (depth_meters <= self.max_depth)
            
            if not np.any(valid_mask):
                self.get_logger().warn('No valid depth data in range')
                return None
            
            # Prepare depth data for colorization
            processed_depth = np.zeros_like(depth_meters)
            processed_depth[valid_mask] = depth_meters[valid_mask]
            
            # Normalize to 0-255 range for colormap application
            if np.max(processed_depth) > np.min(processed_depth[valid_mask]):
                normalized_depth = np.zeros_like(processed_depth, dtype=np.uint8)
                valid_depths = processed_depth[valid_mask]
                norm_valid = ((valid_depths - self.min_depth) / 
                             (self.max_depth - self.min_depth) * 255).astype(np.uint8)
                normalized_depth[valid_mask] = norm_valid
            else:
                normalized_depth = np.zeros_like(processed_depth, dtype=np.uint8)
            
            # Apply colormap
            colorized = cv2.applyColorMap(normalized_depth, self.colormap)
            
            # Set invalid regions to black
            colorized[~valid_mask] = [0, 0, 0]
            
            return colorized
            
        except Exception as e:
            self.get_logger().error(f'Error in depth processing: {e}')
            return None

    def _display_image(self, image):
        """Display the processed image with optional resizing and info overlay."""
        try:
            display_image = image.copy()
            
            # Resize if needed
            if self.resize_factor != 1.0:
                h, w = display_image.shape[:2]
                new_w = int(w * self.resize_factor)
                new_h = int(h * self.resize_factor)
                display_image = cv2.resize(display_image, (new_w, new_h))
            
            # Add FPS information if enabled
            if self.display_fps and hasattr(self, 'current_fps'):
                fps_text = f'FPS: {self.current_fps:.1f}'
                cv2.putText(display_image, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add depth range info
                range_text = f'Range: {self.min_depth:.1f}-{self.max_depth:.1f}m'
                cv2.putText(display_image, range_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display image
            cv2.imshow("Enhanced Depth Colormap", display_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error displaying image: {e}')

    def _update_performance_stats(self, processing_time):
        """Update and log performance statistics."""
        self.processing_times.append(processing_time)
        self.frame_count += 1
        
        # Keep only recent processing times
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        # Calculate FPS every second
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            avg_processing_time = np.mean(self.processing_times) * 1000  # Convert to ms
            
            if self.display_fps:
                self.get_logger().debug(
                    f'Performance: {self.current_fps:.1f} FPS, '
                    f'Avg processing: {avg_processing_time:.1f}ms'
                )
            
            # Reset counters
            self.frame_count = 0
            self.last_fps_time = current_time

    def cleanup(self):
        """Clean up resources."""
        self.get_logger().info('Cleaning up DepthColorMap node...')
        cv2.destroyAllWindows()


def main(args=None):
    """Main entry point for the depth colormap node."""
    rclpy.init(args=args)
    
    node = DepthColorMap()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()