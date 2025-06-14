#!/usr/bin/env python3

import sys
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import yaml

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# Open3D import
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("ERROR: Open3D is required for point cloud generation. Please install it.")
    sys.exit(1)

class PointCloudGenerator(Node):
    """
    Generates point clouds from RGB-D images using Open3D.
    Publishes the point cloud and its TF for visualization in RViz.
    """
    def __init__(self):
        super().__init__('point_cloud_generator')
        
        # Declare parameters
        self.declare_parameter('update_rate', 10.0)                    # Hz
        #self.declare_parameter('camera_intrinsics_file', '')           
        self.declare_parameter("camera_intrinsics_file", "/home/serch/xarm_ws/src/image_converter/camera/kinect_calib.yaml")
        self.declare_parameter('depth_scale', 1000.0)                  # Convert mm to meters
        self.declare_parameter('voxel_size', 1.0)                   
        
        self.declare_parameter('tf_x', 0.2)       
        self.declare_parameter('tf_y', 0.0)  
        self.declare_parameter('tf_z', 0.5)  
        self.declare_parameter('tf_qx', 0.0)
        self.declare_parameter('tf_qy', 0.0)
        self.declare_parameter('tf_qz', 0.0)
        self.declare_parameter('tf_qw', 1.0)
        
        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.intrinsics_file = self.get_parameter('camera_intrinsics_file').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.voxel_size = self.get_parameter('voxel_size').value
        
        self.tf_x = self.get_parameter('tf_x').value
        self.tf_y = self.get_parameter('tf_y').value
        self.tf_z = self.get_parameter('tf_z').value
        self.tf_qx = self.get_parameter('tf_qx').value
        self.tf_qy = self.get_parameter('tf_qy').value
        self.tf_qz = self.get_parameter('tf_qz').value
        self.tf_qw = self.get_parameter('tf_qw').value

        # Timer for periodic processing
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Immediately validate the initial values
        init_params = [
            Parameter('update_rate',            Parameter.Type.DOUBLE, self.update_rate),
            Parameter('tf_x',                   Parameter.Type.DOUBLE, self.tf_x),
            Parameter('tf_y',                   Parameter.Type.DOUBLE, self.tf_y),
            Parameter('tf_z',                   Parameter.Type.DOUBLE, self.tf_z),
            Parameter('tf_qx',                  Parameter.Type.DOUBLE, self.tf_qx),
            Parameter('tf_qy',                  Parameter.Type.DOUBLE, self.tf_qy),
            Parameter('tf_qz',                  Parameter.Type.DOUBLE, self.tf_qz),
            Parameter('tf_qw',                  Parameter.Type.DOUBLE, self.tf_qw),
            Parameter('camera_intrinsics_file', Parameter.Type.STRING, self.intrinsics_file),
            Parameter('depth_scale',            Parameter.Type.DOUBLE, self.depth_scale),
            Parameter('voxel_size',             Parameter.Type.DOUBLE, self.voxel_size),
        ]

        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize variables
        self.rgb_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        self.camera_intrinsic = None
        
        # Load camera intrinsics
        self.load_camera_intrinsics()
        
        # Create subscribers for segmented images
        self.create_subscription(
            Image,
            '/cleaned_img/rgb_image',
            self.rgb_callback,
            qos.qos_profile_sensor_data
        )
        
        self.create_subscription(
            Image,
            '/cleaned_img/depth_image',
            self.depth_callback,
            qos.qos_profile_sensor_data
        )
        
        # Create point cloud publisher
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            'pointcloud/segmented_object',
            10
        )
        
        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info("PointCloudGenerator Start.")

    def load_camera_intrinsics(self):
        """Load camera intrinsics."""
        if not self.intrinsics_file:
            raise RuntimeError("camera_intrinsics_file parameter is required.")
            
        try:
            with open(self.intrinsics_file, 'r') as f:
                calib = yaml.safe_load(f)
            
            # Extract intrinsics from OpenCV format
            width = calib['image_width']
            height = calib['image_height']
            fx = calib["camera_matrix"]["fx"]
            fy = calib["camera_matrix"]["fy"]
            cx = calib["camera_matrix"]["cx"]
            cy = calib["camera_matrix"]["cy"]
            
            # Create Open3D camera intrinsic
            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                fx,
                fy,
                cx,
                cy
            )
            
            self.get_logger().info(
                f"Loaded intrinsics: width={width}, height={height}, "
                f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load camera intrinsics: {e}")

    def rgb_callback(self, msg):
        """Store the segmented RGB image."""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    def depth_callback(self, msg):
        """Store the segmented depth image."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def timer_callback(self):
        """Main processing loop - generate and publish point cloud."""
        if self.rgb_image is None or self.depth_image is None:
            return
            
        try:
            # Create Open3D images
            o3d_color = o3d.geometry.Image(self.rgb_image)
            o3d_depth = o3d.geometry.Image(self.depth_image)
            
            # Create RGB-D image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth,
                depth_scale=self.depth_scale,
                depth_trunc=3.0,  
                convert_rgb_to_intensity=False
            )
            
            # Generate point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.camera_intrinsic
            )
            
            # Transform to ROS coordinate system
            # Camera convention:    X-right, Y-down, Z-forward
            # ROS convention:       X-forward, Y-left, Z-up
            # This flips Y and Z axes to match ROS standards
            pcd.transform([[1, 0, 0, 0], 
                          [0, -1, 0, 0], 
                          [0, 0, -1, 0], 
                          [0, 0, 0, 1]])
            
            if len(pcd.points) > 0:
                # Voxel downsampling
                if self.voxel_size > 0:
                    pcd = pcd.voxel_down_sample(self.voxel_size)
                
                # Publish point cloud and TF
                self.publish_pointcloud(pcd)
                self.publish_tf()
                    
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def publish_pointcloud(self, pcd):
        """Publish the point cloud as PointCloud2 message."""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        if len(points) == 0:
            return
            
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_color_optical_frame'
        
        # Prepare point cloud data with RGB
        cloud_data = []
        for i in range(len(points)):
            rgb = (int(colors[i][0] * 255) << 16 | 
                   int(colors[i][1] * 255) << 8 | 
                   int(colors[i][2] * 255))
            cloud_data.append([
                float(points[i][0]), 
                float(points[i][1]), 
                float(points[i][2]), 
                int(rgb)
            ])
        
        # Define fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # Create and publish message
        pc2_msg = pc2.create_cloud(header, fields, cloud_data)
        self.pointcloud_pub.publish(pc2_msg)
        
    def publish_tf(self):
        """Publish TF transform for the camera frame."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'                  # XARM robot base
        t.child_frame_id = 'camera_color_optical_frame'  # Kinect optical frame
        
        t.transform.translation.x = self.tf_x  
        t.transform.translation.y = self.tf_y  
        t.transform.translation.z = self.tf_z  
        
        t.transform.rotation.x = self.tf_qx
        t.transform.rotation.y = self.tf_qy
        t.transform.rotation.z = self.tf_qz
        t.transform.rotation.w = self.tf_qw
        
        self.tf_broadcaster.sendTransform(t)

    def parameter_callback(self, params):
        """Handle parameter updates."""
        for param in params:
            name = param.name
            value = param.value

            if name == 'update_rate':
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason="update_rate must be > 0.")
                self.update_rate = float(value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
                self.get_logger().info(f"Updated update_rate: {self.update_rate} Hz.")

            elif name in ['depth_scale', 'voxel_size']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, float(value))
                self.get_logger().info(f"Updated {name}: {value}")

            elif name in ['tf_x', 'tf_y', 'tf_z', 'tf_qx', 'tf_qy', 'tf_qz', 'tf_qw']:
                if not isinstance(value, (int, float)):
                    return SetParametersResult(successful=False, reason=f"{name} must be a number.")
                setattr(self, name, float(value))
                self.get_logger().info(f"Updated {name}: {value}.")

            elif name == 'camera_intrinsics_file':
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason="camera_intrinsics_file must be a string.")
                self.intrinsics_file = value
                self.get_logger().info(f"camera_intrinsics_file updated: {value}.")
                self.load_camera_intrinsics()

        return SetParametersResult(successful=True)

    def destroy_node(self):
        """Clean up before shutting down."""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PointCloudGenerator()
    except Exception as e:
        print(f"[FATAL] PointCloudGenerator failed to initialize: {e}", file=sys.stderr)
        rclpy.shutdown()
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted with Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()