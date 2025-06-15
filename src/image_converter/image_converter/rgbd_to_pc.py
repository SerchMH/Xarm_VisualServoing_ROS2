import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import math
import cv2
import open3d as o3d
import numpy as np
import struct
import cv_bridge
from rclpy import qos
import yaml
import message_filters


class PointCloudPub(Node):
    """
    ROS2 Node that subscribes to RGB and depth images, converts them to point clouds,
    and publishes them in both camera frame and world frame coordinates.
    
    This node performs the following operations:
    1. Synchronizes RGB and depth image messages
    2. Converts images to point clouds using camera intrinsics
    3. Applies filtering and downsampling
    4. Publishes point clouds in camera frame and world frame
    5. Manages TF transforms between coordinate frames
    """
    
    def __init__(self):
        super().__init__('point_cloud_publisher')

        # ==================== PARAMETER DECLARATIONS ====================
        # Declare ROS2 parameters with default values for point cloud processing
        self.declare_parameter('enable_radius_filter', False)  # Enable/disable radius-based filtering
        self.declare_parameter('filter_radius', 0.15)          # Radius for filtering points around centroid
        self.declare_parameter('depth_scale', 1.0)             # Scale factor for depth values
        self.declare_parameter('depth_trunc', 3.0)             # Maximum depth value to consider
        self.declare_parameter('world_transform_z', 0.07)      # Z translation from world to camera
        self.declare_parameter('world_transform_x', 0.20)      # X translation from world to camera
        self.declare_parameter('world_transform_y', 0.29)      # Y translation from world to camera
        self.declare_parameter('voxel_size', 0.001)            # Voxel size for downsampling

        # ==================== PARAMETER RETRIEVAL ====================
        # Get parameter values and store them as instance variables
        self.enable_radius_filter = self.get_parameter('enable_radius_filter').get_parameter_value().bool_value
        self.filter_radius = self.get_parameter('filter_radius').get_parameter_value().double_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.depth_trunc = self.get_parameter('depth_trunc').get_parameter_value().double_value
        self.world_transform_z = self.get_parameter('world_transform_z').get_parameter_value().double_value
        self.world_transform_x = self.get_parameter('world_transform_x').get_parameter_value().double_value
        self.world_transform_y = self.get_parameter('world_transform_y').get_parameter_value().double_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value

        # ==================== TF TRANSFORM SETUP ====================
        # Initialize static transform broadcaster for publishing coordinate frame relationships
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Publish identity transform from world to base_link (no rotation or translation)
        self.publish_base_link_to_world_identity()

        # Publish transform from world to camera_base with rotation and translation
        self.publish_static_transforms()

        # ==================== QOS PROFILES ====================
        # Define QoS profile for sensor data (best effort, volatile)
        # This allows for some message loss but prioritizes low latency
        sensor_qos = qos.QoSProfile(
            reliability=qos.ReliabilityPolicy.BEST_EFFORT,  # Allow message loss for performance
            durability=qos.DurabilityPolicy.VOLATILE,       # Don't store messages for late joiners
            history=qos.HistoryPolicy.KEEP_LAST,            # Keep only latest messages
            depth=1                                         # Queue size of 1
        )

        # ==================== MESSAGE SYNCHRONIZATION ====================
        # Create synchronized subscribers for RGB and depth images
        # This ensures we process RGB and depth images that were captured at the same time
        self.depth_sub = message_filters.Subscriber(self, Image, '/cleaned_img/depth_image', qos_profile=sensor_qos)
        self.rgb_sub = message_filters.Subscriber(self, Image, '/cleaned_img/rgb_image', qos_profile=sensor_qos)

        # Create approximate time synchronizer to match RGB and depth messages
        # slop=0.1 means messages within 0.1 seconds are considered synchronized
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)

        # ==================== INITIALIZATION ====================
        # Load camera calibration parameters from YAML file
        self.load_camera_intrinsics()
        
        # Initialize OpenCV bridge for converting ROS images to OpenCV format
        self.bridge = cv_bridge.CvBridge()

        # ==================== PUBLISHERS SETUP ====================
        # Define QoS profile for point cloud publishing (reliable, higher queue depth)
        pointcloud_qos = qos.QoSProfile(
            reliability=qos.ReliabilityPolicy.RELIABLE,      # Ensure message delivery
            durability=qos.DurabilityPolicy.VOLATILE,        # Don't store for late joiners
            history=qos.HistoryPolicy.KEEP_LAST,             # Keep recent messages
            depth=5                                          # Higher queue depth for reliability
        )

        # Create publishers for point clouds in different coordinate frames
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/pointcloud/pointcloud_raw', pointcloud_qos)
        self.world_view_pub = self.create_publisher(PointCloud2, '/pointcloud/world_view', pointcloud_qos)

        self.get_logger().info("✅ Point Cloud Publisher READY")

    def publish_base_link_to_world_identity(self):
        """
        Publish an identity transform from world to base_link frame.
        This creates a direct relationship where base_link and world have the same pose.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'      # Parent frame
        t.child_frame_id = 'link_base'   # Child frame

        # Identity transform: no translation or rotation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Identity quaternion (no rotation)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info("📡 Published static transform: world -> base_link (identity)")

    def publish_static_transforms(self):
        """
        Publish static transform from world frame to camera_base frame.
        This defines where the camera is positioned and oriented relative to the world frame.
        Includes both translation and a 90-degree rotation around the Z-axis.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'        # Parent frame
        t.child_frame_id = 'camera_base'   # Child frame
        
        # Set translation (camera position relative to world origin)
        t.transform.translation.x = self.world_transform_x
        t.transform.translation.y = self.world_transform_y
        t.transform.translation.z = self.world_transform_z
        
        # Set rotation: -90 degrees around Z-axis (yaw rotation)
        # This rotates the camera coordinate system to align with world coordinates
        theta = -math.pi / 2  # -90 degrees in radians
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(theta / 2)  # Z component of quaternion
        t.transform.rotation.w = math.cos(theta / 2)  # W component of quaternion
        
        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info("📡 Published static transform: world -> camera_base")

    def load_camera_intrinsics(self):
        """
        Load camera calibration parameters from a YAML file.
        These parameters are essential for converting 2D image coordinates to 3D world coordinates.
        Falls back to default values if the file cannot be loaded.
        """
        yaml_path = '/home/serch/xarm_ws/src/image_converter/camera/kinect_calib.yaml'
        try:
            # Attempt to load calibration data from YAML file
            with open(yaml_path, 'r') as file:
                calib_data = yaml.safe_load(file)
            
            # Extract camera parameters
            self.camera_width = calib_data['image_width']
            self.camera_height = calib_data['image_height']
            # Convert camera matrix from flat list to 3x3 numpy array
            self.intrinsic_matrix = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
            
        except Exception as e:
            # If loading fails, use default parameters (typical for VGA resolution)
            self.get_logger().error(f"⚠️ Error loading intrinsics: {str(e)}")
            self.camera_width = 640
            self.camera_height = 480
            # Default intrinsic matrix for a typical camera
            # [fx, 0, cx]
            # [0, fy, cy] where fx,fy are focal lengths and cx,cy are principal points
            # [0,  0,  1]
            self.intrinsic_matrix = np.array([
                [525.0, 0.0, 320.0],  # fx=525, cx=320 (center x)
                [0.0, 525.0, 240.0],  # fy=525, cy=240 (center y)
                [0.0, 0.0, 1.0]       # Homogeneous coordinate
            ])

    def synchronized_callback(self, rgb_msg, depth_msg):
        """
        Callback function triggered when synchronized RGB and depth messages are received.
        This ensures we process RGB and depth images that correspond to the same time instant.
        
        Args:
            rgb_msg (sensor_msgs.msg.Image): RGB image message
            depth_msg (sensor_msgs.msg.Image): Depth image message
        """
        try:
            # Process the synchronized image pair
            self.process_images(rgb_msg, depth_msg)
        except Exception as e:
            self.get_logger().error(f"❌ Error processing images: {str(e)}")

    def process_images(self, rgb_msg, depth_msg):
        """
        Main processing function that converts RGB-D images to point clouds.
        
        Pipeline:
        1. Validate image dimensions match
        2. Convert ROS images to OpenCV/numpy format
        3. Create Open3D RGBD image
        4. Generate point cloud from RGBD data
        5. Apply filtering and downsampling
        6. Publish point clouds in camera and world frames
        
        Args:
            rgb_msg (sensor_msgs.msg.Image): RGB image message
            depth_msg (sensor_msgs.msg.Image): Depth image message
        """
        # ==================== INPUT VALIDATION ====================
        # Ensure RGB and depth images have matching dimensions
        if (rgb_msg.width != depth_msg.width or rgb_msg.height != depth_msg.height):
            self.get_logger().warn("RGB and depth image dimensions don't match")
            return

        # ==================== IMAGE CONVERSION ====================
        try:
            # Convert RGB image from ROS format to OpenCV BGR, then to RGB
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Convert depth image based on its encoding
            # Handle different depth image formats and convert to meters
            if depth_msg.encoding == '16UC1':
                # 16-bit unsigned integer, convert to float and scale to meters
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1').astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                # 32-bit float, already in correct format
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1').astype(np.float32)
            else:
                # Handle other encodings with passthrough
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
                if depth_image.dtype == np.uint16:
                    depth_image /= 1000.0  # Convert to meters if needed

        except Exception as e:
            self.get_logger().error(f"❌ Conversion error: {str(e)}")
            return

        # ==================== DATA VALIDATION ====================
        # Check if we have valid depth data (non-zero and reasonable values)
        if np.sum((depth_image > 0) & (depth_image < 10.0)) == 0:
            self.get_logger().warn("No valid depth data found")
            return

        # ==================== OPEN3D SETUP ====================
        # Convert numpy arrays to Open3D image format
        color_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_image)

        # Create Open3D camera intrinsic object with calibration parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=self.camera_width,
            height=self.camera_height,
            fx=self.intrinsic_matrix[0, 0],  # Focal length X
            fy=self.intrinsic_matrix[1, 1],  # Focal length Y
            cx=self.intrinsic_matrix[0, 2],  # Principal point X
            cy=self.intrinsic_matrix[1, 2]   # Principal point Y
        )

        # ==================== RGBD IMAGE CREATION ====================
        # Create RGBD image from color and depth with specified parameters
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,           # Scale factor for depth values
            depth_trunc=self.depth_trunc,           # Maximum depth to consider
            convert_rgb_to_intensity=False          # Keep color information
        )

        # ==================== POINT CLOUD GENERATION ====================
        # Generate 3D point cloud from RGBD image using camera intrinsics
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # Apply voxel downsampling to reduce point density and noise
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # ==================== FILTERING ====================
        # Apply optional radius-based filtering to remove outliers
        if self.enable_radius_filter:
            pcd = self.apply_radius_filter(pcd)
            if len(pcd.points) == 0:
                self.get_logger().warn("No points remaining after radius filtering")
                return

        # Final check for empty point cloud
        if len(pcd.points) == 0:
            self.get_logger().warn("Empty point cloud generated")
            return

        # ==================== PUBLISHING ====================
        # Publish original point cloud in camera frame
        msg_cam = self.convert_o3d_to_ros2_pointcloud2(pcd, "camera_base", rgb_msg.header.stamp)
        self.point_cloud_pub.publish(msg_cam)

        # Transform point cloud to world coordinates and publish
        pcd_transformed = self.apply_world_transform(pcd)
        msg_world = self.convert_o3d_to_ros2_pointcloud2(pcd_transformed, "world", rgb_msg.header.stamp)
        self.world_view_pub.publish(msg_world)

    def apply_radius_filter(self, pcd):
        """
        Apply radius-based filtering to remove points that are too far from the centroid.
        This helps remove outliers and noise from the point cloud.
        
        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud
            
        Returns:
            o3d.geometry.PointCloud: Filtered point cloud
        """
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return pcd

        # Calculate centroid of all points
        centroid = np.mean(points, axis=0)
        
        # Calculate distance from each point to centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Create mask for points within the specified radius
        mask = distances <= self.filter_radius

        # Create new point cloud with filtered points
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points[mask])
        
        # Preserve colors if they exist
        if pcd.has_colors():
            pcd_filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
            
        return pcd_filtered

    def apply_world_transform(self, pcd):
        """
        Transform point cloud from camera coordinates to world coordinates.
        
        The transformation involves:
        1. Converting from camera coordinate system to ROS coordinate system
        2. Applying a 90-degree yaw rotation
        3. Translating to the world position
        
        Camera coordinates: X=right, Y=down, Z=forward
        ROS coordinates: X=forward, Y=left, Z=up
        
        Args:
            pcd (o3d.geometry.PointCloud): Point cloud in camera coordinates
            
        Returns:
            o3d.geometry.PointCloud: Point cloud in world coordinates
        """
        # ==================== ROTATION MATRICES ====================
        # 90-degree rotation around Z-axis (yaw rotation)
        theta = -math.pi / 2  # -90 degrees in radians
        R_yaw_90 = np.array([
            [math.cos(theta), -math.sin(theta), 0, 0],
            [math.sin(theta),  math.cos(theta), 0, 0],
            [0,               0,                1, 0],
            [0,               0,                0, 1]
        ])

        # Transformation from camera coordinate system to ROS coordinate system
        # Camera: X=right, Y=down, Z=forward
        # ROS: X=forward, Y=left, Z=up
        R_cam_to_ros = np.array([
            [0,  0, 1, 0],   # ROS X = Camera Z
            [-1, 0, 0, 0],   # ROS Y = -Camera X  
            [0, -1, 0, 0],   # ROS Z = -Camera Y
            [0,  0, 0, 1]    # Homogeneous coordinate
        ])

        # ==================== TRANSLATION MATRIX ====================
        # Translation matrix to move from camera position to world position
        T_world = np.eye(4)  # 4x4 identity matrix
        T_world[0, 3] = self.world_transform_x  # X translation
        T_world[1, 3] = self.world_transform_y  # Y translation
        T_world[2, 3] = self.world_transform_z  # Z translation

        # ==================== COMBINED TRANSFORMATION ====================
        # Apply transformations in order:
        # 1. Camera to ROS coordinate system
        # 2. 90-degree yaw rotation
        # 3. Translation to world position
        full_transform = T_world @ R_yaw_90 @ R_cam_to_ros

        self.get_logger().info(f"Applying world transform with 90 deg yaw rotation")
        
        # Apply the transformation to the point cloud
        return pcd.transform(full_transform)

    def convert_o3d_to_ros2_pointcloud2(self, o3d_cloud, frame_id="world", timestamp=None):
        """
        Convert Open3D point cloud to ROS2 PointCloud2 message format.
        
        This function creates a properly formatted ROS2 message that can be published
        and visualized in tools like RViz. It handles both geometry (XYZ) and color (RGB) data.
        
        Args:
            o3d_cloud (o3d.geometry.PointCloud): Open3D point cloud
            frame_id (str): TF frame ID for the point cloud
            timestamp: ROS timestamp (uses current time if None)
            
        Returns:
            sensor_msgs.msg.PointCloud2: ROS2 point cloud message
        """
        # ==================== HEADER SETUP ====================
        # Create message header with timestamp and frame information
        header = Header()
        header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
        header.frame_id = frame_id

        # ==================== DATA EXTRACTION ====================
        # Extract points and colors from Open3D point cloud
        points = np.asarray(o3d_cloud.points)
        colors = np.asarray(o3d_cloud.colors) if o3d_cloud.has_colors() else None
        has_colors = colors is not None

        # ==================== FIELD DEFINITIONS ====================
        # Define the structure of each point in the point cloud
        # Each point has X, Y, Z coordinates (and optionally RGB color)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),   # X coordinate
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),   # Y coordinate  
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),   # Z coordinate
        ]
        if has_colors:
            # Add RGB field if colors are available (packed as single 32-bit integer)
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))

        # ==================== DATA PACKING ====================
        # Calculate the size of each point in bytes
        point_step = 16 if has_colors else 12  # 4 bytes × (3 or 4 fields)
        
        # Pack all point data into a byte array
        data = bytearray()
        for i in range(len(points)):
            x, y, z = points[i]
            
            # Skip invalid points (NaN or infinite values)
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
                
            # Pack XYZ coordinates as 32-bit floats
            data.extend(struct.pack('fff', x, y, z))
            
            if has_colors:
                # Convert RGB colors from [0,1] to [0,255] and pack as single integer
                r, g, b = np.clip(colors[i] * 255, 0, 255).astype(np.uint8)
                # Pack RGB as single 32-bit integer: 0x00RRGGBB
                rgb_packed = (int(r) << 16) | (int(g) << 8) | int(b)
                data.extend(struct.pack('I', rgb_packed))

        # ==================== MESSAGE CONSTRUCTION ====================
        # Create and populate the PointCloud2 message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1                    # Unorganized point cloud (height = 1)
        msg.width = len(points)           # Number of points
        msg.fields = fields               # Field definitions
        msg.is_bigendian = False          # Little-endian byte order
        msg.point_step = point_step       # Size of each point in bytes
        msg.row_step = len(data)          # Size of entire point cloud in bytes
        msg.is_dense = False              # May contain invalid points
        msg.data = data                   # Actual point data

        return msg


def main(args=None):
    """
    Main function to initialize and run the ROS2 node.
    
    This function:
    1. Initializes the ROS2 system
    2. Creates the PointCloudPub node
    3. Spins the node to process callbacks
    4. Handles shutdown gracefully
    
    Args:
        args: Command line arguments (optional)
    """
    # Initialize ROS2 system
    rclpy.init(args=args)
    
    try:
        # Create and run the point cloud publisher node
        node = PointCloudPub()
        
        # Spin the node to process incoming messages and callbacks
        # This will continue until interrupted or the node is destroyed
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    except Exception as e:
        # Handle any other exceptions
        print(f"Error: {e}")
    finally:
        # Clean up resources
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
