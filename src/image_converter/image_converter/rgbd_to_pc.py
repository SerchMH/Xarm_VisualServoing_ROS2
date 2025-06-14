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
    def __init__(self):
        super().__init__('point_cloud_publisher')

        self.declare_parameter('enable_radius_filter', False)
        self.declare_parameter('filter_radius', 0.15)
        self.declare_parameter('depth_scale', 1.0)
        self.declare_parameter('depth_trunc', 3.0)
        self.declare_parameter('world_transform_z', 0.07)
        self.declare_parameter('world_transform_x', 0.20)
        self.declare_parameter('world_transform_y', 0.29)
        self.declare_parameter('voxel_size', 0.001)

        self.enable_radius_filter = self.get_parameter('enable_radius_filter').get_parameter_value().bool_value
        self.filter_radius = self.get_parameter('filter_radius').get_parameter_value().double_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.depth_trunc = self.get_parameter('depth_trunc').get_parameter_value().double_value
        self.world_transform_z = self.get_parameter('world_transform_z').get_parameter_value().double_value
        self.world_transform_x = self.get_parameter('world_transform_x').get_parameter_value().double_value
        self.world_transform_y = self.get_parameter('world_transform_y').get_parameter_value().double_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value

        # Initialize static transform broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Publicar transform identidad world -> base_link
        self.publish_base_link_to_world_identity()

        # Publicar transform world -> camera_base (como antes)
        self.publish_static_transforms()

        sensor_qos = qos.QoSProfile(
            reliability=qos.ReliabilityPolicy.BEST_EFFORT,
            durability=qos.DurabilityPolicy.VOLATILE,
            history=qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.depth_sub = message_filters.Subscriber(self, Image, '/cleaned_img/depth_image', qos_profile=sensor_qos)
        self.rgb_sub = message_filters.Subscriber(self, Image, '/cleaned_img/rgb_image', qos_profile=sensor_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.load_camera_intrinsics()
        self.bridge = cv_bridge.CvBridge()

        pointcloud_qos = qos.QoSProfile(
            reliability=qos.ReliabilityPolicy.RELIABLE,
            durability=qos.DurabilityPolicy.VOLATILE,
            history=qos.HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.point_cloud_pub = self.create_publisher(PointCloud2, '/pointcloud/pointcloud_raw', pointcloud_qos)
        self.world_view_pub = self.create_publisher(PointCloud2, '/pointcloud/world_view', pointcloud_qos)

        self.get_logger().info("✅ Point Cloud Publisher READY")

    def publish_base_link_to_world_identity(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'link_base'

        # Identidad: sin traslación ni rotación
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info("📡 Published static transform: world -> base_link (identity)")

    def publish_static_transforms(self):
        """Publish static transforms to define the TF tree"""
        # Transform from camera_base to world
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'camera_base'
        
        # Set translation
        t.transform.translation.x = self.world_transform_x
        t.transform.translation.y = self.world_transform_y
        t.transform.translation.z = self.world_transform_z
        
        # Set rotation (90 degrees around Z-axis)
        # Convert from rotation matrix to quaternion
        theta = -math.pi / 2  # -90 degrees
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(theta / 2)
        t.transform.rotation.w = math.cos(theta / 2)
        
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info("📡 Published static transform: world -> camera_base")

    def load_camera_intrinsics(self):
        yaml_path = '/home/serch/xarm_ws/src/image_converter/camera/kinect_calib.yaml'
        try:
            with open(yaml_path, 'r') as file:
                calib_data = yaml.safe_load(file)
            self.camera_width = calib_data['image_width']
            self.camera_height = calib_data['image_height']
            self.intrinsic_matrix = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
        except Exception as e:
            self.get_logger().error(f"⚠️ Error loading intrinsics: {str(e)}")
            self.camera_width = 640
            self.camera_height = 480
            self.intrinsic_matrix = np.array([
                [525.0, 0.0, 320.0],
                [0.0, 525.0, 240.0],
                [0.0, 0.0, 1.0]
            ])

    def synchronized_callback(self, rgb_msg, depth_msg):
        try:
            self.process_images(rgb_msg, depth_msg)
        except Exception as e:
            self.get_logger().error(f"❌ Error processing images: {str(e)}")

    def process_images(self, rgb_msg, depth_msg):
        if (rgb_msg.width != depth_msg.width or rgb_msg.height != depth_msg.height):
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            if depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1').astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1').astype(np.float32)
            else:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
                if depth_image.dtype == np.uint16:
                    depth_image /= 1000.0

        except Exception as e:
            self.get_logger().error(f"❌ Conversion error: {str(e)}")
            return

        if np.sum((depth_image > 0) & (depth_image < 10.0)) == 0:
            return

        color_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_image)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=self.camera_width,
            height=self.camera_height,
            fx=self.intrinsic_matrix[0, 0],
            fy=self.intrinsic_matrix[1, 1],
            cx=self.intrinsic_matrix[0, 2],
            cy=self.intrinsic_matrix[1, 2]
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        if self.enable_radius_filter:
            pcd = self.apply_radius_filter(pcd)
            if len(pcd.points) == 0:
                return

        if len(pcd.points) == 0:
            return

        # Publicación original (en cámara)
        msg_cam = self.convert_o3d_to_ros2_pointcloud2(pcd, "camera_base", rgb_msg.header.stamp)
        self.point_cloud_pub.publish(msg_cam)

        # Transformación y publicación en world
        pcd_transformed = self.apply_world_transform(pcd)
        msg_world = self.convert_o3d_to_ros2_pointcloud2(pcd_transformed, "world", rgb_msg.header.stamp)
        self.world_view_pub.publish(msg_world)

    def apply_radius_filter(self, pcd):
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return pcd

        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        mask = distances <= self.filter_radius

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points[mask])
        if pcd.has_colors():
            pcd_filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
        return pcd_filtered

    def apply_world_transform(self, pcd):
        # Rotación 90° en yaw (Z)
        theta = -math.pi / 2  # 90 grados en radianes
        R_yaw_90 = np.array([
            [math.cos(theta), -math.sin(theta), 0, 0],
            [math.sin(theta),  math.cos(theta), 0, 0],
            [0,               0,                1, 0],
            [0,               0,                0, 1]
        ])

        R_cam_to_ros = np.array([
            [0,  0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0,  0, 0, 1]
        ])

        T_world = np.eye(4)
        T_world[0, 3] = self.world_transform_x
        T_world[1, 3] = self.world_transform_y
        T_world[2, 3] = self.world_transform_z

        # Primero aplicamos la rotación del frame cámara a ROS
        # Luego aplicamos la rotación de 90° en yaw
        # Finalmente la traslación en world
        full_transform = T_world @ R_yaw_90 @ R_cam_to_ros

        self.get_logger().info(f"Applying world transform with 90 deg yaw rotation")
        return pcd.transform(full_transform)

    def convert_o3d_to_ros2_pointcloud2(self, o3d_cloud, frame_id="world", timestamp=None):
        header = Header()
        header.stamp = timestamp if timestamp else self.get_clock().now().to_msg()
        header.frame_id = frame_id

        points = np.asarray(o3d_cloud.points)
        colors = np.asarray(o3d_cloud.colors) if o3d_cloud.has_colors() else None
        has_colors = colors is not None

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        if has_colors:
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))

        point_step = 16 if has_colors else 12
        data = bytearray()
        for i in range(len(points)):
            x, y, z = points[i]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            data.extend(struct.pack('fff', x, y, z))
            if has_colors:
                r, g, b = np.clip(colors[i] * 255, 0, 255).astype(np.uint8)
                rgb_packed = (int(r) << 16) | (int(g) << 8) | int(b)
                data.extend(struct.pack('I', rgb_packed))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = len(data)
        msg.is_dense = False
        msg.data = data

        return msg


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PointCloudPub()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
