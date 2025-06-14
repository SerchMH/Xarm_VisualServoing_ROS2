import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudToXYPose(Node):
    def __init__(self):
        super().__init__('pointcloud_to_xy_pose')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/pointcloud/world_view',  # Cambia al topic que tienes
            self.pointcloud_callback,
            10
        )

        self.publisher = self.create_publisher(
            PoseStamped,
            '/goal_pose_xy',
            10
        )

        self.get_logger().info("Nodo listo para convertir PointCloud a XY Pose")

    def pointcloud_callback(self, cloud_msg):
        points = []
        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
            points.append([point[0], point[1], point[2]])

        if len(points) == 0:
            self.get_logger().warn("No hay puntos válidos en la nube")
            return

        points_np = np.array(points)

        centroid_x = np.mean(points_np[:, 0])
        centroid_y = np.mean(points_np[:, 1])

        pose = PoseStamped()
        pose.header = cloud_msg.header
        pose.pose.position.x = float(centroid_x)  # <-- conversión explícita
        pose.pose.position.y = float(centroid_y)
        pose.pose.position.z = 0.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.publisher.publish(pose)
        self.get_logger().info(f"Publicado objetivo XY: x={centroid_x:.3f}, y={centroid_y:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToXYPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
