# Import required libraries from ROS2 and Python
import rclpy  # ROS2 Python client library
from rclpy.node import Node  # Base class to create a ROS2 node
from sensor_msgs.msg import PointCloud2  # Standard message type for 3D point clouds
from geometry_msgs.msg import PoseStamped  # Standard message type for stamped poses
import numpy as np  # Used for numerical operations like computing the mean
import sensor_msgs_py.point_cloud2 as pc2  # Utilities to read PointCloud2 messages in Python

# Define a custom node that converts a PointCloud2 message to an XY goal pose
class PointCloudToXYPose(Node):
    def __init__(self):
        super().__init__('pointcloud_to_xy_pose')  # Initialize the node with a name

        # Subscribe to the point cloud topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/pointcloud/world_view',  # Topic name (change to match your setup)
            self.pointcloud_callback,  # Callback function triggered on new message
            10  # Queue size
        )

        # Publisher to output the XY goal pose
        self.publisher = self.create_publisher(
            PoseStamped,
            '/goal_pose_xy',  # Output topic to publish pose
            10
        )

        self.last_centroid = None  # Store the last published centroid
        self.threshold = 0.01  # Minimum distance change (1 cm) to trigger a new publication

        self.get_logger().info("🟢 Node active: Will only publish if the centroid changes significantly")

    # Callback function triggered on receiving a point cloud message
    def pointcloud_callback(self, cloud_msg):
        points = []

        # Extract x, y, z coordinates from point cloud, skipping NaNs
        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
            points.append([point[0], point[1], point[2]])

        # Warn and return if no valid points are found
        if len(points) == 0:
            self.get_logger().warn("⚠️ No valid points in the point cloud")
            return

        # Convert list of points to a NumPy array
        points_np = np.array(points)

        # Compute the centroid in the XY plane
        centroid_x = float(np.mean(points_np[:, 0]))
        centroid_y = float(np.mean(points_np[:, 1]))

        # Check if the change in centroid is significant
        if self.last_centroid:
            dx = abs(centroid_x - self.last_centroid[0])
            dy = abs(centroid_y - self.last_centroid[1])
            if dx < self.threshold and dy < self.threshold:
                # If movement is below the threshold, do not publish
                return

        # Update the last centroid
        self.last_centroid = (centroid_x, centroid_y)

        # Create and populate a PoseStamped message
        pose = PoseStamped()
        pose.header = cloud_msg.header  # Keep the same timestamp and frame as the point cloud
        pose.pose.position.x = centroid_x
        pose.pose.position.y = centroid_y
        pose.pose.position.z = 0.0  # Z is fixed at 0 for XY navigation

        # Set orientation to identity (no rotation)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        # Publish the pose
        self.publisher.publish(pose)
        self.get_logger().info(f"📤 Published XY goal: x={centroid_x:.3f}, y={centroid_y:.3f}")

# Entry point for the ROS2 node
def main(args=None):
    rclpy.init(args=args)  # Initialize ROS2
    node = PointCloudToXYPose()  # Create the node

    try:
        rclpy.spin(node)  # Keep the node running
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
    finally:
        node.destroy_node()  # Clean up the node
        rclpy.shutdown()  # Shutdown ROS2

# Run the main function when executed as a script
if __name__ == '__main__':
    main()
