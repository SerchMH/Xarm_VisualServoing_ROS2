import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from rclpy.action import ActionClient

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class XarmXYMover(Node):
    """
    ROS2 Node for controlling xArm6 robot arm movement in XY plane only.
    This node receives XY goal positions and plans trajectories while maintaining
    a fixed Z height (hover mode). Plans are visualized in RViz but not executed.
    """
    
    def __init__(self):
        super().__init__('xarm_xy_mover')

        # Initialize TF2 buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to goal pose topic for receiving XY movement commands
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose_xy',
            self.goal_pose_callback,
            10  # Queue size
        )

        # Create action client for MoveIt motion planning
        self.action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info("Esperando acción move_action...")
        self.action_client.wait_for_server()
        self.get_logger().info("✅ Servidor move_action conectado")

        self.get_logger().info("🚀 Nodo listo para mover el xArm6 en modo hover (solo XY, visualización en RViz)")

    def get_current_pose(self):
        """
        Get the current pose of the end-effector using TF2 transforms.
        
        Returns:
            PoseStamped: Current pose of the end-effector, or None if transform fails
        """
        try:
            # Look up transform from base to end-effector
            trans = self.tf_buffer.lookup_transform(
                'link_base',      # Target frame (base)
                'link_eef',       # Source frame (end-effector)
                rclpy.time.Time() # Use latest available transform
            )
            
            # Convert transform to PoseStamped message
            pose = PoseStamped()
            pose.header = trans.header
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation
            return pose
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"No se pudo obtener pose actual: {e}")
            return None
        
    def goal_pose_callback(self, msg: PoseStamped):
        """
        Callback function for processing incoming XY goal positions.
        
        Args:
            msg (PoseStamped): Goal pose containing target X and Y coordinates
        """
        # Get current pose to maintain orientation and reference Z
        current_pose = self.get_current_pose()
        if current_pose is None:
            self.get_logger().warn("Pose actual no disponible, esperando...")
            return

        self.get_logger().info(f"🎯 Recibido objetivo XY: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}")

        # Create target pose using input XY and fixed Z
        target_pose = PoseStamped()
        target_pose.header = msg.header
        target_pose.pose.position.x = msg.pose.position.x  # Use input X coordinate
        target_pose.pose.position.y = msg.pose.position.y  # Use input Y coordinate
        
        # Force Z to fixed height (30 cm above base) for hover mode
        target_pose.pose.position.z = 0.3  
        
        # Maintain current orientation to avoid unnecessary rotation
        target_pose.pose.orientation = current_pose.pose.orientation

        # Send the target pose for motion planning
        self.send_plan_goal(target_pose)

    def send_plan_goal(self, pose_stamped: PoseStamped):
        """
        Send motion planning request to MoveIt for the given target pose.
        
        Args:
            pose_stamped (PoseStamped): Target pose for the end-effector
        """
        # Create MoveGroup action goal
        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()

        # Specify the planning group (xarm6 robot configuration)
        goal.request.group_name = "xarm6"

        # Set conservative velocity and acceleration limits for safety
        goal.request.max_velocity_scaling_factor = 0.10    # 10% of max velocity
        goal.request.max_acceleration_scaling_factor = 0.10 # 10% of max acceleration

        # Create position constraint for end-effector
        position_constraint = PositionConstraint()
        position_constraint.header = pose_stamped.header
        position_constraint.link_name = "link_eef"  # End-effector link
        
        # Define small box region around target position (1cm tolerance)
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01]  # 1cm x 1cm x 1cm box
        position_constraint.constraint_region.primitives.append(primitive)
        position_constraint.constraint_region.primitive_poses.append(pose_stamped.pose)

        # Create orientation constraint to maintain current orientation
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = pose_stamped.header
        orientation_constraint.link_name = "link_eef"
        orientation_constraint.orientation = pose_stamped.pose.orientation
        # Very tight orientation tolerances (1 milliradians)
        orientation_constraint.absolute_x_axis_tolerance = 0.001
        orientation_constraint.absolute_y_axis_tolerance = 0.001
        orientation_constraint.absolute_z_axis_tolerance = 0.001
        orientation_constraint.weight = 1.0  # Full weight on orientation constraint

        # Combine constraints into goal constraints
        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        goal.request.goal_constraints = [constraints]

        self.get_logger().info("📬 Enviando solicitud de planificación a MoveIt...")

        # Send goal for PLANNING only (not execution)
        send_goal_future = self.action_client.send_goal_async(goal)

        # Set callback for when planning result is received
        send_goal_future.add_done_callback(self.plan_result_callback)

    def plan_result_callback(self, future):
        """
        Callback for handling the initial response to planning request.
        
        Args:
            future: Future object containing the goal handle
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Planificación rechazada")
            return
            
        self.get_logger().info("Planificación aceptada, esperando resultado...")

        # Wait for planning to complete
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.plan_done_callback)

    def plan_done_callback(self, future):
        """
        Callback for handling completed planning results.
        
        Args:
            future: Future object containing the planning result
        """
        result = future.result().result
        
        # Check if planning was successful
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info("Planificación exitosa, visualiza en RViz")
            # NOTE: Trajectory is only visualized in RViz, not executed
            # The planned trajectory can be accessed with: result.trajectory
        else:
            self.get_logger().error(f"Error en la planificación: {result.error_code.val}")


def main(args=None):
    """
    Main function to initialize and run the xArm XY Mover node.
    
    Args:
        args: Command line arguments (optional)
    """
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create and run the node
    node = XarmXYMover()
    try:
        rclpy.spin(node)  # Keep node running
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
