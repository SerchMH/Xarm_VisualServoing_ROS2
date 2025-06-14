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
    def __init__(self):
        super().__init__('xarm_xy_mover')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose_xy',
            self.goal_pose_callback,
            10
        )

        self.action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info("Esperando acción move_action...")
        self.action_client.wait_for_server()
        self.get_logger().info("✅ Servidor move_action conectado")

        self.get_logger().info("🚀 Nodo listo para mover el xArm6 en modo hover (solo XY, visualización en RViz)")

    def get_current_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'link_base',
                'link_eef',
                rclpy.time.Time()
            )
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
        current_pose = self.get_current_pose()
        if current_pose is None:
            self.get_logger().warn("Pose actual no disponible, esperando...")
            return

        self.get_logger().info(f"🎯 Recibido objetivo XY: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}")

        target_pose = PoseStamped()
        target_pose.header = msg.header
        target_pose.pose.position.x = msg.pose.position.x
        target_pose.pose.position.y = msg.pose.position.y
        
        # Aquí forzamos Z a 0.05 m (5 cm)
        target_pose.pose.position.z = 0.3  
        
        # Mantener la orientación actual
        target_pose.pose.orientation = current_pose.pose.orientation

        self.send_plan_goal(target_pose)



    def send_plan_goal(self, pose_stamped: PoseStamped):
        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()

        goal.request.group_name = "xarm6"

        goal.request.max_velocity_scaling_factor = 0.10
        goal.request.max_acceleration_scaling_factor = 0.10

        position_constraint = PositionConstraint()
        position_constraint.header = pose_stamped.header
        position_constraint.link_name = "link_eef"
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01]
        position_constraint.constraint_region.primitives.append(primitive)
        position_constraint.constraint_region.primitive_poses.append(pose_stamped.pose)

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = pose_stamped.header
        orientation_constraint.link_name = "link_eef"
        orientation_constraint.orientation = pose_stamped.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.001
        orientation_constraint.absolute_y_axis_tolerance = 0.001
        orientation_constraint.absolute_z_axis_tolerance = 0.001
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        goal.request.goal_constraints = [constraints]

        self.get_logger().info("📬 Enviando solicitud de planificación a MoveIt...")

        # Enviar el goal para PLANIFICAR (no ejecutar)
        send_goal_future = self.action_client.send_goal_async(goal)

        # Esperar el resultado de planificación y solo visualizar
        send_goal_future.add_done_callback(self.plan_result_callback)

    def plan_result_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Planificación rechazada")
            return
        self.get_logger().info("Planificación aceptada, esperando resultado...")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.plan_done_callback)

    def plan_done_callback(self, future):
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info("Planificación exitosa, visualiza en RViz")
            # Aquí no se llama a execute(), solo se visualiza la trayectoria planificada en RViz
            # Si quieres puedes obtener la trayectoria con:
            # plan_trajectory = result.trajectory
        else:
            self.get_logger().error(f"Error en la planificación: {result.error_code.val}")


def main(args=None):
    rclpy.init(args=args)
    node = XarmXYMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

