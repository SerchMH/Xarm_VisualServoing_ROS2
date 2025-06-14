import rclpy
from rclpy.node import Node
import rclpy.qos as qos
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
import numpy as np
import time

'''
ROS2 Node mejorado que procesa imágenes RGB y de profundidad del Kinect
con máscaras avanzadas para detectar objetos negros. Mantiene los mismos tópicos
pero con algoritmos de detección mucho más robustos.
'''
class Image2RGBD(Node):
    def __init__(self):
        super().__init__('image_to_rgbd_node')

        # Depth and RGB Image Subscriptions
        self.kinect_rgb_subscriber = self.create_subscription(
            Image, '/rgb/image_raw', self.rgb_update, qos.qos_profile_sensor_data)
        self.kinect_depth_subscriber = self.create_subscription(
            Image, '/depth_to_rgb/image_raw', self.depth_update, qos.qos_profile_sensor_data)

        # Image and Mask Publishers (mantenemos los mismos nombres)
        self.mask_publisher = self.create_publisher(
            Image, '/cleaned_img/mask', qos.qos_profile_sensor_data)
        self.mask_rgb_publisher = self.create_publisher(
            Image, '/cleaned_img/rgb_image', qos.qos_profile_sensor_data)
        self.mask_depth_publisher = self.create_publisher(
            Image, '/cleaned_img/depth_image', qos.qos_profile_sensor_data)
        self.depth_mask_publisher = self.create_publisher(
            Image, '/depth_mask', qos.qos_profile_sensor_data)

        # Variables to store images
        self.latest_rgb = None
        self.latest_depth = None

        self.bridge = CvBridge()
        
        # Parámetros mejorados para detección de negros
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([180,255, 70])  # V más bajo para mejor detexcción de negros
        self.lab_threshold = 50  # Threshold para canal L en LAB
        self.gray_threshold = 40  # Threshold para escala de grises
        self.min_component_area = 100  # Área mínima para componentes válidos
        
        # Kernels morfológicos precomputados
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((7, 7), np.uint8)
        
        # Estadísticas
        self.frame_count = 0
        self.detection_count = 0
        self.last_log_time = time.time()

        self.get_logger().info("Enhanced Image Processing Node Started")
        self.get_logger().info("Improved black object detection with multi-modal approach")

    def rgb_update(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Procesar solo cuando ambas imágenes están disponibles
            if self.latest_depth is not None:
                self.process_image()
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_update(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Procesar solo cuando ambas imágenes están disponibles
            if self.latest_rgb is not None:
                self.process_image()
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def create_enhanced_color_mask(self, rgb_image):
        """
        Crea una máscara de color mejorada usando múltiples espacios de color
        para detectar objetos negros de manera más robusta
        """
        try:
            # Método 1: HSV mejorado
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)
            
            # Método 2: LAB - Canal L para luminosidad
            lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
            l_channel = lab_image[:, :, 0]
            mask_lab = (l_channel < self.lab_threshold).astype(np.uint8) * 255
            
            # Método 3: Escala de grises
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            mask_gray = (gray_image < self.gray_threshold).astype(np.uint8) * 255
            
            # Método 4: Detección basada en intensidad RGB
            # Suma de canales RGB baja indica colores oscuros
            rgb_sum = rgb_image[:, :, 0].astype(np.float32) + \
                     rgb_image[:, :, 1].astype(np.float32) + \
                     rgb_image[:, :, 2].astype(np.float32)
            mask_rgb_sum = (rgb_sum < 120).astype(np.uint8) * 255  # Suma baja = negro
            
            # Combinar todas las máscaras
            # Usar OR para capturar más variaciones de negro
            combined_color_mask = cv2.bitwise_or(mask_hsv, mask_lab)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, mask_gray)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, mask_rgb_sum)
            
            return combined_color_mask
            
        except Exception as e:
            self.get_logger().error(f"Error creating enhanced color mask: {e}")
            # Fallback a método original
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 70]))

    def create_enhanced_depth_mask(self, depth_image):
        """
        Crea una máscara de profundidad mejorada con filtrado de ruido
        """
        try:
            depth_img = depth_image.copy()
            
            # Manejar diferentes tipos de datos
            if depth_img.dtype == np.uint16:
                # Convertir de milímetros a metros
                depth_img = depth_img.astype(np.float32) / 1000.0
            
            # Filtrar valores inválidos (NaN, inf, negativos)
            valid_mask = np.isfinite(depth_img) & (depth_img > 0)
            
            # Definir rangos de profundidad en metros
            lower_depth = 0.01  # 1cm
            upper_depth = 1.0   # 50cm
            
            # Crear máscara básica de profundidad
            depth_mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
            valid_depth = valid_mask & (depth_img >= lower_depth) & (depth_img <= upper_depth)
            depth_mask[valid_depth] = 255
            
            # Aplicar filtro mediano para reducir ruido
            depth_mask = cv2.medianBlur(depth_mask, 5)
            
            # Operaciones morfológicas para limpiar la máscara
            # Eliminar puntos aislados
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # Rellenar huecos pequeños
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, self.kernel_medium)
            
            return depth_mask
            
        except Exception as e:
            self.get_logger().error(f"Error creating enhanced depth mask: {e}")
            # Fallback a método original
            depth_img = depth_image.copy()
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0
            
            depth_mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
            valid_depth = (depth_img >= 0.01) & (depth_img <= 0.5) & (depth_img > 0)
            depth_mask[valid_depth] = 255
            return depth_mask

    def apply_advanced_morphology(self, mask):
        """
        Aplica operaciones morfológicas avanzadas para limpiar la máscara
        """
        try:
            # 1. Apertura para eliminar ruido pequeño
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # 2. Cierre para conectar componentes cercanos
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_large)
            
            # 3. Dilatación suave para expandir ligeramente los objetos
            cleaned = cv2.dilate(cleaned, self.kernel_small, iterations=1)
            
            # 4. Erosión para volver al tamaño original pero manteniendo conexiones
            cleaned = cv2.erode(cleaned, self.kernel_small, iterations=1)
            
            return cleaned
            
        except Exception as e:
            self.get_logger().error(f"Error in advanced morphology: {e}")
            return mask

    def filter_components_by_criteria(self, mask):
        """
        Filtra componentes conectados usando múltiples criterios
        """
        try:
            # Análisis de componentes conectados
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, 8, cv2.CV_32S)
            
            if num_labels <= 1:
                return np.zeros_like(mask), 0
            
            # Filtrar componentes por múltiples criterios
            filtered_mask = np.zeros_like(mask)
            valid_components = 0
            
            for i in range(1, num_labels):  # Saltar el fondo (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Criterios de filtrado
                area_valid = area >= self.min_component_area
                aspect_ratio = width / height if height > 0 else 0
                aspect_valid = 0.2 <= aspect_ratio <= 5.0  # Evitar componentes muy alargados
                size_valid = width >= 10 and height >= 10  # Tamaño mínimo
                
                # Si pasa todos los criterios, añadir a la máscara final
                if area_valid and aspect_valid and size_valid:
                    filtered_mask[labels == i] = 255
                    valid_components += 1
            
            return filtered_mask, valid_components
            
        except Exception as e:
            self.get_logger().error(f"Error filtering components: {e}")
            return mask, 0

    def select_best_component(self, mask):
        """
        Selecciona el mejor componente basado en área y posición central
        """
        try:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, 8, cv2.CV_32S)
            
            if num_labels <= 1:
                return np.zeros_like(mask)
            
            # Calcular centro de la imagen
            img_center_x = mask.shape[1] // 2
            img_center_y = mask.shape[0] // 2
            
            best_score = -1
            best_label = -1
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                centroid_x, centroid_y = centroids[i]
                
                # Calcular distancia al centro (normalizada)
                distance_to_center = np.sqrt(
                    ((centroid_x - img_center_x) / img_center_x) ** 2 +
                    ((centroid_y - img_center_y) / img_center_y) ** 2
                )
                
                # Score combinado: área grande + cerca del centro
                area_score = min(area / 1000.0, 1.0)  # Normalizar área
                distance_score = max(0, 1.0 - distance_to_center)  # Mejor si está cerca del centro
                
                combined_score = 0.7 * area_score + 0.3 * distance_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_label = i
            
            # Crear máscara solo con el mejor componente
            if best_label > 0:
                best_mask = np.where(labels == best_label, 255, 0).astype('uint8')
                return best_mask
            else:
                return np.zeros_like(mask)
                
        except Exception as e:
            self.get_logger().error(f"Error selecting best component: {e}")
            # Fallback al método original
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                return np.where(labels == largest_label, 255, 0).astype('uint8')
            return np.zeros_like(mask)

    def process_image(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return
        
        try:
            self.frame_count += 1
            
            # Verificar que las imágenes tengan el mismo tamaño
            if self.latest_rgb.shape[:2] != self.latest_depth.shape[:2]:
                self.get_logger().warn("RGB and depth images have different dimensions")
                return
            
            # === PASO 1: Crear máscara de color mejorada ===
            color_mask = self.create_enhanced_color_mask(self.latest_rgb)
            
            # === PASO 2: Crear máscara de profundidad mejorada ===
            depth_mask = self.create_enhanced_depth_mask(self.latest_depth)
            
            # === PASO 3: Combinar máscaras ===
            combined_mask = cv2.bitwise_and(color_mask, depth_mask)
            
            # Verificar si hay píxeles válidos
            if np.sum(combined_mask) == 0:
                self.get_logger().warn("No valid pixels found in enhanced combined mask")
                return
            
            # === PASO 4: Aplicar morfología avanzada ===
            morphed_mask = self.apply_advanced_morphology(combined_mask)
            
            # === PASO 5: Filtrar componentes por criterios múltiples ===
            filtered_mask, num_components = self.filter_components_by_criteria(morphed_mask)
            
            if num_components == 0:
                self.get_logger().warn("No valid components found after filtering")
                return
            
            # === PASO 6: Seleccionar el mejor componente ===
            if num_components > 1:
                cleaned_mask = self.select_best_component(filtered_mask)
            else:
                cleaned_mask = filtered_mask
            
            # Verificar que tengamos una máscara válida
            if np.sum(cleaned_mask) == 0:
                self.get_logger().warn("Final mask is empty")
                return
            
            self.detection_count += 1
            
            # === PASO 7: Aplicar máscara a las imágenes ===
            masked_color_image = cv2.bitwise_and(self.latest_rgb, self.latest_rgb, mask=cleaned_mask)
            masked_depth_image = self.latest_depth.copy()
            
            # Para la imagen de profundidad, aplicar la máscara de manera apropiada
            if len(masked_depth_image.shape) == 2:
                masked_depth_image[cleaned_mask == 0] = 0
            else:
                masked_depth_image = cv2.bitwise_and(masked_depth_image, masked_depth_image, mask=cleaned_mask)

            # === PASO 8: Convertir a mensajes ROS y publicar ===
            depth_mask_msg = self.bridge.cv2_to_imgmsg(depth_mask, encoding='mono8')
            mask_message = self.bridge.cv2_to_imgmsg(cleaned_mask, encoding="mono8")
            masked_color_message = self.bridge.cv2_to_imgmsg(masked_color_image, encoding="bgr8")
            
            # Para la imagen de profundidad, usar el encoding apropiado
            if self.latest_depth.dtype == np.uint16:
                masked_depth_message = self.bridge.cv2_to_imgmsg(masked_depth_image, encoding="16UC1")
            else:
                masked_depth_message = self.bridge.cv2_to_imgmsg(masked_depth_image, encoding="passthrough")

            # Publicar las imágenes procesadas (mismos tópicos que antes)
            self.depth_mask_publisher.publish(depth_mask_msg)
            self.mask_publisher.publish(mask_message)
            self.mask_rgb_publisher.publish(masked_color_message)
            self.mask_depth_publisher.publish(masked_depth_message)
            
            # Log estadísticas cada 60 frames (aprox cada 2 segundos a 30fps)
            current_time = time.time()
            if self.frame_count % 60 == 0:
                time_elapsed = current_time - self.last_log_time
                fps = 60.0 / time_elapsed if time_elapsed > 0 else 0
                detection_rate = (self.detection_count / self.frame_count) * 100
                
                self.get_logger().info(
                    f"Enhanced processing: {self.frame_count} frames, "
                    f"FPS: {fps:.1f}, Detection rate: {detection_rate:.1f}%, "
                    f"Components detected: {num_components}"
                )
                self.last_log_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error in enhanced process_image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Image2RGBD()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()