import rclpy
from rclpy.node import Node
import rclpy.qos as qos
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
import numpy as np
import time

'''
Enhanced ROS2 Node for processing RGB and depth images from Kinect sensor
with advanced masks for black object detection. Maintains the same topic structure
but implements much more robust detection algorithms using multiple color spaces
and advanced morphological operations.

Main Features:
- Multi-modal color detection (HSV, LAB, Grayscale, RGB sum)
- Enhanced depth filtering with noise reduction
- Advanced morphological operations
- Component filtering with multiple criteria
- Best component selection based on area and position
- Comprehensive error handling and logging
'''

class Image2RGBD(Node):
    def __init__(self):
        """
        Initialize the enhanced image processing node with improved parameters
        and setup all subscribers, publishers, and processing variables.
        """
        super().__init__('image_to_rgbd_node')

        # =================================================================
        # ROS2 SUBSCRIBERS - Input image streams
        # =================================================================
        # Subscribe to RGB image stream from Kinect
        self.kinect_rgb_subscriber = self.create_subscription(
            Image, '/rgb/image_raw', self.rgb_update, qos.qos_profile_sensor_data)
        
        # Subscribe to depth image stream from Kinect (aligned to RGB)
        self.kinect_depth_subscriber = self.create_subscription(
            Image, '/depth_to_rgb/image_raw', self.depth_update, qos.qos_profile_sensor_data)

        # =================================================================
        # ROS2 PUBLISHERS - Output processed images and masks
        # =================================================================
        # Publisher for the final binary mask of detected objects
        self.mask_publisher = self.create_publisher(
            Image, '/cleaned_img/mask', qos.qos_profile_sensor_data)
        
        # Publisher for RGB image with mask applied (only detected objects visible)
        self.mask_rgb_publisher = self.create_publisher(
            Image, '/cleaned_img/rgb_image', qos.qos_profile_sensor_data)
        
        # Publisher for depth image with mask applied
        self.mask_depth_publisher = self.create_publisher(
            Image, '/cleaned_img/depth_image', qos.qos_profile_sensor_data)
        
        # Publisher for depth-based mask (intermediate processing step)
        self.depth_mask_publisher = self.create_publisher(
            Image, '/depth_mask', qos.qos_profile_sensor_data)

        # =================================================================
        # IMAGE STORAGE VARIABLES
        # =================================================================
        self.latest_rgb = None      # Stores the most recent RGB image
        self.latest_depth = None    # Stores the most recent depth image

        # OpenCV-ROS bridge for image format conversion
        self.bridge = CvBridge()
        
        # =================================================================
        # ENHANCED DETECTION PARAMETERS
        # =================================================================
        # HSV color space thresholds for black object detection
        # H: 0-180 (all hues), S: 0-255 (all saturations), V: 0-70 (low brightness)
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([180, 255, 70])  # Lower V threshold for better black detection
        
        # LAB color space threshold - L channel represents lightness
        self.lab_threshold = 50  # Pixels with L < 50 are considered dark
        
        # Grayscale threshold for dark object detection
        self.gray_threshold = 40  # Pixels with intensity < 40 are considered dark
        
        # Minimum area for valid connected components (filters out noise)
        self.min_component_area = 100  # pixels
        
        # =================================================================
        # MORPHOLOGICAL KERNELS - Precomputed for efficiency
        # =================================================================
        # Small kernel for fine noise removal and edge refinement
        self.kernel_small = np.ones((3, 3), np.uint8)
        
        # Medium kernel for moderate morphological operations
        self.kernel_medium = np.ones((5, 5), np.uint8)
        
        # Large kernel for connecting nearby components and filling gaps
        self.kernel_large = np.ones((7, 7), np.uint8)
        
        # =================================================================
        # PERFORMANCE MONITORING VARIABLES
        # =================================================================
        self.frame_count = 0        # Total frames processed
        self.detection_count = 0    # Frames where objects were detected
        self.last_log_time = time.time()  # For FPS calculation

        # Log initialization success
        self.get_logger().info("Enhanced Image Processing Node Started")
        self.get_logger().info("Improved black object detection with multi-modal approach")

    def rgb_update(self, msg):
        """
        Callback function for RGB image updates.
        Stores the latest RGB image and triggers processing when both RGB and depth are available.
        
        Args:
            msg (sensor_msgs.msg.Image): ROS2 Image message containing RGB data
        """
        try:
            # Convert ROS2 Image message to OpenCV format (BGR8)
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process images only when both RGB and depth are available
            # This ensures synchronized processing of both data streams
            if self.latest_depth is not None:
                self.process_image()
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_update(self, msg):
        """
        Callback function for depth image updates.
        Stores the latest depth image and triggers processing when both RGB and depth are available.
        
        Args:
            msg (sensor_msgs.msg.Image): ROS2 Image message containing depth data
        """
        try:
            # Convert ROS2 Image message to OpenCV format (preserve original encoding)
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # Process images only when both RGB and depth are available
            if self.latest_rgb is not None:
                self.process_image()
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def create_enhanced_color_mask(self, rgb_image):
        """
        Creates an enhanced color mask using multiple color spaces for robust black object detection.
        Combines HSV, LAB, Grayscale, and RGB sum methods for comprehensive coverage.
        
        Args:
            rgb_image (numpy.ndarray): Input RGB image in BGR format
            
        Returns:
            numpy.ndarray: Combined binary mask where black objects are white (255)
        """
        try:
            # =================================================================
            # METHOD 1: HSV Color Space Detection
            # =================================================================
            # Convert to HSV color space (Hue, Saturation, Value)
            # HSV is better for color-based filtering than RGB
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)
            
            # =================================================================
            # METHOD 2: LAB Color Space - L Channel (Lightness)
            # =================================================================
            # LAB color space separates lightness from color information
            # L channel is excellent for detecting dark objects regardless of color
            lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
            l_channel = lab_image[:, :, 0]  # Extract lightness channel
            mask_lab = (l_channel < self.lab_threshold).astype(np.uint8) * 255
            
            # =================================================================
            # METHOD 3: Grayscale Intensity
            # =================================================================
            # Simple grayscale conversion for intensity-based detection
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            mask_gray = (gray_image < self.gray_threshold).astype(np.uint8) * 255
            
            # =================================================================
            # METHOD 4: RGB Sum Method
            # =================================================================
            # Sum of RGB channels: low sum indicates dark colors
            # This method catches dark colors that might be missed by other approaches
            rgb_sum = rgb_image[:, :, 0].astype(np.float32) + \
                     rgb_image[:, :, 1].astype(np.float32) + \
                     rgb_image[:, :, 2].astype(np.float32)
            mask_rgb_sum = (rgb_sum < 120).astype(np.uint8) * 255  # Low sum = dark color
            
            # =================================================================
            # COMBINE ALL MASKS USING LOGICAL OR
            # =================================================================
            # Use OR operation to capture all variations of dark/black objects
            # This ensures we don't miss objects that are detected by only one method
            combined_color_mask = cv2.bitwise_or(mask_hsv, mask_lab)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, mask_gray)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, mask_rgb_sum)
            
            return combined_color_mask
            
        except Exception as e:
            self.get_logger().error(f"Error creating enhanced color mask: {e}")
            # Fallback to simple HSV method if enhanced processing fails
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 70]))

    def create_enhanced_depth_mask(self, depth_image):
        """
        Creates an enhanced depth mask with noise reduction and robust filtering.
        Handles different depth data formats and applies morphological cleaning.
        
        Args:
            depth_image (numpy.ndarray): Input depth image (various formats supported)
            
        Returns:
            numpy.ndarray: Binary mask where valid depth regions are white (255)
        """
        try:
            depth_img = depth_image.copy()
            
            # =================================================================
            # HANDLE DIFFERENT DEPTH DATA FORMATS
            # =================================================================
            # Convert from millimeters to meters if using uint16 format
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0
            
            # =================================================================
            # FILTER INVALID DEPTH VALUES
            # =================================================================
            # Remove NaN, infinite, and negative values that indicate sensor errors
            valid_mask = np.isfinite(depth_img) & (depth_img > 0)
            
            # =================================================================
            # DEFINE DEPTH RANGE OF INTEREST
            # =================================================================
            # Focus on objects within reasonable distance (1cm to 100cm)
            lower_depth = 0.01  # 1cm - minimum valid distance
            upper_depth = 1.0   # 100cm - maximum distance of interest
            
            # =================================================================
            # CREATE BASIC DEPTH MASK
            # =================================================================
            depth_mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
            valid_depth = valid_mask & (depth_img >= lower_depth) & (depth_img <= upper_depth)
            depth_mask[valid_depth] = 255
            
            # =================================================================
            # NOISE REDUCTION AND MORPHOLOGICAL CLEANING
            # =================================================================
            # Apply median filter to reduce salt-and-pepper noise
            depth_mask = cv2.medianBlur(depth_mask, 5)
            
            # Opening: Remove small isolated points (noise)
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # Closing: Fill small gaps and holes within objects
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, self.kernel_medium)
            
            return depth_mask
            
        except Exception as e:
            self.get_logger().error(f"Error creating enhanced depth mask: {e}")
            # Fallback to simple depth thresholding
            depth_img = depth_image.copy()
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0
            
            depth_mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
            valid_depth = (depth_img >= 0.01) & (depth_img <= 0.5) & (depth_img > 0)
            depth_mask[valid_depth] = 255
            return depth_mask

    def apply_advanced_morphology(self, mask):
        """
        Applies advanced morphological operations to clean and refine the mask.
        Uses a sequence of operations to remove noise while preserving object shape.
        
        Args:
            mask (numpy.ndarray): Input binary mask to be cleaned
            
        Returns:
            numpy.ndarray: Cleaned binary mask
        """
        try:
            # =================================================================
            # STEP 1: OPENING - Remove small noise and thin connections
            # =================================================================
            # Opening = Erosion followed by Dilation
            # Removes small objects and separates touching objects
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # =================================================================
            # STEP 2: CLOSING - Connect nearby components and fill holes
            # =================================================================
            # Closing = Dilation followed by Erosion
            # Fills small holes and connects nearby objects
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_large)
            
            # =================================================================
            # STEP 3: GENTLE DILATION - Slightly expand objects
            # =================================================================
            # Expand object boundaries to ensure we don't lose edge pixels
            cleaned = cv2.dilate(cleaned, self.kernel_small, iterations=1)
            
            # =================================================================
            # STEP 4: GENTLE EROSION - Return to original size
            # =================================================================
            # Shrink back to original size while maintaining connections made by closing
            cleaned = cv2.erode(cleaned, self.kernel_small, iterations=1)
            
            return cleaned
            
        except Exception as e:
            self.get_logger().error(f"Error in advanced morphology: {e}")
            return mask

    def filter_components_by_criteria(self, mask):
        """
        Filters connected components using multiple criteria to remove invalid detections.
        Analyzes area, aspect ratio, and minimum size to identify valid objects.
        
        Args:
            mask (numpy.ndarray): Input binary mask with potential objects
            
        Returns:
            tuple: (filtered_mask, number_of_valid_components)
        """
        try:
            # =================================================================
            # CONNECTED COMPONENT ANALYSIS
            # =================================================================
            # Find all connected components and their statistics
            # Returns: number of labels, label matrix, statistics, centroids
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, 8, cv2.CV_32S)
            
            # Check if any components were found (label 0 is background)
            if num_labels <= 1:
                return np.zeros_like(mask), 0
            
            # =================================================================
            # INITIALIZE FILTERING VARIABLES
            # =================================================================
            filtered_mask = np.zeros_like(mask)
            valid_components = 0
            
            # =================================================================
            # EVALUATE EACH COMPONENT AGAINST MULTIPLE CRITERIA
            # =================================================================
            for i in range(1, num_labels):  # Skip background (label 0)
                # Extract component statistics
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # =================================================================
                # FILTERING CRITERIA
                # =================================================================
                # Criterion 1: Minimum area requirement
                area_valid = area >= self.min_component_area
                
                # Criterion 2: Reasonable aspect ratio (avoid very elongated shapes)
                aspect_ratio = width / height if height > 0 else 0
                aspect_valid = 0.2 <= aspect_ratio <= 5.0  # Not too thin or too wide
                
                # Criterion 3: Minimum size in both dimensions
                size_valid = width >= 10 and height >= 10  # At least 10x10 pixels
                
                # =================================================================
                # ADD COMPONENT IF IT PASSES ALL CRITERIA
                # =================================================================
                if area_valid and aspect_valid and size_valid:
                    filtered_mask[labels == i] = 255
                    valid_components += 1
            
            return filtered_mask, valid_components
            
        except Exception as e:
            self.get_logger().error(f"Error filtering components: {e}")
            return mask, 0

    def select_best_component(self, mask):
        """
        Selects the best component from multiple candidates based on area and central position.
        Prioritizes larger objects that are closer to the image center.
        
        Args:
            mask (numpy.ndarray): Input mask with multiple components
            
        Returns:
            numpy.ndarray: Mask containing only the best component
        """
        try:
            # =================================================================
            # ANALYZE ALL COMPONENTS
            # =================================================================
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, 8, cv2.CV_32S)
            
            if num_labels <= 1:
                return np.zeros_like(mask)
            
            # =================================================================
            # CALCULATE IMAGE CENTER FOR POSITION SCORING
            # =================================================================
            img_center_x = mask.shape[1] // 2
            img_center_y = mask.shape[0] // 2
            
            # =================================================================
            # EVALUATE EACH COMPONENT WITH COMBINED SCORING
            # =================================================================
            best_score = -1
            best_label = -1
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                centroid_x, centroid_y = centroids[i]
                
                # =================================================================
                # CALCULATE DISTANCE TO CENTER (NORMALIZED)
                # =================================================================
                distance_to_center = np.sqrt(
                    ((centroid_x - img_center_x) / img_center_x) ** 2 +
                    ((centroid_y - img_center_y) / img_center_y) ** 2
                )
                
                # =================================================================
                # COMBINED SCORING SYSTEM
                # =================================================================
                # Area score: Larger objects get higher scores (normalized to 0-1)
                area_score = min(area / 1000.0, 1.0)
                
                # Position score: Objects closer to center get higher scores
                distance_score = max(0, 1.0 - distance_to_center)
                
                # Weighted combination: 70% area importance, 30% position importance
                combined_score = 0.7 * area_score + 0.3 * distance_score
                
                # Update best component if this one scores higher
                if combined_score > best_score:
                    best_score = combined_score
                    best_label = i
            
            # =================================================================
            # CREATE MASK WITH ONLY THE BEST COMPONENT
            # =================================================================
            if best_label > 0:
                best_mask = np.where(labels == best_label, 255, 0).astype('uint8')
                return best_mask
            else:
                return np.zeros_like(mask)
                
        except Exception as e:
            self.get_logger().error(f"Error selecting best component: {e}")
            # Fallback to largest component selection
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            if num_labels > 1:
                # Find the largest component (excluding background)
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                return np.where(labels == largest_label, 255, 0).astype('uint8')
            return np.zeros_like(mask)

    def process_image(self):
        """
        Main image processing pipeline that combines all detection methods.
        Processes RGB and depth images to create clean masks of detected objects.
        
        This method orchestrates the entire detection pipeline:
        1. Enhanced color mask creation
        2. Enhanced depth mask creation  
        3. Mask combination
        4. Morphological cleaning
        5. Component filtering
        6. Best component selection
        7. Result publishing
        """
        # =================================================================
        # VALIDATE INPUT IMAGES
        # =================================================================
        if self.latest_rgb is None or self.latest_depth is None:
            return
        
        try:
            self.frame_count += 1
            
            # Ensure RGB and depth images have matching dimensions
            if self.latest_rgb.shape[:2] != self.latest_depth.shape[:2]:
                self.get_logger().warn("RGB and depth images have different dimensions")
                return
            
            # =================================================================
            # STEP 1: CREATE ENHANCED COLOR MASK
            # =================================================================
            # Use multi-modal approach for robust color-based detection
            color_mask = self.create_enhanced_color_mask(self.latest_rgb)
            
            # =================================================================
            # STEP 2: CREATE ENHANCED DEPTH MASK
            # =================================================================
            # Filter depth image for valid range and reduce noise
            depth_mask = self.create_enhanced_depth_mask(self.latest_depth)
            
            # =================================================================
            # STEP 3: COMBINE COLOR AND DEPTH MASKS
            # =================================================================
            # Use AND operation: objects must satisfy both color AND depth criteria
            combined_mask = cv2.bitwise_and(color_mask, depth_mask)
            
            # Early exit if no valid pixels found
            if np.sum(combined_mask) == 0:
                self.get_logger().warn("No valid pixels found in enhanced combined mask")
                return
            
            # =================================================================
            # STEP 4: APPLY ADVANCED MORPHOLOGICAL OPERATIONS
            # =================================================================
            # Clean the mask using sophisticated morphological operations
            morphed_mask = self.apply_advanced_morphology(combined_mask)
            
            # =================================================================
            # STEP 5: FILTER COMPONENTS BY MULTIPLE CRITERIA
            # =================================================================
            # Remove invalid components based on size, shape, and area
            filtered_mask, num_components = self.filter_components_by_criteria(morphed_mask)
            
            if num_components == 0:
                self.get_logger().warn("No valid components found after filtering")
                return
            
            # =================================================================
            # STEP 6: SELECT BEST COMPONENT (IF MULTIPLE EXIST)
            # =================================================================
            # Choose the most promising component based on area and position
            if num_components > 1:
                cleaned_mask = self.select_best_component(filtered_mask)
            else:
                cleaned_mask = filtered_mask
            
            # Final validation of the resulting mask
            if np.sum(cleaned_mask) == 0:
                self.get_logger().warn("Final mask is empty")
                return
            
            # Update detection statistics
            self.detection_count += 1
            
            # =================================================================
            # STEP 7: APPLY FINAL MASK TO INPUT IMAGES
            # =================================================================
            # Create masked versions of both RGB and depth images
            
            # Apply mask to RGB image (shows only detected objects in color)
            masked_color_image = cv2.bitwise_and(self.latest_rgb, self.latest_rgb, mask=cleaned_mask)
            
            # Apply mask to depth image
            masked_depth_image = self.latest_depth.copy()
            if len(masked_depth_image.shape) == 2:
                # For 2D depth images, directly zero out non-mask areas
                masked_depth_image[cleaned_mask == 0] = 0
            else:
                # For 3D depth images, use bitwise AND
                masked_depth_image = cv2.bitwise_and(masked_depth_image, masked_depth_image, mask=cleaned_mask)

            # =================================================================
            # STEP 8: CONVERT TO ROS2 MESSAGES AND PUBLISH
            # =================================================================
            
            # Convert intermediate depth mask for debugging/visualization
            depth_mask_msg = self.bridge.cv2_to_imgmsg(depth_mask, encoding='mono8')
            
            # Convert final binary mask
            mask_message = self.bridge.cv2_to_imgmsg(cleaned_mask, encoding="mono8")
            
            # Convert masked RGB image
            masked_color_message = self.bridge.cv2_to_imgmsg(masked_color_image, encoding="bgr8")
            
            # Convert masked depth image with appropriate encoding
            if self.latest_depth.dtype == np.uint16:
                masked_depth_message = self.bridge.cv2_to_imgmsg(masked_depth_image, encoding="16UC1")
            else:
                masked_depth_message = self.bridge.cv2_to_imgmsg(masked_depth_image, encoding="passthrough")

            # =================================================================
            # PUBLISH ALL PROCESSED IMAGES
            # =================================================================
            # Maintain same topic names as original implementation for compatibility
            self.depth_mask_publisher.publish(depth_mask_msg)        # Intermediate depth mask
            self.mask_publisher.publish(mask_message)                # Final binary mask
            self.mask_rgb_publisher.publish(masked_color_message)    # Masked RGB image
            self.mask_depth_publisher.publish(masked_depth_message) # Masked depth image
            
            # =================================================================
            # PERFORMANCE LOGGING
            # =================================================================
            # Log statistics every 60 frames (approximately every 2 seconds at 30fps)
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
    """
    Main entry point for the ROS2 node.
    Initializes ROS2, creates the node instance, and handles the execution loop.
    
    Args:
        args: Command line arguments (optional)
    """
    # Initialize ROS2 communication
    rclpy.init(args=args)
    
    # Create the enhanced image processing node
    node = Image2RGBD()
    
    try:
        # Run the node (blocking call that processes callbacks)
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        pass
    finally:
        # Clean up resources
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
