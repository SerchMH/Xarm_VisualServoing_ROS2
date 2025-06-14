import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import numpy as np
import open3d as o3d
import copy
import time
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


class PointCloudAligner(Node):
    """
    A ROS2 Node that aligns a complete object model to a segmented scan
    using advanced multi-stage alignment with robust error handling.
    """

    def __init__(self):
        super().__init__('pointcloud_aligner')

        # Declare parameters
        self.declare_parameter("model_path", "/home/serch/xarm_ws/src/pose_estimator/point_clouds/oil_pan_full_pc_10000.ply")
        self.declare_parameter("input_topic", "/pointcloud/world_view")
        self.declare_parameter("output_topic", "/aligned_model")
        self.declare_parameter("voxel_size", 0.001)
        self.declare_parameter("enable_scale_estimation", True)
        self.declare_parameter("enable_pca_alignment", True)
        self.declare_parameter("enable_iterative_refinement", True)
        # NUEVO: Parámetros para control de errores
        self.declare_parameter("min_points_threshold", 100)
        self.declare_parameter("max_voxel_size", 1.0)
        self.declare_parameter("min_voxel_size", 0.001)

        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.voxel_size = self.get_parameter("voxel_size").get_parameter_value().double_value
        self.enable_scale_estimation = self.get_parameter("enable_scale_estimation").get_parameter_value().bool_value
        self.enable_pca_alignment = self.get_parameter("enable_pca_alignment").get_parameter_value().bool_value
        self.enable_iterative_refinement = self.get_parameter("enable_iterative_refinement").get_parameter_value().bool_value
        self.min_points_threshold = self.get_parameter("min_points_threshold").get_parameter_value().integer_value
        self.max_voxel_size = self.get_parameter("max_voxel_size").get_parameter_value().double_value
        self.min_voxel_size = self.get_parameter("min_voxel_size").get_parameter_value().double_value


        # Load model and preprocess
        self.get_logger().info(f"Loading complete model from: {self.model_path}")
        
        if self.load_and_prepare_model():
            self.model_features = self.extract_geometric_features(self.complete_model)
            self.publisher_ = self.create_publisher(PointCloud2, self.output_topic, 10)
            self.subscription = self.create_subscription(PointCloud2, self.input_topic, self.pointcloud_callback, 10)
            self.get_logger().info(f"✅ Node initialized successfully!")
        else:
            self.get_logger().error("❌ Failed to initialize. Check model path.")

    def load_and_prepare_model(self):
        """Load and prepare the reference model with optional caching."""
        try:
            cache_path = self.model_path.replace('.ply', '_preprocessed.pcd')
            
            if o3d.io.read_point_cloud(cache_path).has_points():
                self.get_logger().info(f"📦 Loading cached preprocessed model from {cache_path}")
                self.complete_model = o3d.io.read_point_cloud(cache_path)
                return True

            self.get_logger().info("📂 Loading original model and preprocessing...")
            model = o3d.io.read_point_cloud(self.model_path)
            
            if len(model.points) == 0:
                self.get_logger().error("❌ Model file is empty or invalid!")
                return False
            
            model_clean, _ = model.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            if len(model_clean.points) < self.min_points_threshold:
                model_clean = model

            model_clean.scale(0.005, center=(0, 0, 0))
            model_clean.translate(-model_clean.get_center())
            model_clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

            self.complete_model = model_clean
            o3d.io.write_point_cloud(cache_path, model_clean)
            self.get_logger().info(f"✅ Model preprocessed and saved to {cache_path}")
            return True

        except Exception as e:
            self.get_logger().error(f"❌ Error loading model: {str(e)}")
            return False


    def extract_geometric_features(self, pcd):
        """Extract comprehensive geometric features from point cloud."""
        if pcd is None or len(pcd.points) == 0:
            return None
            
        try:
            points = np.asarray(pcd.points)
            
            features = {
                'center': np.mean(points, axis=0),
                'bbox_size': pcd.get_max_bound() - pcd.get_min_bound(),
                'principal_axes': self.compute_principal_axes(points),
                'surface_area': self.estimate_surface_area(pcd),
                'volume': self.estimate_volume(pcd)
            }
            return features
        except Exception as e:
            self.get_logger().warn(f"⚠️ Feature extraction failed: {str(e)}")
            return None

    def compute_principal_axes(self, points):
        """Compute principal axes using PCA with validation."""
        if len(points) < 3:
            return np.eye(3)
        
        try:
            centered_points = points - np.mean(points, axis=0)
            pca = PCA(n_components=4)
            pca.fit(centered_points)
            explained_var = pca.explained_variance_ratio_
            if np.any(explained_var < 0.01):
                self.get_logger().warn("⚠️ Low variance in one PCA axis – may indicate noise or flat surface")

            return pca.components_
        except:
            return np.eye(3)

    def estimate_surface_area(self, pcd):
        """Estimate surface area using mesh approximation."""
        try:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            return mesh.get_surface_area()
        except:
            return 0.0

    def estimate_volume(self, pcd):
        """Estimate volume using convex hull."""
        try:
            hull, _ = pcd.compute_convex_hull()
            return hull.get_volume()
        except:
            return 0.0

    def preprocess_point_cloud(self, pcd, adaptive_voxel=True):
        """Advanced preprocessing optimized for large point clouds."""
        if pcd is None or len(pcd.points) == 0:
            self.get_logger().error("❌ Cannot preprocess empty point cloud")
            return None, None, self.voxel_size

        original_count = len(pcd.points)
        self.get_logger().info(f"🔧 Preprocessing {original_count} points")

        try:
            # Step 1: Early aggressive downsampling for very large clouds
            if original_count > 100000:  # More than 100k points
                self.get_logger().info(f"🔥 Large cloud detected, applying early downsampling")
                # Apply initial coarse downsampling
                early_voxel_size = self.voxel_size * 3.0
                pcd = pcd.voxel_down_sample(early_voxel_size)
                self.get_logger().info(f"📉 Early downsampling: {len(pcd.points)} points remaining")

            # Step 2: Calculate adaptive voxel size
            if adaptive_voxel:
                bbox = pcd.get_max_bound() - pcd.get_min_bound()
                bbox_diagonal = np.linalg.norm(bbox)
                
                if bbox_diagonal > 0:
                    # Target 1000-3000 points for efficiency
                    target_points = min(2000, max(500, len(pcd.points) // 5))
                    estimated_voxel = bbox_diagonal / (target_points ** (1/3)) * 0.3
                    
                    adaptive_voxel_size = max(estimated_voxel, self.voxel_size)
                    adaptive_voxel_size = np.clip(adaptive_voxel_size, 
                                                self.min_voxel_size, 
                                                bbox_diagonal * 0.05)  # More aggressive max limit
                else:
                    adaptive_voxel_size = self.voxel_size
            else:
                adaptive_voxel_size = self.voxel_size

            self.get_logger().info(f"📐 Adaptive voxel size: {adaptive_voxel_size:.6f}")

            # Step 3: Statistical outlier removal (only for manageable sizes)
            pcd_clean = pcd
            if len(pcd.points) <= 50000:  # Only clean if not too large
                try:
                    pcd_temp, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    if len(pcd_temp.points) >= max(50, len(pcd.points) * 0.3):
                        pcd_clean = pcd_temp
                        self.get_logger().info(f"🧹 Outlier removal: {len(pcd_clean.points)} points remaining")
                except Exception as e:
                    self.get_logger().warn(f"⚠️ Outlier removal failed: {str(e)}")

            # Step 4: Progressive downsampling with validation
            down = None
            current_voxel_size = adaptive_voxel_size
            attempts = 0
            max_attempts = 4  # Reduced attempts for efficiency
            
            while attempts < max_attempts:
                try:
                    down = pcd_clean.voxel_down_sample(current_voxel_size)
                    points_after = len(down.points)
                    
                    self.get_logger().info(f"🔄 Downsampling attempt {attempts + 1}: "
                                        f"voxel={current_voxel_size:.6f}, points={points_after}")
                    
                    if points_after >= self.min_points_threshold:
                        break
                    elif points_after >= 20:  # Lower threshold for acceptance
                        self.get_logger().warn(f"⚠️ Low point count ({points_after}) but acceptable")
                        break
                    else:
                        current_voxel_size *= 0.6  # More aggressive reduction
                        attempts += 1
                        
                except Exception as e:
                    self.get_logger().error(f"❌ Downsampling attempt {attempts + 1} failed: {str(e)}")
                    current_voxel_size *= 0.7
                    attempts += 1

            # Step 5: Fallback sampling if downsampling failed
            if down is None or len(down.points) < 10:
                self.get_logger().warn(f"🆘 Downsampling failed, using uniform sampling")
                try:
                    # Calculate step size for uniform sampling
                    target_points = max(self.min_points_threshold, 100)
                    step = max(1, len(pcd_clean.points) // target_points)
                    indices = np.arange(0, len(pcd_clean.points), step)
                    down = pcd_clean.select_by_index(indices)
                    current_voxel_size = self.voxel_size
                    self.get_logger().info(f"✅ Uniform sampling: {len(down.points)} points")
                except Exception as e:
                    self.get_logger().error(f"❌ Uniform sampling failed: {str(e)}")
                    return None, None, current_voxel_size

            # Step 6: Normal estimation with adaptive parameters
            try:
                # Use smaller search radius for large clouds
                search_radius = current_voxel_size * (2.0 if len(down.points) > 1000 else 3.0)
                max_nn = min(30, max(10, len(down.points) // 50))  # Adaptive max neighbors
                
                down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn))
            except Exception as e:
                self.get_logger().warn(f"⚠️ Normal estimation failed: {str(e)}")

            # Step 7: FPFH feature computation with adaptive parameters
            fpfh = None
            try:
                fpfh_radius = current_voxel_size * (5.0 if len(down.points) > 1000 else 10.0)
                max_nn_fpfh = min(100, max(50, len(down.points) // 20))  # Adaptive max neighbors
                
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    down,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=max_nn_fpfh))
                
                self.get_logger().info(f"🎯 FPFH features: {fpfh.data.shape[1]}")
                
                # Quality check
                if fpfh.data.shape[1] < 10:
                    self.get_logger().warn(f"⚠️ Very few FPFH features, trying larger radius")
                    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                        down,
                        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=max_nn_fpfh * 2))
                    
            except Exception as e:
                self.get_logger().error(f"❌ FPFH computation failed: {str(e)}")
                return None, None, current_voxel_size

            # Final validation
            if fpfh is None or fpfh.data.shape[1] < 10:
                self.get_logger().error("❌ Insufficient features for registration")
                return None, None, current_voxel_size

            efficiency_ratio = len(down.points) / original_count
            self.get_logger().info(f"✅ Preprocessing complete: {len(down.points)} points "
                                f"({efficiency_ratio:.1%} of original), {fpfh.data.shape[1]} features")
            
            return down, fpfh, current_voxel_size

        except Exception as e:
            self.get_logger().error(f"❌ Preprocessing failed: {str(e)}")
            return None, None, self.voxel_size

    def ros_to_open3d(self, msg):
        """Convert ROS2 PointCloud2 message to Open3D point cloud with validation."""
        try:
            points = np.array([
                [x, y, z]
                for x, y, z in point_cloud2.read_points(msg, field_names=("x","y","z"), skip_nans=True)
            ])
            
            if len(points) == 0:
                self.get_logger().warn("⚠️ Received empty point cloud!")
                return None
                
            if len(points) < self.min_points_threshold:
                self.get_logger().warn(f"⚠️ Too few points received: {len(points)}")
                return None
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Remove obvious outliers
            pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            if len(pcd_clean.points) >= self.min_points_threshold:
                return pcd_clean
            else:
                self.get_logger().warn("⚠️ Outlier removal too aggressive, using original")
                return pcd
                
        except Exception as e:
            self.get_logger().error(f"❌ ROS to Open3D conversion failed: {str(e)}")
            return None

    def open3d_to_ros(self, cloud, header):
        """Convert Open3D PointCloud to ROS2 PointCloud2."""
        if cloud is None or len(cloud.points) == 0:
            return None
            
        try:
            points = np.asarray(cloud.points).astype(np.float32)

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            return point_cloud2.create_cloud(header, fields, points)
        except Exception as e:
            self.get_logger().error(f"❌ Open3D to ROS conversion failed: {str(e)}")
            return None

    def estimate_scale_factor(self, model, target):
        """Estimate optimal scale factor using memory-efficient robust methods."""
        if not self.enable_scale_estimation or model is None or target is None:
            return 1.0
            
        try:
            model_points = np.asarray(model.points)
            target_points = np.asarray(target.points)
            
            if len(model_points) < 10 or len(target_points) < 10:
                self.get_logger().warn("⚠️ Too few points for scale estimation")
                return 1.0
            
            # MEMORY-EFFICIENT APPROACH: Sample points to avoid memory issues
            max_points_for_analysis = 2000  # Reasonable limit for distance calculations
            
            # Sample points if datasets are too large
            if len(model_points) > max_points_for_analysis:
                indices = np.random.choice(len(model_points), max_points_for_analysis, replace=False)
                model_sample = model_points[indices]
            else:
                model_sample = model_points
                
            if len(target_points) > max_points_for_analysis:
                indices = np.random.choice(len(target_points), max_points_for_analysis, replace=False)
                target_sample = target_points[indices]
            else:
                target_sample = target_points
                
            self.get_logger().info(f"📊 Scale analysis using {len(model_sample)} model points, {len(target_sample)} target points")
            
            # Method 1: Statistical distance analysis (memory-efficient)
            statistical_scale = self._estimate_scale_from_distances(model_sample, target_sample)
            
            # Method 2: Bounding box analysis
            model_bbox = np.max(model_points, axis=0) - np.min(model_points, axis=0)
            target_bbox = np.max(target_points, axis=0) - np.min(target_points, axis=0)
            
            if np.any(model_bbox <= 1e-6) or np.any(target_bbox <= 1e-6):
                bbox_scale = statistical_scale
            else:
                # Use median of dimension ratios to be robust against incomplete scans
                bbox_ratios = target_bbox / model_bbox
                # Filter out extreme ratios that might indicate missing dimensions
                valid_ratios = bbox_ratios[(bbox_ratios > 0.1) & (bbox_ratios < 10.0)]
                if len(valid_ratios) > 0:
                    bbox_scale = np.median(valid_ratios)
                else:
                    bbox_scale = statistical_scale
            
            # Method 3: Centroid-based distance analysis
            centroid_scale = self._estimate_scale_from_centroids(model_points, target_points)
            
            # Method 4: PCA-based dimension analysis
            pca_scale = self._estimate_scale_from_pca(model_sample, target_sample)
            
            # Combine methods with weights
            scales = [statistical_scale, bbox_scale, centroid_scale, pca_scale]
            weights = [0.4, 0.3, 0.2, 0.1]  # Give more weight to statistical method
            
            # Filter out extreme values
            valid_scales = [(s, w) for s, w in zip(scales, weights) if 0.1 <= s <= 10.0]
            
            if not valid_scales:
                self.get_logger().warn("⚠️ All scale estimates are extreme, using 1.0")
                return 1.0
            
            # Calculate weighted average of valid scales
            scales_only, weights_only = zip(*valid_scales)
            final_scale = np.average(scales_only, weights=weights_only)
            final_scale = np.clip(final_scale, 0.1, 10.0)
            
            self.get_logger().info(f"📏 Scale estimates - Statistical: {statistical_scale:.3f}, "
                                f"BBox: {bbox_scale:.3f}, Centroid: {centroid_scale:.3f}, PCA: {pca_scale:.3f}")
            self.get_logger().info(f"📏 Final scale factor: {final_scale:.3f}")
            
            return final_scale
            
        except Exception as e:
            self.get_logger().error(f"❌ Scale estimation failed: {str(e)}")
            return 1.0

    def _estimate_scale_from_distances(self, model_sample, target_sample):
        """Memory-efficient distance-based scale estimation."""
        try:
            # Use smaller random samples for distance calculations
            n_samples = min(500, len(model_sample), len(target_sample))
            
            model_indices = np.random.choice(len(model_sample), n_samples, replace=False)
            target_indices = np.random.choice(len(target_sample), n_samples, replace=False)
            
            model_subset = model_sample[model_indices]
            target_subset = target_sample[target_indices]
            
            # Calculate pairwise distances for subsets
            model_distances = cdist(model_subset, model_subset)
            target_distances = cdist(target_subset, target_subset)
            
            # Get non-zero distances and use percentiles
            model_nonzero = model_distances[model_distances > 1e-6]
            target_nonzero = target_distances[target_distances > 1e-6]
            
            if len(model_nonzero) == 0 or len(target_nonzero) == 0:
                return 1.0
            
            # Use multiple percentiles for robustness
            percentiles = [25, 50, 75, 90]
            model_percs = np.percentile(model_nonzero, percentiles)
            target_percs = np.percentile(target_nonzero, percentiles)
            
            scale_ratios = target_percs / (model_percs + 1e-8)
            return np.median(scale_ratios)
            
        except Exception as e:
            self.get_logger().warn(f"⚠️ Distance-based scale estimation failed: {str(e)}")
            return 1.0

    def _estimate_scale_from_centroids(self, model_points, target_points):
        """Scale estimation based on centroid distances."""
        try:
            model_center = np.mean(model_points, axis=0)
            target_center = np.mean(target_points, axis=0)
            
            # Calculate average distance from centroid
            model_centroid_dists = np.linalg.norm(model_points - model_center, axis=1)
            target_centroid_dists = np.linalg.norm(target_points - target_center, axis=1)
            
            model_avg_dist = np.mean(model_centroid_dists)
            target_avg_dist = np.mean(target_centroid_dists)
            
            if model_avg_dist > 1e-6:
                return target_avg_dist / model_avg_dist
            else:
                return 1.0
                
        except Exception as e:
            self.get_logger().warn(f"⚠️ Centroid-based scale estimation failed: {str(e)}")
            return 1.0

    def _estimate_scale_from_pca(self, model_sample, target_sample):
        """Scale estimation using PCA eigenvalues."""
        try:
            model_centered = model_sample - np.mean(model_sample, axis=0)
            target_centered = target_sample - np.mean(target_sample, axis=0)
            
            # Compute covariance matrices
            model_cov = np.cov(model_centered.T)
            target_cov = np.cov(target_centered.T)
            
            # Get eigenvalues
            model_eigenvals = np.linalg.eigvals(model_cov)
            target_eigenvals = np.linalg.eigvals(target_cov)
            
            # Sort eigenvalues in descending order
            model_eigenvals = np.sort(model_eigenvals)[::-1]
            target_eigenvals = np.sort(target_eigenvals)[::-1]
            
            # Calculate scale based on the ratio of dominant eigenvalues
            valid_ratios = []
            for i in range(min(len(model_eigenvals), len(target_eigenvals))):
                if model_eigenvals[i] > 1e-8:
                    ratio = np.sqrt(target_eigenvals[i] / model_eigenvals[i])
                    if 0.1 <= ratio <= 10.0:  # Filter extreme ratios
                        valid_ratios.append(ratio)
            
            if valid_ratios:
                return np.median(valid_ratios)
            else:
                return 1.0
                
        except Exception as e:
            self.get_logger().warn(f"⚠️ PCA-based scale estimation failed: {str(e)}")
            return 1.0
    def validate_scale_factor(self, model, target, scale_factor):
        """Validate if the scale factor makes sense geometrically."""
        try:
            # Crear modelo escalado temporalmente
            temp_model = copy.deepcopy(model)
            temp_model.scale(scale_factor, center=(0, 0, 0))
            
            # Comparar características geométricas
            model_bbox = temp_model.get_max_bound() - temp_model.get_min_bound()
            target_bbox = target.get_max_bound() - target.get_min_bound()
            
            # Verificar que las dimensiones escaladas sean razonables
            dimension_ratios = model_bbox / (target_bbox + 1e-8)
            
            # Si alguna dimensión difiere por más de 3x, la escala probablemente está mal
            if np.any(dimension_ratios > 3.0) or np.any(dimension_ratios < 0.33):
                self.get_logger().warn(f"⚠️ Scale validation failed: dimension ratios {dimension_ratios}")
                return False
            
            return True
            
        except Exception as e:
            self.get_logger().warn(f"⚠️ Scale validation error: {str(e)}")
            return False
        
    def align_with_pca(self, model, target):
        """Align using PCA-based orientation matching."""
        if not self.enable_pca_alignment or model is None or target is None:
            return np.eye(4)
            
        try:
            model_axes = self.compute_principal_axes(np.asarray(model.points))
            target_axes = self.compute_principal_axes(np.asarray(target.points))
            
            rotation_matrix = target_axes @ model_axes.T
            
            # Ensure proper rotation matrix
            U, _, Vt = np.linalg.svd(rotation_matrix)
            rotation_matrix = U @ Vt
            
            # Handle reflection
            if np.linalg.det(rotation_matrix) < 0:
                Vt[-1, :] *= -1
                rotation_matrix = U @ Vt
            
            transformation = np.eye(4)
            transformation[:3, :3] = rotation_matrix
            
            return transformation
        except Exception as e:
            self.get_logger().warn(f"⚠️ PCA alignment failed: {str(e)}")
            return np.eye(4)

    def multi_scale_ransac(self, source, target, source_fpfh, target_fpfh, voxel_size):
        """Perform RANSAC at multiple scales with robust error handling."""
        # VALIDACIÓN CRÍTICA: Verificar correspondencias antes de RANSAC
        if (source is None or target is None or 
            source_fpfh is None or target_fpfh is None):
            self.get_logger().error("❌ Invalid inputs for RANSAC")
            return self._create_identity_result()

        if (len(source.points) < 10 or len(target.points) < 10 or
            source_fpfh.data.shape[1] < 10 or target_fpfh.data.shape[1] < 10):
            self.get_logger().error(f"❌ Insufficient data for RANSAC: "
                                  f"source_pts={len(source.points)}, target_pts={len(target.points)}, "
                                  f"source_features={source_fpfh.data.shape[1]}, target_features={target_fpfh.data.shape[1]}")
            return self._create_identity_result()

        best_result = None
        best_fitness = 0.0
        
        # Start with larger, more permissive thresholds
        distance_multipliers = [3.0, 2.5, 2.0, 1.5, 1.0]
        
        for i, multiplier in enumerate(distance_multipliers):
            distance_threshold = voxel_size * multiplier
            
            try:
                self.get_logger().info(f"🎯 RANSAC attempt {i+1}/{len(distance_multipliers)}, threshold: {distance_threshold:.4f}")
                
                result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    source, target,
                    source_fpfh, target_fpfh,
                    mutual_filter=False,  # Less strict filtering
                    max_correspondence_distance=distance_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,  # Reduced from 4 to 3
                    checkers=[
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),  # More permissive
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                    ],
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 500)  # Reduced iterations
                )
                
                self.get_logger().info(f"📊 RANSAC result {i+1} - Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.6f}")
                
                if result.fitness > best_fitness:
                    best_fitness = result.fitness
                    best_result = result
                
                # Early termination if good result found
                if result.fitness > 0.2:  # Lower threshold for early termination
                    self.get_logger().info(f"✅ Acceptable RANSAC result found!")
                    break
                    
            except Exception as e:
                self.get_logger().warn(f"⚠️ RANSAC attempt {i+1} failed: {str(e)}")
                continue
        
        if best_result is None or best_fitness == 0.0:
            self.get_logger().warn("⚠️ All RANSAC attempts failed, using identity transformation")
            return self._create_identity_result()
        
        self.get_logger().info(f"🏆 Best RANSAC fitness: {best_fitness:.4f}")
        return best_result

    def _create_identity_result(self):
        """Create a fallback identity transformation result."""
        class IdentityResult:
            def __init__(self):
                self.transformation = np.eye(4)
                self.fitness = 0.0
                self.inlier_rmse = float('inf')
        return IdentityResult()

    def iterative_icp_refinement(self, source, target, voxel_size, max_iterations=3):
        """Perform iterative ICP refinement with robust error handling."""
        if source is None or target is None:
            return source, np.eye(4)

        if len(source.points) < 10 or len(target.points) < 10:
            self.get_logger().warn(f"⚠️ Too few points for ICP refinement")
            return source, np.eye(4)

        current_source = copy.deepcopy(source)
        total_transformation = np.eye(4)
        
        # More conservative thresholds
        thresholds = [voxel_size * 4.0, voxel_size * 2.0, voxel_size * 1.0]
        
        for i, threshold in enumerate(thresholds):
            try:
                self.get_logger().info(f"🔄 ICP iteration {i+1}/{len(thresholds)}, threshold: {threshold:.4f}")
                
                result = o3d.pipelines.registration.registration_icp(
                    current_source, target,
                    max_correspondence_distance=threshold,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-4,  # Less strict
                        relative_rmse=1e-4,     # Less strict
                        max_iteration=30        # Fewer iterations
                    )
                )
                
                if result.fitness > 0 and not np.isinf(result.inlier_rmse):
                    current_source.transform(result.transformation)
                    total_transformation = result.transformation @ total_transformation
                    
                    self.get_logger().info(f"📈 ICP iteration {i+1} - Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.6f}")
                    
                    # Early termination conditions
                    if result.fitness > 0.8 and result.inlier_rmse < voxel_size * 0.5:
                        self.get_logger().info("🎉 ICP converged early!")
                        break
                else:
                    self.get_logger().warn(f"⚠️ ICP iteration {i+1} produced invalid result, continuing...")
                    
            except Exception as e:
                self.get_logger().warn(f"⚠️ ICP iteration {i+1} failed: {str(e)}")
                continue
        
        return current_source, total_transformation

    def pointcloud_callback(self, msg):
        start_time = time.time()
        self.get_logger().info("🚀 Received new scan, starting advanced alignment...")

        # Convert and validate target cloud
        target = self.ros_to_open3d(msg)
        if target is None:
            self.get_logger().error("❌ Failed to convert target cloud")
            return

        self.get_logger().info(f"📊 Target cloud: {len(target.points)} points")

        # Step 1: Initial model preparation
        aligned_model = copy.deepcopy(self.complete_model)

        # Step 2: Scale estimation and application
        if self.enable_scale_estimation:
                scale_factor = self.estimate_scale_factor(aligned_model, target)
                
                # Validar la escala antes de aplicarla
                if scale_factor != 1.0 and self.validate_scale_factor(aligned_model, target, scale_factor):
                    self.get_logger().info(f"✅ Applying validated scale factor: {scale_factor:.3f}")
                    aligned_model.scale(scale_factor, center=(0, 0, 0))
                elif scale_factor != 1.0:
                    self.get_logger().warn(f"⚠️ Scale factor {scale_factor:.3f} failed validation, using 1.0")
                    # Intentar con una escala más conservadora
                    conservative_scale = 1.0 + (scale_factor - 1.0) * 0.5
                    if self.validate_scale_factor(aligned_model, target, conservative_scale):
                        self.get_logger().info(f"✅ Applying conservative scale: {conservative_scale:.3f}")
                        aligned_model.scale(conservative_scale, center=(0, 0, 0))
                    else:
                        self.get_logger().info("📏 Using original scale (1.0)")

        # Step 3: Centroid alignment
        try:
            model_center = aligned_model.get_center()
            target_center = target.get_center()
            aligned_model.translate(target_center - model_center)
        except Exception as e:
            self.get_logger().warn(f"⚠️ Centroid alignment failed: {str(e)}")

        # Step 4: PCA-based orientation alignment
        if self.enable_pca_alignment:
            pca_transform = self.align_with_pca(aligned_model, target)
            if not np.allclose(pca_transform, np.eye(4)):
                aligned_model.transform(pca_transform)
                # Re-center after rotation
                aligned_model.translate(target_center - aligned_model.get_center())

        # Step 5: Preprocessing for registration
        aligned_down, aligned_fpfh, adaptive_voxel = self.preprocess_point_cloud(aligned_model)
        target_down, target_fpfh, _ = self.preprocess_point_cloud(target)

        # VALIDACIÓN CRÍTICA: Verificar que el preprocessing fue exitoso
        if (aligned_down is None or target_down is None or 
            aligned_fpfh is None or target_fpfh is None):
            self.get_logger().error("❌ Preprocessing failed, publishing centroid-aligned model as fallback")
            aligned_msg = self.open3d_to_ros(aligned_model, msg.header)
            if aligned_msg is not None:
                self.publisher_.publish(aligned_msg)
            return

        self.get_logger().info(f"✅ Preprocessed clouds - Model: {len(aligned_down.points)}, Target: {len(target_down.points)}")

        # Step 6: Multi-scale RANSAC registration
        ransac_result = self.multi_scale_ransac(
            aligned_down, target_down, 
            aligned_fpfh, target_fpfh, 
            adaptive_voxel
        )

        # Apply RANSAC transform to full model
        if ransac_result is not None and hasattr(ransac_result, 'transformation'):
            aligned_model.transform(ransac_result.transformation)

        # Step 7: Ensure normals are computed for ICP
        try:
            aligned_model.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=adaptive_voxel * 2, max_nn=30))
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=adaptive_voxel * 2, max_nn=30))
        except Exception as e:
            self.get_logger().warn(f"⚠️ Normal estimation failed: {str(e)}")

        # Step 8: Iterative ICP refinement
        if self.enable_iterative_refinement:
            final_model, icp_transform = self.iterative_icp_refinement(
                aligned_model, target, adaptive_voxel
            )
        else:
            # Single ICP pass with error handling
            try:
                icp_result = o3d.pipelines.registration.registration_icp(
                    aligned_model, target,
                    max_correspondence_distance=adaptive_voxel * 2.0,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                final_model = copy.deepcopy(aligned_model)
                if hasattr(icp_result, 'transformation'):
                    final_model.transform(icp_result.transformation)
                else:
                    final_model = aligned_model
            except Exception as e:
                self.get_logger().warn(f"⚠️ Single ICP failed: {str(e)}")
                final_model = aligned_model

        # Step 9: Final quality assessment and publishing
        try:
            if final_model is not None and len(final_model.points) > 0:
                final_distance = np.mean(final_model.compute_point_cloud_distance(target))
                
                aligned_msg = self.open3d_to_ros(final_model, msg.header)
                if aligned_msg is not None:
                    self.publisher_.publish(aligned_msg)
                    
                    total_time = time.time() - start_time
                    self.get_logger().info(f"✅ Advanced alignment complete!")
                    self.get_logger().info(f"📊 Final metrics - Mean distance: {final_distance:.6f}m, Processing time: {total_time:.2f}s")
                else:
                    self.get_logger().error("❌ Failed to convert final result to ROS message")
            else:
                self.get_logger().error("❌ Final model is invalid")
                
        except Exception as e:
            self.get_logger().error(f"❌ Final processing failed: {str(e)}")
        
        distances = np.asarray(final_model.compute_point_cloud_distance(target))
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        inlier_ratio = np.sum(distances < adaptive_voxel * 1.5) / len(distances)

        self.get_logger().info(f"📏 Final metrics:")
        self.get_logger().info(f"   ➤ Mean distance: {mean_distance:.6f} m")
        self.get_logger().info(f"   ➤ Max (Hausdorff) distance: {max_distance:.6f} m")
        self.get_logger().info(f"   ➤ Inlier ratio: {inlier_ratio:.2%}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudAligner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("👋 Shutting down PointCloud Aligner...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
