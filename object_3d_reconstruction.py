import sys
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import open3d as o3d
from scipy.spatial import ConvexHull
from PIL import Image
from datetime import datetime

# Add the pyorbbecsdk/examples directory to Python path
sdk_example_path = os.path.join(os.path.dirname(__file__), "pyorbbecsdk", "examples")
if sdk_example_path not in sys.path:
    sys.path.append(sdk_example_path)

from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFormat
from utils import frame_to_bgr_image

def find_center_scissors(detections, frame_width, frame_height):
    """Find the scissors closest to the center of the frame."""
    if not detections:
        return None
    
    frame_center = np.array([frame_width/2, frame_height/2])
    min_distance = float('inf')
    center_scissors = None
    
    for detection in detections:
        # Check for both singular and plural forms
        if detection['class_name'].lower() not in ['scissors', 'scissor']:
            continue
            
        bbox = detection['bbox']
        object_center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        distance = np.linalg.norm(object_center - frame_center)
        
        if distance < min_distance:
            min_distance = distance
            center_scissors = detection
    
    return center_scissors

def create_point_cloud(depth_data, bbox, intrinsics):
    """Create a point cloud from depth data within the bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    
    print(f"Processing bounding box: ({x1}, {y1}, {x2}, {y2})")
    
    # Add more padding to capture more context
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(depth_data.shape[1], x2 + padding)
    y2 = min(depth_data.shape[0], y2 + padding)
    
    # Extract depth data for the object
    object_depth = depth_data[y1:y2, x1:x2].copy()
    
    # Apply median filter to reduce noise
    object_depth = cv2.medianBlur(object_depth.astype(np.float32), 5)
    
    # Create meshgrid for pixel coordinates
    rows, cols = object_depth.shape
    u, v = np.meshgrid(range(x1, x2), range(y1, y2))
    
    # Convert to 3D points
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    Z = object_depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack coordinates and reshape
    points = np.stack([X, Y, Z], axis=-1)
    points = points.reshape(-1, 3)
    
    # Remove invalid points
    valid_points = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0)
    points = points[valid_points]
    
    if len(points) > 0:
        # Remove outliers based on Z-depth
        z_mean = np.mean(points[:, 2])
        z_std = np.std(points[:, 2])
        z_inliers = np.abs(points[:, 2] - z_mean) < 2 * z_std
        points = points[z_inliers]
        
        # Center the points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale the points to a reasonable size
        max_dim = np.max(np.abs(points))
        if max_dim > 0:
            points = points * (100.0 / max_dim)
        
        # Sort points by depth for better visualization
        depth_order = np.argsort(points[:, 2])
        points = points[depth_order]
    
    print(f"Number of valid points after filtering: {len(points)}")
    return points

def reconstruct_3d_object(points):
    """Create a 3D reconstruction from point cloud."""
    if len(points) < 4:
        print("Not enough points for reconstruction")
        return None
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Optional: Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
    
    try:
        # Create mesh using Ball Pivoting instead of Poisson
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        
        # Compute vertex normals for better visualization
        mesh.compute_vertex_normals()
        
        print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        return mesh
    
    except Exception as e:
        print(f"Mesh creation failed: {e}")
        return pcd  # Return point cloud if mesh creation fails

class PhotogrammetryCapture:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.target_class = "pan"  # or "frying pan"
        self.images_folder = "photogrammetry_images"
        self.min_images = 20  # Minimum recommended images for good reconstruction
        self.captured_images = []
        
        # Create output directory
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
            
    def capture_images(self):
        cap = cv2.VideoCapture(0)
        image_count = 0
        last_capture_time = datetime.now()
        capture_delay = 2.0  # Seconds between automatic captures
        
        print("\nPhotogrammetry Capture System")
        print("-----------------------------")
        print("Instructions:")
        print("1. Slowly rotate the object or move the camera around it")
        print("2. Keep the target object centered")
        print("3. System will automatically capture when object is detected clearly")
        print("4. Try to get different angles (top, sides, slightly below)")
        print(f"5. Need minimum {self.min_images} images for good reconstruction")
        print("\nControls:")
        print("SPACE - Force capture current frame")
        print("R - Reset capture session")
        print("Q - Quit and save captured images")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create copy for visualization
            display_frame = frame.copy()
            
            # Run YOLO detection
            results = self.model.predict(frame, conf=0.3, save=False)
            
            # Find target object
            target_detected = False
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                class_name = self.model.names[int(class_id)].lower()
                
                if self.target_class in class_name:
                    # Draw green box for target object
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    target_detected = True
                    
                    # Auto capture if enough time has passed
                    time_since_last = (datetime.now() - last_capture_time).total_seconds()
                    if time_since_last >= capture_delay and target_detected:
                        self.save_image(frame, image_count)
                        last_capture_time = datetime.now()
                        image_count += 1
                else:
                    # Draw red box for other objects
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # Add overlay information
            cv2.putText(display_frame, f"Images: {image_count}/{self.min_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if target_detected:
                cv2.putText(display_frame, "Target Detected", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Photogrammetry Capture', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                image_count = 0
                self.captured_images = []
                print("\nCapture session reset")
            elif key == ord(' '):
                self.save_image(frame, image_count)
                image_count += 1
                last_capture_time = datetime.now()
        
        cap.release()
        cv2.destroyAllWindows()
        
        if image_count >= self.min_images:
            print(f"\nCapture complete! {image_count} images saved to {self.images_folder}")
            print("\nNext steps:")
            print("1. Use these images with Meshroom, RealityCapture, or similar photogrammetry software")
            print("2. Process images to create 3D mesh")
            print("3. Clean up and export mesh for simulation")
        else:
            print(f"\nWarning: Only {image_count} images captured. More images recommended for good reconstruction.")
    
    def save_image(self, frame, count):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.images_folder}/capture_{count:03d}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        self.captured_images.append(filename)
        print(f"Saved image {count + 1}")

def main():
    # Initialize YOLO model
    model = YOLO("yolov8n.pt")
    current_detections = []
    last_yolo_time = time.time()

    # Camera intrinsics (replace with your camera's values)
    intrinsics = {
        'fx': 600,  # focal length x
        'fy': 600,  # focal length y
        'cx': 320,  # principal point x
        'cy': 240   # principal point y
    }

    # Initialize pipeline
    config = Config()
    pipeline = Pipeline()

    # Create visualizer with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Reconstruction", width=800, height=600)
    
    # Set up render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 5.0  # Increased point size
    
    # Set up camera view
    ctr = vis.get_view_control()

    try:
        # Configure color and depth streams
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        if not color_profile_list or not depth_profile_list:
            raise RuntimeError("Failed to get sensor profile lists")

        color_profile = color_profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        depth_profile = depth_profile_list.get_default_video_stream_profile()

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

        pipeline.start(config)
        print("Pipeline started. Press 'q' to exit, 'r' to reconstruct current center object.")

        while True:
            try:
                frames = pipeline.wait_for_frames(1000)
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    continue

                # Get depth data
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                depth_scale = depth_frame.get_depth_scale()
                depth_data = depth_data.astype(np.float32) * depth_scale

                # Run YOLO detection every 5 seconds
                current_time = time.time()
                if current_time - last_yolo_time >= 5.0:
                    results = model.predict(source=color_image, conf=0.3, save=False)
                    
                    current_detections = []
                    for result in results[0].boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = result
                        class_name = model.names[int(class_id)]
                        current_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': score,
                            'class_name': class_name
                        })
                    last_yolo_time = current_time

                # Find and highlight center scissors
                center_scissors = find_center_scissors(current_detections, color_image.shape[1], color_image.shape[0])
                
                # Draw all detections
                for detection in current_detections:
                    x1, y1, x2, y2 = map(int, detection['bbox'])
                    # Green for scissors, blue for others
                    color = (0, 255, 0) if detection == center_scissors else (255, 0, 0)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(color_image, detection['class_name'], (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display the image
                cv2.imshow("Object Detection", color_image)

                key = cv2.waitKey(1) & 0xFF
                print(f"Key pressed: {key}")  # Debug print
                
                if key == ord('q'):
                    print("Quit key pressed")
                    break
                elif key == ord('r') and center_scissors:
                    print(f"\nReconstructing scissors...")
                    
                    # Create 3D reconstruction of scissors
                    points = create_point_cloud(depth_data, center_scissors['bbox'], intrinsics)
                    
                    if points is not None and len(points) > 0:
                        try:
                            # Create point cloud visualization
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(points)
                            
                            # Color the points based on depth (Z coordinate)
                            points = np.asarray(pcd.points)
                            colors = np.zeros((len(points), 3))
                            min_z = np.min(points[:, 2])
                            max_z = np.max(points[:, 2])
                            normalized_depth = (points[:, 2] - min_z) / (max_z - min_z + 1e-6)
                            
                            # Create a more distinctive color gradient
                            colors[:, 0] = np.clip(1 - normalized_depth, 0, 1)  # Red
                            colors[:, 1] = np.clip(normalized_depth * 0.5, 0, 1)  # Green
                            colors[:, 2] = np.clip(normalized_depth, 0, 1)  # Blue
                            
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                            
                            # Clear previous geometry and add new point cloud
                            vis.clear_geometries()
                            vis.add_geometry(pcd)
                            
                            # Set a better initial view
                            view_param = vis.get_view_control()
                            view_param.set_lookat(np.mean(points, axis=0))
                            view_param.set_front([0, 0, -1])
                            view_param.set_up([0, -1, 0])
                            view_param.set_zoom(0.8)
                            
                            # Update visualization
                            vis.poll_events()
                            vis.update_renderer()
                            print("3D visualization updated")
                            print(f"Point cloud created with {len(points)} points")
                            print("Use mouse to rotate view:")
                            print("- Left button + drag: Rotate")
                            print("- Right button + drag: Pan")
                            print("- Mouse wheel: Zoom")
                            
                        except Exception as e:
                            print(f"Error during visualization: {e}")
                    else:
                        print("No valid points for reconstruction")

                # Add debug prints for detections
                if current_detections:
                    print("Current detections:", [d['class_name'] for d in current_detections])
                
                # Update visualization window even if no new geometry
                vis.poll_events()
                vis.update_renderer()

            except Exception as e:
                print(f"Frame processing error: {e}")
                continue

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("Pipeline stopped")

if __name__ == "__main__":
    capture = PhotogrammetryCapture()
    capture.capture_images() 