#Current iteration of the code, utilizes both the camera's RBG and depth sensors along with Yolo to detect objects and display their average depths from the bounding boxes. 

import sys
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Add the current script's directory to the Python path
print("\n".join(sys.path))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add the pyorbbecsdk/examples directory to Python path
sdk_example_path = os.path.join(os.path.dirname(__file__), "pyorbbecsdk", "examples")
if sdk_example_path not in sys.path:
    sys.path.append(sdk_example_path)

# Importing the SDK components like the other scripts
from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFormat
from utils import frame_to_bgr_image

# Camera intrinsics (replace with actual values from your camera)
fx, fy = 600, 600  # Focal lengths
cx, cy = 320, 240  # Optical center

def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """Convert 2D pixel coordinates to 3D world coordinates."""
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def analyze_object_depth(depth_data, bbox):
    """Analyze depth data within a bounding box to identify potential grasp points."""
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Extract depth data for the object
    object_depth = depth_data[y1:y2, x1:x2]
    
    # Basic depth statistics
    avg_depth = np.mean(object_depth)
    min_depth = np.min(object_depth)
    max_depth = np.max(object_depth)
    
    # Calculate depth gradients (can help identify edges/handles)
    depth_gradients = np.gradient(object_depth)
    gradient_magnitude = np.sqrt(depth_gradients[0]**2 + depth_gradients[1]**2)
    
    # Find regions with high gradient (potential grasp points)
    high_gradient_threshold = np.percentile(gradient_magnitude, 90)
    potential_grasp_points = np.where(gradient_magnitude > high_gradient_threshold)
    
    # Convert grasp points to original image coordinates
    grasp_points = [(y1 + y, x1 + x) for y, x in zip(*potential_grasp_points)]
    
    return {
        'avg_depth': avg_depth,
        'min_depth': min_depth,
        'max_depth': max_depth,
        'grasp_points': grasp_points,
        'depth_map': object_depth
    }

def main():
    # Initialize YOLO model
    model = YOLO("yolov8n.pt")
    last_yolo_time = time.time()
    
    # Store detection results
    current_detections = []

    # Initialize pipeline
    config = Config()
    pipeline = Pipeline()

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
        print("Pipeline started. Press 'q' to exit.")

        while True:
            # Capture frames (5000ms = 5 seconds timeout)
            frames = pipeline.wait_for_frames(5000)
            if not frames:
                print("No frames received!")
                continue

            # Extract color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Missing color or depth frame!")
                continue

            # Convert color frame to BGR for YOLO
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert color frame to image!")
                continue

            # Get color frame dimensions
            color_height, color_width = color_image.shape[:2]

            # Convert depth frame to a NumPy array and normalize for visualization
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(depth_frame.get_height(), depth_frame.get_width())
            depth_scale = depth_frame.get_depth_scale()
            depth_data = depth_data.astype(np.float32) * depth_scale  # Convert to millimeters

            # Create colored depth map and resize to match color frame
            depth_colormap = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
            depth_colormap = cv2.resize(depth_colormap, (color_width, color_height))

            # Also resize the depth_data for accurate measurements
            depth_data = cv2.resize(depth_data, (color_width, color_height))

            # Run YOLO detection every 5 seconds
            current_time = time.time()
            if current_time - last_yolo_time >= 5.0:
                # Run YOLO detection on RGB image
                results = model.predict(source=color_image, conf=0.6, save=False)
                
                # Update stored detections
                current_detections = []
                for result in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    depth_info = analyze_object_depth(depth_data, result)
                    current_detections.append({
                        'bbox': list(map(int, [x1, y1, x2, y2])),
                        'class_id': int(class_id),
                        'class_name': model.names[int(class_id)],
                        'depth_info': depth_info
                    })
                last_yolo_time = current_time

            # Draw current detections on every frame
            for detection in current_detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                depth_info = detection['depth_info']

                # Draw on color image
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{class_name} ({depth_info['avg_depth']:.2f}mm)"
                cv2.putText(color_image, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw on depth colormap
                cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(depth_colormap, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Visualize grasp points on both views
                for point in depth_info['grasp_points'][:10]:
                    px, py = int(point[1]), int(point[0])
                    # Draw on color image
                    cv2.circle(color_image, (px, py), 2, (0, 255, 0), -1)
                    # Draw on depth colormap
                    cv2.circle(depth_colormap, (px, py), 2, (0, 255, 0), -1)

            # Display both views side by side
            combined_view = np.hstack((color_image, depth_colormap))
            cv2.imshow("RGB and Depth View", combined_view)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()
