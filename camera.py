#Current iteration of the code, utilizes both the camera's RBG and depth sensors along with Yolo to detect objects and display their average depths from the bounding boxes. 

import sys
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

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

def main():
    # Initialize YOLO model
    model = YOLO("yolov8n.pt")

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
            # Capture frames
            frames = pipeline.wait_for_frames(100)
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

            # Convert depth frame to a NumPy array
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(depth_frame.get_height(), depth_frame.get_width())
            depth_scale = depth_frame.get_depth_scale()
            depth_data = depth_data.astype(np.float32) * depth_scale  # Convert to millimeters

            # Run YOLO detection
            results = model.predict(source=color_image, conf=0.3, save=False)

            # Process YOLO results and map to depth
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                class_name = model.names[int(class_id)]

                # Get depth information
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                object_depth = depth_data[y1:y2, x1:x2]
                avg_depth = np.mean(object_depth)

                # Calculate 3D coordinates for the center of the bounding box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                x, y, z = pixel_to_3d(center_x, center_y, avg_depth, fx, fy, cx, cy)

                # Draw bounding box and annotate
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{class_name} ({avg_depth:.2f}mm)"
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                print(f"Detected: {class_name}, Score: {score:.2f}, 3D Coordinates: ({x:.2f}, {y:.2f}, {z:.2f})")

            # Display results
            cv2.imshow("YOLO Detection with Depth", color_image)

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
