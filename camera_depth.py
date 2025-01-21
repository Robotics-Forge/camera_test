#Test of the camera's depth perception features, inconsequential. 

import sys
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure YOLO is installed and functional

# Add the pyorbbecsdk/examples directory to Python path
sdk_example_path = os.path.join(os.path.dirname(__file__), "pyorbbecsdk", "examples")
if sdk_example_path not in sys.path:
    sys.path.append(sdk_example_path)

# Try importing the SDK components
from pyorbbecsdk import Config, OBSensorType, Pipeline

ESC_KEY = 27
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 5000  # 5000mm

# Camera intrinsics (replace with your camera's values)
fx, fy = 600, 600  # Focal lengths
cx, cy = 320, 240  # Principal point (optical center)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # Replace with your YOLO model

def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """Convert 2D pixel coordinates to 3D world coordinates."""
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def main():
    config = Config()
    pipeline = Pipeline()
    last_yolo_time = time.time()  # To control YOLO detection frequency

    try:
        # Configure the pipeline
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None, "Depth sensor not found!"
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None, "No default video stream profile for depth!"
        config.enable_stream(depth_profile)

        pipeline.start(config)
        print("Pipeline started successfully!")

        depth_image_colored = None  # Initialize to avoid unassigned reference

        while True:
            try:
                # Wait for frames with a longer timeout
                frames = pipeline.wait_for_frames(500)  # 500 ms timeout
                if frames is None:
                    print("No frames received from pipeline!")
                    continue

                # Get the depth frame
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    print("Depth frame is not available!")
                    continue

                # Process depth frame
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()

                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                # Convert to millimeters and clamp values
                depth_data = depth_data.astype(np.float32) * scale
                depth_data = np.clip(depth_data, MIN_DEPTH, MAX_DEPTH)

                # Normalize for display
                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image_colored = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

                # Run YOLO detection every 10 seconds
                if time.time() - last_yolo_time >= 10:
                    print("Running YOLO detection...")
                    results = yolo_model.predict(source=depth_image_colored, conf=0.5, save=False, save_txt=False)
                    
                    for result in results:
                        for box in result.boxes:
                            # Extract box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            label = result.names[box.cls[0].item()]
                            object_depth = depth_data[y1:y2, x1:x2]
                            average_depth = np.mean(object_depth)

                            # Calculate 3D coordinates
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            x, y, z = pixel_to_3d(center_x, center_y, average_depth, fx, fy, cx, cy)
                            print(f"Detected {label} at (x: {x:.2f}, y: {y:.2f}, z: {z:.2f})")

                            # Draw bounding box and label on the image
                            cv2.rectangle(depth_image_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(depth_image_colored, f"{label} ({average_depth:.2f}mm)", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    last_yolo_time = time.time()

                # Display the depth frame with detections
                if depth_image_colored is not None:
                    cv2.imshow("Depth Viewer with YOLO", depth_image_colored)

                # Break loop on user interrupt
                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:  # Exit on 'q' or ESC key
                    print("Exiting...")
                    break

            except KeyboardInterrupt:
                print("User interrupted the process!")
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pipeline.stop()
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()
