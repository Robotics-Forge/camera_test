#Test of the camera's RBG (Color) sensor along with compatibility with YOLO. 

import sys
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Add the pyorbbecsdk/examples directory to Python path
sdk_example_path = os.path.join(os.path.dirname(__file__), "pyorbbecsdk", "examples")
if sdk_example_path not in sys.path:
    sys.path.append(sdk_example_path)

from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFormat
from utils import frame_to_bgr_image

def main():
    # Initialize YOLO model
    model = YOLO("yolov8n.pt")
    current_detections = []
    last_yolo_time = time.time()

    # Initialize pipeline
    config = Config()
    pipeline = Pipeline()

    try:
        # Get color stream profile
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is None:
            raise RuntimeError("Failed to get color sensor profile list")

        # Try to get specific color profile
        color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        pipeline.start(config)
        print("Pipeline started. Press 'q' to exit...")
        
        while True:
            try:
                # Capture frames with shorter timeout
                frames = pipeline.wait_for_frames(1000)
                if frames is None:
                    print("No frames received, retrying...")
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    print("Failed to get color frame, retrying...")
                    continue

                # Convert frame to BGR image
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    print("Failed to convert frame to image, retrying...")
                    continue

                # Run YOLO detection every 5 seconds
                current_time = time.time()
                if current_time - last_yolo_time >= 5.0:
                    print("Running YOLO detection...")
                    results = model.predict(source=color_image, conf=0.3, save=False)
                    
                    # Update stored detections
                    current_detections = []
                    for result in results[0].boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = result
                        class_name = model.names[int(class_id)]
                        current_detections.append({
                            'bbox': list(map(int, [x1, y1, x2, y2])),
                            'score': score,
                            'class_name': class_name
                        })
                    last_yolo_time = current_time

                # Draw current detections
                for detection in current_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    class_name = detection['class_name']
                    score = detection['score']
                    
                    # Draw blue rectangle and text
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(color_image, f"{class_name}: {score:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 0, 0), 2)

                # Display results
                cv2.imshow("YOLO Detection", color_image)

                # Add frame rate control
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User requested exit")
                    break

                # Check if window was closed
                if cv2.getWindowProperty("YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed")
                    break

            except Exception as frame_error:
                print(f"Frame processing error: {frame_error}")
                time.sleep(0.1)  # Brief pause before retrying
                continue

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped")

if __name__ == "__main__":
    main()