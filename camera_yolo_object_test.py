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
        print("Pipeline started. Waiting 5 seconds before capturing frame...")
        
        # Show live preview for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            frames = pipeline.wait_for_frames(100)
            if frames is not None:
                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    color_image = frame_to_bgr_image(color_frame)
                    if color_image is not None:
                        cv2.imshow("Camera Preview", color_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return

        # Capture final frame for YOLO
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            raise RuntimeError("Failed to get frames")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("Failed to get color frame")

        # Convert frame to BGR image
        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            raise RuntimeError("Failed to convert frame to image")

        # Convert to PIL Image for consistent handling
        pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # Run YOLO detection with lower confidence threshold
        print("Running YOLO detection...")
        results = model.predict(source=color_image, conf=0.01, save=False)

        # Draw results manually like in YOLO.py
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            class_name = model.names[int(class_id)]
            print(f"Detected: {class_name}, Score: {score:.2f}, Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
            
            # Draw blue rectangle and text
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            cv2.putText(color_image, f"{class_name}: {score:.2f}", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)

        # Display results
        cv2.imshow("YOLO Detection", color_image)
        print("Press any key to exit")
        cv2.waitKey(0)

        # Save the annotated image
        cv2.imwrite("camera_output_yolo.jpg", color_image)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped")

if __name__ == "__main__":
    main()