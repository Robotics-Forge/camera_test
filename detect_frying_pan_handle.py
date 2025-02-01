import cv2
import numpy as np
from ultralytics import YOLO
import time

class PanHandleDetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        self.pan_classes = ['bowl', 'cup', 'dining table', 'bottle']
        print("Model loaded successfully")

    def analyze_geometry(self, obj_region):
        """Analyze object geometry to find handle and important points"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(obj_region, cv2.COLOR_BGR2GRAY)
            
            # Threshold to get binary image
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Get the largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get geometric features
            # Center point
            M = cv2.moments(main_contour)
            if M["m00"] == 0:
                return None
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # Rightmost point (potential handle)
            rightmost = tuple(main_contour[main_contour[:,:,0].argmax()][0])
            
            # Leftmost point
            leftmost = tuple(main_contour[main_contour[:,:,0].argmin()][0])
            
            # Get rotated rectangle for orientation
            rect = cv2.minAreaRect(main_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get convex hull for shape analysis
            hull = cv2.convexHull(main_contour)
            
            return {
                'contour': main_contour,
                'center': (center_x, center_y),
                'rightmost': rightmost,
                'leftmost': leftmost,
                'rect': rect,
                'box': box,
                'hull': hull
            }
            
        except Exception as e:
            print(f"Error in geometry analysis: {e}")
            return None

    def detect_and_analyze(self, frame):
        try:
            # Get model predictions
            results = self.model(frame, verbose=False)
            annotated_frame = frame.copy()
            
            # Process each detection
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = [int(x) if i < 4 else x for i, x in enumerate(result)]
                class_name = self.model.names[int(cls)]
                
                if class_name in self.pan_classes and conf > 0.3:
                    print(f"\nAnalyzing {class_name} ({conf:.2f})")
                    
                    # Extract object region
                    obj_region = frame[y1:y2, x1:x2]
                    if obj_region.size == 0:
                        continue
                    
                    # Analyze geometry
                    geometry = self.analyze_geometry(obj_region)
                    
                    if geometry:
                        # Adjust points to full frame coordinates
                        center_x = x1 + geometry['center'][0]
                        center_y = y1 + geometry['center'][1]
                        handle_x = x1 + geometry['rightmost'][0]
                        handle_y = y1 + geometry['rightmost'][1]
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw object center
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)
                        cv2.putText(annotated_frame, "Center", (center_x-30, center_y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Draw handle point
                        cv2.circle(annotated_frame, (handle_x, handle_y), 8, (0, 0, 255), -1)
                        cv2.putText(annotated_frame, "Handle", (handle_x+10, handle_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Draw grasp direction
                        cv2.arrowedLine(annotated_frame, 
                                      (center_x, center_y),
                                      (handle_x, handle_y),
                                      (0, 255, 255), 2)
                        
                        # Draw rotated rectangle
                        box = np.int0(geometry['box']) + np.array([x1, y1])
                        cv2.drawContours(annotated_frame, [box], 0, (255, 0, 255), 2)
                        
                        # Draw convex hull
                        hull = geometry['hull'] + np.array([x1, y1])
                        cv2.drawContours(annotated_frame, [hull], 0, (255, 255, 0), 2)
                        
                        # Add measurements
                        width = x2 - x1
                        height = y2 - y1
                        handle_length = np.sqrt((handle_x-center_x)**2 + (handle_y-center_y)**2)
                        
                        measurements = [
                            f"Width: {width}px",
                            f"Height: {height}px",
                            f"Handle Length: {int(handle_length)}px",
                            f"Confidence: {conf:.2f}"
                        ]
                        
                        for i, text in enumerate(measurements):
                            cv2.putText(annotated_frame, text,
                                      (10, 30 + i*25),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return frame

def process_webcam(detector):
    """Handle webcam input"""
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    print("\nWebcam active!")
    print("Controls:")
    print("- 'q': Quit")
    print("- 's': Save frame")
    print("- 'd': Force detection")
    
    last_detection_time = 0
    detection_interval = 5  # seconds
    frame_count = 0
    
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Geometry Analysis', cv2.WINDOW_NORMAL)
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("No frame received!")
                break
            
            current_time = time.time()
            
            # Show raw feed
            cv2.imshow('Camera Feed', frame)
            
            # Run detection and analysis
            if current_time - last_detection_time >= detection_interval:
                print("\nAnalyzing frame...")
                analyzed_frame = detector.detect_and_analyze(frame)
                cv2.imshow('Geometry Analysis', analyzed_frame)
                last_detection_time = current_time
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'analysis_{frame_count}.jpg'
                cv2.imwrite(filename, analyzed_frame)
                print(f"Saved {filename}")
                frame_count += 1
            elif key == ord('d'):
                analyzed_frame = detector.detect_and_analyze(frame)
                cv2.imshow('Geometry Analysis', analyzed_frame)
                last_detection_time = current_time
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Initializing geometry analyzer...")
    detector = PanHandleDetector()
    process_webcam(detector)
    print("Done!")

if __name__ == "__main__":
    main()
