import cv2
import numpy as np
from ultralytics import YOLO
import os

class SavedImageAnalyzer:
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
            
            # Edge detection for better handle identification
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Get the largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get geometric features
            M = cv2.moments(main_contour)
            if M["m00"] == 0:
                return None
                
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # Find potential handle points
            rightmost = tuple(main_contour[main_contour[:,:,0].argmax()][0])
            leftmost = tuple(main_contour[main_contour[:,:,0].argmin()][0])
            
            # Get orientation
            rect = cv2.minAreaRect(main_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            return {
                'contour': main_contour,
                'center': (center_x, center_y),
                'rightmost': rightmost,
                'leftmost': leftmost,
                'rect': rect,
                'box': box
            }
            
        except Exception as e:
            print(f"Error in geometry analysis: {e}")
            return None

    def analyze_image(self, image_path):
        """Analyze a single image"""
        print(f"\nAnalyzing {image_path}...")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read {image_path}")
            return None
            
        # Create output image
        output = frame.copy()
        
        try:
            # Get YOLO predictions
            results = self.model(frame, verbose=False)
            
            found_objects = False
            
            # Process each detection
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = [int(x) if i < 4 else x for i, x in enumerate(result)]
                class_name = self.model.names[int(cls)]
                
                # Check confidence and class
                if conf > 0.3:  # Lowered threshold to detect more objects
                    found_objects = True
                    print(f"Found {class_name} with confidence {conf:.2f}")
                    
                    # Extract object region
                    obj_region = frame[y1:y2, x1:x2]
                    if obj_region.size == 0:
                        continue
                    
                    # Analyze geometry
                    geometry = self.analyze_geometry(obj_region)
                    
                    if geometry:
                        # Adjust coordinates to full frame
                        center_x = x1 + geometry['center'][0]
                        center_y = y1 + geometry['center'][1]
                        handle_x = x1 + geometry['rightmost'][0]
                        handle_y = y1 + geometry['rightmost'][1]
                        
                        # Draw all the geometric information
                        # Main bounding box
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Center point
                        cv2.circle(output, (center_x, center_y), 5, (255, 0, 0), -1)
                        cv2.putText(output, "Center", (center_x-30, center_y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Handle point
                        cv2.circle(output, (handle_x, handle_y), 8, (0, 0, 255), -1)
                        cv2.putText(output, "Handle", (handle_x+10, handle_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Grasp direction
                        cv2.arrowedLine(output, 
                                      (center_x, center_y),
                                      (handle_x, handle_y),
                                      (0, 255, 255), 2)
                        
                        # Object orientation
                        box = np.int0(geometry['box']) + np.array([x1, y1])
                        cv2.drawContours(output, [box], 0, (255, 0, 255), 2)
                        
                        # Add measurements
                        width = x2 - x1
                        height = y2 - y1
                        handle_length = np.sqrt((handle_x-center_x)**2 + (handle_y-center_y)**2)
                        angle = np.degrees(np.arctan2(handle_y-center_y, handle_x-center_x))
                        
                        measurements = [
                            f"Object: {class_name}",
                            f"Width: {width}px",
                            f"Height: {height}px",
                            f"Handle Length: {int(handle_length)}px",
                            f"Handle Angle: {int(angle)}°",
                            f"Confidence: {conf:.2f}"
                        ]
                        
                        for i, text in enumerate(measurements):
                            cv2.putText(output, text,
                                      (10, 30 + i*25),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (255, 255, 255), 2)
            
            if not found_objects:
                print("No objects detected with sufficient confidence")
                cv2.putText(output, "No objects detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return output
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

def main():
    analyzer = SavedImageAnalyzer()
    
    # Look for analysis images in current directory
    image_files = [f for f in os.listdir('.') if f.startswith('analysis_') and f.endswith('.jpg')]
    
    if not image_files:
        print("No analysis images found!")
        return
    
    print(f"Found {len(image_files)} analysis images")
    
    # Process each image
    for image_file in image_files:
        # Analyze image
        result = analyzer.analyze_image(image_file)
        
        if result is not None:
            # Show result
            cv2.imshow(f'Analysis of {image_file}', result)
            
            # Save result
            output_name = f'analyzed_{image_file}'
            cv2.imwrite(output_name, result)
            print(f"Saved analysis to {output_name}")
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 