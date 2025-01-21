# YOLO model from Ultralytics, detects objects in an image, ended up the best

from ultralytics import YOLO
from PIL import Image, ImageDraw

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the nano version, small and fast

# Load the image
image_path = r"C:\Users\Theor\Downloads\20250118_141344.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Run inference
results = model.predict(source=image_path, save=False, conf=0.01)  # Lower confidence threshold if needed

# Convert results to PIL image for visualization
draw = ImageDraw.Draw(image)
for result in results[0].boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    class_name = model.names[int(class_id)]
    print(f"Detected: {class_name}, Score: {score:.2f}, Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
    # Draw all boxes for visualization
    draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
    draw.text((x1, y1), f"{class_name}: {score:.2f}", fill="blue")


# Show the image with bounding boxes
image.show()

# Optional: Save the annotated image
image.save("output_yolo.jpg")
