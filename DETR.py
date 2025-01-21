# DETR model from Hugging face, detects objects in an image

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch

# Load the model and processor
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Load the image
image_path = r"C:\Users\Theor\Downloads\20250118_141344.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Process the outputs
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)

# Print all detected objects
for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
    label_name = model.config.id2label.get(label.item(), "Unknown")
    print(f"Detected: {label_name}, Score: {score:.2f}, Box: {box.tolist()}")

# Draw all bounding boxes for visualization
draw = ImageDraw.Draw(image)
for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
    label_name = model.config.id2label.get(label.item(), "Unknown")
    box = [int(i) for i in box]  # Convert to integer
    draw.rectangle(box, outline="blue", width=2)  # Draw all detections
    draw.text((box[0], box[1]), f"{label_name}: {score:.2f}", fill="blue")

# Show the image with all bounding boxes
image.show()

# Optional: Save the image with annotations
image.save("output.jpg")
