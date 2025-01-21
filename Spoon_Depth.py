from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch
import numpy as np

# Load DETR model and processor
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

# Load MiDaS model for depth estimation
from transformers import DPTForDepthEstimation, DPTImageProcessor

depth_model_name = "Intel/dpt-large"
depth_processor = DPTImageProcessor.from_pretrained(depth_model_name)
depth_model = DPTForDepthEstimation.from_pretrained(depth_model_name)

# Load the RGB image
image_path = r"C:\Users\Theor\Downloads\20250118_141344.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Run DETR for object detection
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Process DETR results
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)

# Extract bounding box for the spoon
spoon_label = None
for key, value in model.config.id2label.items():
    if "spoon" in value.lower():  # Adjust as needed based on label names
        spoon_label = key
        break

if spoon_label is None:
    raise ValueError("Spoon label not found in the model's label set.")

# Identify spoon bounding box
spoon_bbox = None
for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
    if label.item() == spoon_label:  # Check if the detected object is a spoon
        spoon_bbox = [int(coord) for coord in box.tolist()]
        break

if spoon_bbox is None:
    raise ValueError("No spoon detected in the image.")

print(f"Spoon bounding box: {spoon_bbox}")

# Run MiDaS for depth estimation
depth_inputs = depth_processor(images=image, return_tensors="pt")
with torch.no_grad():
    depth_outputs = depth_model(**depth_inputs)
    depth_map = depth_outputs.predicted_depth.squeeze().cpu().numpy()

# Resize depth map to match original image dimensions
depth_map_resized = np.array(Image.fromarray(depth_map).resize(image.size, resample=Image.BILINEAR))

# Ensure bounding box is within resized depth map dimensions
depth_map_height, depth_map_width = depth_map_resized.shape
x1 = max(0, min(spoon_bbox[0], depth_map_width - 1))
y1 = max(0, min(spoon_bbox[1], depth_map_height - 1))
x2 = max(0, min(spoon_bbox[2], depth_map_width - 1))
y2 = max(0, min(spoon_bbox[3], depth_map_height - 1))

# Crop depth map using spoon bounding box
cropped_depth = depth_map_resized[y1:y2, x1:x2]

# Debug bounding box and cropped depth
print(f"Depth Map Dimensions: {depth_map_width}x{depth_map_height}")
print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
print(f"Cropped Depth Shape: {cropped_depth.shape}")

# Handle empty depth map crop
if cropped_depth.size == 0:
    print("Error: Cropped depth map is empty. Check bounding box or depth map alignment.")
else:
    # Analyze depth data
    average_depth = np.mean(cropped_depth)
    min_depth = np.min(cropped_depth)
    max_depth = np.max(cropped_depth)

    print(f"Depth Data for Spoon:")
    print(f"  - Average Depth: {average_depth}")
    print(f"  - Min Depth: {min_depth}")
    print(f"  - Max Depth: {max_depth}")

# Optional: Visualize depth map and bounding box
draw = ImageDraw.Draw(image)
draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
image.show()

# Normalize depth map for visualization
depth_min, depth_max = depth_map_resized.min(), depth_map_resized.max()
depth_normalized = (depth_map_resized - depth_min) / (depth_max - depth_min)
depth_image = Image.fromarray((depth_normalized * 255).astype("uint8"))
depth_image.show()
