# MiDaS model from Hugging face, estimates depth in an image

from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import torch
import numpy as np

# Load MiDaS model and processor
model_name = "Intel/dpt-large"  # MiDaS model
processor = DPTImageProcessor.from_pretrained(model_name)
model = DPTForDepthEstimation.from_pretrained(model_name)

# Load the image
image_path = r"C:\Users\Theor\Downloads\s-l1200.png"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform depth estimation
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

# Normalize depth for visualization
depth_min = predicted_depth.min()
depth_max = predicted_depth.max()
depth_normalized = (predicted_depth - depth_min) / (depth_max - depth_min)
depth_image = Image.fromarray((depth_normalized * 255).astype("uint8"))
depth_image.show()  # Show depth map
