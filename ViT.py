# ViT model from Hugging face, classifies images, used for object detection

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the pre-trained model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Load and preprocess the image
image_path = r"C:\Users\Theor\Downloads\20250118_141344.jpg"
image = Image.open(image_path).convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

# Get class label
labels = model.config.id2label
print(f"Predicted class: {labels[predicted_class]}")










