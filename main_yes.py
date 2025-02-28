# room items segmentation
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load the SAM model
sam_checkpoint = "sam_vit_b.pth"  # Ensure this path is correct
model_type = "vit_b"  # Model type can be 'vit_b', 'vit_l', or 'vit_h'
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Initialize the automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Load and preprocess the image
image_path = "room1.jpg"  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks for all objects in the image
masks = mask_generator.generate(image_rgb)

# Create an output image to visualize the masks
output_image = image_rgb.copy()
for mask in masks:
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)
    output_image[mask['segmentation']] = output_image[mask['segmentation']] * 0.5 + color * 0.5

# Display the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Segmented Image')
plt.axis('off')

plt.show()

