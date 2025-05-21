import torch
from PIL import Image
import numpy as np
from transformers import pipeline
import io

class GenderClassificationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gender",)
    FUNCTION = "classify_gender"
    CATEGORY = "Image Processing"

    def __init__(self):
        # Initialize the pipeline once during node creation
        self.pipe = pipeline("image-classification", model="prithivMLmods/Realistic-Gender-Classification")

    def classify_gender(self, image):
        try:
            # Convert ComfyUI's image tensor (B, H, W, C) to PIL Image
            # ComfyUI image is a torch tensor in range [0,1]
            image = image[0].cpu().numpy()  # Take first image if batch
            image = (image * 255).astype(np.uint8)  # Convert to uint8
            pil_image = Image.fromarray(image)

            # Perform classification
            results = self.pipe(pil_image)

            # Find the result with the highest score
            top_result = max(results, key=lambda x: x['score'])
            label = top_result['label'].lower()

            # Extract gender from label (e.g., "female portrait" -> "Female")
            if 'female' in label:
                gender = "Female"
            elif 'male' in label:
                gender = "Male"
            else:
                gender = "Unknown"

            return (gender,)

        except Exception as e:
            return (f"Error: {str(e)}",)