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
            
            image = image[0].cpu().numpy()  
            image = (image * 255).astype(np.uint8)  
            pil_image = Image.fromarray(image)

            
            results = self.pipe(pil_image)

        
            top_result = max(results, key=lambda x: x['score'])
            label = top_result['label'].lower()

            
            if 'female' in label:
                gender = "Female"
            elif 'male' in label:
                gender = "Male"
            else:
                gender = "Unknown"

            return (gender,)

        except Exception as e:
            return (f"Error: {str(e)}",)
