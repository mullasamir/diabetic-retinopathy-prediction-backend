from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image, target_size=(98, 98)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
