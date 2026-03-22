import numpy as np
import pandas as pd
from PIL import Image
import io
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Load model once
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = model.predict(img)
    return embedding.flatten()

def get_embedding_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = model.predict(img)
    return embedding.flatten()

def preprocess_tabular(user_metrics, scaler):
    user_array = np.array(user_metrics).reshape(1, -1)
    return scaler.transform(user_array)
