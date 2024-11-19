from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Response

import tensorflow
import pandas as pd
from PIL import Image
import pickle
import shutil
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os
import json
from pathlib import Path
from datetime import datetime

app = FastAPI()

features_list = pickle.load(open("embeddings.pkl", "rb"))
img_files_list = pickle.load(open("products.pkl", "rb"))

UPLOAD_DIRECTORY = Path("uploads")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])


def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)

    return result_normlized

def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices

def sanitize_filename(filename: str) -> str:
    return f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

def get_indices(lst, targets):
    indices = []
    for target in targets[0]:
        try:
            indices.append(lst[target])
        except:
            print('ERROR: indices not found.', target)
    return indices


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/check_gpu/")
def check_gpu():
    gpus = tensorflow.config.list_physical_devices('GPU')
    return {"gpus": [gpu.name for gpu in gpus] if gpus else "No GPU detected"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file to the UPLOAD_DIRECTORY
    sanitized_filename = sanitize_filename(file.filename)
    file_path = UPLOAD_DIRECTORY / sanitized_filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    features = extract_img_features(file_path, model)

    img_indicess = recommendd(features, features_list)
    img_indicess = img_indicess.tolist()
    image_list = get_indices(img_files_list, img_indicess)
    return {"filename": file.filename, "message": "predicted", 'IDs': image_list}