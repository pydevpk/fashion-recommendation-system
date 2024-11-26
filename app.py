from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates  
from fastapi import UploadFile
from fastapi import File
from fastapi import Response

# import faiss
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
from sklearn.preprocessing import normalize
import os
import json
import joblib
from pathlib import Path
from datetime import datetime
import requests

app = FastAPI()

templates = Jinja2Templates(directory="templates")

data = pd.read_csv("ASHI_FINAL_DATA.csv")

# Load saved encoders and model
encoders = joblib.load('column_encoders.pkl')
combined_features = joblib.load('combined_features.pkl')
r_model = joblib.load('categorical_recommendation_model.pkl')

features_list = pickle.load(open("embeddings.pkl", "rb"))
img_files_list = pickle.load(open("products.pkl", "rb"))
# print(features_list, img_files_list[8297])

UPLOAD_DIRECTORY = Path("uploads")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])


def sanitize_filename() -> str:
    return f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

def download_image(image_url, save_directory="uploads"):
    try:
        # Make a GET request to fetch the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes (e.g., 404)

        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Use the provided filename or extract it from the URL
        filename = sanitize_filename()
        
        save_path = os.path.join(save_directory, filename)

        # Write the image to the file
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"Image saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"Failed to download image: {e}")
        return None
    

def normalize_query(query):
    # Normalize and encode query
    query_encoded_columns = []
    for col, value in query.items():
        normalized_value = sorted(set(value.split(', ')))
        query_encoded = encoders[col].transform([normalized_value])
        query_encoded_columns.append(query_encoded)
    
    return query_encoded_columns

def find_categorical_similarity(product_row, filtered_data, k):
    query = {
            "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
            "METAL_COLOR": product_row["METAL_COLOR"],
            "COLOR_STONE": product_row["COLOR_STONE"],
            "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
        }
    
    normalize_query_data = normalize_query(query)
    # Combine encoded query features
    query_combined_features = np.hstack(normalize_query_data)

    # Get nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    neighbors.fit(filtered_data)
    # print(combined_features, type(combined_features))
    distence, indices = neighbors.kneighbors(query_combined_features)
    # print(distence, 'ddddddddddddddddddd', indices)
    # distances, indices = r_model.kneighbors(query_combined_features, n_neighbors=20)

    # Fetch recommended ITEM_IDs
    recommended_item_ids = data.iloc[indices[0]]['ITEM_ID'].tolist()
    # print("Recommended ITEM_IDs:", recommended_item_ids)
    return indices


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
    neighbors = NearestNeighbors(n_neighbors=60, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices
    


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


@app.get("/recoomendate/{item_id}")
def get_recommendate(item_id: int):
    if item_id in data["ITEM_ID"].values:
        product_row = data.loc[data["ITEM_ID"] == item_id].iloc[0]
        categorical_prediction = find_categorical_similarity(product_row=product_row)

        image_url = product_row['IMAGE_URL_VIEW_1']
        download_file = download_image(image_url=image_url)
        if download_file is None and len(categorical_prediction) == 0:
            return {"status": False, "message":"Invalid image or item ID, please check again!"}
        features = extract_img_features(download_file, model)
        img_indicess = recommendd(features, features_list)
        img_indicess = img_indicess.tolist()
        image_list = get_indices(img_files_list, img_indicess)

        # Find common elements in the order of `image_based`
        common_elements = [item for item in image_list if int(item) in categorical_prediction]
        common_elements = common_elements[:100]

        return {"status": True, "message":"Predicted successfully.", 'image_based': image_list, 'categorical_based': categorical_prediction, 'array': common_elements}
    else:
        return {"status": False, "message":"Item ID not found, please check again!"}


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: int):
    try:
        product_row = data.loc[data["ITEM_ID"] == id].iloc[0]
    except:
        return templates.TemplateResponse(
        request=request, name="item.html", context={"Error": True}
    )
    p_r = product_row

    image_url = product_row['IMAGE_URL_VIEW_1']
    download_file = download_image(image_url=image_url)
    if download_file is None:
        return {"status": False, "message":"Invalid image or item ID, please check again!"}
    features = extract_img_features(download_file, model)
    img_indicess = recommendd(features, features_list)
    img_indicess = img_indicess.tolist()
    image_list = get_indices(img_files_list, img_indicess)
    filtered_data = []
    image_list2 = [int(item) for item in image_list]
    indices = data.index[data["ITEM_ID"].isin(image_list2)].tolist()
    # print(indices, 'iiiiiiiiiiiiiiiiiiiiiiiiiiiii')
    for ind in indices:
        filtered_data.append(combined_features[ind])
    # print(np.array(combined_features).shape, '--------------------------------------------------------', np.array(filtered_data).shape)
    k = 30
    if len(filtered_data)< 30:
        k = len(filtered_data)
    categorical_prediction = find_categorical_similarity(product_row=product_row, filtered_data=filtered_data, k=k)
    final_indices = []
    for i in categorical_prediction[0]:
        final_indices.append(indices[i])
    final_data_ids = data.iloc[final_indices]['ITEM_ID'].tolist()
    common_elements = [item for item in image_list if int(item) in categorical_prediction]
    common_elements = common_elements[:100]

    search_query = {
            "ITEM_ID": product_row["ITEM_ID"],
            "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
            "METAL_COLOR": product_row["METAL_COLOR"],
            "COLOR_STONE": product_row["COLOR_STONE"],
            "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
            "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
            "ITEM_TYPE": product_row["ITEM_TYPE"],
            "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"]
        }

    image_based = []
    for i in image_list:
        i = int(i)
        product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
        query = {
            "ITEM_ID": product_row["ITEM_ID"],
            "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
            "METAL_COLOR": product_row["METAL_COLOR"],
            "COLOR_STONE": product_row["COLOR_STONE"],
            "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
            "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
            "ITEM_TYPE": product_row["ITEM_TYPE"],
            "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"]
        }
        image_based.append(query)
    attribute_based = []
    for i in final_data_ids:
        product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
        PRODUCT_STYLE = p_r['PRODUCT_STYLE']
        ITEM_TYPE = p_r['ITEM_TYPE']
        if set(sorted(ITEM_TYPE.split(','))) == set(sorted(product_row["ITEM_TYPE"].split(','))):
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"]
            }
            attribute_based.append(query)
    common = []
    for i in common_elements:
        i = int(i)
        product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
        # if set(PRODUCT_STYLE.split(',')) == set(product_row["PRODUCT_STYLE"].split(',')) and set(ITEM_TYPE.split(',')) == set(product_row["ITEM_TYPE"].split(',')):
        query = {
            "ITEM_ID": product_row["ITEM_ID"],
            "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
            "METAL_COLOR": product_row["METAL_COLOR"],
            "COLOR_STONE": product_row["COLOR_STONE"],
            "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
            "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
            "ITEM_TYPE": product_row["ITEM_TYPE"],
            "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"]
        }
        common.append(query)
    return templates.TemplateResponse(
        request=request, name="item.html", context={"image_based": image_based, "attribute_based":attribute_based, "common":common, "search_query":search_query}
    )