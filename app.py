from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates  
from fastapi import UploadFile
from fastapi import File
from fastapi import Response

import pandas as pd
import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import joblib
from pathlib import Path
from datetime import datetime

app = FastAPI()

templates = Jinja2Templates(directory="templates")

data = pd.read_csv("ASHI_FINAL_DATA.csv")

# Load saved encoders and model
encoders = joblib.load('column_encoders.pkl')
combined_features = joblib.load('combined_features.pkl')

features_list = pickle.load(open("embeddings.pkl", "rb"))
img_files_list = pickle.load(open("products.pkl", "rb"))

UPLOAD_DIRECTORY = Path("uploads")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


def sanitize_filename() -> str:
    return f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

"""
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
"""

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
            "ITEM_TYPE": product_row["ITEM_TYPE"]
        }
    
    normalize_query_data = normalize_query(query)
    # Combine encoded query features
    query_combined_features = np.hstack(normalize_query_data)

    # Get nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    neighbors.fit(filtered_data)
    distence, indices = neighbors.kneighbors(query_combined_features)
    # recommended_item_ids = data.iloc[indices[0]]['ITEM_ID'].tolist()
    return indices


"""
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
"""

def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=150, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors(features)

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



@app.get("/recoomendate/{item_id}")
def get_recommendate(item_id: int):
    if item_id in data["ITEM_ID"].values:
        try:
            product_row = data.loc[data["ITEM_ID"] == item_id].iloc[0]
        except:
            return {"status": False, "message":"Item ID not found, please check again!"}

        p_r = product_row

        product_index = data.index[data["ITEM_ID"] == item_id].to_list()[0]


        features_list_array = np.array(features_list)
        features = np.array([features_list_array[product_index]])

        normalized_features_1 = features.reshape(features.shape[0], -1)
        normalized_features_2 = features_list_array.reshape(features_list_array.shape[0], -1)
        
        img_indicess = recommendd(normalized_features_1, normalized_features_2)
        img_indicess = img_indicess.tolist()
        image_list = get_indices(img_files_list, img_indicess)
        filtered_data = []
        for ind in img_indicess[0]:
            filtered_data.append(combined_features[ind])

        k = 150
        if len(filtered_data)< 150:
            k = len(filtered_data)
        categorical_prediction = find_categorical_similarity(product_row=product_row, filtered_data=filtered_data, k=k)
        
        final_indices = []
        for i in categorical_prediction[0]:
            final_indices.append(img_indicess[0][i])
        final_data_ids = data.iloc[final_indices]['ITEM_ID'].tolist()

        attribute_based = []
        for i in final_data_ids:
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            PRODUCT_STYLE = p_r['PRODUCT_STYLE']
            ITEM_TYPE = p_r['ITEM_TYPE']
            Category_type = p_r['CATEGORY_TYPE']
            if set(sorted(ITEM_TYPE.split(','))) == set(sorted(product_row["ITEM_TYPE"].split(','))) and set(sorted(Category_type.split(','))) == set(sorted(product_row["CATEGORY_TYPE"].split(','))):
                attribute_based.append(i)

        return {"status": True, "message":"Predicted successfully.", 'array': attribute_based}
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

    product_index = data.index[data["ITEM_ID"] == id].to_list()[0]


    features_list_array = np.array(features_list)
    features = np.array([features_list_array[product_index]])

    normalized_features_1 = features.reshape(features.shape[0], -1)
    normalized_features_2 = features_list_array.reshape(features_list_array.shape[0], -1)
    # print(features_list, 'fffffffffffffffffff')
    
    img_indicess = recommendd(normalized_features_1, normalized_features_2)
    img_indicess = img_indicess.tolist()
    image_list = get_indices(img_files_list, img_indicess)
    filtered_data = []
    # image_list2 = [int(item) for item in image_list]
    # indices = data.index[data["ITEM_ID"].isin(image_list2)].tolist()
    for ind in img_indicess[0]:
        filtered_data.append(combined_features[ind])

    k = 150
    if len(filtered_data)< 150:
        k = len(filtered_data)
    categorical_prediction = find_categorical_similarity(product_row=product_row, filtered_data=filtered_data, k=k)
    
    final_indices = []
    for i in categorical_prediction[0]:
        final_indices.append(img_indicess[0][i])
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
        Category_type = p_r['CATEGORY_TYPE']
        if set(sorted(ITEM_TYPE.split(','))) == set(sorted(product_row["ITEM_TYPE"].split(','))) and set(sorted(Category_type.split(','))) == set(sorted(product_row["CATEGORY_TYPE"].split(','))):
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
    print(len(attribute_based), 'llllllllllllllllllllllllll')
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



import csv
def calculate_prediction_count():
    final_data = []
    for index, row in data.iterrows():
        id = row['ITEM_ID']
        print("PROCESSING: ", id, '------------INDEX----------------- ', index)
        try:
            product_row = data.loc[data["ITEM_ID"] == id].iloc[0]
        except:
            continue

        p_r = product_row

        product_index = data.index[data["ITEM_ID"] == id].to_list()[0]


        features_list_array = np.array(features_list)
        features = np.array([features_list_array[product_index]])

        normalized_features_1 = features.reshape(features.shape[0], -1)
        normalized_features_2 = features_list_array.reshape(features_list_array.shape[0], -1)

        img_indicess = recommendd(normalized_features_1, normalized_features_2)
        img_indicess = img_indicess.tolist()
        image_list = get_indices(img_files_list, img_indicess)
        filtered_data = []
        for ind in img_indicess[0]:
            filtered_data.append(combined_features[ind])

        k = 100
        if len(filtered_data)< 100:
            k = len(filtered_data)
        categorical_prediction = find_categorical_similarity(product_row=product_row, filtered_data=filtered_data, k=k)

        final_indices = []
        for i in categorical_prediction[0]:
            final_indices.append(img_indicess[0][i])
        final_data_ids = data.iloc[final_indices]['ITEM_ID'].tolist()
        final_data.append({
            "input_id": id,
            "count": len(final_data_ids)
        })
        print("PROCESSED: ", id, '++++++++++++++++++++++++++++INDEX++++++++++++++++++++++++++ ', index)
    
    with open('prediction_count.csv', 'w', newline='') as csvfile:
        fieldnames = ['input_id', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_data)


# calculate_prediction_count()