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
from rules import apply_lj_product_rule, apply_silver_platinum_rule, apply_exact_matching_rule, distinct_and_sort_by_best_seller, inject_related_style_shapes, aggregate_arrays, get_similar_name_styles, apply_lj_product_rule_df, apply_silver_platinum_rule_df, get_similar_category_style

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

CACHED_RESULT = {}


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
    neighbors = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='euclidean')
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
        global CACHED_RESULT
        try:
            product_row = data.loc[data["ITEM_ID"] == item_id].iloc[0]
        except:
            return {"status": False, "message":"Item ID not found, please check again!"}
        
        if item_id in CACHED_RESULT:
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}

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

        k = 50
        if len(filtered_data)< 50:
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

        array_0_1 = apply_lj_product_rule(attribute_based, data, item_id)
        array_0_2 = apply_silver_platinum_rule(attribute_based, data, item_id)
        array_0 = array_0_1+array_0_2

        final_result = []
        print('RULES ARRAY ------------------------------------------------')
        print(len(array_0))
        print(array_0)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
        
        print('ARRAY-1 ------------------------------------------------')
        array_1 = apply_exact_matching_rule(array_0, data, item_id)
        print('LENGTH: ' ,len(array_1))
        print("ARRAY: ", array_1)
        array_1 = distinct_and_sort_by_best_seller(array_1, data)
        print('LENGTH: ' ,len(array_1))
        print("ARRAY: ", array_1)
        if len(array_1) >= 6:
            array_1_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0.4, action="positive")
            array_1 += distinct_and_sort_by_best_seller(array_1_plus, data)
            injections = inject_related_style_shapes(array_1, data, item_id)
            array_1 += injections
            final_result += array_1
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_1
        print('LENGTH: ' ,len(array_1))
        print("ARRAY: ", array_1)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
        
        print('ARRAY-2 ------------------------------------------------')
        array_2 = apply_exact_matching_rule(array_0, data, item_id, 0.2, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_2))
        print("ARRAY: ", array_2)
        array_2 = distinct_and_sort_by_best_seller(array_2, data)
        print('LENGTH: ' ,len(array_2))
        print("ARRAY: ", array_2)
        if len(array_2) >= 6:
            array_2_plus = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_2 += distinct_and_sort_by_best_seller(array_2_plus, data)
            injections = inject_related_style_shapes(array_2, data, item_id)
            array_2 += injections
            final_result += array_2
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_2
        print('LENGTH: ' ,len(array_2))
        print("ARRAY: ", array_2)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-3 ------------------------------------------------')
        array_3 = apply_exact_matching_rule(array_0, data, item_id, 0.2, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_3))
        print("ARRAY: ", array_3)
        array_3 = distinct_and_sort_by_best_seller(array_3, data)
        print('LENGTH: ' ,len(array_3))
        print("ARRAY: ", array_3)
        if len(array_3) >= 6:
            array_3_plus = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_3 += distinct_and_sort_by_best_seller(array_3_plus, data)
            injections = inject_related_style_shapes(array_3, data, item_id)
            array_3 += injections
            final_result += array_3
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_3
        print('LENGTH: ' ,len(array_3))
        print("ARRAY: ", array_3)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-4 ------------------------------------------------')
        array_4 = apply_exact_matching_rule(array_0, data, item_id, 0.2, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_4))
        print("ARRAY: ", array_4)
        array_4 = distinct_and_sort_by_best_seller(array_4, data)
        print('LENGTH: ' ,len(array_4))
        print("ARRAY: ", array_4)
        if len(array_4) >= 6:
            array_4_plus = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_4 += distinct_and_sort_by_best_seller(array_4_plus, data)
            injections = inject_related_style_shapes(array_4, data, item_id)
            array_4 += injections
            final_result += array_4
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_4
        print('LENGTH: ' ,len(array_4))
        print("ARRAY: ", array_4)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-5 ------------------------------------------------')
        array_5 = apply_exact_matching_rule(array_0, data, item_id, 0.4)
        print('LENGTH: ' ,len(array_5))
        print("ARRAY: ", array_5)
        array_5 = distinct_and_sort_by_best_seller(array_5, data)
        print('LENGTH: ' ,len(array_5))
        print("ARRAY: ", array_5)
        if len(array_5) >= 6:
            array_5_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, action="positive")
            array_5 += distinct_and_sort_by_best_seller(array_5_plus, data)
            injections = inject_related_style_shapes(array_5, data, item_id)
            array_5 += injections
            final_result += array_5
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_5
        print('LENGTH: ' ,len(array_5))
        print("ARRAY: ", array_5)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-6 ------------------------------------------------')
        array_6 = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_6))
        print("ARRAY: ", array_6)
        array_6 = distinct_and_sort_by_best_seller(array_6, data)
        print('LENGTH: ' ,len(array_6))
        print("ARRAY: ", array_6)
        if len(array_6) >= 6:
            array_6_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_6 += distinct_and_sort_by_best_seller(array_6_plus, data)
            injections = inject_related_style_shapes(array_6, data, item_id)
            array_6 += injections
            final_result + array_6
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_6
        print('LENGTH: ' ,len(array_6))
        print("ARRAY: ", array_6)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-7 ------------------------------------------------')
        array_7 = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_7))
        print("ARRAY: ", array_7)
        array_7 = distinct_and_sort_by_best_seller(array_7, data)
        print('LENGTH: ' ,len(array_7))
        print("ARRAY: ", array_7)
        if len(array_7) >= 6:
            array_7_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_7 += distinct_and_sort_by_best_seller(array_7_plus, data)
            injections = inject_related_style_shapes(array_7, data, item_id)
            array_7 += injections
            final_result += array_7
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_7
        print('LENGTH: ' ,len(array_7))
        print("ARRAY: ", array_7)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-8 ------------------------------------------------')
        array_8 = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_8))
        print("ARRAY: ", array_8)
        array_8 = distinct_and_sort_by_best_seller(array_8, data)
        print('LENGTH: ' ,len(array_8))
        print("ARRAY: ", array_8)
        if len(array_8) >= 6:
            array_8_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_8 += distinct_and_sort_by_best_seller(array_8_plus, data)
            injections = inject_related_style_shapes(array_8, data, item_id)
            array_8 += injections
            final_result += array_8
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_8
        print('LENGTH: ' ,len(array_8))
        print("ARRAY: ", array_8)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-9 ------------------------------------------------')
        array_9 = apply_exact_matching_rule(array_0, data, item_id, 0.6)
        print('LENGTH: ' ,len(array_9))
        print("ARRAY: ", array_9)
        array_9 = distinct_and_sort_by_best_seller(array_9, data)
        print('LENGTH: ' ,len(array_9))
        print("ARRAY: ", array_9)
        if len(array_9) >= 6:
            array_9_plus = apply_exact_matching_rule(array_0, data, item_id, 1, action="positive")
            array_9 += distinct_and_sort_by_best_seller(array_9_plus, data)
            injections = inject_related_style_shapes(array_9, data, item_id)
            array_9 += injections
            final_result += array_9
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_9
        print('LENGTH: ' ,len(array_9))
        print("ARRAY: ", array_9)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-10 ------------------------------------------------')
        array_10 = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_10))
        print("ARRAY: ", array_10)
        array_10 = distinct_and_sort_by_best_seller(array_10, data)
        print('LENGTH: ' ,len(array_10))
        print("ARRAY: ", array_10)
        if len(array_10) >= 6:
            array_10_plus = apply_exact_matching_rule(array_0, data, item_id, 1, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_10 += distinct_and_sort_by_best_seller(array_10_plus, data)
            injections = inject_related_style_shapes(array_10, data, item_id)
            array_10 += injections
            final_result += array_10
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_10
        print('LENGTH: ' ,len(array_10))
        print("ARRAY: ", array_10)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-11 ------------------------------------------------')
        array_11 = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_11))
        print("ARRAY: ", array_11)
        array_11 = distinct_and_sort_by_best_seller(array_11, data)
        print('LENGTH: ' ,len(array_11))
        print("ARRAY: ", array_11)
        if len(array_11) >= 6:
            array_11_plus = apply_exact_matching_rule(array_0, data, item_id, 1, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_11 += distinct_and_sort_by_best_seller(array_11_plus, data)
            injections = inject_related_style_shapes(array_11, data, item_id)
            array_11 += injections
            final_result += array_11
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_11
        print('LENGTH: ' ,len(array_11))
        print("ARRAY: ", array_11)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-12 ------------------------------------------------')
        array_12 = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_12))
        print("ARRAY: ", array_12)
        array_12 = distinct_and_sort_by_best_seller(array_12, data)
        print('LENGTH: ' ,len(array_12))
        print("ARRAY: ", array_12)
        if len(array_12) >= 6:
            array_12_plus = apply_exact_matching_rule(array_0, data, item_id, 1, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_12 += distinct_and_sort_by_best_seller(array_12_plus, data)
            injections = inject_related_style_shapes(array_12, data, item_id)
            array_12 += injections
            final_result += array_12
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_12
        print('LENGTH: ' ,len(array_12))
        print("ARRAY: ", array_12)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-13 ------------------------------------------------')
        array_13 = apply_exact_matching_rule(array_0, data, item_id, 1)
        print('LENGTH: ' ,len(array_13))
        print("ARRAY: ", array_13)
        array_13 = distinct_and_sort_by_best_seller(array_13, data)
        print('LENGTH: ' ,len(array_13))
        print("ARRAY: ", array_13)
        if len(array_13) >= 6:
            array_13_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, action="positive")
            array_13 += distinct_and_sort_by_best_seller(array_13_plus, data)
            injections = inject_related_style_shapes(array_13, data, item_id)
            array_13 += injections
            final_result += array_13
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_13
        print('LENGTH: ' ,len(array_13))
        print("ARRAY: ", array_13)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-14 ------------------------------------------------')
        array_14 = apply_exact_matching_rule(array_0, data, item_id, 1, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_14))
        print("ARRAY: ", array_14)
        array_14 = distinct_and_sort_by_best_seller(array_14, data)
        print('LENGTH: ' ,len(array_14))
        print("ARRAY: ", array_14)
        if len(array_14) >= 6:
            array_14_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, base_properties=["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_14 += distinct_and_sort_by_best_seller(array_14_plus, data)
            injections = inject_related_style_shapes(array_14, data, item_id)
            array_14 += injections
            final_result += array_14
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_14
        print('LENGTH: ' ,len(array_14))
        print("ARRAY: ", array_14)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-15 ------------------------------------------------')
        array_15 = apply_exact_matching_rule(array_0, data, item_id, 1, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_15))
        print("ARRAY: ", array_15)
        array_15 = distinct_and_sort_by_best_seller(array_15, data)
        print('LENGTH: ' ,len(array_15))
        print("ARRAY: ", array_15)
        if len(array_15) >= 6:
            array_15_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, base_properties=["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_15 += distinct_and_sort_by_best_seller(array_15_plus, data)
            injections = inject_related_style_shapes(array_15, data, item_id)
            array_15 += injections
            final_result += array_15
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_15
        print('LENGTH: ' ,len(array_15))
        print("ARRAY: ", array_15)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-16 ------------------------------------------------')
        array_16 = apply_exact_matching_rule(array_0, data, item_id, 1, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_16))
        print("ARRAY: ", array_16)
        array_16 = distinct_and_sort_by_best_seller(array_16, data)
        print('LENGTH: ' ,len(array_16))
        print("ARRAY: ", array_16)
        if len(array_16) >= 6:
            array_16_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, base_properties=["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
            array_16 += distinct_and_sort_by_best_seller(array_16_plus, data)
            injections = inject_related_style_shapes(array_16, data, item_id)
            array_16 += injections
            final_result += array_16
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_16
        print('LENGTH: ' ,len(array_16))
        print("ARRAY: ", array_16)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-17 ------------------------------------------------')
        array_17 = apply_exact_matching_rule(array_0, data, item_id, 0)
        print('LENGTH: ' ,len(array_17))
        print("ARRAY: ", array_17)
        array_17 = distinct_and_sort_by_best_seller(array_17, data)
        print('LENGTH: ' ,len(array_17))
        print("ARRAY: ", array_17)
        if len(array_17) >= 6:
            injections = inject_related_style_shapes(array_17, data, item_id)
            array_17 += injections
            final_result += array_17
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_17
        print('LENGTH: ' ,len(array_17))
        print("ARRAY: ", array_17)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-18 ------------------------------------------------')
        array_18 = apply_exact_matching_rule(array_0, data, item_id, 0, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_18))
        print("ARRAY: ", array_18)
        array_18 = distinct_and_sort_by_best_seller(array_18, data)
        print('LENGTH: ' ,len(array_18))
        print("ARRAY: ", array_18)
        if len(array_18) >= 6:
            injections = inject_related_style_shapes(array_18, data, item_id)
            array_18 += injections
            final_result += array_18
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_18
        print('LENGTH: ' ,len(array_18))
        print("ARRAY: ", array_18)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-19 ------------------------------------------------')
        array_19 = apply_exact_matching_rule(array_0, data, item_id, 0, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_19))
        print("ARRAY: ", array_19)
        array_19 = distinct_and_sort_by_best_seller(array_19, data)
        print('LENGTH: ' ,len(array_19))
        print("ARRAY: ", array_19)
        if len(array_19) >= 6:
            injections = inject_related_style_shapes(array_19, data, item_id)
            array_19 += injections
            final_result += array_19
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_19
        print('LENGTH: ' ,len(array_19))
        print("ARRAY: ", array_19)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-20 ------------------------------------------------')
        array_20 = apply_exact_matching_rule(array_0, data, item_id, 0, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
        print('LENGTH: ' ,len(array_20))
        print("ARRAY: ", array_20)
        array_20 = distinct_and_sort_by_best_seller(array_20, data)
        print('LENGTH: ' ,len(array_20))
        print("ARRAY: ", array_20)
        if len(array_20) >= 6:
            injections = inject_related_style_shapes(array_20, data, item_id)
            array_20 += injections
            final_result += array_20
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_20
        print('LENGTH: ' ,len(array_20))
        print("ARRAY: ", array_20)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-21 ------------------------------------------------')
        injections = inject_related_style_shapes(final_result, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_21_1 = apply_lj_product_rule_df(data, item_id)
        array_21_2 = apply_silver_platinum_rule_df(data, item_id)
        array_21_ = list(set(array_21_1+array_21_2))
        array_21 = get_similar_name_styles(array_21_, data, item_id)
        print('LENGTH: ' ,len(array_21))
        print("ARRAY: ", array_21)
        array_21 = distinct_and_sort_by_best_seller(array_21, data)
        print('LENGTH: ' ,len(array_21))
        print("ARRAY: ", array_21)
        array_21 += injections
        if len(array_21) >= 6:
            # array_21 += injections
            final_result += array_21
            CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
            return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_21
        print('LENGTH: ' ,len(array_21))
        print("ARRAY: ", array_21)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print('ARRAY-22 ------------------------------------------------')
        array_22_1 = apply_lj_product_rule_df(data, item_id)
        array_22_2 = apply_silver_platinum_rule_df(data, item_id)
        array_22_ = list(set(array_22_1+array_22_2))
        print(len(array_22_), '---------------------------')
        array_22 = get_similar_category_style(array_22_, data, item_id)
        print('LENGTH: ' ,len(array_22))
        print("ARRAY: ", array_22)
        array_22 = distinct_and_sort_by_best_seller(array_22, data)
        print('LENGTH: ' ,len(array_22))
        print("ARRAY: ", array_22)
        # if len(array_22) >= 6:
        #     injections = inject_related_style_shapes(array_22, data, item_id)
        #     print(len(injections), 'jjjjjjjjjjjjjjjjjjjjjjjj')
        #     array_22 += injections
        #     final_result += array_22
        #     CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        #     return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
        final_result += array_22
        print('LENGTH: ' ,len(array_22))
        print("ARRAY: ", array_22)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

        # final_array = array_1+array_2+array_3+array_4+array_5+array_6+array_7+array_8+array_9+array_10+array_11+array_12+array_13+array_14+array_15+array_16+array_17+array_18+array_19+array_20+array_21+array_22
        print('ffffffffffffffffffffffffffffffffffffff')
        print('LENGTH: ' ,len(final_result))
        print("ARRAY: ", final_result)
        # attribute_based = inject_related_style_shapes(attribute_based, data, item_id)

        # final_array = aggregate_arrays(final_result)
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        print('ffffffffffffffffffffffffffffffffffffff')
        print('LENGTH: ' ,len(CACHED_RESULT[item_id]))
        print("ARRAY: ", CACHED_RESULT[item_id])

        return {"status": True, "message":"Predicted successfully.", 'array': CACHED_RESULT[item_id]}
    else:
        return {"status": False, "message":"Item ID not found, please check again!"}

def reapply(array_0, item_id):
    array_1 = apply_exact_matching_rule(array_0, data, item_id)
    array_1 = distinct_and_sort_by_best_seller(array_1, data)

@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: int):
    global CACHED_RESULT
    try:
        product_row = data.loc[data["ITEM_ID"] == id].iloc[0]
    except:
        return templates.TemplateResponse(
        request=request, name="item.html", context={"Error": True}
    )

    search_query = {
            "ITEM_ID": product_row["ITEM_ID"],
            "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
            "METAL_COLOR": product_row["METAL_COLOR"],
            "COLOR_STONE": product_row["COLOR_STONE"],
            "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
            "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
            "ITEM_TYPE": product_row["ITEM_TYPE"],
            "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
            "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
        }
    
    if id in CACHED_RESULT:
        attribute_based = []
        for i in CACHED_RESULT[id]:
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
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

    k = 50
    if len(filtered_data)< 50:
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
    
    item_id = id
    array_0_1 = apply_lj_product_rule(attribute_based, data, item_id)
    array_0_2 = apply_silver_platinum_rule(attribute_based, data, item_id)
    array_0 = array_0_1+array_0_2

    final_result = []
    print('RULES ARRAY ------------------------------------------------')
    print(len(array_0))
    print(array_0)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    print('ARRAY-1 ------------------------------------------------')
    array_1 = apply_exact_matching_rule(array_0, data, item_id)
    print('LENGTH: ' ,len(array_1))
    print("ARRAY: ", array_1)
    array_1 = distinct_and_sort_by_best_seller(array_1, data)
    print('LENGTH: ' ,len(array_1))
    print("ARRAY: ", array_1)
    if len(array_1) >= 6:
        array_1_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0.4, action="positive")
        array_1 += distinct_and_sort_by_best_seller(array_1_plus, data)
        print('LENGTH: ' ,len(array_1))
        print("ARRAY: ", array_1)
        injections = inject_related_style_shapes(array_1, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_1 += injections
        final_result += array_1
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_1
    print('LENGTH: ' ,len(array_1))
    print("ARRAY: ", array_1)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    print('ARRAY-2 ------------------------------------------------')
    array_2 = apply_exact_matching_rule(array_0, data, item_id, 0.2, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_2))
    print("ARRAY: ", array_2)
    array_2 = distinct_and_sort_by_best_seller(array_2, data)
    print('LENGTH: ' ,len(array_2))
    print("ARRAY: ", array_2)
    if len(array_2) >= 6:
        array_2_plus = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_2 += distinct_and_sort_by_best_seller(array_2_plus, data)
        print('LENGTH: ' ,len(array_2))
        print("ARRAY: ", array_2)
        injections = inject_related_style_shapes(array_2, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_2 += injections
        final_result += array_2
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_2
    print('LENGTH: ' ,len(array_2))
    print("ARRAY: ", array_2)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-3 ------------------------------------------------')
    array_3 = apply_exact_matching_rule(array_0, data, item_id, 0.2, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_3))
    print("ARRAY: ", array_3)
    array_3 = distinct_and_sort_by_best_seller(array_3, data)
    print('LENGTH: ' ,len(array_3))
    print("ARRAY: ", array_3)
    if len(array_3) >= 6:
        array_3_plus = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_3 += distinct_and_sort_by_best_seller(array_3_plus, data)
        injections = inject_related_style_shapes(array_3, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_3 += injections
        final_result += array_3
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_3
    print('LENGTH: ' ,len(array_3))
    print("ARRAY: ", array_3)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-4 ------------------------------------------------')
    array_4 = apply_exact_matching_rule(array_0, data, item_id, 0.2, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_4))
    print("ARRAY: ", array_4)
    array_4 = distinct_and_sort_by_best_seller(array_4, data)
    print('LENGTH: ' ,len(array_4))
    print("ARRAY: ", array_4)
    if len(array_4) >= 6:
        array_4_plus = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_4 += distinct_and_sort_by_best_seller(array_4_plus, data)
        injections = inject_related_style_shapes(array_4, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_4 += injections
        final_result += array_4
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_4
    print('LENGTH: ' ,len(array_4))
    print("ARRAY: ", array_4)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-5 ------------------------------------------------')
    array_5 = apply_exact_matching_rule(array_0, data, item_id, 0.4)
    print('LENGTH: ' ,len(array_5))
    print("ARRAY: ", array_5)
    array_5 = distinct_and_sort_by_best_seller(array_5, data)
    print('LENGTH: ' ,len(array_5))
    print("ARRAY: ", array_5)
    if len(array_5) >= 6:
        array_5_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, action="positive")
        array_5 += distinct_and_sort_by_best_seller(array_5_plus, data)
        injections = inject_related_style_shapes(array_5, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_5 += injections
        final_result += array_5
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_5
    print('LENGTH: ' ,len(array_5))
    print("ARRAY: ", array_5)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-6 ------------------------------------------------')
    array_6 = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_6))
    print("ARRAY: ", array_6)
    array_6 = distinct_and_sort_by_best_seller(array_6, data)
    print('LENGTH: ' ,len(array_6))
    print("ARRAY: ", array_6)
    if len(array_6) >= 6:
        array_6_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_6 += distinct_and_sort_by_best_seller(array_6_plus, data)
        injections = inject_related_style_shapes(array_6, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_6 += injections
        final_result + array_6
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_6
    print('LENGTH: ' ,len(array_6))
    print("ARRAY: ", array_6)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-7 ------------------------------------------------')
    array_7 = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_7))
    print("ARRAY: ", array_7)
    array_7 = distinct_and_sort_by_best_seller(array_7, data)
    print('LENGTH: ' ,len(array_7))
    print("ARRAY: ", array_7)
    if len(array_7) >= 6:
        array_7_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_7 += distinct_and_sort_by_best_seller(array_7_plus, data)
        injections = inject_related_style_shapes(array_7, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_7 += injections
        final_result += array_7
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_7
    print('LENGTH: ' ,len(array_7))
    print("ARRAY: ", array_7)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-8 ------------------------------------------------')
    array_8 = apply_exact_matching_rule(array_0, data, item_id, 0.4, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_8))
    print("ARRAY: ", array_8)
    array_8 = distinct_and_sort_by_best_seller(array_8, data)
    print('LENGTH: ' ,len(array_8))
    print("ARRAY: ", array_8)
    if len(array_8) >= 6:
        array_8_plus = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_8 += distinct_and_sort_by_best_seller(array_8_plus, data)
        injections = inject_related_style_shapes(array_8, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_8 += injections
        final_result += array_8
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_8
    print('LENGTH: ' ,len(array_8))
    print("ARRAY: ", array_8)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-9 ------------------------------------------------')
    array_9 = apply_exact_matching_rule(array_0, data, item_id, 0.6)
    print('LENGTH: ' ,len(array_9))
    print("ARRAY: ", array_9)
    array_9 = distinct_and_sort_by_best_seller(array_9, data)
    print('LENGTH: ' ,len(array_9))
    print("ARRAY: ", array_9)
    if len(array_9) >= 6:
        array_9_plus = apply_exact_matching_rule(array_0, data, item_id, 1, action="positive")
        array_9 += distinct_and_sort_by_best_seller(array_9_plus, data)
        injections = inject_related_style_shapes(array_9, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_9 += injections
        final_result += array_9
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_9
    print('LENGTH: ' ,len(array_9))
    print("ARRAY: ", array_9)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-10 ------------------------------------------------')
    array_10 = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_10))
    print("ARRAY: ", array_10)
    array_10 = distinct_and_sort_by_best_seller(array_10, data)
    print('LENGTH: ' ,len(array_10))
    print("ARRAY: ", array_10)
    if len(array_10) >= 6:
        array_10_plus = apply_exact_matching_rule(array_0, data, item_id, 1, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_10 += distinct_and_sort_by_best_seller(array_10_plus, data)
        injections = inject_related_style_shapes(array_10, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_10 += injections
        final_result += array_10
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_10
    print('LENGTH: ' ,len(array_10))
    print("ARRAY: ", array_10)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-11 ------------------------------------------------')
    array_11 = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_11))
    print("ARRAY: ", array_11)
    array_11 = distinct_and_sort_by_best_seller(array_11, data)
    print('LENGTH: ' ,len(array_11))
    print("ARRAY: ", array_11)
    if len(array_11) >= 6:
        array_11_plus = apply_exact_matching_rule(array_0, data, item_id, 1, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_11 += distinct_and_sort_by_best_seller(array_11_plus, data)
        injections = inject_related_style_shapes(array_11, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_11 += injections
        final_result += array_11
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_11
    print('LENGTH: ' ,len(array_11))
    print("ARRAY: ", array_11)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-12 ------------------------------------------------')
    array_12 = apply_exact_matching_rule(array_0, data, item_id, 0.6, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_12))
    print("ARRAY: ", array_12)
    array_12 = distinct_and_sort_by_best_seller(array_12, data)
    print('LENGTH: ' ,len(array_12))
    print("ARRAY: ", array_12)
    if len(array_12) >= 6:
        array_12_plus = apply_exact_matching_rule(array_0, data, item_id, 1, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_12 += distinct_and_sort_by_best_seller(array_12_plus, data)
        injections = inject_related_style_shapes(array_12, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_12 += injections
        final_result += array_12
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_12
    print('LENGTH: ' ,len(array_12))
    print("ARRAY: ", array_12)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-13 ------------------------------------------------')
    array_13 = apply_exact_matching_rule(array_0, data, item_id, 1)
    print('LENGTH: ' ,len(array_13))
    print("ARRAY: ", array_13)
    array_13 = distinct_and_sort_by_best_seller(array_13, data)
    print('LENGTH: ' ,len(array_13))
    print("ARRAY: ", array_13)
    if len(array_13) >= 6:
        array_13_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, action="positive")
        array_13 += distinct_and_sort_by_best_seller(array_13_plus, data)
        injections = inject_related_style_shapes(array_13, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_13 += injections
        final_result += array_13
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_13
    print('LENGTH: ' ,len(array_13))
    print("ARRAY: ", array_13)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-14 ------------------------------------------------')
    array_14 = apply_exact_matching_rule(array_0, data, item_id, 1, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_14))
    print("ARRAY: ", array_14)
    array_14 = distinct_and_sort_by_best_seller(array_14, data)
    print('LENGTH: ' ,len(array_14))
    print("ARRAY: ", array_14)
    if len(array_14) >= 6:
        array_14_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, base_properties=["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_14 += distinct_and_sort_by_best_seller(array_14_plus, data)
        injections = inject_related_style_shapes(array_14, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_14 += injections
        final_result += array_14
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_14
    print('LENGTH: ' ,len(array_14))
    print("ARRAY: ", array_14)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-15 ------------------------------------------------')
    array_15 = apply_exact_matching_rule(array_0, data, item_id, 1, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_15))
    print("ARRAY: ", array_15)
    array_15 = distinct_and_sort_by_best_seller(array_15, data)
    print('LENGTH: ' ,len(array_15))
    print("ARRAY: ", array_15)
    if len(array_15) >= 6:
        array_15_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, base_properties=["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_15 += distinct_and_sort_by_best_seller(array_15_plus, data)
        injections = inject_related_style_shapes(array_15, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_15 += injections
        final_result += array_15
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_15
    print('LENGTH: ' ,len(array_15))
    print("ARRAY: ", array_15)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-16 ------------------------------------------------')
    array_16 = apply_exact_matching_rule(array_0, data, item_id, 1, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_16))
    print("ARRAY: ", array_16)
    array_16 = distinct_and_sort_by_best_seller(array_16, data)
    print('LENGTH: ' ,len(array_16))
    print("ARRAY: ", array_16)
    if len(array_16) >= 6:
        array_16_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, base_properties=["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        array_16 += distinct_and_sort_by_best_seller(array_16_plus, data)
        injections = inject_related_style_shapes(array_16, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_16 += injections
        final_result += array_16
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_16
    print('LENGTH: ' ,len(array_16))
    print("ARRAY: ", array_16)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-17 ------------------------------------------------')
    array_17 = apply_exact_matching_rule(array_0, data, item_id, 0)
    print('LENGTH: ' ,len(array_17))
    print("ARRAY: ", array_17)
    array_17 = distinct_and_sort_by_best_seller(array_17, data)
    print('LENGTH: ' ,len(array_17))
    print("ARRAY: ", array_17)
    if len(array_17) >= 6:
        # array_17_plus = apply_exact_matching_rule(array_0, data, item_id, price_tolerance=0, base_properties=["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"], action="positive")
        # array_17 += distinct_and_sort_by_best_seller(array_17_plus, data)
        injections = inject_related_style_shapes(array_17, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_17 += injections
        final_result += array_17
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_17
    print('LENGTH: ' ,len(array_17))
    print("ARRAY: ", array_17)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-18 ------------------------------------------------')
    array_18 = apply_exact_matching_rule(array_0, data, item_id, 0, ["METAL_KARAT_DISPLAY", "COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_18))
    print("ARRAY: ", array_18)
    array_18 = distinct_and_sort_by_best_seller(array_18, data)
    print('LENGTH: ' ,len(array_18))
    print("ARRAY: ", array_18)
    if len(array_18) >= 6:
        injections = inject_related_style_shapes(array_18, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_18 += injections
        final_result += array_18
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_18
    print('LENGTH: ' ,len(array_18))
    print("ARRAY: ", array_18)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-19 ------------------------------------------------')
    array_19 = apply_exact_matching_rule(array_0, data, item_id, 0, ["COLOR_STONE", "CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_19))
    print("ARRAY: ", array_19)
    array_19 = distinct_and_sort_by_best_seller(array_19, data)
    print('LENGTH: ' ,len(array_19))
    print("ARRAY: ", array_19)
    if len(array_19) >= 6:
        injections = inject_related_style_shapes(array_19, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_19 += injections
        final_result += array_19
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_19
    print('LENGTH: ' ,len(array_19))
    print("ARRAY: ", array_19)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-20 ------------------------------------------------')
    array_20 = apply_exact_matching_rule(array_0, data, item_id, 0, ["CATEGORY_TYPE", "ITEM_TYPE", "PRODUCT_STYLE"])
    print('LENGTH: ' ,len(array_20))
    print("ARRAY: ", array_20)
    array_20 = distinct_and_sort_by_best_seller(array_20, data)
    print('LENGTH: ' ,len(array_20))
    print("ARRAY: ", array_20)
    if len(array_20) >= 6:
        injections = inject_related_style_shapes(array_20, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        array_20 += injections
        final_result += array_20
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_20
    print('LENGTH: ' ,len(array_20))
    print("ARRAY: ", array_20)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-21 ------------------------------------------------')
    injections = inject_related_style_shapes(final_result, data, item_id)
    print('LENGTH injections: ' ,len(injections))
    print("ARRAY injections: ", injections)
    array_21_1 = apply_lj_product_rule_df(data, item_id)
    array_21_2 = apply_silver_platinum_rule_df(data, item_id)
    array_21_ = list(set(array_21_1+array_21_2))
    array_21 = get_similar_name_styles(array_21_, data, item_id)
    print('LENGTH: ' ,len(array_21))
    print("ARRAY: ", array_21)
    array_21 = distinct_and_sort_by_best_seller(array_21, data)
    print('LENGTH: ' ,len(array_21))
    print("ARRAY: ", array_21)
    array_21 += injections
    if len(array_21) >= 6:
        # injections = inject_related_style_shapes(array_21, data, item_id)
        print('LENGTH injections: ' ,len(injections))
        print("ARRAY injections: ", injections)
        # array_21 += injections
        final_result += array_21
        try:
            final_result.remove(item_id)
        except:pass
        CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
        attribute_based = []
        for i in aggregate_arrays(item_id, final_result):
            product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
            query = {
                "ITEM_ID": product_row["ITEM_ID"],
                "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
                "METAL_COLOR": product_row["METAL_COLOR"],
                "COLOR_STONE": product_row["COLOR_STONE"],
                "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
                "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
                "ITEM_TYPE": product_row["ITEM_TYPE"],
                "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
                "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
            }
            attribute_based.append(query)

        return templates.TemplateResponse(
            request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
        )
    final_result += array_21
    print('LENGTH: ' ,len(array_21))
    print("ARRAY: ", array_21)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ARRAY-22 ------------------------------------------------')
    array_22_1 = apply_lj_product_rule_df(data, item_id)
    array_22_2 = apply_silver_platinum_rule_df(data, item_id)
    array_22_ = list(set(array_22_1+array_22_2))
    print(len(array_22_), '---------------------------')
    array_22 = get_similar_category_style(array_22_, data, item_id)
    print('LENGTH: ' ,len(array_22))
    print("ARRAY: ", array_22)
    array_22 = distinct_and_sort_by_best_seller(array_22, data)
    print('LENGTH: ' ,len(array_22))
    print("ARRAY: ", array_22)
    # if len(array_22) >= 6:
    #     injections = inject_related_style_shapes(array_22, data, item_id)
    #     print('LENGTH injections: ' ,len(injections))
    #     print("ARRAY injections: ", injections)
    #     array_22 += injections
    #     final_result += array_22
    #     try:
    #         final_result.remove(item_id)
    #     except:pass
    #     CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
    #     attribute_based = []
    #     for i in aggregate_arrays(item_id, final_result):
    #         product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
    #         query = {
    #             "ITEM_ID": product_row["ITEM_ID"],
    #             "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
    #             "METAL_COLOR": product_row["METAL_COLOR"],
    #             "COLOR_STONE": product_row["COLOR_STONE"],
    #             "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
    #             "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
    #             "ITEM_TYPE": product_row["ITEM_TYPE"],
    #             "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
    #             "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
    #         }
    #         attribute_based.append(query)

    #     return templates.TemplateResponse(
    #         request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
    #     )
    final_result += array_22
    try:
        final_result.remove(item_id)
    except:pass
    CACHED_RESULT[item_id] = aggregate_arrays(item_id, final_result)
    print('LENGTH: ' ,len(array_22))
    print("ARRAY: ", array_22)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

    print('ffffffffffffffffffffffffffffffffffffff')
    print('LENGTH: ' ,len(final_result))
    print("ARRAY: ", final_result)

    final_array = aggregate_arrays(item_id, final_result)
    print('ffffffffffffffffffffffffffffffffffffff')
    print('LENGTH: ' ,len(final_array))
    print("ARRAY: ", final_array)

    

    attribute_based = []
    for i in final_array:
        product_row = data.loc[data["ITEM_ID"] == i].iloc[0]
        query = {
            "ITEM_ID": product_row["ITEM_ID"],
            "METAL_KARAT_DISPLAY": product_row["METAL_KARAT_DISPLAY"],
            "METAL_COLOR": product_row["METAL_COLOR"],
            "COLOR_STONE": product_row["COLOR_STONE"],
            "CATEGORY_TYPE": product_row["CATEGORY_TYPE"],
            "PRODUCT_STYLE": product_row["PRODUCT_STYLE"],
            "ITEM_TYPE": product_row["ITEM_TYPE"],
            "IMAGE_URL_VIEW_1": product_row["IMAGE_URL_VIEW_1"],
            "C_LEVEL_PRICE": product_row["C_LEVEL_PRICE"]
        }
        attribute_based.append(query)

    return templates.TemplateResponse(
        request=request, name="item.html", context={"attribute_based":attribute_based, "search_query":search_query}
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