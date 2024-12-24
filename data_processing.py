import os
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from numpy.linalg import norm
import cv2
import pickle
import boto3
import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


BUCKET_NAME = 'ashi-similar-styles-ai-engine'
DATA_PREFIX = 'v1/'

s3 = boto3.client('s3')

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    s3.upload_file(file_name, bucket, object_name)


def augment_image(img_path):
    img = cv2.imread(img_path)  # Load the image

    # Split the image into 4 quadrants
    height, width, _ = img.shape
    top_left = img[:height//2, :width//2]
    top_right = img[:height//2, width//2:]
    bottom_left = img[height//2:, :width//2]
    bottom_right = img[height//2:, width//2:]
    # Resize the split parts to be used as separate images
    images = [img, top_left, top_right, bottom_left]
    
    # Zoom: Zoom in by cropping a central region and resizing back
    zoomed_in = cv2.resize(img[height//4:3*height//4, width//4:3*width//4], (224, 224))

    # Rotate: Rotate the image by 45 degrees
    M = cv2.getRotationMatrix2D((width//2, height//2), 45, 1)
    rotated = cv2.warpAffine(img, M, (width, height))
    rotated = cv2.resize(rotated, (224, 224))
    images.append(rotated)

    M = cv2.getRotationMatrix2D((width//2, height//2), 90, 1)
    rotated = cv2.warpAffine(img, M, (width, height))
    rotated = cv2.resize(rotated, (224, 224))
    images.append(rotated)

    M = cv2.getRotationMatrix2D((width//2, height//2), 180, 1)
    rotated = cv2.warpAffine(img, M, (width, height))
    rotated = cv2.resize(rotated, (224, 224))
    images.append(rotated)

    M = cv2.getRotationMatrix2D((width//2, height//2), 270, 1)
    rotated = cv2.warpAffine(img, M, (width, height))
    rotated = cv2.resize(rotated, (224, 224))
    images.append(rotated)

    # Combine the original and augmented images
    images.append(zoomed_in)
    
    return images

# Function to normalize comma-separated values
def normalize_column(column):
    return column.apply(lambda x: sorted(set(x.split(', '))))

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize to (224, 224)
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        processed_images.append(normalized_result)
    
    return np.vstack(processed_images)  # Stack all images together


def download_image(row):
    try:
        product_id = row['ITEM_ID']
        image_url = row['IMAGE_URL_VIEW_1']
        image_path = f"data/images/{product_id}.jpg"
        
        s3.download_file(BUCKET_NAME, DATA_PREFIX + image_path, image_path)
        return "success"
    except Exception as e:
        print(f"Failed to download {row['ITEM_ID']}: {e}")
        return "failure"
    

def download_images(df):
    os.makedirs('data/images', exist_ok=True)
    with ThreadPoolExecutor(max_workers=100) as executor:
            results = list(tqdm(executor.map(download_image, [row for _, row in df.iterrows()]), total=len(df)))
        
    success_count = results.count("success")
    failure_count = results.count("failure")

    print(f"Successes: {success_count}, Failures: {failure_count}")


def start_training(df):
    
    products = []
    feature_list = []
    # Initialize encoders for each column
    encoders = {}
    encoded_columns = []
    count = 0
    for index, row in df.iterrows():
        products.append(row['ITEM_ID'])
        image_path = f"data/images/{row['ITEM_ID']}.jpg"

        images = augment_image(image_path)
        features = preprocess_images(images)
        feature_list.append(features)
        count += 1
        print(count, 'IMAGES are proceed---------------------------------')

    # Normalize all categorical columns
    for col in df.columns[1:-2]:  # Skip ITEM_ID and IMAGE_URL_VIEW_1, PRODUCT_STYLE, ITEM_TYPE
        df[col] = normalize_column(df[col])

    for col in df.columns[1:-2]:  # Skip ITEM_ID and IMAGE_URL_VIEW_1, PRODUCT_STYLE, ITEM_TYPE
        encoder = MultiLabelBinarizer()
        encoded_col = encoder.fit_transform(df[col])
        encoders[col] = encoder  # Save encoder for later use
        encoded_columns.append(encoded_col)

    
    
    combined_features = np.hstack(encoded_columns)

    joblib.dump(encoders, 'column_encoders.pkl')
    print("column_encoders.pkl saved")
    joblib.dump(combined_features, 'combined_features.pkl')
    print("combined_features.pkl saved")

    pickle.dump(feature_list,open('embeddings.pkl','wb'))
    print("embeddings.pkl saved")
    pickle.dump(products,open('products.pkl','wb'))
    print("products.pkl saved")
    
if __name__ == "__main__":
    
    s3.download_file(BUCKET_NAME, f'{DATA_PREFIX}data/ASHI_FINAL_DATA.csv', 'ASHI_FINAL_DATA.csv')
    df = pd.read_csv('ASHI_FINAL_DATA.csv')
    df = df[['ITEM_ID', 'METAL_KARAT_DISPLAY', 'METAL_COLOR', 'COLOR_STONE',
       'CATEGORY_TYPE', 'ITEM_TYPE', 'PRODUCT_STYLE', 'IMAGE_URL_VIEW_1']]
    download_images(df)
    start_training(df)
    save_to_s3('embeddings.pkl', BUCKET_NAME, f'{DATA_PREFIX}embeddings/embeddings.pkl')
    save_to_s3('products.pkl', BUCKET_NAME, f'{DATA_PREFIX}embeddings/products.pkl')
    save_to_s3('column_encoders.pkl', BUCKET_NAME, f'{DATA_PREFIX}embeddings/column_encoders.pkl')
    save_to_s3('combined_features.pkl', BUCKET_NAME, f'{DATA_PREFIX}embeddings/combined_features.pkl')