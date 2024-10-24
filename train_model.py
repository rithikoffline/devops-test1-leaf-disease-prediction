import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import cv2

# Load the CSV file
data = pd.read_csv('/path/to/leaf.csv')

# Paths and Labels
image_paths = data['image_path_column']  # Replace 'image_path_column' with the actual column name
labels = data['label_column']  # Replace 'label_column' with the actual column name

# Image size and batch size
IMG_SIZE = 128
BATCH_SIZE = 32

# Preprocessing Function
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize pixel values
    return img

# Preprocess images and labels
images = []
for path in tqdm(image_paths):
    images.append(preprocess_image(path))
images = np.array(images)

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(i
