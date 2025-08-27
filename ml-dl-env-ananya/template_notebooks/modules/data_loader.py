import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from sklearn.datasets import load_iris
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder

def load_excel_file(excel_file_path):
    df = pd.read_excel(excel_file_path)
    return df

def load_custom_data(file_path):
    df = pd.read_csv(file_path)
    return df

#For built in datasets some are given
def load_mnist():
    (X_train, y_train),(X_test,y_test) = fashion_mnist.load_data()
    return X_train,y_train,X_test,y_test

def load_iris_data():
    data = load_iris(as_frame=True)
    df = data.frame
    return 



def load_image_dataset_npz(path, flatten=False):
    """
    Load image dataset from a .npz file (like MNIST).
    Returns X (image array), y (labels).
    """
    with np.load(path) as data:
        X = data['images'] if 'images' in data else data['x']
        y = data['labels'] if 'labels' in data else data['y']

    if flatten:
        X = X.reshape((X.shape[0], -1))

    return X, y


def load_image_dataset_from_folders(folder_path, image_size=(28, 28)):
    """
    Load image dataset from folders where each subfolder is a class.
    Returns X (image array), y (labels).
    """
    X = []
    y = []
    classes = os.listdir(folder_path)

    for label in classes:
        class_dir = os.path.join(folder_path, label)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                X.append(img)
                y.append(label)

    X = np.array(X)
    X = X.reshape(-1, image_size[0], image_size[1], 1)
    y = LabelEncoder().fit_transform(y)

    return X, y


