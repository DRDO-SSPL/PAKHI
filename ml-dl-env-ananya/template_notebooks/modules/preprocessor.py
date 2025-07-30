#modules/preprocessor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
import numpy as np
import cv2
import pandas as pd


def show_tabular_info(df):
    print("\n Dataset Overview:\n")
    print(f"Shape: {df.shape}")
    print(f"Size: {df.size}")
    print("\nColumn Data Types:")
    print(df.dtypes)

    print("\nStatistical Summary (for Numeric Columns):")
    print(df.describe())

    print("\n Null Values Check:")
    print(df.isnull().sum())

    print("\n First 5 Rows:")
    print(df.head())

    print("\nℹ Dataset Info:")
    df.info()

def clean_tabular_data(df, threshold=0.6):
    """
    Handle missing values smartly before preprocessing.
    - Drop columns with too many missing values (above threshold)
    - Fill numeric columns with mean
    - Fill categorical columns with mode or 'Unknown'
    """
    print("\n Cleaning Data...")

    # Drop columns with too many nulls
    null_ratios = df.isnull().mean()
    to_drop = null_ratios[null_ratios > threshold].index
    if len(to_drop) > 0:
        print(f" Dropping columns with >{int(threshold * 100)}% nulls: {list(to_drop)}")
        df = df.drop(columns=to_drop)

    # Fill numeric nulls with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill object/categorical nulls
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    return df

def preprocess_tabular(df, target_column):
    # Step 1: Clean data before preprocessing
    df = clean_tabular_data(df)

    # Step 2: Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Step 3: Encode target if categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Step 4: One-hot encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        unique_vals = X[col].dropna().unique()
        if set(unique_vals) == {'yes', 'no'}:
            X[col] = X[col].map({'yes': 1, 'no': 0})
        elif len(unique_vals) == 2:
            X[col] = LabelEncoder().fit_transform(X[col])
        else:
            X = pd.get_dummies(X, columns=[col], drop_first=True)

    # Step 5: Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 6: Train-test split
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def preprocess_images(X, y=None, img_size=(28, 28), grayscale=True, split=True):
    """
    Preprocess image data for ML/CNN models.
    
    Parameters:
    - X: list or array of images (numpy arrays)
    - y: labels (optional)
    - img_size: desired (height, width) of images
    - grayscale: whether to convert images to grayscale shape
    - split: whether to split into train/test (default: True)

    Returns:
    - If split=True: X_train, X_test, y_train, y_test
    - If split=False: X_processed, y (can be None)
    """

    X_resized = []
    for img in X:
        img = cv2.resize(img, img_size)
        img = img.astype("float32") / 255.0
        if grayscale:
            img = img.reshape(img_size[0], img_size[1], 1)
        X_resized.append(img)
    
    X_resized = np.array(X_resized)

    if y is not None:
        if isinstance(y[0], str):
            y = LabelEncoder().fit_transform(y)
        y = np.array(y)

        if split:
            return train_test_split(X_resized, y, test_size=0.2, random_state=42)
        else:
            return X_resized, y
    else:
        return X_resized, None