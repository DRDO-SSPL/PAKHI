# modules/ml_models.py

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_model(X_train, y_train,model_type):
    """
    Train a classification or regression model based on model_type.
    
    Supported classification models: 
        'logistic', 'decision_tree', 'svm', 'naive_bayes', 'random_forest', 'knn'
    
    Supported regression models:
        'linear', 'decision_tree_reg', 'svm_reg', 'random_forest_reg'
    """
    model_type = model_type.lower()

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "svm":
        model = SVC()
    elif model_type == "naive_bayes":
        model = GaussianNB()
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "knn":
        model = KNeighborsClassifier()
    
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "decision_tree_reg":
        model = DecisionTreeRegressor()
    elif model_type == "svm_reg":
        model = SVR()
    elif model_type == "random_forest_reg":
        model = RandomForestRegressor()
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Automatically evaluates a model based on its type.
    Returns accuracy for classifiers and MSE for regressors.
    """
    y_pred = model.predict(X_test)

    # Automatically detect if task is classification or regression
    is_classification = (y_test.dtype.name == "int64" or y_test.dtype.name == "object") and len(np.unique(y_test)) < 50

    if not is_classification:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Regression Evaluation:\nMSE: {mse:.4f}\nR²: {r2:.4f}")
        return {"mse": mse, "r2": r2}
    else:
        # Classification metrics
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        elif y_pred.dtype != int:
            y_pred = (y_pred > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        print(f"Classification Accuracy: {acc:.4f}")
        return {"accuracy": acc}
