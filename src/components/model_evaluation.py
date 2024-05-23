import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
import joblib 

def define_models():
    # Define the models and their hyperparameters
    models = {
        "Random Forest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                "max_features": [None, 'sqrt', 'log2'],
                "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['auto', 'sqrt', 'log2']
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(),
            "params": {
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.6, 0.8, 1.0]
            }
        },
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}  # No hyperparameters to tune
        },
        "XGBRegressor": {
            "model": XGBRegressor(),
            "params": {
                "n_estimators": [100, 200, 300, 400, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            }
        },
        "CatBoosting Regressor": {
            "model": CatBoostRegressor(verbose=False),
            "params": {
                "iterations": [100, 200, 300, 400, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "depth": [3, 4, 5, 6, 7, 8, 9, 10]
            }
        },
        "AdaBoost Regressor": {
            "model": AdaBoostRegressor(),
            "params": {
                "n_estimators": [int(x) for x in np.linspace(start=50, stop=500, num=10)],
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            }
        }
    }
    
    return models

def fit_models(x_train, y_train):
    models = define_models()
    best_estimators = {}

    for name, model_info in models.items():
        print(f"Fitting {name}...")
        model = model_info['model']
        params = model_info['params']
        
        if params:
            search = RandomizedSearchCV(model, params, n_iter=100, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
            search.fit(x_train, y_train)
            best_model =search.best_estimator_
        else:
            model.fit(x_train, y_train)
            best_model = model

        best_estimators[name] = best_model

        joblib.dump(best_model, f"modelpkl/{name}_model.pkl")


    return best_estimators

def calculate_r2score(y_test, y_pred):
    return r2_score(y_test, y_pred)

def predict_output(X_test, best_estimators):
    predictions = {}
    
    for name, model in best_estimators.items():
        y_pred = model.predict(x_test)
        predictions[name] = y_pred

    return predictions

# Example usage
if __name__ == "__main__":
    

    x_train = pd.read_csv(r'processed_dataset\x_train_preprocessed.csv')
    x_test = pd.read_csv(r'processed_dataset\x_test_preprocessed.csv')
    y_train = pd.read_csv(r'train_dataset\y_train.csv')
    y_test = pd.read_csv(r'test_dataset\y_test.csv')

    best_estimators = fit_models(x_train, y_train)
    predictions = predict_output(x_test, best_estimators)
    for name, y_pred in predictions.items():
        r2 = calculate_r2score(y_test, y_pred)
        print(f"{name} - R2 Score: {r2:.4f}")

