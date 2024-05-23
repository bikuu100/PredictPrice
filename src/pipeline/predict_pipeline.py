import pandas as pd
import numpy as np
import joblib  # Use joblib to load the model


def load_preprocessor(file_path):
    # Load the preprocessor object from file
    preprocessor = joblib.load(file_path)
    return preprocessor

def load_model(file_path):
    # Load the model object from file
    model = joblib.load(file_path)
    return model

def preprocess_data(data, preprocessor):
    preprocessed_data = preprocessor.transform(data)
    return preprocessed_data

def predict_pipeline(input_data):
    
        # Load the preprocessor and model
        preprocessor = load_preprocessor("processed_dataset\preprocessor.pkl")
        model = load_model("modelpkl\CatBoosting Regressor_model.pkl")

        # Preprocess the input data
        preprocessed_data = preprocess_data(pd.DataFrame([input_data]), preprocessor)  # Wrap input_data in a DataFrame
        print("Preprocessed data:", preprocessed_data)

        # Make predictions
        predictions = model.predict(preprocessed_data)
        print("Predictions:", predictions)

        return predictions
    