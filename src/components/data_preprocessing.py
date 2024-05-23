import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import os
import logging

# Set up logging
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(log_directory, 'data_preprocessing.log')),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

numerical_features = ['screen_size', 'main_camera_mp', 'selfie_camera_mp', 'int_memory', 'ram', 'battery', 'weight', 'release_year', 'days_used', 'new_price']
categorical_features = ['brand_name', 'os', '4g', '5g']

def custom_preprocessor(numerical_features, categorical_features):
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    return preprocessor

# Get data from train_test_split.py
x_train = pd.read_csv(r'train_dataset\X_train.csv')
x_test = pd.read_csv(r'test_dataset\X_test.csv')
y_train = pd.read_csv(r'train_dataset\y_train.csv')
y_test = pd.read_csv(r'test_dataset\y_test.csv')

# Now you can use x_train, x_test, y_train, y_test in your script

preprocessor = custom_preprocessor(numerical_features, categorical_features)

# Fit and transform x_train
x_train_preprocessed_1 = preprocessor.fit_transform(x_train)
x_train_preprocessed = x_train_preprocessed_1.toarray()

# Transform x_test
x_test_preprocessed_1 = preprocessor.transform(x_test)
x_test_preprocessed = x_test_preprocessed_1.toarray()

# Convert to DataFrame
x_train_preprocessed_df = pd.DataFrame(x_train_preprocessed)
x_test_preprocessed_df = pd.DataFrame(x_test_preprocessed)

# Save to CSV
output_directory = 'processed_dataset'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

x_train_preprocessed_df.to_csv(os.path.join(output_directory, 'x_train_preprocessed.csv'), index=False)
x_test_preprocessed_df.to_csv(os.path.join(output_directory, 'x_test_preprocessed.csv'), index=False)

logger.info("Preprocessed training and testing data saved successfully.")

# Save the preprocessor as .pkl file in the specified directory
output_file_path = os.path.join(output_directory, 'preprocessor.pkl')
joblib.dump(preprocessor, output_file_path)

def main():
    # Your main code here
    pass

if __name__ == "__main__":
    main()
