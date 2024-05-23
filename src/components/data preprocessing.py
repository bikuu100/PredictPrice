import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from split_data import train_test_split_and_save

# Configure logging
log_file = 'data_preprocessing.log'

# Remove any existing log handlers to avoid duplicate logging issues
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Verify logging is set up
logging.info('Starting the data preprocessing script - this should be logged.')

numerical_features = ['screen_size', 'main_camera_mp', 'selfie_camera_mp', 'int_memory', 'ram', 'battery', 'weight', 'release_year', 'days_used', 'new_price']
categorical_features = ['brand_name', 'os', '4g', '5g']

def custom_preprocessor(numerical_features, categorical_features):
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
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

def preprocess_and_save(X_train, X_test, y_train, y_test, train_dir, test_dir):
    logging.info('Starting data preprocessing')

    preprocessor = custom_preprocessor(numerical_features, categorical_features)
    logging.info('Preprocessor created successfully')

    # Fit and transform X_train
    logging.info('Fitting preprocessor on training data')
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    logging.info('Transforming test data')
    X_test_preprocessed = preprocessor.transform(X_test)

    # Convert to DataFrame
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed)

    # Save to CSV
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    X_train_preprocessed_df.to_csv(os.path.join(train_dir, 'X_train_preprocessed.csv'), index=False)
    X_test_preprocessed_df.to_csv(os.path.join(test_dir, 'X_test_preprocessed.csv'), index=False)
    y_train.to_csv(os.path.join(train_dir, 'y_train.csv'), index=False, header=[y_train.name])
    y_test.to_csv(os.path.join(test_dir, 'y_test.csv'), index=False, header=[y_test.name])
    logging.info('Preprocessed data saved successfully')

    # Save the preprocessor as .pkl file in the specified directory
    pkl_dir = 'preprocessor_directory'
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    output_file_path = os.path.join(pkl_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, output_file_path)
    logging.info('Preprocessor model saved successfully as %s', output_file_path)

def main():
    logging.info('Starting main function')
    
    # Use the train_test_split_and_save function to get split data
    data_path = r'Notebook\used_device_data.csv'  # Ensure this is the correct path to your CSV file
    target_column = 'used_price'  # Replace with the actual target column name
    train_dir = 'train_dataset'
    test_dir = 'test_dataset'

    try:
        X_train, X_test, y_train, y_test = train_test_split_and_save(data_path, target_column, train_dir, test_dir)
        logging.info('Data split successfully')
    except Exception as e:
        logging.error('Error splitting data: %s', e)
        raise

    # Call the preprocessing function
    try:
        preprocess_and_save(X_train, X_test, y_train, y_test, train_dir, test_dir)
    except Exception as e:
        logging.error('Error in preprocessing and saving data: %s', e)
        raise

if __name__ == "__main__":
    logging.info('Script started')
    main()
    logging.info('Script finished successfully')
