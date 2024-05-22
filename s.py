import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

# Configure logging
log_file = 'data_preprocessing.log'
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_and_save(X_train, X_test, y_train, y_test, categorical_features, numerical_features, train_dir, test_dir):
   
    logging.info('Starting preprocessing of data')

    # Create preprocessing pipelines for both numerical and categorical data
    logging.info('Creating preprocessing pipelines')

    # Pipeline for numerical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Fit the preprocessor on the training data and transform both train and test data
    logging.info('Fitting preprocessor on the training data')
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        logging.info('Preprocessing of training data completed successfully')
    except NotFittedError as e:
        logging.error('Error fitting preprocessor: %s', e)
        raise

    logging.info('Transforming test data using the fitted preprocessor')
    try:
        X_test_processed = preprocessor.transform(X_test)
        logging.info('Preprocessing of test data completed successfully')
    except NotFittedError as e:
        logging.error('Error transforming test data: %s', e)
        raise

    # Convert processed arrays back to DataFrames for easier handling and saving
    X_train_processed = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
    X_test_processed = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())

    # Create directories if they don't exist
    logging.info('Creating directories if they do not exist')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save the processed datasets
    logging.info('Saving processed training and testing datasets')
    train_X_path = os.path.join(train_dir, 'X_train_processed.csv')
    train_y_path = os.path.join(train_dir, 'y_train.csv')
    test_X_path = os.path.join(test_dir, 'X_test_processed.csv')
    test_y_path = os.path.join(test_dir, 'y_test.csv')

    X_train_processed.to_csv(train_X_path, index=False)
    y_train.to_csv(train_y_path, index=False, header=[y_train.name])
    X_test_processed.to_csv(test_X_path, index=False)
    y_test.to_csv(test_y_path, index=False, header=[y_test.name])

    logging.info('Processed training and testing datasets saved successfully')
    logging.info('Processed training datasets: %s, %s', train_X_path, train_y_path)
    logging.info('Processed testing datasets: %s, %s', test_X_path, test_y_path)

    return X_train_processed, X_test_processed, y_train, y_test

categorical_features = ['brand_name', 'os', '4g', '5g']
numerical_features =['screen_size', 'main_camera_mp', 'selfie_camera_mp', 'int_memory', 'ram', 'battery', 'weight', 'release_year', 'days_used', 'new_price']
X_train=pd.read_csv(r'train_dataset\X_train.csv')
X_test=pd.read_csv(r'test_dataset\X_test.csv')
y_train=pd.read_csv(r'train_dataset\y_train.csv')
y_test=pd.read_csv(r'test_dataset\y_test.csv')
train_dir = 'preprocessed_train_dataset'
test_dir = 'preprocessed_test_dataset'
X_train_processed, X_test_processed, y_train, y_test = preprocess_and_save(X_train, X_test, y_train, y_test, categorical_features, numerical_features, train_dir, test_dir)
