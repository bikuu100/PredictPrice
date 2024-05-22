import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configure logging
log_file = 'train_test_split.log'
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_test_split_and_save(data_path, target_column, train_dir, test_dir, test_size=0.2, random_state=None):
    """
    Perform train-test split on a dataset, save the split datasets into specified directories,
    and return X_train, X_test, y_train, and y_test.

    Args:
    - data_path (str): Path to the dataset file.
    - features_columns (list): List of column names representing features.
    - target_column (str): Name of the target column.
    - train_dir (str): Directory to save the training dataset.
    - test_dir (str): Directory to save the testing dataset.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train (DataFrame): Features of the training dataset.
    - X_test (DataFrame): Features of the testing dataset.
    - y_train (Series): Target variable of the training dataset.
    - y_test (Series): Target variable of the testing dataset.
    """

    logging.info('Loading dataset from %s', data_path)
    # Load the dataset
    data = pd.read_csv(data_path)
    logging.info('Dataset loaded successfully')

    # Separate features and target variable
    logging.info('Separating features and target variable')
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Perform train-test split
    logging.info('Performing train-test split with test size = %f and random state = %s', test_size, str(random_state))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info('Train-test split completed successfully')

    # Create directories if they don't exist
    logging.info('Creating directories if they do not exist')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save the split datasets
    logging.info('Saving training and testing datasets')
    train_X_path = os.path.join(train_dir, 'X_train.csv')
    train_y_path = os.path.join(train_dir, 'y_train.csv')
    test_X_path = os.path.join(test_dir, 'X_test.csv')
    test_y_path = os.path.join(test_dir, 'y_test.csv')

    X_train.to_csv(train_X_path, index=False)
    y_train.to_csv(train_y_path, index=False, header=[target_column])
    X_test.to_csv(test_X_path, index=False)
    y_test.to_csv(test_y_path, index=False, header=[target_column])

    logging.info('Training and testing datasets saved successfully')
    logging.info('Training datasets: %s, %s', train_X_path, train_y_path)
    logging.info('Testing datasets: %s, %s', test_X_path, test_y_path)

    return X_train, X_test, y_train, y_test



# Example usage:
data_path = r'Notebook\used_device_data.csv'
target_column = 'used_price'
train_dir = 'train_dataset'
test_dir = 'test_dataset'
X_train, X_test, y_train, y_test = train_test_split_and_save(data_path, target_column, train_dir, test_dir, test_size=0.2, random_state=42)
