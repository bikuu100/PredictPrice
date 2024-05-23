import pandas as pd
import numpy as np
import joblib
import logging
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

# Set up logging
logging.basicConfig(filename='logs/model_evaluation.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load preprocessed data
x_train = pd.read_csv('processed_dataset/x_train_preprocessed.csv')
y_train = pd.read_csv('train_dataset/y_train.csv')
x_test = pd.read_csv('processed_dataset/x_test_preprocessed.csv')
y_test = pd.read_csv('test_dataset/y_test.csv')

# Convert DataFrame to numpy array
x_train_array = x_train.values
x_test_array = x_test.values
y_train_array = y_train.values
y_test_array = y_test.values

# Define the ANN model with hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32), activation='relu', input_dim=x_train_array.shape[1]))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(units=hp.Int('units_3', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
    return model

# Hyperparameter tuning
def tune_model(x_train, y_train):
    tuner = RandomSearch(
        build_model,
        objective='val_mean_absolute_error',
        max_trials=10,
        executions_per_trial=3,
        directory='tuner',
        project_name='ann_tuning'
    )
    
    tuner.search(x_train, y_train, epochs=50, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(x_train, y_train, epochs=50, validation_split=0.2, verbose=2)
    return model

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    results = model.evaluate(x_test, y_test, verbose=0)
    logging.info(f"Model Evaluation - Loss: {results[0]}, MAE: {results[1]}")
    return results

# Main function
def main():
    model = tune_model(x_train_array, y_train_array)
    evaluate_model(model, x_test_array, y_test_array)

    # Log model performance
    logging.info("Training and evaluation completed.")
    logging.info(f"Train Evaluation: {model.evaluate(x_train_array, y_train_array, verbose=0)}")
    logging.info(f"Test Evaluation: {model.evaluate(x_test_array, y_test_array, verbose=0)}")

    # Save the best model
    model.save('modelpkl/ann_model.h5')

if __name__ == "__main__":
    main()
