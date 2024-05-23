import pandas as pd
import numpy as np
from keras.models import load_model

# Load the trained ANN model
model = load_model("modelpkl/ann_model.h5")

# Load the preprocessed test data
x_test = pd.read_csv('processed_dataset/x_test_preprocessed.csv')
y_test = pd.read_csv('test_dataset/y_test.csv')
x_test_array = x_test.values
y_test_array = y_test.values

# Make predictions
predictions = model.predict(x_test_array)

# Calculate absolute errors
errors = np.abs(predictions - y_test_array)

# Define a threshold for accuracy calculation (e.g., 10%)
threshold = 0.1 * y_test_array

# Check if absolute errors are within the threshold
accurate_predictions = errors <= threshold

# Calculate accuracy
accuracy = np.mean(accurate_predictions)
print("Accuracy:", accuracy)
