
---

# Mobile Phone Price Prediction

This project involves a detailed analysis and prediction of used mobile phone prices using machine learning and deep learning techniques. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and deployment of the model as a web application using Flask and AWS.

Table of Contents
Dataset Features
Data Analysis and Visualization
Exploratory Data Analysis (EDA)
Visualizations
Key Insights
Model Development
Deep Learning Model
Preprocessing Steps
Model Training and Evaluation
Model Deployment
Flask Application
AWS Deployment
Usage
Running the Flask App Locally
Deploying to AWS
Conclusion

├── app.py                     # Flask application file
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data                       # Directory for storing data
│   └── student_performance.csv
├── notebooks                  # Jupyter notebooks for data exploration and analysis
│   └── EDA.ipynb & Model Training.ipynb
├── try                        # Source code for data processing and model training
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict_pipeline.py
└── templates                  # HTML templates for Flask app
    └── index.html


## Dataset Features

### Numerical Features:
- `screen_size`
- `main_camera_mp`
- `selfie_camera_mp`
- `int_memory`
- `ram`
- `battery`
- `weight`
- `release_year`
- `days_used`
- `new_price`
- `used_price` (target)
- `price_drop`
- `price_drop_range`
- `days_used_bins`
- `days_used_rounded`

### Categorical Features:
- `brand_name`
- `os`
- `4g`
- `5g`

## Data Analysis and Visualization

### Exploratory Data Analysis (EDA)
- Analyzed the dataset to understand distributions, trends, and patterns.
- Handled missing values by filling them with appropriate statistics (mean, median, mode).

### Visualizations
- **Histograms and KDE Plots**: Visualized the distribution of numerical features.
- **Pairplot**: Examined pairwise relationships between numerical features.
- **Correlation Heatmap**: Identified highly correlated features with 'used_price'.
- **Boxplots**: Visualized the distribution of numerical features grouped by categorical variables.
- **Bar Plots**: Showed the frequency of each category in categorical features.
- **Count Plots and Stacked Bar Plots**: Illustrated the distribution and count of categorical features across different categories.

### Key Insights
- **Correlation Analysis**: Identified numerical features most correlated with 'used_price'.
- **Feature Importance**: Used machine learning models to determine the importance of each feature in predicting 'used_price'.
- **Distribution Analysis**: Examined the distribution of 'used_price' and other numerical features to identify outliers or unusual patterns.
- **Categorical Analysis**: Analyzed the distribution of categorical features and their impact on 'used_price'.

## Model Development

### Deep Learning Model
- Built and trained a deep learning model using TensorFlow and Keras to predict 'used_price' based on numerical and categorical features.
- Evaluated the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.

### Preprocessing Steps
1. **Scaling Numerical Features**: Used `StandardScaler` to standardize numerical features.
2. **One-Hot Encoding Categorical Features**: Used `OneHotEncoder` to encode categorical features.
3. **Combining Preprocessing Steps**: Utilized `ColumnTransformer` to apply transformations.

### Model Training and Evaluation
```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Load the dataset
df = pd.read_csv('your_file.csv')

# Display correlation matrix
numerical_features = ['screen_size', 'main_camera_mp', 'selfie_camera_mp', 'int_memory', 'ram', 'battery', 'weight', 'release_year', 'days_used', 'new_price', 'used_price', 'price_drop']
correlation_matrix = df[numerical_features].corr()

# Extract correlation with 'used_price' and sort
correlation_with_used_price = correlation_matrix['used_price'].sort_values(ascending=False)
print(correlation_with_used_price)

# Preprocess numerical features
numerical_transformer = StandardScaler()

# Preprocess categorical features
categorical_features = ['brand_name', 'os', '4g', '5g']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features[:-1]),  # Exclude 'used_price' from features
        ('cat', categorical_transformer, categorical_features)
    ])

# Separate features and target variable
X = df.drop('used_price', axis=1)
y = df['used_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Preprocess the testing data
X_test_preprocessed = preprocessor.transform(X_test)

# Build the deep learning model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train_preprocessed.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer with one neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_preprocessed, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predict the values for the test set
y_pred = model.predict(X_test_preprocessed).flatten()

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Calculate R^2 Score
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')
```

## Model Deployment

### Flask Application
- Developed a Flask web application to provide a user interface for predicting used mobile phone prices based on input features.
- Implemented API endpoints for model inference.

### AWS Deployment
- Deployed the Flask web application on AWS Elastic Beanstalk for scalable and reliable access.
- Configured the environment for deploying the model, ensuring proper dependencies and runtime settings.

## Usage

### Running the Flask App Locally
1. Clone the repository:
    ```sh
    git clone https://github.com/your_username/mobile-price-prediction.git
    cd mobile-price-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Flask app:
    ```sh
    python app.py
    ```

4. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the application.

### Deploying to AWS
1. Set up an AWS account and configure Elastic Beanstalk.
2. Create a new Elastic Beanstalk application and environment.
3. Deploy the Flask app using the Elastic Beanstalk CLI:
    ```sh
    eb init -p python-3.7 mobile-price-prediction
    eb create mobile-price-env
    eb deploy
    ```

4. Access the deployed application via the provided AWS URL.

## Conclusion

This project provides a comprehensive approach to predicting used mobile phone prices using advanced data analysis techniques and deep learning models. The deployment of the model as a web application ensures easy accessibility and usability for end-users. By leveraging AWS for deployment, the application is scalable and reliable.

---
