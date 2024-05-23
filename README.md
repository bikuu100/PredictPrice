Certainly! Here's a detailed README file for your project, which includes a table of contents, project structure, and descriptions of each component.

---

# Phone Price Prediction

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [Deployment](#deployment)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Phone Price Prediction project aims to predict the resale price of phones based on various features such as screen size, camera specifications, memory, battery, and brand. The project includes data preprocessing, model training, evaluation, and deployment of a web application using Flask.

## Project Structure
```
Phone_Price_Prediction/
│
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_preprocessing.py
│   │   ├── model_evaluation.py
│   │   └── train_test_split.py
│   │
│   ├── app.py
│   ├── predict_pipeline.py
│   └── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── home.html
│   └── prediction_result.html
│
├── processed_dataset/
│   ├── preprocessor.pkl
│   └── X_train_preprocessed.csv
│   └── X_test_preprocessed.csv
│
├── modelpkl/
│   └── Random_Forest_model.pkl
│
├── data/
│   └── raw_data.csv
│
└── README.md
```

## Data Description
- `raw_data.csv`: The dataset containing features like screen size, camera specifications, memory, battery, brand, etc., and the target variable `used_price`.

## Exploratory Data Analysis (EDA)
- The EDA and model training were performed in a Jupyter Notebook located in `notebooks/EDA_and_Model_Training.ipynb`. This notebook includes:
  - Data cleaning and handling missing values.
  - Visualizations of feature distributions and relationships.
  - Feature engineering and selection.

## Data Preprocessing
- `data_preprocessing.py`: This script contains the `custom_preprocessor` function which handles missing values, scaling numerical features, and encoding categorical features using a ColumnTransformer and Pipelines.

## Model Training
- `EDA_and_Model_Training.ipynb`: This notebook contains the steps for training various machine learning models. The Random Forest model was selected as the best model and saved as `Random_Forest_model.pkl`.

## Model Evaluation
- `model_evaluation.py`: This script contains functions to evaluate the model performance on test data using metrics like RMSE, MAE, and R^2.

## Web Application
- `app.py`: The Flask web application script. This script:
  - Loads the pre-trained model and preprocessor.
  - Handles web requests for predicting phone prices.
  - Renders HTML templates for the web interface.
- `predict_pipeline.py`: Contains the `predict_pipeline` function to preprocess input data and make predictions using the trained model.

## Deployment
- The web application is deployed on AWS. The deployment process includes:
  - Setting up an EC2 instance.
  - Installing necessary dependencies from `requirements.txt`.
  - Configuring the Flask application to run on the server.

## How to Run
1. **Clone the repository**:
    ```sh
    git clone https://github.com/bikuu100/PredictPrice.git
    cd PredictPrice
    ```

2. **Set up a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r src/requirements.txt
    ```

4. **Run the Flask web application**:
    ```sh
    python src/app.py
    ```

5. **Open your web browser and go to** `http://127.0.0.1:5000`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to add.

## License
This project is licensed under the MIT License.

---

This README provides a comprehensive overview of your project, including all necessary steps and descriptions. Feel free to adjust any sections as per your specific project details.