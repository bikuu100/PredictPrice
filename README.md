Gadget Price Prediction
This project aims to predict the price of used gadgets using various regression models. By taking numerous factors into consideration, we evaluate the performance of multiple models and identify the best one. After extensive hyperparameter tuning, CatBoost Regressor was found to be the most effective model. The project is fully documented in a Jupyter Notebook (.ipynb), and the model coding is implemented in Python scripts (.py) located in the src folder. Additionally, a web application was created using Flask and deployed on AWS.

Table of Contents
Project Overview
Dataset
Models Evaluated
Results
Hyperparameter Tuning
Deployment
Repository Structure
How to Run
Conclusion
Project Overview
The goal of this project is to accurately predict the price of used gadgets. This involves:

Data preprocessing and feature engineering.
Model training using various regression algorithms.
Evaluating and comparing model performance.
Hyperparameter tuning of the best models.
Deployment of the best model using Flask and AWS.
Dataset
The dataset includes various features that influence the price of used gadgets, such as:

Brand
Model
Year of Manufacture
Condition
Usage Duration
Technical Specifications (e.g., RAM, Storage)
Accessories Included
Market Trends
Models Evaluated
The following regression models were evaluated:

Linear Regression
Lasso Regression
Ridge Regression
K-Neighbors Regressor
Decision Tree
Random Forest Regressor
XGBRegressor
CatBoost Regressor
AdaBoost Regressor
Results
After evaluating the models, Random Forest and CatBoost Regressor were identified as the top-performing models. The performance metrics for these models are as follows:

Random Forest Regressor:

Training Set:
Root Mean Squared Error: 9.9718
Mean Absolute Error: 6.1817
R² Score: 0.9676
Test Set:
Root Mean Squared Error: 23.9984
Mean Absolute Error: 16.5604
R² Score: 0.7852
CatBoost Regressor:

Training Set:
Root Mean Squared Error: 12.6317
Mean Absolute Error: 9.9678
R² Score: 0.9480
Test Set:
Root Mean Squared Error: 24.8179
Mean Absolute Error: 16.4798
R² Score: 0.7703
Hyperparameter Tuning
Hyperparameter tuning was performed to optimize the models. The best parameters for CatBoost Regressor were:

Learning Rate: 0.01
Iterations: 500
Depth: 6
After tuning, CatBoost Regressor showed the best performance:

Training Set:
Root Mean Squared Error: 21.2684
Mean Absolute Error: 15.1630
R² Score: 0.8526
Test Set:
Root Mean Squared Error: 22.8235
Mean Absolute Error: 16.3499
R² Score: 0.8057
Deployment
The best model (CatBoost Regressor) was deployed using Flask to create a web application. The application is hosted on AWS to provide an accessible platform for users to predict gadget prices.

Repository Structure
scss
Copy code
├── notebooks
│   └── Gadget_Price_Prediction.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── app.py
├── templates
│   └── index.html
├── static
│   └── styles.css
├── requirements.txt
├── README.md
└── Dockerfile
How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/gadget-price-prediction.git
cd gadget-price-prediction
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Flask application:

bash
Copy code
python src/app.py
Access the web app:
Open your browser and go to http://localhost:5000.

Deployment on AWS:
The application is deployed on AWS. You can access it via the provided AWS URL.

Conclusion
This project demonstrates a comprehensive approach to predicting the price of used gadgets using machine learning. By evaluating multiple models and performing hyperparameter tuning, the CatBoost Regressor was found to be the most effective. The project also includes a web application for easy access to predictions, which has been successfully deployed on AWS.