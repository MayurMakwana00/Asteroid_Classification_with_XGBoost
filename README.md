NASA Asteroid Hazard Classification

This project uses machine learning to classify whether a Near-Earth Asteroid is hazardous or non-hazardous based on its physical and orbital features.
The notebook demonstrates a complete end-to-end machine learning workflow, including data preprocessing, visualization, model training, and evaluation using multiple algorithms.

Project Objective

The goal of this project is to:
Analyze asteroid data from NASA
Clean and preprocess the dataset
Train classification models
Compare model performance
Predict whether an asteroid is hazardous

Dataset

The dataset contains information about Near-Earth Objects (NEOs), including:
Estimated diameter
Relative velocity
Miss distance
Orbital parameters
Hazardous label (target variable)

Target Column: 'Hazardous'
1 → Hazardous asteroid
0 → Non-hazardous asteroid


Machine Learning Workflow
1. Data Loading

Dataset loaded using pandas
Basic inspection with:
.info()
.describe()
.isnull().sum()


2. Data Cleaning

Dropped unnecessary columns:
Neo Reference ID
Name
Close Approach Date
Orbit Determination Date
Filled missing numerical values using median

3. Encoding
Categorical features converted using Label Encoding

4. Data Visualization
Countplot for hazardous vs non-hazardous asteroids
Correlation heatmap for numerical features

5. Feature Scaling
Used StandardScaler to normalize features

6. Model Training

Two models were trained:
Logistic Regression
Baseline classification model
Used for comparison
XGBoost Classifier
Gradient boosting algorithm
Used for improved performance

7. Model Evaluation

Metrics used:
Accuracy
ROC-AUC score
Confusion Matrix
ROC Curve comparison

Technologies Used

Python
Jupyter Notebook
pandas
numpy
matplotlib
seaborn
scikit-learn
XGBoost

Nasa-Asteroid-Classification/
│
├── Nasa_Asteroid_Classification.ipynb
├── nasa.csv
└── README.md

How to Run the Project:

1. Clone the repository
2. git clone https://github.com/MayurMakwana00/Nasa-Asteroid-Classification.git

2.Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

3. Open the notebook:
jupyter notebook

Run all cells

Model Comparison

The notebook compares:
Logistic Regression
XGBoost Classifier
Based on:
Accuracy
ROC-AUC score
XGBoost generally performs better due to its ensemble learning approach.

Future Improvements

Hyperparameter tuning
Feature selection
Cross-validation
Deployment as a web app
Real-time asteroid risk prediction

Author

Mayur Makwana

Machine Learning Enthusiast
