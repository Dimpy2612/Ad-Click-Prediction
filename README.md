Ad Click Prediction
Project Overview

Ad Click Prediction is a machine learning project that predicts whether a user will click on an online advertisement based on user demographics, internet usage, income, time spent, and advertisement-related information.

The project uses Python and machine learning techniques to analyze advertising data, identify important features, and build a Logistic Regression classification model to predict ad clicks.

Objectives
Analyze and preprocess advertisement click data.
Explore the relationship between user characteristics and ad-click behavior.
Identify relevant features using statistical tests.
Build a machine learning model to predict whether a user clicks an advertisement.
Evaluate model performance using classification metrics and cross-validation.
Technologies Used
Programming Language: Python
Data Analysis: Pandas, NumPy
Data Visualization: Matplotlib
Statistical Analysis: SciPy
Machine Learning: Scikit-learn
Model: Logistic Regression
Dataset Description

The project uses the Ad click data.csv dataset.

The dataset contains information about users and their interaction with online advertisements.

Features Used
Feature	Description
Time_Spent	Time spent on the website or advertisement-related page
Age	Age of the user
Avg_Income	Average income associated with the user
Internet_Usage	Internet usage of the user
Ad_Topic	Topic of the advertisement
City_code	City code associated with the user
Male	Gender indicator
Time_Period	Time period associated with the user or interaction
Clicked	Target variable indicating whether the advertisement was clicked

The code also removes duplicate records and drops the VistID, Country_Name, and Year columns before model development.

Project Workflow
Data Loading: Load the dataset using Pandas.
Data Cleaning: Remove duplicate records and inspect missing values.
Exploratory Data Analysis (EDA): Examine dataset statistics, target distribution, categorical variables, histograms, and box plots.
Feature Analysis: Use ANOVA for continuous variables and Chi-square tests for categorical variables to examine their associations with the target.
Feature Selection: Select relevant numerical and categorical predictors.
Data Preprocessing: Convert categorical variables using one-hot encoding and transform the Male column into numeric values.
Feature Scaling: Apply MinMaxScaler to scale predictor values.
Model Training: Train a Logistic Regression classifier.
Model Evaluation: Generate a classification report, confusion matrix, weighted F1-score, and 10-fold cross-validation scores.
Machine Learning Algorithm
Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification problems. In this project, it predicts whether an advertisement will be clicked based on the selected input features.

The model is configured with the following parameters:

C = 1
penalty = 'l2'
solver = 'newton-cg'
max_iter = 500

The dataset is divided into training and testing sets using a 70:30 split.

Model Evaluation

The model is evaluated using the following metrics:

Classification Report: Provides precision, recall, F1-score, and support for each class.
Confusion Matrix: Shows correct and incorrect predictions for each class.
Weighted F1-Score: Measures classification performance while accounting for class support.
10-Fold Cross-Validation: Evaluates the model across multiple data splits to estimate its performance more reliably.

The actual performance scores are generated when the Python script is executed on the dataset.

Project Structure
Ad-Click-Prediction/
│
├── ADCode.py
├── Ad click data.csv
├── README.md
├── AD REPORT.pdf
├── Ad click prediction paper.pdf
├── Final PPT.pdf
├── AcceptanceLetter.pdf
├── Plagrism Report.pdf
└── payment_receipt.pdf


Note: The structure above reflects the repository files shared in the project description. Some PDF filenames may differ in the actual repository.

Installation and Setup
Prerequisites

Install Python 3 and the required Python libraries.

Step 1: Clone the Repository
git clone https://github.com/Dimpy2612/Ad-Click-Prediction.git

Step 2: Navigate to the Project Directory
cd Ad-Click-Prediction

Step 3: Install Dependencies
pip install pandas numpy matplotlib scipy scikit-learn

Step 4: Configure the Dataset Path

Update the CSV file path in ADCode.py to match the location of your dataset.

For example:

AdClickData = pd.read_csv(
    "Ad click data.csv",
    encoding="latin1"
)


Ensure that the CSV file is present in the project directory.

Step 5: Run the Project
python ADCode.py


The script performs data analysis, displays visualizations, trains the Logistic Regression model, and prints the evaluation results.

Expected Output

After execution, the program displays:

Dataset dimensions before and after duplicate removal.
Dataset summary and descriptive statistics.
Charts showing target distribution and categorical feature distributions.
Histograms and box plots for continuous variables.
Statistical feature-selection results.
Training and testing dataset dimensions.
Classification report and confusion matrix.
Weighted F1-score and cross-validation results.

Note: Numerical results depend on the dataset and the execution environment.

Applications
Online advertising analysis
Digital marketing optimization
User engagement analysis
Advertisement click-through behavior prediction
Data-driven advertising strategies
Future Enhancements
Compare Logistic Regression with Random Forest, Decision Tree, and other classification algorithms.
Improve preprocessing by handling missing values and class imbalance where necessary.
Use a pipeline to prevent data leakage during scaling and cross-validation.
Tune model hyperparameters to improve predictive performance.
Evaluate ROC-AUC, precision, recall, and other relevant metrics.
Deploy the trained model through a web application for interactive predictions.
Conclusion

The Ad Click Prediction project demonstrates how data analysis, statistical feature selection, and supervised machine learning can be used to predict online advertisement click behavior. By applying Logistic Regression and evaluating its performance with multiple metrics, the project provides a foundation for understanding and predicting user engagement with digital advertisements.

Author

Dimpy

GitHub: Dimpy2612

License

A license has not been specified for this repository. Add an appropriate license file if you intend to distribute or reuse the project publicly.
