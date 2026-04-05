import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
AdClickData = pd.read_csv("/Users/dmkgarg/Downloads/Ad click data.csv", encoding='latin1')
print('Shape before deleting duplicate values:', AdClickData.shape)

# Remove duplicates
AdClickData = AdClickData.drop_duplicates()
print('Shape After deleting duplicate values:', AdClickData.shape)

# Preview data
print(AdClickData.head(10))

# Target variable distribution
GroupedData = AdClickData.groupby('Clicked').size()
GroupedData.plot(kind='bar', figsize=(4,3))
plt.show()

# Data info
print(AdClickData.info())
print(AdClickData.describe(include='all'))
print(AdClickData.nunique())

# Drop useless columns
UselessColumns = ["VistID", "Country_Name", "Year"]
AdClickData = AdClickData.drop(UselessColumns, axis=1)

# Function to plot bar charts
def PlotBarCharts(inpData, colsToPlot):
    fig, subPlot = plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(40,6))
    fig.suptitle('Bar charts of: ' + str(colsToPlot))

    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar', ax=subPlot[plotNumber])

    plt.show()

# Plot categorical variables
PlotBarCharts(
    inpData=AdClickData,
    colsToPlot=["Ad_Topic","City_code","Male","Time_Period","Weekday","Month"]
)

# Histograms
AdClickData.hist(["Time_Spent", "Age", "Avg_Income", "Internet_Usage"], figsize=(18,10))
plt.show()

# Check missing values
print(AdClickData.isnull().sum())

# Continuous columns
ContinuousColsList = ["Time_Spent", "Age", "Avg_Income", "Internet_Usage"]

# Box plots
fig, PlotCanvas = plt.subplots(nrows=1, ncols=len(ContinuousColsList), figsize=(18,5))

for PredictorCol, i in zip(ContinuousColsList, range(len(ContinuousColsList))):
    AdClickData.boxplot(column=PredictorCol, by='Clicked', ax=PlotCanvas[i])

plt.show()

# ANOVA function
def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    from scipy.stats import f_oneway
    SelectedPredictors = []

    print('##### ANOVA Results #####\n')

    for predictor in ContinuousPredictorList:
        CategoryGroupLists = inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])

    return SelectedPredictors

ContinuousVariables = ["Time_Spent", "Age", "Avg_Income", "Internet_Usage"]
FunctionAnova(AdClickData, 'Clicked', ContinuousVariables)

# Chi-Square function
def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    from scipy.stats import chi2_contingency
    SelectedPredictors = []

    for predictor in CategoricalVariablesList:
        CrossTabResult = pd.crosstab(inpData[TargetVariable], inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)

        if (ChiSqResult[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])

    return SelectedPredictors

CategoricalVariables = ["Ad_Topic","City_code","Male","Time_Period","Weekday","Month"]
FunctionChisq(AdClickData, 'Clicked', CategoricalVariables)

# Final feature selection
SelectedColumns = ["Time_Spent", "Age", "Avg_Income", "Internet_Usage",
                   "Ad_Topic", "City_code", "Male", "Time_Period"]

DataForML = AdClickData[SelectedColumns]

# Save data
DataForML.to_pickle('DataForML.pkl')

# Convert Male to numeric
DataForML['Male'] = DataForML['Male'].replace({'Yes':1, 'No':0})

# One-hot encoding
DataForML_Numeric = pd.get_dummies(DataForML)

# Add target variable
DataForML_Numeric['Clicked'] = AdClickData['Clicked']

# Define target and predictors
TargetVariable = 'Clicked'
X = DataForML_Numeric.drop(columns=[TargetVariable]).values
y = DataForML_Numeric[TargetVariable].values

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg', max_iter=500)

LOG = clf.fit(X_train, y_train)
prediction = LOG.predict(X_test)

# Evaluation
from sklearn import metrics

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
print('F1 Score (Accuracy):', round(F1_Score, 2))

# Cross-validation
from sklearn.model_selection import cross_val_score

Accuracy_Values = cross_val_score(LOG, X, y, cv=10, scoring='f1_weighted')

print('\nCross-validation scores:\n', Accuracy_Values)
print('\nFinal Average Accuracy:', round(Accuracy_Values.mean(), 2))