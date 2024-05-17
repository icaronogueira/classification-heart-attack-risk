# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:50:57 2024

@author: Ícaro de Lima Nogueira
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graph_utils
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# load the data from a kaggle dataset
data = pd.read_csv('heart_attack_prediction_dataset.csv')

# ====================  DATA ANALYSIS ====================================
# sets a pandas configuration for showing every column information
pd.set_option('display.max_columns', None)
# first 15 registries of dataframe
print(data[:15])
# description of data columns and dtypes
data.info()
# description of data using statistical indicators
data.describe()
# check for missing values
data.isnull().sum()


data = data.drop(['Patient ID'], axis=1)

# Generates the target variable distribution histogram
graph_utils.plot_histogram(
    data['Heart Attack Risk'], 'Heart Attack Risk', 'Frequency', 'Heart Attack Risk Histogram')


# Let's do the correlation between the target variable and each individual feature.

# Before I identify the numerical and categorical features, a split on the feature
# 'Blood Pressure' is needed
data['Systolic_BP', 'Diastolic_BP'] = data['Blood Pressure'].str.split(
    '/', expand=True)
data['Systolic_BP'] = pd.to_numeric(data['Systolic_BP'])
data['Diastolic_BP'] = pd.to_numeric(data['Diastolic_BP'])

data.drop('Blood Pressure', axis=1, inplace=True)


# Now, I identify the numerical and categorical features.
# doing the individual features histograms should be useful for this (and looking
# for outliers)
for column in data.columns[1:]:
    graph_utils.plot_histogram(
        data[column], column, 'Frequency', column+' histogram')


# now a little manual work
numerical_features = ['Age', 'Cholesterol',
                      'Systolic_Blood_Pressure', 'Diastolic_Blood_Pressure', 'Heart Rate',
                      'Exercise Hours Per Week', 'Stress Level', 'Sedentary Hours Per Day',
                      'Income', 'BMI', 'Triglycerides']
categorical_features = ['Sex', 'Diabetes', 'Family History', 'Smoking',
                        'Obesity', 'Alcohol Consumption', 'Diet', 'Previous Heart Problems',
                        'Medication Use', 'Physical Activity Days Per Week', 'Sleep Hours Per Day',]
# 'Country', 'Continent', 'Hemisphere']
target = data['Heart Attack Risk']


# Analizing correlation between numeric features
correlation_matrix = data[numerical_features].corr()
print(correlation_matrix)
# Printing only revelant Pearson values (say absolute values bigger than 0.1)
threshold = 0.1
print(correlation_matrix[np.abs(correlation_matrix) > threshold])
# THere is no significant correlation between any features in the dataset. That's a good signal that
# all of our features bring some new information that should be learned by the model


# Analyzing correlation between target variable and categorical features
for feature in categorical_features:
    print("\nHeart Attack Risk by ", feature)
    print(data.groupby(feature)['Heart Attack Risk'].sum())


# LABEL ENCODING ON THE SEX AND DIET FEATURES

sex_encoder = LabelEncoder()
diet_encoder = LabelEncoder()

sex_encoder.fit(data['Sex'])
diet_encoder.fit(data['Diet'])
data['sex_encoded'] = sex_encoder.transform(data['Sex'])
data['diet_encoded'] = diet_encoder.transform(data['Diet'])

data = pd.get_dummies(
    data, columns=['Country', 'Hemisphere', 'Continent', 'Smoking', 'Family History'], drop_first=False, dtype=int)


data = data.drop(['Sex', 'Diet'], axis=1)


# DATA SPLITTING AND SCALING

X = data.drop('Heart Attack Risk', axis=1)
y = data['Heart Attack Risk']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("Training set shape: ", X_train.shape, y_train.shape)
print("Validation set shape: ", X_val.shape, y_val.shape)


# SCALING ALTERNATIVES
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Gradient Boosting Machine


gbm = GradientBoostingClassifier(
    learning_rate=0.01, max_depth=3, n_estimators=100)
gbm.fit(X_train_scaled, y_train)
accuracy = gbm.score(X_val_scaled, y_val)
print(f"Accuracy: {accuracy}")
