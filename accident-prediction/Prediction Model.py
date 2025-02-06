#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

# Correct file path handling
file_path = os.path.join('accident-prediction', 'accidents_india.csv')
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isna().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Fill remaining missing values with column means
df.Sex_Of_Driver = df.Sex_Of_Driver.fillna(df.Sex_Of_Driver.mean())
df.Vehicle_Type = df.Vehicle_Type.fillna(df.Vehicle_Type.mean())
df.Speed_limit = df.Speed_limit.fillna(df.Speed_limit.mean())
df.Road_Type = df.Road_Type.fillna(df.Road_Type.mean())
df.Number_of_Pasengers = df.Number_of_Pasengers.fillna(df.Speed_limit.mean())

# Correlation heatmap
corr = df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="RdYlBu", annot=True, fmt=".1f")
plt.show()

# Label Encoding
c = LabelEncoder()
df['Day'] = c.fit_transform(df['Day_of_Week'])
df.drop('Day_of_Week', axis=1, inplace=True)

l = LabelEncoder()
df['Light'] = l.fit_transform(df['Light_Conditions'])
df.drop('Light_Conditions', axis=1, inplace=True)

s = LabelEncoder()
df['Severity'] = s.fit_transform(df['Accident_Severity'])
df.drop('Accident_Severity', axis=1, inplace=True)

# Display the first few rows after preprocessing
print(df.head())

# Split data into features (x) and target (y)
x = df.drop(['Pedestrian_Crossing', 'Special_Conditions_at_Site', 'Severity'], axis=1)
y = df['Severity']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.86, random_state=42)

# Decision Tree Classifier
reg = DecisionTreeClassifier(criterion='gini')
reg.fit(x_train, y_train)
print("Decision Tree Accuracy:", reg.score(x_test, y_test))

# Confusion Matrix for Decision Tree
yp = reg.predict(x_test)
cm = confusion_matrix(y_test, yp)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Save the Decision Tree model
with open('test1.pkl', 'wb') as f:
    pickle.dump(reg, f)

# Load the saved model
with open('test1.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Test the loaded model
input_data = [2, 10, 201, 10, 10, 8, 3]  # Example input
prediction = loaded_model.predict([input_data])
print("Prediction from loaded model:", prediction)

# Random Forest Classifier
r_forest = RandomForestClassifier(criterion='entropy', random_state=42)
r_forest.fit(x_train, y_train)
print("Random Forest Accuracy:", r_forest.score(x_test, y_test))

# SVM Classifier
svm_model = svm.SVC(C=50, kernel='linear')
svm_model.fit(x_train, y_train)
print("SVM Accuracy:", svm_model.score(x_test, y_test))

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
print("Logistic Regression Accuracy:", log_reg.score(x_test, y_test))

# Feature Importance Plot for Decision Tree
plt.figure(figsize=(10, 6))
plt.barh(range(len(x.columns)), reg.feature_importances_, align='center')
plt.yticks(range(len(x.columns)), x.columns)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Decision Tree")
plt.show()

# Decision Tree Visualization
from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(reg, filled=True, feature_names=x.columns, class_names=['Low', 'Medium', 'High'])
plt.show()