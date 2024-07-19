# Diabetes Predictor using Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter

# Ignore future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load the dataset from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Display basic information about the dataset
def display_dataset_info(data):
    print('Columns of Dataset')
    print(data.columns)
    print('\nFirst 5 records of the dataset')
    print(data.head())
    print('\nDimension of diabetes data: {}'.format(data.shape))

# Split the dataset into training and testing sets
def split_data(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data.loc[:, data.columns != 'Outcome'],
                                                        data['Outcome'], stratify=data['Outcome'], random_state=66)
    return X_train, X_test, Y_train, Y_test

# Train and evaluate the logistic regression model
def train_and_evaluate(X_train, X_test, Y_train, Y_test, C=1.0):
    model = LogisticRegression(C=C).fit(X_train, Y_train)
    print('Training set accuracy (C={:.2f}): {:.3f}'.format(C, model.score(X_train, Y_train)))
    print('Test set accuracy (C={:.2f}): {:.3f}\n'.format(C, model.score(X_test, Y_test)))

# Main function to execute the workflow
def main():
    print("--------------------Created By Mahesh Pawar---------------------")
    print("---------Diabetes Predictor using Logistic Regression-----------\n")

    file_path = 'diabetes.csv'  # Path to the CSV file
    data = load_data(file_path)  # Load the data
    display_dataset_info(data)  # Display dataset info

    X_train, X_test, Y_train, Y_test = split_data(data)  # Split the data
    train_and_evaluate(X_train, X_test, Y_train, Y_test, C=1.0)  # Train and evaluate the model with C=1.0
    train_and_evaluate(X_train, X_test, Y_train, Y_test, C=0.01)  # Train and evaluate the model with C=0.01

# Execute the main function
if __name__ == "__main__":
    main()
