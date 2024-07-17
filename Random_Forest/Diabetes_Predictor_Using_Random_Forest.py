# Diabetes Predictor using Random Forest Classifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter

# Ignore future warnings
simplefilter(action='ignore', category=FutureWarning)  

# Load the dataset from a given file path
def load_data(file_path):
    return pd.read_csv(file_path)

# Display the dataset information
def display_dataset_info(data):
    print("Columns of Dataset:")
    print(data.columns, "\n")  # Print the column names of the dataset

    print("First 5 records of dataset:")
    print(data.head(), "\n")  # Print the first 5 records of the dataset

    print("Dimension of diabetes dataset: {}\n".format(data.shape))  # Print the shape of the dataset

# Split the data into training and testing sets
def split_data(data):
    X = data.loc[:, data.columns != 'Outcome']  # Features (all columns except 'Outcome')
    y = data['Outcome']  # Target (the 'Outcome' column)
    return train_test_split(X, y, stratify=y, random_state=66)  # Ensure reproducibility and maintain class proportions

# Train the Random Forest model
def train_random_forest(X_train, Y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=0)  # Initialize Random Forest with 100 trees
    model.fit(X_train, Y_train)  # Fit the model to the training data
    return model

# Print accuracy of the model on training and test sets
def print_accuracy(model, X_train, Y_train, X_test, Y_test):
    print("Accuracy on training set: {:.3f}".format(model.score(X_train, Y_train)))  # Training accuracy
    print("Accuracy on test set: {:.3f}".format(model.score(X_test, Y_test)))  # Test accuracy

# Plot feature importance
def plot_feature_importance(model, data):
    plt.figure(figsize=(8, 5))  # Set the figure size
    n_features = data.shape[1] - 1  # Number of features (excluding target)
    plt.barh(range(n_features), model.feature_importances_, align='center')  # Horizontal bar plot
    Diabetes_features = [x for i, x in enumerate(data.columns) if i != 8]  # Feature names (excluding target)
    plt.yticks(np.arange(n_features), Diabetes_features)  # Set the y-axis ticks to the feature names
    plt.xlabel("Feature Importance")  # Set the x-axis label
    plt.ylabel("Feature")  # Set the y-axis label
    plt.ylim(-1, n_features)  # Set the y-axis limits
    plt.title("Random Forest: Feature Importance")  # Set the plot title
    plt.savefig('Random_Forest/Random_Forest.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

# Main function to orchestrate the execution of all tasks
def main():
    print("--------------------Created by Mahesh Pawar--------------------")
    print("------Diabetes Predictor using Random Forest algorithm-------\n")

    # Define the path to the dataset
    file_path = 'Random_Forest/diabetes.csv'

    # Load the data
    diabetes_data = load_data(file_path)

    # Display dataset information
    display_dataset_info(diabetes_data)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = split_data(diabetes_data)

    # Train the Random Forest model
    random_forest_model = train_random_forest(X_train, Y_train)

    # Print accuracy of the model
    print_accuracy(random_forest_model, X_train, Y_train, X_test, Y_test)

    # Plot feature importance
    plot_feature_importance(random_forest_model, diabetes_data)

# Execute the main function
if __name__ == "__main__":
    main()
