# Diabetes Predictor using K Nearest Neighbours

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the dataset from a given file path
def load_data(file_path):
    return pd.read_csv(file_path)

# Display the dataset information
def display_dataset_info(data):
    print('Columns of dataset:')
    print(data.columns, "\n")
    print('First 5 records of dataset:')
    print(data.head(), "\n")
    print(f'Dimension of diabetes data: {data.shape}\n')

# Split the data into training and testing sets
def split_data(data):
    X = data.loc[:, data.columns != 'Outcome']
    y = data['Outcome']
    return train_test_split(X, y, stratify=y, random_state=66)

# Train KNN models with different numbers of neighbors and record their accuracies
def train_knn_models(X_train, y_train, X_test, y_test, max_neighbors=10):
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, max_neighbors + 1)

    for n_neighbors in neighbors_settings:
        # Build the KNN model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        # Record the training set accuracy
        training_accuracy.append(knn.score(X_train, y_train))

        # Record the test set accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    return neighbors_settings, training_accuracy, test_accuracy

# Plot the training and testing accuracies for different numbers of neighbors
def plot_accuracy(neighbors_settings, training_accuracy, test_accuracy):
    plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Neighbors")
    plt.legend()
    plt.title("KNN: Varying Number of Neighbors")
    plt.savefig('K Nearest Neighbours/KNN_Compare_model.png')
    plt.show()

# Evaluate the KNN model with the specified number of neighbors and print the accuracy
def evaluate_model(X_train, y_train, X_test, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    print(f'Accuracy of KNN classifier on training set: {knn.score(X_train, y_train):.2f}')
    print(f'Accuracy of KNN classifier on test set: {knn.score(X_test, y_test):.2f}')
    return knn

# Main function to orchestrate the execution of all the tasks
def main():
    print("--------------------Created By Mahesh Pawar---------------------")
    print("---------Diabetes Predictor using K Nearest Neighbours----------\n")

    # Define the path to the dataset
    file_path = 'K Nearest Neighbours/diabetes.csv'

    # Load the data
    diabetes_data = load_data(file_path)

    # Display dataset information
    display_dataset_info(diabetes_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(diabetes_data)

    # Train KNN models and get accuracies
    neighbors_settings, training_accuracy, test_accuracy = train_knn_models(X_train, y_train, X_test, y_test)

    # Plot the training and testing accuracies
    plot_accuracy(neighbors_settings, training_accuracy, test_accuracy)

    # Evaluate the best model (with 9 neighbors)
    best_n_neighbors = 9
    evaluate_model(X_train, y_train, X_test, y_test, best_n_neighbors)

# Execute the main function
if __name__ == "__main__":
    main()