import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the diabetes dataset from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Display dataset information
def display_data_info(data):
    print('Columns of dataset:')
    print(data.columns, "\n")
    print('First 5 records of dataset:')
    print(data.head(), "\n")
    print(f'Dimension of diabetes data: {data.shape}\n')

# Split the dataset into training and testing sets
def split_data(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data.loc[:, data.columns != 'Outcome'], data['Outcome'],
                                        stratify=data['Outcome'], random_state=66)
    return X_train, X_test, Y_train, Y_test

# Train a Decision Tree classifier and display accuracy
def train_and_evaluate(X_train, Y_train, X_test, Y_test, max_depth=None):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    tree.fit(X_train, Y_train)
    train_accuracy = tree.score(X_train, Y_train)
    test_accuracy = tree.score(X_test, Y_test)
    print("------------------------------------------------------------------------------------")
    print(f'Accuracy of Decision Tree classifier on training set: {train_accuracy:.3f}')
    print(f'Accuracy of Decision Tree classifier on test set: {test_accuracy:.3f}')
    return tree

# Plot feature importances of the model
def plot_feature_importance(model, data, output_path):
    plt.figure(figsize=(8, 6))
    n_features = data.shape[1] - 1
    plt.barh(range(n_features), model.feature_importances_, align='center')
    Diabetes_features = [x for i, x in enumerate(data.columns) if i != n_features]
    plt.yticks(np.arange(n_features), Diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.savefig(output_path)
    plt.show()

# Main function to execute the script
def main():
    print("--------------------Created By Mahesh Pawar---------------------")
    print("------------Diabetes Predictor using Decision Tree--------------\n")
    
    Diabetes = load_data('Decision_Tree/diabetes.csv')
    display_data_info(Diabetes)
    X_train, X_test, Y_train, Y_test = split_data(Diabetes)
    # Train and evaluate the model without max_depth
    train_and_evaluate(X_train, Y_train, X_test, Y_test)
    # Train and evaluate the model with max_depth = 3
    tree = train_and_evaluate(X_train, Y_train, X_test, Y_test, max_depth=3)
    print("Feature importance : {}\n".format(tree.feature_importances_))
    plot_feature_importance(tree, Diabetes, 'Decision_Tree/Decision_Tree_model.png')

# Execute the main function
if __name__ == "__main__":
    main()