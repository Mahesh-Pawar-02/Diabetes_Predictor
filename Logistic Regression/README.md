# Diabetes Predictor using Logistic Regression

This project implements a diabetes predictor using a logistic regression model. The predictor is built using Python and several data science 
libraries, including pandas, numpy, scikit-learn, and matplotlib.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Author](#author)

## Introduction
Diabetes is a chronic disease that affects millions of people worldwide. Early diagnosis and proper management can significantly improve 
patients' quality of life. This project aims to build a predictive model that can classify whether a person has diabetes based on certain 
medical attributes.

## Dataset
The dataset used for this project is the Pima Indians Diabetes Database, which is available on Kaggle. The dataset consists of several medical 
predictor variables and one target variable, `Outcome`, which indicates whether the patient has diabetes (1) or not (0).

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install the required libraries using pip:

pip install pandas numpy matplotlib scikit-learn

## Results
The script prints the accuracy of the logistic regression model on both the training and test sets for different values of the regularization 
parameter C.

## Conclusion
This project demonstrates how to build and evaluate a logistic regression model to predict diabetes based on medical data. The model can be 
further improved by exploring different preprocessing techniques, feature selection methods, and more advanced machine learning algorithms.

## Author
Created By Mahesh Dinkar Pawar