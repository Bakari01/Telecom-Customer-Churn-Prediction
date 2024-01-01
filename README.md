# Telecom Customer Churn Prediction using Classification Algorithms. 

## Table of Contents
1. [Introduction](#introduction)
2. [Data Collection](#data-collection)
3. [Data Cleaning](#data-cleaning)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Modeling](#modeling)
7. [Evaluation](#evaluation)
8. [Conclusion](#conclusion)
9. [References](#references)

## Introduction
This project aims to predict customer churn in a telecom company using various customer features such as tenure, monthly charges, type of services used, etc. The goal is to identify customers who are likely to churn, so that the company can proactively engage with these customers and try to retain them.

## Data Collection
The data was collected from the Kaggle dataset "Telco Customer Churn". It consists of 7043 instances with 21 features describing various aspects of a customer's account.

## Data Cleaning
The data cleaning process involved handling missing values and removing duplicates, checking for inconsistent data types and unusual observations. 

## Exploratory Data Analysis
During the exploratory data analysis, we found that churn is highly correlated with features like tenure, monthly charges, and type of contract. We visualized these relationships using bar charts and box plots.

## Feature Engineering
We standardized the numerical features using `Standard Scaler` as we as encoding categorical variables to `0` and `1` because they were binary variables and a boolean one - churn where True was encoded to `1` and `0` represented False.

## Modeling
The following models were trained:
* Support Vector Classifier
* Logistic Regression,
* Decision Trees,
* Random Forest,
* LightGBM,
* XGBoost,
* KNN,
* Gradient Boosting,
* AdaBoost,
* SGD,
* Adaline
Grid Search Cross Validation was used to tune the hyperparameters of the models. 

## Evaluation
We evaluated our models using accuracy, precision, recall, and the F1 score, AUC Score. Our gradient boosting model performed better with a higher F1 score and AUC Score.

## Conclusion
Our models were able to predict customer churn with reasonable accuracy. However, there are some limitations. For example, our models might not perform well on data from other telecom companies because the dataset we used is specific to one company. Future work could involve collecting more data from other companies and improving the models' performance on a more diverse dataset.

## References
1. "Telco Customer Churn", Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn
2. "A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library", Machine Learning Mastery: https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
