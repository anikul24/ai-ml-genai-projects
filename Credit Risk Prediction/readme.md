# 💳 Credit Risk Prediction — End-to-End MLOps Project

This project predicts **credit default risk** using customer financial data.  
It demonstrates a **complete Machine Learning lifecycle**, including data ingestion from **Snowflake**, model training and hyperparameter tuning, and experiment tracking using **MLflow**.  

The goal is to build a production-ready, explainable, and reproducible credit scoring pipeline.

Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 

Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

The goal of this competition is to build a model that borrowers can use to help make the best financial decisions and this project uses historical credit data to train machine learning models capable of classifying high-risk customers.


## Context
Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years.

## Content
SeriousDlqin2yrs	:- Person experienced 90 days past due delinquency or worse 	Y/N
RevolvingUtilizationOfUnsecuredLines	:- Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits	percentage
age	Age of borrower in years	:- integer
NumberOfTime30-59DaysPastDueNotWorse	:- Number of times borrower has been 30-59 days past due but no worse in the last 2 years.	integer
DebtRatio	:- Monthly debt payments, alimony,living costs divided by monthy gross income	percentage
MonthlyIncome	:- Monthly income	real
NumberOfOpenCreditLinesAndLoans	:- Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)	integer
NumberOfTimes90DaysLate	:- Number of times borrower has been 90 days or more past due.	integer
NumberRealEstateLoansOrLines	:- Number of mortgage and real estate loans including home equity lines of credit	integer
NumberOfTime60-89DaysPastDueNotWorse	:- Number of times borrower has been 60-89 days past due but no worse in the last 2 years.	integer
NumberOfDependents	:- Number of dependents in family excluding themselves (spouse, children etc.)	integer


## Kaggel Link
https://www.kaggle.com/competitions/GiveMeSomeCredit/overview


## 1) Problem Statement

- This project understands the financial data and predicts probability score of someone will experience financial distress in the next two years.


## 2) Data Collection
- Dataset Source - https://www.kaggle.com/competitions/GiveMeSomeCredit/data
- The data consists of separe training and test datasets.



## 🧠 Project Overview

Credit risk prediction is crucial for financial institutions to assess the likelihood of a borrower defaulting on a loan.  

---

## 🏗️ Architecture & Design

```mermaid
flowchart TD
    A[Data Source: csv -> Snowflake ] --> B[Data Ingestion from Snowflake & EDA]
    B --> C[Feature Engineering & Preprocessing]
    C --> D[Model Training: Logistic, GradientBoost]
    D --> E[Hyperparameter Tuning & Evaluation]
    E --> F[Model Tracking: MLflow]
    F --> G[Model Deployment: MLflow UI / API]
    G --> H[Result Storage: export to Snowflake tables / Local CSV]
    H --> A
