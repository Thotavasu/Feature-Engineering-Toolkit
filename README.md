# Reusable Feature Engineering Toolkit (Regression + Classification)

A small, production-style feature engineering toolkit that builds consistent preprocessing pipelines using scikit-learn.

## Why this project matters
In real ML systems, **feature consistency** is often harder than modeling:
- training vs inference mismatches cause silent accuracy drops
- new categories in production can crash a service
- missing values break pipelines
- inconsistent preprocessing across notebooks causes chaos

This project solves those issues by packaging preprocessing into a reusable, configurable module.

## Datasets used
1. **Ames Housing (Regression)**: Predict `SalePrice` from mixed numeric + categorical features with missing values.
2. **IBM Telco Customer Churn (Classification)**: Predict `Churn` (Yes/No) from customer/service features.
