# Predictive Analytics Using Historical Data

## Project: Startup Profit Prediction

This project uses the `50_Startups.csv` historical dataset to estimate startup profit from:
- R&D Spend
- Administration
- Marketing Spend
- State

## Model
Multiple Linear Regression. The State column is converted into numeric features using one-hot encoding.

## Workflow
1. Load and inspect historical startup data.
2. Split the data into training and testing sets.
3. Train a regression model.
4. Evaluate it using MAE, RMSE, and R-squared.
5. Plot actual vs predicted profit.
6. Predict profit for a hypothetical startup investment.

## Files
- `50_Startups.csv` — dataset
- `predictive_analytics.py` — model training and prediction code
- `actual_vs_predicted.png` — generated when the script runs

## Run
Install dependencies:
`pip install pandas scikit-learn matplotlib`

Run:
`python predictive_analytics.py`

## Note
This is a supervised regression task using historical startup records. It predicts profit from investment inputs; it is not a time-series forecast of future monthly trends.
