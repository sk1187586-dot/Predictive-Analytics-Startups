# Predictive Analytics Using Historical Data
# Startup Profit Prediction using Multiple Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (keep this file in the same folder as this script)
data = pd.read_csv("50_Startups.csv")

X = data[["R&D Spend", "Administration", "Marketing Spend", "State"]]
y = data["Profit"]

preprocessor = ColumnTransformer(
    transformers=[("state", OneHotEncoder(drop="first", handle_unknown="ignore"), ["State"])],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("Root Mean Squared Error:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
print("R-squared:", round(r2_score(y_test, predictions), 4))

# Compare actual and predicted profit for test records
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Startup Profit")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# Example prediction: enter hypothetical investment values
new_startup = pd.DataFrame({
    "R&D Spend": [100000],
    "Administration": [120000],
    "Marketing Spend": [200000],
    "State": ["New York"]
})
estimated_profit = model.predict(new_startup)[0]
print("Estimated startup profit:", round(estimated_profit, 2))
