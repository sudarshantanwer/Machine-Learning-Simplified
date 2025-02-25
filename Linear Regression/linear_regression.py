import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Reload the saved Linear Regression model from disk
model = load('linear_regression_model.joblib')

# Example data for predictions
X_test = np.array([[1], [2], [3], [4], [5], [6], [7]])  # Input features

# Predict values using the model (use numpy arrays to avoid the warning)
y_predicted = model.predict(X_test)

# Example actual values for comparison (use the values relevant to your dataset)
y_actual = np.array([57370.68, 61862.06, 66353.44, 70844.82, 75336.2, 79827.58, 84318.96])

# Plot actual vs. predicted values using Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_actual, color='blue', label='Actual Values')  # Actual data points
plt.plot(X_test, y_predicted, color='red', label='Predicted Line')  # Prediction line
plt.xlabel('Feature (X)')  # X-axis label
plt.ylabel('Target (y)')  # Y-axis label
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Print the predictions
print("Predictions:", y_predicted)
