import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Reload the saved Linear Regression model from disk
model = load('linear_regression_model.joblib')

# Example data for predictions (input features)
X_test = np.array([[1], [2]])  # Ensure this matches the input format used during training

# Predict values using the model
y_predicted = model.predict(X_test)

# Example feature values (for the X-axis)
# You can use indices [1, 2, ...] or use the feature values (X_test) to represent the X-axis
features = [1, 2]  # Use corresponding features for visualization

# Plot predicted values using Matplotlib
plt.figure(figsize=(8, 6))

# Scatter plot for predicted points
plt.scatter(features, y_predicted, color='red', label='Predicted Values')

# Optionally, plot the line for predicted values
plt.plot(features, y_predicted, color='blue', linestyle='--', label='Predicted Line')

# Customize the plot
plt.xlabel('Feature (X)')
plt.ylabel('Predicted Values (y)')
plt.title('Linear Regression: Predicted Values')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Print the predictions
print("Predictions:", y_predicted)
