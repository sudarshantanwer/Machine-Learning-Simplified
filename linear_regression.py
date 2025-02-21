# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (you can replace this with your own dataset)
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [40000, 35000, 50000, 55000, 60000, 6000, 70000, 75000, 80000, 85000]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Separate features and target variable
X = df[['Experience']]  # Independent variable
y = df['Salary']        # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Evaluate the model
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")

# Visualize the data and regression line
plt.scatter(X, y, color='blue')  # Plot original data points
plt.plot(X, model.predict(X), color='red')  # Plot the regression line
plt.title('Linear Regression: Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
