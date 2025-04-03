import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump, load

# Step 1: Create a simulated dataset
data = {'Income': [50000, 60000, 80000, 90000, 120000, 40000, 20000, 100000, 110000, 30000],
        'Loan_Amount': [10000, 15000, 20000, 18000, 25000, 8000, 12000, 22000, 23000, 5000],
        'Credit_Score': [700, 720, 750, 680, 780, 500, 400, 730, 740, 450],
        'Age': [25, 30, 35, 40, 50, 20, 22, 45, 48, 28],
        'Default': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1]}  # 0 = No Default, 1 = Default

df = pd.DataFrame(data)

# Step 2: Split data into Features (X) and Target (y)
X = df[['Income', 'Loan_Amount', 'Credit_Score', 'Age']]
y = df['Default']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Build a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of Default (class 1)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

dump(model, 'trained_model.joblib')

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Step 7: Visualize the probability of default for each customer in the test set
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(y_test)), y_pred_proba, c=y_test, cmap='coolwarm', edgecolor='k', s=100)

# Formatting the plot
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary (0.5)')
plt.title('Probability of Loan Default for Test Data')
plt.xlabel('Customer Index')
plt.ylabel('Probability of Default')
plt.colorbar(label='Actual Default (0=No, 1=Yes)')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Predict the probability of default for a new customer
new_customer = [[75000, 18000, 720, 32]]  # Income, Loan_Amount, Credit_Score, Age
probability = model.predict_proba(new_customer)[0][1]  # Probability of default
print(f"\nProbability of default for a new customer: {probability:.2f}")
