# Step 1: Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Fetch dataset using ucimlrepo
maternal_health_risk = fetch_ucirepo(id=863)

# Step 3: Load the dataset into a DataFrame
# Check the keys in the dataset to understand its structure
print("Data Keys:", maternal_health_risk.keys())
print("Variables:", maternal_health_risk.variables)

# Step 4: Manually assign column names based on the dataset structure
columns = [var['name'] for var in maternal_health_risk.variables]  # Extract column names from 'variables'
df = pd.DataFrame(maternal_health_risk.data, columns=columns)

# Step 5: Basic data exploration
print("Dataset shape:", df.shape)
print("First few rows of the dataset:")
print(df.head())

# Step 6: Check for missing values
print("Missing values in the dataset:\n", df.isnull().sum())

# Step 7: Visualize distribution of the target variable ('RiskLevel')
plt.figure(figsize=(8, 5))
sns.countplot(x='RiskLevel', data=df, palette="coolwarm")
plt.title("Distribution of Risk Levels in the Dataset")
plt.show()

# Step 8: Split the data into features and target
X = df.drop(columns=['RiskLevel'])
y = df['RiskLevel']

# Step 9: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 11: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Display classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 12: Feature importance visualization (optional)
importances = model.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importance Ranking")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
