# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Updated dataset with more records
data = {
    'Credit_Score': [720, 660, 690, 710, 580, 620, 740, 680, 690, 700, 
                     750, 580, 630, 670, 710, 660, 700, 620, 760, 680],
    'Income': [50000, 60000, 55000, 62000, 45000, 48000, 75000, 55000, 
               60000, 63000, 80000, 42000, 53000, 62000, 70000, 50000, 
               65000, 48000, 78000, 54000],
    'Loan_Amount': [20000, 30000, 25000, 32000, 22000, 26000, 40000, 28000, 
                    25000, 29000, 35000, 20000, 24000, 30000, 32000, 28000, 
                    31000, 25000, 37000, 23000],
    'Default': [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 
                0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Credit_Score', 'Income', 'Loan_Amount']]  # Features
y = df['Default']  # Target (Default)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier with a depth limit
clf = DecisionTreeClassifier(max_depth=3)  # Limiting depth to avoid overfitting

# Train the model using the training data
clf.fit(X_train, y_train)

# Make predictions using the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=['Credit_Score', 'Income', 'Loan_Amount'], class_names=['No Default', 'Default'])
plt.show()
