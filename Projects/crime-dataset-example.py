import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# You can download the dataset from UCI and place the file path accordingly
# Here, assuming 'liver-disorders.csv' is the file from UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'

# No headers in original dataset, so define column names manually
column_names = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']

# Load the data into a DataFrame
df = pd.read_csv(url, names=column_names)

# Step 2: Basic data exploration
print("First few rows of the dataset:")
print(df.head())

# Step 3: Handle any missing values (if present, though in this dataset, there are none)
# For example: df.fillna(df.mean(), inplace=True)

# Step 4: Visualize feature distributions
plt.figure(figsize=(10, 6))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.show()

# Step 5: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Liver Disorder Features")
plt.show()

# Step 6: Boxplot to visualize distributions and outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.iloc[:, :-1], orient='h')
plt.title("Boxplot of Liver Disorder Features")
plt.show()

# Step 7: Pairplot of all features with the 'drinks' variable
sns.pairplot(df, hue='drinks', markers='+')
plt.suptitle("Pairplot of Liver Disorder Features Colored by Drinks", y=1.02)
plt.show()
