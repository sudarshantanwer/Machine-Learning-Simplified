import pandas as pd
import numpy as np

# Load the dataset
file_path = "data.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# 1. Remove duplicate rows
df = df.drop_duplicates()

# 2. Standardize Name column (remove extra spaces & capitalize first letter)
df["Name"] = df["Name"].str.strip().str.capitalize()

# 3. Standardize Gender values
df["Gender"] = df["Gender"].str.lower().map({
    "m": "Male", "male": "Male",
    "f": "Female", "female": "Female"
})

# 4. Normalize City names, fill missing with "Unknown"
city_mapping = {
    "new york": "New York", "ny": "New York",
    "los angeles": "Los Angeles", "la": "Los Angeles",
    "san francisco": "San Francisco", "sf": "San Francisco",
    "boston": "Boston", "hou": "Houston", "houston": "Houston",
    "chicago": "Chicago"
}
df["City"] = df["City"].str.strip().str.lower().map(city_mapping)
df["City"].fillna("Unknown", inplace=True)

# 5. Convert Salary to Numeric (remove "$" & convert to integer), fix outliers
df["Salary"] = df["Salary"].replace('[\$,]', '', regex=True).astype(float)
q1, q3 = df["Salary"].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = max(30000, q1 - 1.5 * iqr)  # Set minimum salary
upper_bound = q3 + 1.5 * iqr  # Cap high outliers
df["Salary"] = np.clip(df["Salary"], lower_bound, upper_bound)
df["Salary"].fillna(df["Salary"].median(), inplace=True)

# 6. Fix Age Column (set invalid values as NaN, then fill with median)
df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = np.nan
df["Age"].fillna(df["Age"].median(), inplace=True)

# 7. Convert Joining Date to standard format (YYYY-MM-DD) & fill missing values
df["Joining_Date"] = pd.to_datetime(df["Joining_Date"], errors='coerce')
most_common_date = df["Joining_Date"].mode()[0]
earliest_date = df["Joining_Date"].min()
df["Joining_Date"].fillna(most_common_date if most_common_date else earliest_date, inplace=True)

# 8. Fix Height (Assume >100 is cm, convert to inches), fill missing with gender-based median
df.loc[df["Height"] > 100, "Height"] = df["Height"] / 2.54
median_height_male = df[df["Gender"] == "Male"]["Height"].median()
median_height_female = df[df["Gender"] == "Female"]["Height"].median()
df.loc[(df["Gender"] == "Male") & (df["Height"].isna()), "Height"] = median_height_male
df.loc[(df["Gender"] == "Female") & (df["Height"].isna()), "Height"] = median_height_female

# 9. Handle Missing Values in Fraud column
df["Fraud"].fillna(0, inplace=True)

# 10. Handle Data Leakage (Drop Total_Sales_2025)
df.drop(columns=["Total_Sales_2025"], inplace=True)

# 11. Remove decimal part (convert to integer)
df['Height'] = df['Height'].astype(int)

# Save the cleaned dataset
cleaned_file = "final_cleaned_data.csv"
df.to_csv(cleaned_file, index=False)

print(f"Data fully cleaned and saved to {cleaned_file}")
