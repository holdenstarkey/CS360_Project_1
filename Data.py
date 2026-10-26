import pandas as pd

input_file = 'data/Training_Validation_Testing Insurance CS360 - Validation.csv'
features_file = 'data/Validation_cleaned_features.csv'
target_file = 'data/Validation_cleaned_target.csv'

# Load dataset
df = pd.read_csv(input_file)

# Remove rows with missing values
df = df.dropna()

# Separate features and target
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# Convert features into numerical values
X = pd.get_dummies(X, columns=["sex", "smoker", "region"], dtype=int)

# Save features and target to separate CSV files
X.to_csv(features_file, index=False)
y.to_frame(name="charges").to_csv(target_file, index=False)