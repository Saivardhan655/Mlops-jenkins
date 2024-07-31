import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Set paths for input and output directories
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "raw_data")
RAW_DATA_FILE = os.environ.get("RAW_DATA_FILE", "adult.csv")
raw_data_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

# Read dataset
df = pd.read_csv(raw_data_path, sep=",")

# Income to binary
df['income'].replace(['<=50K', '>50K'], [0, 1], inplace=True)

# Drop useless variables
df.drop(['fnlwgt', 'education.num'], axis=1, inplace=True)

# Remove rows with missing data
df = df.loc[(df['workclass'] != '?') & (df['occupation'] != '?') & (df['native.country'] != '?')]

# Split into dependent and independent variables
X = df.drop('income', axis=1)
y = df['income']

# Split X into continuous and categorical variables
X_continuous = X[['age', 'capital.gain', 'capital.loss', 'hours.per.week']]
X_categorical = X[['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']]

# Get the dummies
X_encoded = pd.get_dummies(X_categorical)

# Concatenate data
data = pd.concat([y, X_continuous, X_encoded], axis=1)

# Split into train and test
train, test = train_test_split(data, test_size=0.3, stratify=data['income'])

# Define output directory and ensure it exists
PROCESSED_DATA_DIR = os.path.abspath(os.environ.get("PROCESSED_DATA_DIR", "preprocessed_data"))
print(f"Checking directory: {PROCESSED_DATA_DIR}")

if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)
    print(f"Created directory: {PROCESSED_DATA_DIR}")
else:
    print(f"Directory already exists: {PROCESSED_DATA_DIR}")

# Define output file paths
train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# Save csv files
try:
    print(f"Saving train data to {train_path}")
    train.to_csv(train_path, index=False)
    print(f"Saving test data to {test_path}")
    test.to_csv(test_path, index=False)
except Exception as e:
    print(f"Error occurred: {e}")
