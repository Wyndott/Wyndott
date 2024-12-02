import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data
url = 'https://bit.ly/33ebs2R'
try:
    df = pd.read_csv(url, delimiter=",")
    print(df.head())
    print(df.info())
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Extract input variables (all rows, all columns but last column)
X = df.iloc[:, :-1].values
print("X shape:", X.shape)

# Extract output column (all rows, last column)
Y = df.iloc[:, -1].values
print("Y shape:", Y.shape)

# Perform logistic regression
model = LogisticRegression(penalty=None, max_iter=1000)  # Corrected 'penalty' parameter
try:
    model.fit(X, Y)
except Exception as e:
    print(f"Error fitting model: {e}")
    raise

# Print coefficients and intercept
print("Coefficients:", model.coef_.flatten())
print("Intercept:", model.intercept_.flatten())
