# SimpleLinearRegressionProject

## why we use reshape in linear regression


Let me explain why we often use reshape in linear regression, especially in Python with libraries like NumPy and scikit-learn.

1. Linear regression expects 2D input (features matrix)

In scikit-learn, the input X must be a 2D array with shape:

(
𝑛
_
𝑠
𝑎
𝑚
𝑝
𝑙
𝑒
𝑠
,
𝑛
_
𝑓
𝑒
𝑎
𝑡
𝑢
𝑟
𝑒
𝑠
)
(n_samples,n_features)

n_samples = number of data points (rows)

n_features = number of features (columns)

Example:

If you have 10 data points with 1 feature, X should be shaped (10, 1).

But if you just create a NumPy array like:

X = np.array([1, 2, 3, 4, 5])


it has shape (5,) → a 1D array, not acceptable for scikit-learn.

That’s why we do:

X = X.reshape(-1, 1)


Now X has shape (5, 1) → 5 rows, 1 column ✅

2. Target y can stay 1D

The output (dependent variable, y) is expected as a 1D array with shape (n_samples,).
So usually, we reshape only X, not y.

3. Common reshape patterns

X.reshape(-1, 1) → convert 1D to 2D column vector

X.reshape(1, -1) → convert 1D to 2D row vector (rare in regression)

✅ Example in regression

import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([1, 2, 3, 4, 5])     # Shape (5,)
y = np.array([2, 4, 6, 8, 10])

# Reshape X to be 2D
X = X.reshape(-1, 1)  # Shape (5,1)

# Fit model
model = LinearRegression()
model.fit(X, y)

print(model.coef_, model.intercept_)


Without reshape, you’d get an error like:
Expected 2D array, got 1D array instead.

👉 So in short: We use reshape in linear regression to convert a 1D array of features into the 2D format required by scikit-learn (n_samples, n_features).
