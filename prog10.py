import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the Locally Weighted Regression (LWR) function
def locally_weighted_regression(X, y, tau, x_query):
    """ 
    X: training features (n x 1)
    y: training labels (n x 1)
    tau: bandwidth parameter (controls the locality)
    x_query: the query point (scalar)
    """
    m = X.shape[0]
    X_ = np.hstack([np.ones((m, 1)), X])  # Add intercept term (bias)
    W = np.zeros((m, m))  # Initialize the weight matrix
    
    # Step 2: Compute the weights for each data point based on proximity to the query point
    for i in range(m):
        W[i, i] = np.exp(-(X[i] - x_query) ** 2 / (2 * tau ** 2))
    
    # Step 3: Compute the theta (parameters) using weighted normal equation
    theta = np.linalg.pinv(X_.T @ W @ X_) @ X_.T @ W @ y
    
    # Step 4: Make prediction for the query point
    x_query_ = np.array([1, x_query])  # Add bias to the query point
    return x_query_ @ theta

# Step 5: Generate synthetic data for regression (dummy dataset)
np.random.seed(0)
X_train = np.linspace(0, 10, 100).reshape(-1, 1)  # 100 points from 0 to 10
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0])  # Sine wave with noise

# Step 6: Perform Locally Weighted Regression for each test point
tau = 0.5  # Bandwidth parameter (controls the smoothness)
X_test = np.linspace(0, 10, 1000)  # 1000 test points for plotting smooth curve
y_pred = np.array([locally_weighted_regression(X_train, y_train, tau, x) for x in X_test])

# Step 7: Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_test, y_pred, color='red', label=f'LWR (tau={tau})', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
