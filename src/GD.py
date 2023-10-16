import numpy as np

# Simulated dataset for linear regression
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initialize model parameters (weights and bias)
w = 0  # Weight
b = 0  # Bias

# Hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Gradient Descent
for i in range(num_iterations):
    # Predictions with the current model parameters
    y_pred = w * X + b

    # Calculate the gradient of the cost function with respect to w and b
    dw = (1 / len(X)) * np.sum((y_pred - y) * X)
    db = (1 / len(X)) * np.sum(y_pred - y)

    # Update the model parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Calculate the cost (Mean Squared Error) for monitoring
    cost = (1 / (2 * len(X))) * np.sum((y_pred - y) ** 2)

    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")

# The final trained model parameters
print("Final model parameters:")
print(f"w = {w}")
print(f"b = {b}")
