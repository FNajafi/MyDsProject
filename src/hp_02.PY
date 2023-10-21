import numpy as np

# Sample feature data (size and number of bedrooms)
X = np.array([
    [1400, 3],
    [1600, 3],
    [1700, 2],
    [1875, 4],
    [1100, 2],
    [1550, 3],
    [2350, 4],
    [2450, 5],
    [1425, 3],
    [1700, 2]
])

# Sample target data (house prices)
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# Initialize model parameters
w1 = 0.1  # Weight for feature 1 (size)
w2 = 0.1  # Weight for feature 2 (number of bedrooms)
b = 0.1  # Bias

# Hypothesis Function
def hypothesis(x):   #Passing (x) means we pass an entire row to the function and then later in the fucntion we unfoald the array by specifiying the x[0] and x[1]
    return w1 * x[0] + w2 * x[1] + b

# Cost Function (Mean Squared Error)
def cost_function(X, y):
    m = len(y)
    predictions = np.array([hypothesis(x) for x in X])
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Gradient Descent
def gradient_descent(X, y, w1, w2, b, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        predictions = np.array([hypothesis(x) for x in X])
        dw1 = (1 / m) * np.sum((predictions - y) * X[:, 0])  # Derivative with respect to w1
        dw2 = (1 / m) * np.sum((predictions - y) * X[:, 1])  # Derivative with respect to w2
        db = (1 / m) * np.sum(predictions - y)  # Derivative with respect to b
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b -= learning_rate * db
    return w1, w2, b

# Training
learning_rate = 0.0000001
num_iterations = 1000
w1, w2, b = gradient_descent(X, y, w1, w2, b, learning_rate, num_iterations)

# Prediction for a new house
new_house_size = 2000
new_bedrooms = 4
predicted_price = hypothesis([new_house_size, new_bedrooms])
print(f"Predicted price for a house with size {new_house_size} sq. ft and {new_bedrooms} bedrooms: ${predicted_price[0]:.2f}")
