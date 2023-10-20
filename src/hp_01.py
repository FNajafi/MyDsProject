import numpy as np

# Step 1: Data Collection (Sample data)
# Let's create some sample data for illustration.
# In practice, you would load a dataset from a file or database.
# For simplicity, we'll use random data here.


# Feature (input) data (size of the house)
X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])

# Target (output) data (house prices)
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# Step 2: Data Preprocessing (None needed in this example)

# Step 3: Model Initialization
# Initialize the model parameters: weights (w) and bias (b)
w = 0.1
b = 0.1

# Step 4: Define the Hypothesis Function
def hypothesis(x):
    return w * x + b

# Step 5: Cost Function (Mean Squared Error)
def cost_function(X, y):
    m = len(y)
    predictions = hypothesis(X)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Step 6: Gradient Descent
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    m = len(y)
    for _ in range(num_iterations):
        predictions = hypothesis(X)
        dw = (1 / m) * np.sum((predictions - y) * X)
        db = (1 / m) * np.sum(predictions - y)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b

# Step 7: Training
learning_rate = 0.0001
num_iterations = 1000
w, b = gradient_descent(X, y, w, b, learning_rate, num_iterations)

# Step 8: Prediction
new_house_size = 2000  # New house size for prediction
predicted_price = hypothesis(new_house_size)
print(f"Predicted price for a house with size {new_house_size} sq. ft: ${predicted_price:.2f}")

# Step 9: Evaluation (MAE, RMSE, R-squared, etc. - not implemented here)
def mean_abs_error (y_true , y_pred):
    return np.mean (np.absolute(y_true - y_pred))

def rt_mean_sqrt_error (y_true , y_pred):
    return np.sqrt(np.mean(y_true - y_pred) **2)

# Calculate R-squared (Coefficient of Determination)
def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    ssr = np.sum((y_pred - y_true) ** 2)  # Sum of Squared Residuals
    sst = np.sum((y_true - y_mean) ** 2)  # Total Sum of Squares
    r2 = 1 - (ssr / sst)
    return r2

# Example usage:
r_squared_value = r_squared(y, hypothesis(X))
print(f"R-squared (R²) Score: {r_squared_value:.4f}")

# R-squared ranges from 0 to 1, with 1 indicating a perfect fit.


y_true = y 
y_pred = hypothesis(X)

mae = mean_abs_error (y_true , y_pred)
rmse = rt_mean_sqrt_error (y_true , y_pred)

print (f"mean abslout error is : {mae:.2f}")
print (f" root mean square error is : {rmse:.2f}")
# Step 10: Deployment (Not covered in this basic example)
