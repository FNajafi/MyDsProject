import numpy as np 

X = np.array([2000,4000,5000,4500,3500]) #this is the feature matrix 
y = np.array([20000,40000,50000,45000,35000]) # this is the target variable


#model initialization 
w = 0.1
b = 0.1

def hypothesis (x):
    return w * x +b

def cost_fucntion(X , y):
    m = len (y)
    prediction = hypothesis (X)
    return (1 / (2 * m)) * np.sum ((prediction - y) **2)

def GD (X , y , w , b , lr, epoch):
    m = len (y)
    for i in range (epoch):
        prediction = hypothesis (X)
        dw = ( 1 / m ) * np.sum((prediction - y) * X)
        db = (1 / m ) * np.sum (prediction - y)
        w -= lr * dw
        b -= lr * db
    return w , b

lr = 0.0001
epoch = 1000
w , b = GD (X, y, w, b, lr, epoch)

new_house = 6000
new_house_predicted_price = hypothesis(new_house)
print (f"the {new_house} sqr ft price is : $ {new_house_predicted_price:.2f}")