import numpy as np


X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

w = 0.1
b = 0.1

def hypothesis (x):
    return w * x + b

def cost_function (X,y):
    m = len (y)
    prediction = hypothesis (X)
    return (1 / ( 2 * m )) * np.sum ((prediction - y ) ** 2)   
 
def GD (X , y, w, b, lr , epoch):
    m = len(y)
    for i in range (epoch):
        prediction = hypothesis (X)
        dw = (1 / m) * np.sum ((prediction - y) * X ) #this is the order partial derivtive of the cost function in respect to w
        db = (1 / m) * np.sum (prediction - y)     #this is the order partial derivtive of the cost function in respect to b
        w -= lr * dw
        b -= lr * db
    return w, b 

lr = 0.0001
epoch = 1000
w , b = GD (X , y, w, b, lr, epoch)


new_house_size = 2000
predicted_price = hypothesis(new_house_size)
print (f"pridicted price for a house with size {new_house_size} sqr.ft : ${predicted_price:.2f} ")
                               
           
    