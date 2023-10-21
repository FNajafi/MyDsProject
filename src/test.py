import numpy as np 

X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

w = 0.1
b = 0.1

def hypothesis (x):
    return w * x + b

def cost_func (X , y):
    m = len (y)
    prediction = hypothesis (X)
    return (1/(2*m)) * np.sum((prediction - y)** 2)

def GD (X , y, w, b, lr, epoch):
    m = len (y)
    for i in range (epoch):
        prediction = hypothesis (X)
        dw = (1 / m) * np.sum(prediction - y) * X
        wb = (1/ m) * np.sum(prediction - y)
        w -= lr * dw
        b -= lr * wb
    return w, b

lr = 0.000001
epoch = 2000
w, b = GD (X, y, w, b, lr, epoch)

new_house = 2000
new_house_price = hypothesis (new_house)
print (f"the house with {new_house} sqr feet price is $: {new_house_price:.2f}")