import numpy as np 
import matplotlib.pyplot as plt

X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

w = 0.1
b = 0.1


def hypothesis (x):
    return w* x + b

def cost_func(X, y):
    m = len (y)
    prediction = hypothesis (X)
    return (1 /(2*m))* np.sum((prediction - y) **2)

def GD (X, y, w, b, lr, epoch):
    m = len (y)
    cost_history = []
    for _ in range (epoch):
        prediction = hypothesis (X)    # since x is a numpy array , it will give you the result in a numpy array. 
        dw = (1/m)* np.sum ((prediction - y)) * X
        db = (1/m) * np.sum ((prediction - y))
        w -= lr * dw 
        b -= lr * db
        cost = cost_func (X,y)
        cost_history.append (cost)
    return w, b , cost_history


lr = 0.0001
epoch = 1000
w, b , cost_history = GD (X, y, w, b,lr, epoch)