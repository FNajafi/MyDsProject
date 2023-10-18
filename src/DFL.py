import numpy as np

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

y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

w1 = 0.1
w2 = 0.1
b = 0.1

def hypothesis (x):
    return w1 * x[0] + w2 * x[1] + b

def cost_fucntion (X , y):
    m = len (y)
    prediction = np.array ([hypothesis(x) for x in X])
    return (1 / (2 * m)) * np.sum((prediction - y) ** 2)
    
def GD (X , y , w1 , w2 , lr, epoch):
    m = len (y)
    for i in range(epoch):
        prediction = np.array( hypothesis(x) for x in X)
        