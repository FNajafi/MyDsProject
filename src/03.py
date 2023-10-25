import numpy as np

X = np.array ([
    [1400, 2, 3], 
    [1600, 3, 4], 
    [1700, 4, 5], 
    [1875, 5, 8], 
    [1100, 5, 3], 
    [1550, 4, 4], 
    [2350, 5, 3], 
    [2450, 4, 8], 
    [1425, 5, 8], 
    [1700, 7, 4]
])


y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])


#lodel initilaizaiton
w1 = 0.1
w2 = 0.1 
w3 = 0.1 
b = 0.1 


def hypothesis (x):
    return w1 * x [0] + w2 * x[1] + w3 * x[2] + b

def cost_fucntion(X , y):
    m = len (y)
    prediction = np.array([hypothesis(x)for x in X])
    return (1/(2*m))* np.sum (((prediction - y) ** 2))
def GD (X, y, w1, w2, w3, b, lr, epoch):
    m = len (y)
    for i in range (epoch):
        prediction = np.array ([hypothesis(x) for x in X])
        dw1 = (1 / m ) * np.sum ((prediction - y) * X[:, 0])
        dw2 = (1 / m ) * np.sum ((prediction - y) * X[:, 1])
        dw3 = (1 / m ) * np.sum ((prediction - y) * X[:, 2])
        db = (1/ m) * np.sum ((prediction - y))
        w1 -= lr * dw1
        w2 -= lr * dw2
        w3 -= lr * dw3
        b -= lr * db
    return w1, w2, w3, b

lr = 0.00001
epoch = 1000
w1, w2, w3, b = GD (X, y, w1, w2, w3, b, lr, epoch)

new_house_size = 3000
new_house_bed = 5
new_house_living= 7 
new_house_price = hypothesis ([new_house_size, new_house_bed, new_house_living])
print (f"the house with the size of {new_house_size} , and {new_house_bed} bedrooms , and {new_house_living} living rroms is coming with the $: {new_house_price:.2f}")



#Evaluation of the model 

y_actual = y
y_pred = np.array([hypothesis(x) for x in X])

def MAE (y_pred , y_actual):
    return np.mean(np.absolute(y_actual - y_pred))
print (f"the MAE of is : {MAE(y_pred, y_actual)}")