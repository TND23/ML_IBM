import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



x = []
y = [] 
for i in range(10):
    x.append([i*2])

for i in range(10,20):
    y.append([int(i*2 * (1 + (np.random.rand() * .2)))])

print(x)
print(y)

LR=LinearRegression()
LR.fit(x,y)
b=LR.predict(np.array(x))

print(b)