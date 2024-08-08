import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

try:
    df = pd.read_csv("../downloads/FuelConsumptionCo2.csv")
    df.describe()
    cols = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']
    cdf = df[cols]

    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(train_x, train_y)

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()
    
    # test_x = np.asanyarray(test[['ENGINESIZE']])
    # test_y = np.asanyarray(test[['CO2EMISSIONS']])
    # test_y_ = regr.predict(test_x)


    # plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    # plt.xlabel("Engine size")
    # plt.ylabel("Emission")
    # plt.show()
except Exception as e:
    print(e)