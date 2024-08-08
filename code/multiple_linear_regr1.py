import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

#%matplotlib inline

try:
    df = pd.read_csv("../downloads/FuelConsumptionCo2.csv")
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    np.random.seed(1)
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]
    # plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    # plt.xlabel("Engine size")
    # plt.ylabel("Emission")
    # plt.title("Training Data set")
    # plt.show()    
    training_set = []

    regr = linear_model.LinearRegression()
#    training_set = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']'

    training_set = ['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'ENGINESIZE','CYLINDERS']
    x_train = np.asanyarray(train[training_set])
    y_train = np.asanyarray(train[['CO2EMISSIONS']])

    x_test = np.asanyarray(test[training_set])

    regr.fit(x_train, y_train)
    # The coefficients
    #print('Coefficients: ', regr.coef_)    
    y_pred = regr.predict(x_test)
    y_hat = regr.predict(test[training_set])
    x_test = np.asanyarray(test[training_set])
    y_test = np.asanyarray(test[['CO2EMISSIONS']])
    print("Mean Squared Error (MSE) : %.2f" % np.mean((y_hat - y_test) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x_test, y_test))

    # plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
    # plt.xlabel("Engine size")
    # plt.ylabel("Emission")
    #plt.plot(x_test, y_pred, color="red", linewidth = 0.5)

    
    plt.show()
except Exception as e:
    print(e)
