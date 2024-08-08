import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    # get the data set
    df = pd.read_csv("../downloads/TTGovtSalaries_latest.csv")

    row_limit = 1000
    msk = np.random.rand(row_limit) < .8 
    np.random.seed(1)
    le_sex = preprocessing.LabelEncoder()
    le_agy = preprocessing.LabelEncoder()
    le_jobclass = preprocessing.LabelEncoder()
    le_race = preprocessing.LabelEncoder()

    le_sex_fit = []
    le_agy_fit = []
    le_jobclass_fit = []
    le_race_fit = []

    cols = ['AGY', 'JOBCLASS', 'RACE', 'SEX', 'HRSWKD', 'RATE']
    # convert categorical data into numerical
    for agy in df['AGY'].unique():
        le_agy_fit.append(agy)
    for race in df['RACE'].unique():
        le_race_fit.append(race)
    for sex in df['SEX'].unique():
        le_sex_fit.append(sex)
    for jobclass in df['JOBCLASS'].unique():
        le_jobclass_fit.append(jobclass)
    
    le_sex.fit(le_sex_fit)
    le_race.fit(le_race_fit)
    le_jobclass.fit(le_jobclass_fit)
    le_agy.fit(le_agy_fit)

    print(le_race_fit)

    with_rates = df[df.RATE > 0]

    cdf = with_rates[cols].head(row_limit)
    X = cdf.values
    X[:,0] = le_agy.transform(X[:,0])
    X[:,1] = le_jobclass.transform(X[:,1])
    X[:,2] = le_race.transform(X[:,2])        
    X[:,3] = le_sex.transform(X[:,3])
    
    print(X)
#    print(le_agy_fit)

    #X = df.drop(labels=['AGY', 'RACE', 'JOBCLASS', 'SEX'], axis=1)
    #Y = df['RATE']
    

    #X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.3, random_state=101)

    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # print(predictions)

    # salary_present = df[cols].notna()['RATE']

    # # masked_cdf = with_rates.mask(with_rates.RATE > 100, 10000)
    # # print(masked_cdf.sort_values(by='RATE', ascending=False)[['AGY', 'RATE', 'NAME']].head(100))
    # viz = cdf.hist()
    # plt.show()
    # cdf = df[df[cols].notnull().all(1)].head(row_limit)
    # print(cdf.shape)
    # cdf = cdf[cols]
    # print(cdf.shape)
    # #split the data into a training set and a testing set
    # train = cdf[msk]
    # test = cdf[~msk]

    #regr = linear_model.LinearRegression
    # select columns to look at in the training set
    #training_set = ['AGY', 'RACE', 'HRSWKD' 'SEX']
    # x_train = np.asanyarray(train[['AGY', 'RACE', 'HRSWKD' 'SEX']])
    # y_train = np.asanyarray(train['RATE'])
    # print(x_train)
    #x_test = np.asanyarray(test[training_set])
    #regr.fit(x_train, y_train)



except Exception as e:
    print(e)
