import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("../downloads/drug200.csv", delimiter=",")
       
    X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

    # convert categorical data, assign independent vars
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F','M'])
    X[:,1] = le_sex.transform(X[:,1]) 

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
    X[:,2] = le_BP.transform(X[:,2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit([ 'NORMAL', 'HIGH'])
    X[:,3] = le_Chol.transform(X[:,3]) 

    # assign independent var
    y = df["Drug"] 

    X_test, X_train, y_test, y_train = train_test_split(X,y, test_size=.3, random_state=3)
    # create the decision tree
    drugtree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
#    print(drugtree)
    drugtree.fit(X_train, y_train)
    predtree = drugtree.predict(X_test)
    print(predtree[0:5])
    print(y_test[0:5])
    tree.plot_tree(drugtree)
    plt.show()
except Exception as e:
    print(e)
