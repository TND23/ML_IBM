import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


govt_sal_df = pd.read_csv("../downloads/TTGovtSalaries_latest.csv")
fuel_df = pd.read_csv("../downloads/FuelConsumptionCo2.csv")

# find the column names of fuel_df

# find the max annual salary of govt_sal_df

# create an index that increments by 10 for the fuel_df dataframe

# access the index 10 for the fuel_df dataframe

# write a method to access the item at any given index  

# access the value at column 3 row 5 for the fuelconsumption dataframe

# replace all NaN salaries in the govt salaries dataframe with 1000

# ensure that NaN values are not present for the salaries

# replace all last names of 'Johnson' and 'Hernandez' with 'Smith' and 'Sanchez'

# cast all salaries as floats for the govt salaries dataframe

# figure out a question to show understanding of these methods: 
# .items  .at  .T .array .index .pivot .notna .drop_duplicates .merge .sort_values .unique


