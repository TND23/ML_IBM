from sklearn import linear_model
import pandas as pd

# IQ, Age, GPA
data_set = [[100, 15, 2.5], [95, 14, 2.0], [110, 18, 3.5], [120, 17, 4.0], [90, 14, 2.0], [80, 13, 1.8], [130, 18, 3.8]
            [105, 13, 3.3], [102, 13, 3.35], [98, 16, 3.1], [93, 11, 2.7], [111, 13, 3.7], [101, 12, 3.0]]

x_train = data_set[6:]
reg = linear_model.LinearRegression()

reg.fit(data_set, [99, 18, 100])
print(reg.coef_)