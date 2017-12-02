import matplotlib.pyplot as plt

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import math
na_values = ['#REF!', '#N/A', 'N/A']
df = pd.read_csv("10Kpercentchange2.csv", na_values=na_values)



df = df.dropna()
#df.fillna(value=0, inplace=True)

Y = df['Percent Change']
X = df['textblob sentiment']


X = X.reshape(len(X),1)
Y = Y.reshape(len(Y),1)

X_train = X[:-1000]
X_test = X[-1000:]

Y_train = Y[:-1000]
Y_test = Y[-1000:]

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)


print(mean_squared_error(Y_test, Y_pred))
print(r2_score(Y_test, Y_pred))
plt.scatter(X_train, Y_train, color='green')
plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred, color='blue', linewidth=3)

plt.show()


