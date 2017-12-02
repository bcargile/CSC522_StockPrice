import pandas as pd
import numpy as np
import statsmodels.api as sm

na_values = ['#REF!', '#N/A', 'N/A']
df = pd.read_csv("10Kpercentchange2.csv", na_values=na_values)
df = df.dropna()
X = df[['flesch reading ease', 'textblob sentiment', 'NLTK Negative', 'NLTK Neutral', 'NLTK Positive']]
Y = df['Percent Change']

df.head()

X = sm.add_constant(X)
est = sm.OLS(Y,X).fit()

print(est.summary())

print('Hello')
