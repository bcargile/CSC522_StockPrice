from numpy import genfromtxt
from numpy import mean
import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import r2_score
import math
from statistics import stdev
from sklearn import metrics
from sklearn import preprocessing
import pandas
import random

# "% change day","% change week","% change month","10day Vol",
# "flesch reading ease","flesch kincaid","gunning fog","smog index",
# "ARI","coleman","linsear","dale chall", "textblob sentiment",
# "NLTK Negative", "NLTK Neutral", "NLTK Positive"
weights = [0, 0.3413693598496798, 1, 0.3551114630547942, 0.034776772608068884, 0, 0, 0, 0, 0, 0, 0.008950269739634642, 0, 0.006135350556587371, 0, 0]
#weights = [0, 0.480, 0.799, 0, 0.515, 0, 0, 0, 0, 0, 0, 0.325, 0, 0.883, 0, 0]

#CIK,Ticker,Date,% change day,% change week,% change month,10day Vol,Percent Change,Increased?,flesch reading ease,flesch kincaid,gunning fog,smog index,ARI,coleman,linsear,dale chall,textblob sentiment,NLTK Negative,NLTK Neutral,NLTK Positive
data = pandas.read_csv("alldata.csv")

print(data)

#data = data.loc[data["flesch reading ease"] != "#REF!"]
data = data.dropna()
data = data.loc[data["Percent Change"] > -50]
data = data.loc[data["Percent Change"] < 50]

print(data)

input_unscaled = data.as_matrix(["% day","% week","% month","10day Vol","flesch reading ease","flesch kincaid","gunning fog","smog index","ARI","coleman","linsear","dale chall", "TextBlob", "NLTK neg", "NLTK neu", "NLTK pos"])

print(input_unscaled)

output = data.as_matrix(["Percent Change"])

print(output)


input = preprocessing.scale(input_unscaled)

print(input)

for i in range(0, 16):
    input[:,i] *= weights[i]

knn = KNeighborsRegressor(n_neighbors=125)
kf = KFold(n_splits=len(input))
errors = 0
fold = 0
all_predicted = []
all_actual = []
for train_index, test_index in kf.split(input):
    fold += 1
    train_in, test_in = input[train_index], input[test_index]
    train_out, test_out = output[train_index], output[test_index]
    knn.fit(train_in, train_out)
    predicted_out = knn.predict(test_in)
    actual_out = test_out
    sum_error = 0
    for i in range(len(predicted_out)):
        sum_error += (predicted_out[i]-actual_out[i])**2
    errors += sum_error
    print("MSE for fold " + str(fold) + ": " + str(sum_error/len(predicted_out)))
    all_predicted.append(predicted_out[0])
    all_actual.append(actual_out[0])
    #print("Proportion of errors: " + str(sum_error/len(predicted_out)))
    #print("Accuracy: " + str(metrics.accuracy_score(actual_out, predicted_out)))
    #print("Precision: " + str(metrics.precision_score(actual_out, predicted_out)))
    #print("Recall: " + str(metrics.recall_score(actual_out, predicted_out)))
    #print("F1 Score: " + str(metrics.f1_score(actual_out, predicted_out)))

mse = errors/len(input)
print("MSE: " + str(mse))
print("R2: " + str(r2_score(all_actual, all_predicted)))
