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
weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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

minMse = -1
maxWeights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
zeroWeights = [True, False, False, False, False, True, True, True, True, True, True, False, True, False, True, True]

for weightLock in range(0, 10):
    scaling = (10-weightLock)/10

    for run in range(0, 100):

        for i in range(0, 16):
            if zeroWeights[i]:
                weights[i] = 0
            else:
                weights[i] = maxWeights[i] + random.random()*scaling-(scaling/2)
                if weights[i] < 0:
                    weights[i] = 0
                if weights[i] > 1:
                    weights[i] = 1

        input = preprocessing.scale(input_unscaled)

        print(input)

        for i in range(0, 16):
            input[:,i] *= weights[i]

        knn = KNeighborsRegressor(n_neighbors=100)
        kf = KFold(n_splits=10)
        errors = 0
        fold = 0
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
            #print("Proportion of errors: " + str(sum_error/len(predicted_out)))
            #print("Accuracy: " + str(metrics.accuracy_score(actual_out, predicted_out)))
            #print("Precision: " + str(metrics.precision_score(actual_out, predicted_out)))
            #print("Recall: " + str(metrics.recall_score(actual_out, predicted_out)))
            #print("F1 Score: " + str(metrics.f1_score(actual_out, predicted_out)))

        mse = errors/len(input)
        print("MSE: " + str(mse))
        if mse < minMse or minMse == -1:
            print("New Min MSE: " + str(mse))
            print("Weights: " + str(weights))
            maxWeights = weights[:]
            minMse = mse

    print("Weightlock " + str(weightLock) + " mse: " + str(minMse))
    print("Weights: " + str(maxWeights))
print("Final accuracy: " + str(minMse))
print("Final Weights: " + str(maxWeights))