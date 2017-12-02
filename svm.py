from sklearn import svm
from numpy import genfromtxt
from numpy import mean
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
import math
from statistics import stdev
from sklearn import metrics
from sklearn import preprocessing
import pandas
import random
from sklearn import svm

data = pandas.read_csv("alldata.csv")

data = data.dropna()
data = data.loc[data["Percent Change"] > -50]
data = data.loc[data["Percent Change"] < 50]

input_unscaled = data.as_matrix(["% day","% week","% month","10day Vol","flesch reading ease","flesch kincaid","gunning fog","smog index","ARI","coleman","linsear","dale chall", "TextBlob", "NLTK neg", "NLTK neu", "NLTK pos"])

input = preprocessing.scale(input_unscaled)

output = data.as_matrix(["Increased?"])

print(output)

clf = svm.SVC(kernel='linear')

clf.fit(input, output)  
	
kf = KFold(n_splits=10)
errors = 0
fold = 0
for train_index, test_index in kf.split(input):
	fold += 1
	train_in, test_in = input[train_index], input[test_index]
	train_out, test_out = output[train_index], output[test_index]
	clf.fit(train_in, train_out)
	predicted_out = clf.predict(test_in)
	
	actual_out = test_out
	sum_error = 0
	for i in range(len(predicted_out)):
		sum_error += math.fabs(predicted_out[i]-actual_out[i])
	errors += sum_error						
			
print("\nTotal errors: " + str(errors) + " out of " + str(len(input)) + " == " + str(errors/len(input)))
accuracy = 1-errors/len(input)
print("\nTotal accuracy: " + str(accuracy))			