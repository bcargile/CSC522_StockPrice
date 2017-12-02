import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

textStat=(9,10,11,12,13,14,15,16)
sentiment=(17,18,19,20)
maxleaves = (3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)
highest = 0

for temp1 in textStat:
	for temp2 in sentiment:
		for temp3 in maxleaves:
			stock_data= pd.read_csv('10KStatsUpdated.csv', header=0, 
				#names=['increased','CIK', 'ticker', 'date', 'day','week','month', 'volatility', 'change','fleschRead', 'fleshKincaid', 'gunningFog', 'smog', 'ARI', 'coleman', 'linsear', 'dale', 'textBlob', 'NLTKneg', 'NLTKneu','NLTKpos'],
				usecols=[0,4,5,6,7,temp1,temp2], 
				dtype='a', float_precision=None)
			stock_data = stock_data.dropna(axis=0, how='any')

			#Declare features and target values
			X = stock_data.values[:, 1:]
			Y = stock_data.values[:,0]

			#Make a 70/30 split of training/testing data
			X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)

			#Make 2D array a 1D list for .fit
			y_train = list(y_train)

			decision = DecisionTreeClassifier(criterion='entropy', max_depth=temp3)
			decision.fit(X_train, y_train)

			y_pred = decision.predict(X_test)

			
			if (accuracy_score(y_test, y_pred)*100) > highest:
				highest = accuracy_score(y_test, y_pred)*100
				textstatCol = temp1
				sentimentCol = temp2
				maxdepth = temp3  

print ("\nNumber of records used:",len(stock_data))
print ("\nHighest accuracy using different combinations of sentiment & readability attributes:")
print ("\tAccuracy: ", highest)
print ("\tReadability column used: ", textstatCol)
print ("\tSentiment column used: ", sentimentCol)
print ("\tMax_depth used: ", maxdepth)

stock_data= pd.read_csv('10KStatsUpdated.csv', header=0, 
	#names=['increased','CIK', 'ticker', 'date', 'day','week','month', 'volatility', 'change','fleschRead', 'fleshKincaid', 'gunningFog', 'smog', 'ARI', 'coleman', 'linsear', 'dale', 'textBlob', 'NLTKneg', 'NLTKneu','NLTKpos'],
	usecols=[0,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20], 
	dtype='a', float_precision=None)
stock_data = stock_data.dropna(axis=0, how='any')

#Declare features and target values
X = stock_data.values[:, 1:]
Y = stock_data.values[:,0]

#Make a 70/30 split of training/testing data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)

#Make 2D array a 1D list for .fit
y_train = list(y_train)

decision = DecisionTreeClassifier(criterion='entropy', max_depth=temp3)
decision.fit(X_train, y_train)

y_pred = decision.predict(X_test)

print ("Using all available attributes and max_depth used above, accuracy score is: ",accuracy_score(y_test, y_pred)*100)