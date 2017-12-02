# CSC522_StockPrice
This code is for CSC 522 at NCSU. It contains the code for the 10K and Stock Price Machine Learning Project


decisiontree.py  - This program imports a csv file of the stock data, builds a decision tree, and outputs the accuracy of the
decision tree along with the columns of the file that was used to calculate that accuracy. Volatility and
price trends will always be incorporated in the decision tree, however different combinations of scoring
systems for readability and sentiment are tested. The program tests these combinations for every tree
max depth between 3 and 30, and outputs the max depth that was used to calculate the highest
accuracy. Only the highest calculated accuracy will be shown in the output. The program also outputs
the accuracy of the tree if all attributes and the same max depth was used. Simply save the csv
containing the stock data to your directory and run the program to view these statistics.
