# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:43:15 2016

In this script, we test the adaline algorithm.
"""
import pandas as pd

# Read in iris data from machine learning database. 
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)
    
# View data frame head and tail.
print(df.head())

import matplotlib.pyplot as plt
import numpy as np

# Extract the first 100 rows of the fourth column of the data frame. 
y = df.iloc[0:100, 4].values
print(y[0:10])

y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values

"""
Plot data before executing adaline algorithm. We are looking for
linear separability.

"""
plt.scatter(X[0:50, 0], X[0:50, 1], 
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], 
            color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

"""
We can see that the data is linearly separable. Now, let us run the 
perceptron algorithm and look at the number of errors make during each 
epoch.

"""
from adaline import Adaline
adln = Adaline(eta=0.001, n_iter=10)
adln.fit(X, y)
plt.plot(range(1, len(adln.errors_) + 1), adln.errors_,
         marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
