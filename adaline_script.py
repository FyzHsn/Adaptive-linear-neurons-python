# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:43:15 2016

In this script, we test the adaline algorithm.
"""
import pandas as pd

# Read in iris data from machine learning database. 
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)
    
import matplotlib.pyplot as plt
import numpy as np

# Extract the first 100 rows of the fourth column of the data frame. 
y = df.iloc[0:100, 4].values

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
plt.title('Species vs petal and sepal length - Separable dataset')
plt.savefig('SetosaVersicolorFig.png')
plt.clf()
#plt.show()

"""
We can see that the data is linearly separable. Now, let us run the 
perceptron algorithm and look at the number of errors make during each 
epoch.

"""
from adalineGD import AdalineGD
from adalineSGD import AdalineSGD

# Standardize data
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

adln = AdalineGD(eta=0.01, n_iter=25)
adln.fit(X_std, y)
adln1 = AdalineSGD(eta=0.01, n_iter=25)
adln1.fit(X_std, y)

plt.plot(range(1, len(adln1.errors_) + 1), adln1.errors_,
         marker='x', color='red', label='AdalineSGD')
plt.plot(range(1, len(adln.errors_) + 1), adln.errors_,
         marker='o', color='blue', label='AdalineGD')
plt.xlabel('Epoch #')
plt.ylabel('Errors')
plt.legend(loc='upper right')
plt.title('Errors in the [S]tochastic [G]radient\n ' \
          ' [D]escent (SGD) and GD in the Adaline algorithm')
plt.savefig('SGDvsGDErrors.png')
plt.clf()
#plt.show()

# plot showing convergence of the cost function
plt.plot(range(1, len(adln1.errors_) + 1), adln1.cost_,
         marker='x', color='red', label='AdalineSGD')
plt.plot(range(1, len(adln.errors_) + 1), adln.cost_,
         marker='o', color='blue', label='AdalineGD')
plt.xlabel('Epoch #')
plt.ylabel('Errors')
plt.legend(loc='upper right')
plt.title('Cost function in the [S]tochastic [G]radient\n ' \
          ' [D]escent (SGD) and GD in the Adaline algorithm')
plt.savefig('SGDvsGDCost.png')
plt.clf()
#plt.show()

"""Next up we look at non-separable data """
# Extract the last 100 rows of the fourth column of the data frame. 
y = df.iloc[51:150, 4].values

y = np.where(y == "Iris-versicolor", -1, 1)
X = df.iloc[51:150, [0, 2]].values

"""
Plot data before executing adaline algorithm. We are looking for
non-separability.

"""
plt.scatter(X[0:50, 0], X[0:50, 1], 
            color='red', marker='o', label='versicolor')
plt.scatter(X[50:100, 0], X[50:100, 1], 
            color='blue', marker='x', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.title('Species vs petal and sepal length - Non-separable dataset')
plt.savefig('VersicolorVirginicaFig.png')
plt.clf()
#plt.show()

"""
We can see that the data is linearly separable. Now, let us run the 
perceptron algorithm and look at the number of errors make during each 
epoch.

"""
from adalineGD import AdalineGD
from adalineSGD import AdalineSGD

# Standardize data
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

adln = AdalineGD(eta=0.01, n_iter=35)
adln.fit(X_std, y)
adln1 = AdalineSGD(eta=0.01, n_iter=35)
adln1.fit(X_std, y)

plt.plot(range(1, len(adln1.errors_) + 1), adln1.errors_,
         marker='x', color='red', label='AdalineSGD')
plt.plot(range(1, len(adln.errors_) + 1), adln.errors_,
         marker='o', color='blue', label='AdalineGD')
plt.xlabel('Epoch #')
plt.ylabel('Errors')
plt.legend(loc='upper right')
plt.title('Errors in the [S]tochastic [G]radient\n ' \
          ' [D]escent (SGD) and GD in the Adaline algorithm')
plt.savefig('SGDvsGDErrorsNS.png')
plt.clf()
#plt.show()

# plot showing convergence of the cost function
plt.plot(range(1, len(adln1.errors_) + 1), adln1.cost_,
         marker='x', color='red', label='AdalineSGD')
plt.plot(range(1, len(adln.errors_) + 1), adln.cost_,
         marker='o', color='blue', label='AdalineGD')
plt.xlabel('Epoch #')
plt.ylabel('Errors')
plt.legend(loc='upper right')
plt.title('Cost function of the [S]tochastic [G]radient\n ' \
          ' [D]escent (SGD) and GD in the Adaline algorithm')
plt.savefig('SGDvsGDCostNS.png')
plt.clf()
#plt.show()
