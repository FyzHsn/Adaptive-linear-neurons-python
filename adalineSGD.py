# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:23:01 2016

* Adaptive Linear Neuron algorithm implemented via the stochastic gradient
descent method. 

* Unlike the gradient descent method which employs a batch learning style, 
  here we go back to learning from individual points. 
  
* This leads to faster convergence, though it is generally noisier. 

* It has the benefit of avoiding shallow cost function minimization as well. 

* Furthermore, a hybrid of the normal and stochastic gradient descent method
  exists known as mini-batch algorithms.
"""
import numpy as np
class AdalineSGD(object):
    """Adaline classifier.
    
    Parameters
    ----------
    eta : float
        Learning rate 0.0 and 1.0.
    n_iter : int
        Number of passes over the training dataset.
        
    Attributes
    ----------
    w_ : 1d-array
        Weights used to fit the dataset. _ after the variable indicates that 
        the variable was not created on instantiation of the object
    cost_ : list
        Cost function after weight changes over each epoch.
    errors_ : 
        Number of errors per epoch. It's a measure of progress in learning.
    
    """
    def __init__(self, eta=0.001, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    """ Method to fit data. """
    def fit(self, X, y):
        
        # Initialize extended weight vector, and arrays to keep track of the
        # cost and error function per epoch. This is a straighforward way to
        # measure the performance of the algorithm and parameters.
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        self.errors_ = []
        
        # Loop through each epoch
        for i in range(self.n_iter):
            
            # Initialize temp. error/cost var. to 0 at the beginning of the
            # epoch
            error = 0
            cost = 0
            
            # Shuffle the rows of the data set
            self._shuffle(X, y)
            
            # Loop through the datasets
            for j in range(X.shape[0]):
                
                # update number of errors
                difference = self.predict(X[j, ]) - y[j]
                error += int(difference != 0)
                
                # update weights and cost function
                cost += self._weight_update(X[j, ], y[j])
            
            # Append the number of errors and cost function at the end of each
            # epoch
            self.errors_.append(error)
            self.cost_.append(cost/X.shape[0])
        
        return self
        
    """Shuffle the rows of input vector and classification. Single leading
       underscore imples that the method will not be imported when this class
       is imported. 
       
    """
    def _shuffle(self, X, y):
        
        # randomize row labels
        r = np.random.permutation(len(y))
        
        # Note that multiple vectors can be returned at once.
        return X[r], y[r]
        
    # update weights according to the stochastic gradient descent method
    def _weight_update(self, X, y):
        difference = (y - self.activation(X))
        self.w_[0] += self.eta * difference
        self.w_[1:] += self.eta * difference * X        
   
        # Compute the cost function
        cost =  (difference**2) / 2.0        
        return cost       
          
    """Method to compute the net input. """
    def net_input(self, X):
        
        # order matters for the np.dot since for a matrix it becomes matrix
        # multiplication
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    """Compute the activation function. """
    def activation(self, X):

        # linear activation function        
        return self.net_input(X)
        
    """Method to compute quantizer and perform binary classification. """
    def predict(self, X):
        
        # We use the Heaviside function as the quantizer.
        return np.where(self.activation(X) >= 0.0, 1, -1)

        
        
        
        
        
        
        
        
    

