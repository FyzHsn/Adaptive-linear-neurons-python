# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:42:47 2016

Here, we take the perceptron algorithm and introduce the standard square
error (SSE) cost function to modify it to be the ADALINE algorithm. The
adaline algorithm is short for Adaptive Linear Neuron algorithm. The
weights are updated according to the gradient of the cost function since
it is convex and differentiable.

"""
import numpy as np
class AdalineGD(object):
    """Adaline classifier.
    
    Parameters
    ----------
    eta : float
        Learning rate between 0.0 and 1.0.
    n_iter : int
        Number of passes over the training dataset.
    
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting. Underscore after a variable name indicates 
        that the variable was not created on instantiation of the object.
    cost_ : list
        Cost function of sample batch and weight vector per epoch.
    errors_ : list
        List of errors after weight update per epoch
        
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data according to the adaline algorithm.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training dataset, where n_samples is the number of samples
            and n_features is the number of features.
        y : {array-like}, shape = [n_samples]
            Binary classification of dataset.
                    
        Returns
        -------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        self.errors_ = []
        
        for i in range(self.n_iter):
            err = 0
            # compute errors per epoch
            for j in range(X.shape[0]):
                status = y[j] - self.predict(X[j, ]) 
                err += int(status != 0.0)
                
            # update weights    
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
            
            
            self.errors_.append(err)
            
        return self
        
    def net_input(self, X):
        """Calculate the dot product of the features and the weights. """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation."""
        return self.net_input(X)
        
    def predict(self, X):
        """Return class label by using the Heaviside activation
        function. """        
        return np.where(self.activation(X) >= 0.0, 1, -1)
        
        
        
        
        
        
        
        
        
    

