Adaptive-linear-neurons-python
==============================  

This repo contains notes and algorithms related to the Adaptive Linear Neurons (ADALINE) algorithm.

adaline.py - algorithm  
adaline_script.py - the script in which I test various properties of Adaline
Adaline_notes.Rmd - playing with Adaline in R
[Adaline_notes.html](http://htmlpreview.github.io/?https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/Adaline_notes.html) - html file corresponding to the Rmd file 

Questions that I am interested in:  
1. Learning rate and convergence of the cost function.  
2. Differences when applied to separable and non-separable data.  
3. Differences when applied to standardized vs its non-standard dataset counterpart.  
4. How are the cost and error function updated?  

I have explored these topics in the html file.

We test the stochastic and regular gradient descent methods on the iris data set once again. Moreover, we test it on both separable and non-separable data. Species setosa and versicolor are separable based on petal and sepal lengths.
![](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/SetosaVersicolorFig.png?raw=true)  

Meanwhile, species Versicolor and Virginica are non-separable.  
![](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/VersicolorVirginicaFig.png?raw=true)  

Comparison of error and cost function for the separable data:  
![](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/SGDvsGDErrors.png?raw=true)  
![](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/SGDvsGDCost.png?raw=true)

Comparison of error and cost function for the non-separable data:    
![](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/SGDvsGDErrorsNS.png?raw=true)    
![](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/SGDvsGDCostNS.png?raw=true)  

Lessons:    
1. Stochastic gradient descent is good for online learning.   
2. The cost and error function using SGD converges faster than GD. This is due to the more frequent weight updates in the former.  
3. SGD is noisier due to more frequent weight updates.   
4. SGD is better at avoiding shallow minima in the cost function but has difficulty converging to the global minima. For this reason a changing learning rate is required for better performance.     
5. From before, we have that standardized data leads to faster convergence of weights.     
6. There are learning rates for which the SGD converges but the GD doesn't. Hence, GD has a smaller domain of convergence as a function of the learning rate.   
7. Standardizing the dataset still seems to improve cost function convergence and the GD cost function can converge instead of blowing up as for the unstandardized data.   
