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
![lab](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/SetosaVersicolorFig.png?raw=true)

Meanwhile, species Versicolor and Virginica are non-separable.
![](https://github.com/FyzHsn/Adaptive-linear-neurons-python/blob/master/VersicolorVirginicaFig.png?raw=true)
