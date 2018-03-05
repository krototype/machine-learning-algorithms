# -*- coding: utf-8 -*-
"""
IMPLEMENTATION OF MULTIVARIATE LINEAR REGRESSION USING GRADIENT DESCENT

this code is created by Abhinav Srivastava, just for the purpose of learning
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#global variable alpha
alpha=0.01

#it computes the total error using any regression line
def computeCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

'''this function calculates the differentiation of cost , and hence helps
in reaching the best solution we could reach
important parameters:
    X - set of independent variables(here only 1)
    y - dependent variable
    alpha - Descending rate
    theta - value of coefficients(like a and b for ax+b line)
'''
def func(X,y,theta):
    diff=np.dot(X,theta)-y
    m=len(X)
    return (1.0/m)*np.dot(np.transpose(X),diff)

#our function here performs the iteration for maximum 1000 times
def gradient_descent(X,y,alpha):
    theta=np.matrix(np.full((1,X.shape[1]),0,dtype=int)).transpose()
    gradient=func(X,y,theta)
    iter=0
    #descending to lower errors each and every time
    while not np.all(np.absolute(gradient))<=1e-4:
        theta=theta-alpha*gradient
        gradient=func(X,y,theta)
        iter+=1
        if iter>1000:
            break;
        print(computeCost(X,y,theta.T))
    print("in iter "+str(iter)+" cost came as "+str(computeCost(X,y,theta.T)))
    return theta


data=pd.read_csv("ex1data2.txt",names=["size","bedrooms","price"])

#describing data
data.head()

"""As the data was not properly distributed , so if we would use the data without any normalization
then the data with higher value will mainly determine the result, so normalizing"""
data=(data-data.mean())/(data.std())

#a row of ones inserted
data.insert(0,'Ones',1)

#converting to dependent and independent variables
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X=np.matrix(X.values)
y=np.matrix(y.values)

#finding the initial theta according to the dataset
theta=np.full((1,X.shape[1]),0,dtype=int)

y=y.transpose()
print(computeCost(X,y,theta))

theta=gradient_descent(X,y,alpha)
ans=computeCost(X,y,theta.T)
print("The minimum we get is "+str(ans))

