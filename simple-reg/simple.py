# -*- coding: utf-8 -*-
"""
IMPLEMENTATION OF SIMPLE LINEAR REGRESSION USING GRADIENT REGRESSION

this code is created by Abhinav Srivastava, just for the purpose of learning
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

alpha=0.01

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
    theta=np.matrix(np.array([0,0])).transpose()
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
    print("in iter "+str(iter)+"cost came as "+str(computeCost(X,y,theta.T)))
    return theta

#importing data
data=pd.read_csv("ex1data1.txt",names=["population","profit"])

#data ploting
data.plot(kind="scatter",x="population",y="profit")

#data description
data.describe()

#a row of ones inserted
data.insert(0,'Ones',1)

#converting to dependent and independent variables
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X=np.matrix(X.values)
y=np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

y=y.transpose()
print(computeCost(X,y,theta))

#calculating the cost using gradient descent
theta=gradient_descent(X,y,alpha)
cost=computeCost(X,y,theta.T)
print("the gradient descent gives an error of :"+str(cost))


#making the graph
x = np.linspace(data.population.min(), data.population.max(), 100)  
f = theta[0, 0] + (theta[1, 0] * x)

fig, ax = plt.subplots()  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.population, data.profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('population')  
ax.set_ylabel('profit')  
ax.set_title('Predicted Profit vs. Population Size')  



