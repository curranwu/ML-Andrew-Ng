'''
Author: curran.wu
Date: 2022-05-01 01:52:47
LastEditors: curran.wu
LastEditTime: 2022-05-01 10:28:53
FilePath: \ML-Andrew-Ng\Exercise1\main1.py
Description: 

Copyright (c) 2022 by curran.wu, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pylab as plt

def loadTrainData():
    data = np.loadtxt('Exercise1/ex1data1.txt', delimiter=',')
    # data = np.loadtxt('ex1data1.txt', delimiter=',')
    return data

def plotData(x, y):
    plt.plot(x, y, 'rx', ms=10)
    plt.xlabel('Population of city in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()

def computeCost(x, y, theta):
    ly = np.size(y, 0)
    cost = (x.dot(theta) - y).dot(x.dot(theta) - y) / (2 * ly)
    return cost

def gradientDescent(x, y, theta, alpha, num_iters):
    m = np.size(y, 0)
    j_history = np.zeros(num_iters)

    for i in range(num_iters):
        deltaJ = x.T.dot(x.dot(theta) - y) / m
        theta = theta - alpha * deltaJ
        j_history[i] = computeCost(x, y, theta)
    return theta, j_history

if __name__ == '__main__':
    # load the data
    data = loadTrainData()
    X = data[:, 0]; Y = data[:, 1]
    m = np.size(Y, 0)
    plotData(X, Y)
    
    # init the parameter
    X = np.vstack((np.ones((m,)), X)).T
    theta = np.zeros((2,))             
    iterations = 1500
    alpha = 0.01
    J = computeCost(X, Y, theta)
    print(J)

    # traning
    theta, j_history = gradientDescent(X, Y, theta, alpha, iterations)
    print('Theta found by gradient descent: ', theta)

    plt.plot(X[:, 1], Y, 'rx', ms=10, label='Training data')
    plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')
    plt.xlabel('Population of City in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.legend(loc='upper right')
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]).dot(theta)
    print('For population = 35,000, we predict a profit of ', predict1*10000)
    predict2 = np.array([1, 7.0]).dot(theta)
    print('For population = 70,000, we predict a profit of ', predict2*10000)
    # _ = input('Press [Enter] to continue.')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing cost func')
    plt.plot(j_history, ms=10, label='iter num')
    plt.show()

