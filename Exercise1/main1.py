'''
Author: curran.wu
Date: 2022-05-01 01:52:47
LastEditors: curran.wu
LastEditTime: 2022-05-01 09:38:14
FilePath: /ML-Andrew-Ng/Exercise1/main1.py
Description: 

Copyright (c) 2022 by curran.wu, All Rights Reserved. 
'''
import numpy as np
import matplotlib.pylab as plt

def loadTrainData():
    # data = np.loadtxt('Exercise1/ex1data1.txt', delimiter=',')
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    return data

def plotData(x, y):
    plt.plot(x, y, 'rx', ms=10)
    plt.xlabel('Population of city in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()



if __name__ == '__main__':
    # load the data
    train_data = loadTrainData()

    plotData(train_data[:,0], train_data[:,1])
    # plt the data set
    # plotData(train_data)
    # fit the linear regression

    # predict the result 
