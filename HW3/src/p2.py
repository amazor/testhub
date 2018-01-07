'''
Problem 2 HW3
Bootstrapping to Find Confidence Interal

Created on Nov 19, 2017
@author: Amir
'''
import numpy as np
from numpy import average
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf
from p1 import LinearRegressor, addBias, train, getInput
from sklearn.model_selection import KFold
from cycler import cycler
from scipy.stats import norm


def bootstrap(confidence, samples, numIter = 10000):
    avgList = []
    print("Samples: " + str(samples))
    print()
    print("removing outliers > 1")
    sampleError = [ s for s in samples if s < 1]    
    print("Number of Samples Removed: " + str(len(samples) - len(sampleError)))
    print()
    print("New Samples: " + str(sampleError))
    for i in range(10000):
        newSample = np.random.choice(sampleError, size = len(sampleError), replace = True)
        avgList.append(average(newSample))
    
    mu = average(avgList)
    z = norm.ppf((1+confidence)/2, len(sampleError)-1)
    sigma = z * np.var(avgList)
    
    # stdError = np.var(avgList)/np.sqrt(len(sampleError))
    # sigma = t.ppf((1+confidence)/2, len(sampleError)-1)
    interval = (mu - sigma, mu + sigma)
    # sd = t.std(len(sampleError)-1)
    return interval, mu, sigma, avgList

def run(confidence, numKFolds, lam, numIter = 10000):
    x, y, _ = getInput()
    # lambda (use p1 to find optimal lambda)
    l = lam    
    sampleError = []
    
    count = 1
    #10-Fold Means (splits data into shuffled 9 training 1 testing with seed 69)
    kf = KFold(n_splits=numKFolds, shuffle = True, random_state = 69)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]     #input  set
        y_train, y_test = y[train_index], y[test_index]     #output set
        print("training fold: " + str(count))
        error, _ = train(x_train, y_train, x_test, y_test, l)
        sampleError.append(np.sum(error)) 
        count +=1   
    #Bootstrapping with t-distribution
    print()
    interval, mu, sigma, avgList= bootstrap(confidence, sampleError, numIter)
    
    #        PLOTTING
    n, bins, patches = plt.hist(avgList, 100,  normed=1 , ec='black')
    y = normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.title("Histogram of Averages Bootrapped from Averages of 10 Fold CR")
    plt.xlabel("Bootstrapped Average Error")
    plt.ylabel("Number of Errors inside Interval")
    print()
    print("Confidence Interval: " + str(interval))
    plt.show()
if __name__ == "__main__":
    run(0.95, 10, 0.08)
    