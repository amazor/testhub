'''
Created on Nov 19, 2017

@author: Amir
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from cycler import cycler


def findNumNonZeros(x):
    sum = 0
    (iRange, jRange) = x.shape
    for i in range(iRange):
        for j in range(jRange):
             if  (x[i,j]) != 0:
                sum += 1
    return sum
class LinearRegressor(object):
    def __init__(self, inputMatrix, outputMatrix):
        self.inputMatrix = inputMatrix
        self.outputMatrix = outputMatrix
        
  
    def RLS(self, lambdaCoeff, regressionType):
        if regressionType is "ridge":
            regularizationMatrix = np.zeros((self.inputMatrix.shape[1],self.inputMatrix.shape[1]))
            np.fill_diagonal(regularizationMatrix, lambdaCoeff) 
    #         regularizationMatrix[0,0] = 0
            RLS = np.dot(self.inputMatrix.T, self.inputMatrix)
            RLS = np.add(RLS, regularizationMatrix)
            RLS = np.linalg.inv(RLS)
            RLS = np.dot(RLS, self.inputMatrix.T)
            RLS = np.dot(RLS, self.outputMatrix)
            return RLS
        elif regressionType is "lasso":
            clf = linear_model.Lasso(alpha=lambdaCoeff)
            clf.fit(self.inputMatrix, self.outputMatrix)
            return(np.asmatrix(clf.coef_).T)
            
        elif regressionType is "elastic":
            clf = linear_model.ElasticNet(alpha=lambdaCoeff)
            clf.fit(self.inputMatrix, self.outputMatrix)
            return(np.asmatrix(clf.coef_).T)
            

def addBias(X): #adds column of 1s to beginning of matrix 
        bias = np.ones((X.shape[0], 1))
        return np.c_[bias, X]


def train(x_train, y_train, x_test, y_test, lam): 
    r = LinearRegressor(x_train, y_train)
    w = (r.RLS(lam,"elastic"))
    guess = np.dot(x_test,w)
    error = np.subtract(guess, y_test)
    error = np.multiply(error, error)
    return error, w   

# def getInput(datatype = "growth_rate"):
#     print("Getting Data From File")
#     print()
#     data = np.loadtxt("ecs171.dataset.txt" , skiprows = 1, usecols = range(1,4501))
#     if datatype is "growth_rate":
#         y = np.asmatrix(data[:,4]).T
# #     elif datatype is "strain":
# #         y = np.asmatrix(data[:,0]).T
# #     elif datatype is "medium":
# #         y = np.asmatrix(data[:,1]).T
# #     elif datatype is "env_perturb":
# #         y = np.asmatrix(data[:,2]).T
# #     elif datatype is "gene_perturb":
# #         y = np.asmatrix(data[:,3]).T
# #         
#     x = addBias(np.asmatrix(data[:,5:]))
# 
#     return x, y
def getInput():
    print("Getting Data From File")

    data = np.loadtxt("ecs171.dataset.txt" , skiprows = 1, usecols = range(6,4501))
    
    stringdata = np.loadtxt("ecs171.dataset.txt" ,dtype = bytes,  skiprows = 1, usecols = range(5)).astype(str)
    x = addBias(np.asmatrix(data[:,1:]))
    y = np.asmatrix(data[:,0]).T
    
    return x, y, stringdata
def sampleData(maxLambda = 0.5, iter = 50, numKFolds = 10):
    x,y, _ = getInput()

    errorMatrix = [[] for _ in range(numKFolds)]
    nonZeroMatrix = [[] for _ in range(numKFolds)]
    count = 0
    
    kf = KFold(n_splits=numKFolds, shuffle = True, random_state = 69)
    for train_index, test_index in kf.split(x):
        print("training fold: " + str(count+1))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lam = []
        sumOfErrors = []
        zeroWeights = []
        
        for i in range(1,iter+1):
            l = (i*maxLambda/iter)
            lam.append(l)
            error, coeff = train(x_train, y_train, x_test, y_test, l)
            zeroWeights.append(findNumNonZeros(coeff))
            sumOfErrors.append(np.sum(error))
        errorMatrix[count] = sumOfErrors
        nonZeroMatrix[count] = zeroWeights
        count = count+1
        
        
    return errorMatrix, nonZeroMatrix, lam

def run(maxLambda, iter, numKFolds):

    errorMatrix, nonZeroMatrix, lambdas = sampleData(maxLambda, iter, numKFolds)
    fig = plt.figure()
    pl1 = fig.add_subplot(121)
    pl2 = fig.add_subplot(122)
    pl1.set_prop_cycle(cycler('marker', ['.', 's', 'x', 'd']) *
                       cycler('color', ['b', 'g', 'r', 'y', 'c', 'm', 'k']))
    pl2.set_prop_cycle(cycler('marker', ['.', 's', 'x', 'd']) *
                       cycler('color', ['b', 'g', 'r', 'y', 'c', 'm', 'k']))
    for i in range(numKFolds):
        pl1.plot(lambdas,errorMatrix[i], label = 'error' + str(i+1))
        pl2.plot(lambdas,nonZeroMatrix[i], label = 'nonZeroWeights' + str(i+1))
    pl1.legend(loc="best", borderaxespad=0)
    pl1.set_title("Lambda vs Testing Error")
    pl2.set_title("Lambda vs Number Non-Zero Coefficients")
    pl1.set_xlabel("lambda")
    pl2.set_xlabel("lambda")
    pl1.set_ylabel("error (RSS)")
    pl2.set_ylabel("# non-zero coef")
    fig.suptitle("Problem 1: Linear Regression with Elastic Net Regularization")
    
    pl2.legend(loc="best", borderaxespad=0)
    # pl1.set_xLabel("lambda")
    # pl1.set_ylabel("RSS")
    # pl2.set_xLabel("lambda")
    # pl2.set_ylabel("# non-zero coefficients")
    # pl1.set_title("Errors for each fold")
    # pl2.set_title("number of non-zero weights for each fold")
    #  
    # plt.suptitle("Problem 1")
     
    plt.show()
if __name__ == "__main__":
    run(0.5, 25, 10)   
    
    