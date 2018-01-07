'''
Created on Nov 20, 2017

@author: Amir
'''
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from p1 import LinearRegressor, getInput, train


x,y = getInput()

t = np.array([0 for _ in range(195)])
ok = 0
numKFolds = 10
for i in range(150, 300):
    print("test: " + str(i+1))
    count = 0
    kf = KFold(n_splits=numKFolds, shuffle = True, random_state = i)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        error, coeff = train(x_train, y_train, x_test, y_test, 0.04)
        if np.sum(error) > 5:
            t[test_index] = t[test_index] + 1
            ok -=1
            break
        count = count+1
    ok += 1
for i in range(195):
    if t[i] > 10:
        print(i,t[i])
        print("amount with low error: " + str(ok))
        
        
