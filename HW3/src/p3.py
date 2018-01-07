'''
Created on Nov 23, 2017

@author: Amir
'''

from p1 import LinearRegressor, getInput, train
import numpy as np

def run():
    x,y, _ = getInput()
    lr = LinearRegressor(x, y)
    w = lr.RLS(0.08, "lasso")
    avgInputs = (np.mean(x,axis = 0))
    guess = np.dot(avgInputs, w)
    print("mean input prediction: " + str(guess[0,0]))    
    
if __name__ == "__main__":
    run()
