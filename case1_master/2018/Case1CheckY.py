# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:49:52 2018

@author: dnor
"""

def Case1CheckY(yhat, Case1AnswerPath = r'Case1_Answer.csv'):
    """
    Input:
        yhat, the 1000 predictions you have created
        Case1AnswerPath, expected to be added in a folder which is in path
        
    output:
        Calculated RMSE
    
    Checks yhat according to y, and gives the RMSE
    yhat is your predictions, and y is the data from Case1_Answer.
    Makes sure that Case1_Answer is in the root folder, or is atleast added to path.
    Otherwise feed the entire path as Case1AnswerPath.
    
    Usage could be as;
    import numpy as np
    yhat = np.loadtxt(r'C:\Users\dnor\Desktop\02582\Case1\yhat.csv')
    # Or just have your yhat saved in kernel and feed it
    RMSE = Case1CheckY(yhat)
    """
    import numpy as np
    import pandas as pd
    # We just use pandas as to not care about the irritating class index at the end
    T = pd.read_csv(Case1AnswerPath)
    y = np.asarray(T['Y'].loc[100:])
    
    if np.size(y) != np.size(yhat):
        print("Sizes do not fit. make sure yhat is (1000,) dim")
        rRMSE = None
    else:
        rRMSE = np.sqrt(np.mean((y-yhat) ** 2)) / np.sqrt(np.mean((y-np.mean(y)) ** 2))
    
    return rRMSE
