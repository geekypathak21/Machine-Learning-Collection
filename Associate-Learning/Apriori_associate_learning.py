# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 19:12:14 2018

@author: Himanshu
"""
# importing the required library
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transactions=[]
for i in range(0,7501):
    transactions.append(str([dataset.values[i,j] for j in range(0,20)]))
#Training Apriori on the dataset
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
#Visualising the results
result=list(rules)    
