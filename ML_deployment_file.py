# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:25:19 2024

@author: Dell
"""
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
os.chdir(r"D:\MLLLL")

medical_df=pd.read_csv("insurance (4).csv")
medical_df

medical_df.info

medical_df.describe()

medical_df.shape

medical_df.isnull().sum()

medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)

x = medical_df.drop('charges',axis=1)
y = medical_df['charges']





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)


lg = LinearRegression()
lg.fit(x_train,y_train) # 80 model will be train
y_pred = lg.predict(x_test) 

r2_score(y_test,y_pred)

input_df = (19, 1, 30,0,0,1) # Example input values, replace with your actual data
np_df = np.asarray(input_df)
input_df_reshaped = np_df.reshape(1,-1)
prediction = lg.predict(input_df_reshaped)
print(prediction)
