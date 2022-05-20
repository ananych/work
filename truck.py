#%%
import csv
import datetime
import json
import math
import os
import pickle
import random
import subprocess
import time
from enum import Enum
from random import sample


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pylab as plt


import scipy
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.utils import np_utils
# from lazypredict.S upervised import LazyClassifier
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import plotly.express as px
# import plotly.graph_objects as go
# import pylab as plt
# from plotly.subplots import make_subplots
from scipy.signal import (decimate, filtfilt, find_peaks, firwin, hilbert,
                          lfilter, peak_prominences, savgol_filter)
from scipy.stats import kurtosis, norm, skew
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras.layers import (GRU, LSTM, Activation, AveragePooling1D,
                                     BatchNormalization, Bidirectional, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D,
                                     GlobalAveragePooling2D, Input,
                                     MaxPooling1D, MaxPooling2D, Reshape,
                                     TimeDistributed, concatenate)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import filter
import performance as pf
import read_data as rd
from alg_freq_domain import Alg_freq_domain
from sklearn import  ensemble, preprocessing
from xgboost import XGBClassifier
#%%
with open('ss_denoise_overlaped_20220518.pickle', 'rb') as f:
    x=pickle.load(f)
    

with open('y_denoise_overlaped_20220518.pickle', 'rb') as f:
    y=pickle.load(f)





ind=y>10
y=y[ind]
x=x[:,ind]




clfy=[]
x_logs=[]
for i in tqdm(range(len(y))):
    A=x[25:55,i]
    x_log=[np.mean(A),np.std(A),kurtosis(A),skew(A),np.sqrt(np.mean(A**2)),np.sqrt(np.mean(A**2))/np.mean(np.abs(A))]

    x_logs.append(x_log)
    if np.abs(y[i]*32/60-np.argmax(x[:200,i]))<5:
        clfy.append(1)
    else:
        clfy.append(0)


clfy=np.array(clfy)
x_logs=np.array(x_logs)
data=np.concatenate((x_logs,clfy.reshape(-1,1)),axis=1)
data=np.array(pd.DataFrame(data).dropna(how='any'))

x_logs=np.array(data[:,:6])
clfy=np.array(data[:,6])

#%%

addx=x_logs[clfy==1,:]
addy=clfy[clfy==1]
x_logs = np.concatenate((x_logs,addx,addx,addx,addx,addx),axis=0)
clfy=np.concatenate((clfy,addy,addy,addy,addy,addy))
ran=random.sample(list(np.arange(len(clfy))), len(clfy))










clf=svm.SVC(kernel='rbf',gamma='auto')
clf.fit(x_logs[ran[:80000],:], clfy[ran[:80000]])
print(clf.score('訓練集: ',x_logs[ran[:80000],:], clfy[ran[:80000]]))
#print(clf.score('測試集: ',x_logs[ran[800000:],:], clfy[ran[800000:]]))


regr_1 = DecisionTreeClassifier(max_depth=50) #最大深度為2的決策樹
regr_1.fit(x_logs[ran[:80000],:], clfy[ran[:80000]])


print(regr_1.score('dt訓練集: ',x_logs[ran[:80000],:], clfy[ran[:80000]]))
print(regr_1.score('測試集: ',x_logs[ran[800000:],:], clfy[ran[800000:]]))
print(regr_1.feature_importances_)  



forest = ensemble.RandomForestClassifier(n_estimators = 50)
forest_fit = forest.fit(x_logs[ran[:80000],:], clfy[ran[:80000]])
print('rf訓練集: ',forest.score(x_logs[ran[:80000],:], clfy[ran[:80000]]))
print('測試集: ',forest.score(x_logs[ran[800000:],:], clfy[ran[800000:]]))
print(forest.feature_importances_)  





# 建立 XGBClassifier 模型
xgboostModel = XGBClassifier(n_estimators=50, learning_rate= 0.5)
# 使用訓練資料訓練模型
xgboostModel.fit(x_logs[ran[:80000],:], clfy[ran[:80000]])
# 使用訓練資料預測分類
print('xg訓練集: ',xgboostModel.score(x_logs[ran[:80000],:], clfy[ran[:80000]]))
print('測試集: ',xgboostModel.score(x_logs[ran[800000:],:], clfy[ran[800000:]]))
