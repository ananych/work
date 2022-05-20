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
import pywt
import pywt.data
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

folder_name='truck_data'
def read(file_path,log_filename,truth):
    #get nss data and align data
    # file_path = "C:\\Users\\reduc\\Desktop\\20211029_roadtest APP狀態\\raw"
    # log_filename="2021_10_29_15_20.log"
    pressure_data, acc_x, acc_y, acc_z, start_time = rd.read_pressure_acc(file_path, log_filename)
    for i in range(len(acc_y)):
        acc_y[i] = 65535 - acc_y[i] if acc_y[i] > 50000 else acc_y[i]
        acc_z[i] = 65535 - acc_z[i] if acc_z[i] > 50000 else acc_z[i]
        acc_x[i] = 65535 - acc_x[i] if acc_x[i] > 50000 else acc_x[i]
    acc_data = (acc_x ** 2 + acc_y ** 2 + acc_z ** 2) ** 0.5

    algo = Alg_freq_domain(fs=64, fft_window_size=32)
    algo.acc_x = acc_x
    algo.acc_y = acc_y
    algo.acc_z = acc_z
    a,d=plot_signal_decomp(pressure_data, 'db2', "DWT: Ecg sample - Symmlets5")
    print(len(d[3][1:-1]),len(pressure_data))
    # algo.get_heart_rate(pressure_data, acc_x,acc_y,acc_z)
    algo.get_heart_rate(d[2][1:-1], acc_x,acc_y,acc_z)

    y=pd.read_csv(truth)

    if algo.ss.shape[1]>len(y):
        algo.ss=algo.ss[:,:len(y)]
    else:
        y=y[:algo.ss.shape[1]]


    return algo.bpm , np.array(y)

dalist=os.listdir(os.path.join(folder_name,'BCG'))    
ind=dalist[100]
r1=os.path.join(folder_name,"BCG")
gr1=os.path.join(folder_name,"GoldenHR",ind[:-4]+'.csv')
p,truth =read(r1,ind,gr1)

px.line(y=[p,truth[:,0]])






#%%
##dwt


mode = pywt.Modes.smooth


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)#選取小波函式
    a = data
    ca = []#近似分量
    cd = []#細節分量
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)#進行5階離散小波變換
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重構

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
    return rec_a,rec_d

#plot_signal_decomp(data1, 'coif5', "DWT: Signal irregularity")
#plot_signal_decomp(data2, 'sym5',
#                   "DWT: Frequency and phase change - Symmlets5")
a,d=plot_signal_decomp(p, 'db2', "DWT: Ecg sample - Symmlets5")
plt.show()





#%% wp

def wpd_plt(signal,n):


    #wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db1',mode='symmetric',maxlevel=n)

    #計算每一個節點的係數，存在map中，key為'aa'等，value為列表
    map = {}
    map[1] = signal
    for row in range(1,n+1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data



    #作圖


    plt.figure(figsize=(15, 10))
    plt.subplot(n+1,1,1) #繪製第一個圖
    plt.plot(map[1])



    for i in range(2,n+2):
        level_num = pow(2,i-1)  #從第二行圖開始，計算上一行圖的2的冪次方
        #獲取每一層分解的node：比如第三層['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i-1, 'freq')]  
        for j in range(1,level_num+1):
            plt.subplot(n+1,level_num,level_num*(i-1)+j)
            plt.plot(map[re[j-1]]) #列表從0開始  






