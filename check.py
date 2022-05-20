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

import numpy as np
import pandas as pd
import plotly.express as px
import pylab as plt
import scipy
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, norm, skew
from sklearn import datasets, metrics, preprocessing, svm
from tqdm import tqdm

import read_data as rd
from alg_freq_domain import Alg_freq_domain






##############
#load model
##############

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

    algo.get_heart_rate(pressure_data, acc_x,acc_y,acc_z)

    y=pd.read_csv(truth)

    if algo.ss.shape[1]>len(y):
        algo.ss=algo.ss[:,:len(y)]
    else:
        y=y[:algo.ss.shape[1]]


    return algo, algo.ss_unnorm[:200,:],algo.nss_unnorm[:200,:] , np.array(y)


with open('re_svm_20220516.pickle', 'rb') as f:
    clf=pickle.load(f)

##############
#讀資料存下 ss nss y
## load data
##############

dalist=os.listdir(os.path.join('20229999_mingtai','raw'))    
ind=dalist[0]
r1=os.path.join("20229999_mingtai","raw")
gr1=os.path.join("20229999_mingtai","ground_truth_bpm",ind[:-4]+'.csv')
algo, x,nss,y =read(r1,ind,gr1)


for i in tqdm(range(1,len(dalist))):
    dalist=os.listdir(os.path.join('20229999_mingtai','raw'))    
    ind=dalist[i]
    r1=os.path.join("20229999_mingtai","raw")
    gr1=os.path.join("20229999_mingtai","ground_truth_bpm",ind[:-4]+'.csv')

    if ind[:-4]+'.csv' in os.listdir(os.path.join('20229999_mingtai','ground_truth_bpm')):
        _, xx,xnss,yy =read(r1,ind,gr1)
        nss=np.concatenate((nss,xnss),axis=1)
        y = np.concatenate((y,yy))
        x = np.concatenate((x,xx),axis=1)


##############
#寫成input form
#
##############

ind=y[:,0]>10
y=y[ind,0]
nss=nss[:,ind]
x=x[:,ind]



clfy=np.zeros(len(y))
x_logs=[]
for i in tqdm(range(len(clfy))):
    A=x[:200,i]-nss[:200,i]
    x_log=[np.mean(A),np.std(A),np.max(A),np.min(A),kurtosis(A),skew(A)]
    x_logs.append(x_log)
    if np.abs(y[i]-np.argmax(x[:,i]))<5:
        clfy[i]=1
x_logs=np.array(x_logs)


##############
#pred and print
#
##############

true=clfy
prediction=clf.predict(x_logs)


print('True:', true)
print('Pred:', prediction)

print('Precision:', metrics.precision_score(true, prediction))
print('Recall:', metrics.recall_score(true, prediction))
print('F1:', metrics.f1_score(true, prediction))
print("confusion_matrix:\n",metrics.confusion_matrix(true, prediction))


