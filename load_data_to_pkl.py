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
from random import sample

import numpy as np
import pandas as pd
import plotly.express as px


import scipy
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, norm, skew
from sklearn import datasets, metrics, preprocessing, svm
from tqdm import tqdm

import read_data as rd
from alg_freq_domain import Alg_freq_domain

#####
#依照需要的資料更改66行的return
#
#
#
#####
folder_name='truck_data'

output_name_x='ss_denoise_overlaped_20220518.pickle'
# output_name_x_g='x_G_unnorm_20220516.pickle'
output_name_y='y_denoise_overlaped_20220518.pickle'



##############
#讀資料存下 
#
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


    return algo, algo.ss_denoise_overlaped[:200,:] , np.array(y)




## load data
dalist=os.listdir(os.path.join(folder_name,'BCG'))    
ind=dalist[0]
r1=os.path.join(folder_name,"BCG")
gr1=os.path.join(folder_name,"GoldenHR",ind[:-4]+'.csv')
algo, x,y =read(r1,ind,gr1)

#len(dalist)
for i in tqdm(range(1,len(dalist))):
    dalist=os.listdir(os.path.join(folder_name,'BCG'))    
    ind=dalist[i]
    r1=os.path.join(folder_name,"BCG")
    gr1=os.path.join(folder_name,"GoldenHR",ind[:-4]+'.csv')

    if ind[:-4]+'.csv' in os.listdir(os.path.join(folder_name,'GoldenHR')):
        _, xx,yy =read(r1,ind,gr1)
        # nss=np.concatenate((nss,xnss),axis=1)
        y = np.concatenate((y,yy))
        x = np.concatenate((x,xx),axis=1)




ind=y[:,0]>10
y=y[ind,0]
x=x[:,ind]









#%%
with open(output_name_x, 'wb') as f:
    pickle.dump(x,f)

# with open(output_name_x_g, 'wb') as f:
#     pickle.dump(nss,f)


with open(output_name_y, 'wb') as f:
    pickle.dump(y,f)

