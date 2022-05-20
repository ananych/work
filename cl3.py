
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
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.utils import np_utils
from scipy.interpolate import interp1d
from scipy.optimize import minimize
# import plotly.express as px
# import plotly.graph_objects as go
# import pylab as plt
# from plotly.subplots import make_subplots
from scipy.signal import (decimate, filtfilt, find_peaks, firwin, hilbert,
                          lfilter, peak_prominences, savgol_filter)
from sklearn import datasets, metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
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

    algo.get_heart_rate(pressure_data, acc_data)

    y=pd.read_csv(truth)

    if algo.ss.shape[1]>len(y):
        algo.ss=algo.ss[:,:len(y)]
    else:
        y=y[:algo.ss.shape[1]]


    return algo.ss[:300,:] , np.array(y)


def read_denois(file_path,log_filename,truth):
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

    algo.get_heart_rate(pressure_data, acc_data)

    y=pd.read_csv(truth)
    denoise=algo.ss-algo.nss
    if denoise.shape[1]>len(y):
        denoise=denoise[:,:len(y)]
    else:
        y=y[:denoise.shape[1]]


    return denoise[:300,:] , np.array(y)
    
    

r1=os.path.join("20210924_verification_candidate","20210922_biologue_road","raw")
gr1=os.path.join("20210924_verification_candidate","20210922_biologue_road","ground_truth_bpm","2021_09_22_14_09.csv")

r2=os.path.join("20211029_roadtest","raw")
gr2=os.path.join("20211029_roadtest","ground_truth_bpm","2021_10_29_15_20.csv")

r3=os.path.join("20211110_tantring_truck","raw")
gr3=os.path.join("20211110_tantring_truck","ground_truth_bpm","2021_11_10_14_51.csv")

r4=os.path.join("20211110_tantring_truck","raw")
gr4=os.path.join("20211110_tantring_truck","ground_truth_bpm","2021_11_10_16_07.csv")

r5=os.path.join("20211110_tantring_truck","raw")
gr5=os.path.join("20211110_tantring_truck","ground_truth_bpm","2021_11_11_11_12.csv")

r6=os.path.join("20211109_roadtest","raw")
gr6=os.path.join("20211109_roadtest","ground_truth_bpm","2021_11_09_14_59.csv")

r7=os.path.join("20211109_roadtest","raw")
gr7=os.path.join("20211109_roadtest","ground_truth_bpm","2021_11_09_15_26.csv")

r8=os.path.join("20211027_tantring","raw")
gr8=os.path.join("20211027_tantring","ground_truth_bpm","2021_10_27_15_27_main.csv")

r9=os.path.join("20211216_roadtest","raw")
gr9=os.path.join("20211216_roadtest","ground_truth_bpm","2021_12_16_10_43_Danny.csv")

r10=os.path.join("20211216_roadtest","raw")
gr10=os.path.join("20211216_roadtest","ground_truth_bpm","2021_12_16_11_08_sFrank.csv")

r11=os.path.join("20210924_verification_candidate","20210922_biologue_road","raw")
gr11=os.path.join("20210924_verification_candidate","20210922_biologue_road","ground_truth_bpm","2021_09_22_14_40.csv")

#r12=os.path.join("20211021_roadtest","raw")
#gr12=os.path.join("20211021_roadtest","ground_truth_bpm","2021_10_21_15_33.csv")

r13=os.path.join("20211011_roadtest","raw")
gr13=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_11_15_13.csv")

r14=os.path.join("20211011_roadtest","raw")
gr14=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_11_15_34.csv")

r15=os.path.join("20211011_roadtest","raw")
gr15=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_11_15_52.csv")

r16=os.path.join("20211011_roadtest","raw")
gr16=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_12_11_50.csv")

####
r17=os.path.join("20220211_office","10分鐘靜止","raw")
gr17=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_12_11_50.csv")

r18=os.path.join("20220211_office","10分鐘靜止","raw")
gr18=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_12_11_50.csv")

r19=os.path.join("20220211_office","10分鐘靜止","raw")
gr19=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_12_11_50.csv")

r20=os.path.join("20220211_office","10分鐘靜止","raw")
gr20=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_12_11_50.csv")

r21=os.path.join("20220211_office","10分鐘靜止","raw")
gr21=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_12_11_50.csv")

r22=os.path.join("20220211_office","10分鐘靜止","raw")
gr22=os.path.join("20211011_roadtest","ground_truth_bpm","2021_10_12_11_50.csv")
#####


tx , ty = read(r1,"2021_09_22_14_09.log",gr1)
x2 , y2 = read(r2,"2021_10_29_15_20.log",gr2)
x3 , y3 = read(r3,"2021_11_10_14_51.log",gr3)
x4 , y4 = read(r4,"2021_11_10_16_07.log",gr4)
x5 , y5 = read(r5,"2021_11_11_11_12.log",gr5)
x15 , y15 = read(r6,"2021_11_09_14_59.log",gr6)
x7 , y7 = read(r7,"2021_11_09_15_26.log",gr7)
x10 , y10 = read(r8,"2021_10_27_15_27_main.log",gr8)
x8 , y8 = read(r11,"2021_09_22_14_40.log",gr11)
x6 , y6 = read(r9,"2021_12_16_10_43_Danny.log",gr9)
x9 , y9 = read(r10,"2021_12_16_11_08_sFrank.log",gr10)

#x11 , y11 = read(r12,"2021_10_21_15_33.log",gr12)
x12 , y12 = read(r13,"2021_10_11_15_13.log",gr13)
x13 , y13 = read(r14,"2021_10_11_15_34.log",gr14)
x14 , y14 = read(r15,"2021_10_11_15_52.log",gr15)
x1 , y1 = read(r16,"2021_10_12_11_50.log",gr16)

y = np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y12,y13,y14,y15))
x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x12,x13,x14,x15),axis=1)


ind=sample(range(len(y)), len(y))
x=x[:,ind]
y=y[ind]
#%%
class spec_NN:
    def __init__(self, data,gt):
        self.data=data
        self.gt=gt
        self.xx=[]
        self.yy=[]
    def prepro(self):
        
        # self.data=data
        # self.gt=gt
        x=self.data.T
        y=np.array(self.gt)

        for i in range(len(x)-299):
            self.xx.append(x[i:i+300])
            self.yy.append(y[i+299])
        self.xx=np.array(self.xx)
        self.xx=self.xx.reshape(len(self.xx),300,300,1)
        self.yy=np.array(self.yy)
        self.yy=self.yy.reshape(len(self.yy),)
        


        self.lstmxx=x.reshape(len(x),1,300)
        self.lstmy= y.reshape(len(y),)
        
        self.yyy = np.zeros((len(x),300))
        self.lstmyy=np.zeros((len(x),300))
        for i in range(len(self.lstmy)):
            ind=self.lstmy[i]
            # self.lstmyy[i][ind+1]=0.125
            # self.lstmyy[i][ind]=0.125
            # self.lstmyy[i][ind-1]=0.5
            # self.lstmyy[i][ind-2]=0.125
            # self.lstmyy[i][ind-3]=0.125

            self.yyy[i][ind-1]=1
    
          
            self.lstmyy[i][ind+1]=0
            self.lstmyy[i][ind]=0
            self.lstmyy[i][ind-1]=1
            self.lstmyy[i][ind-2]=0
            self.lstmyy[i][ind-3]=0
        self.mxx=self.data.T.reshape(len(self.data.T),300,1)
        
        
        self.y_30=[]

        for i in range(len(y)):
            self.y_30.append(int(y[i]/5))
        self.y_30_onehot=np.zeros((len(self.y_30),30))
        for i in range(len(self.y_30)):
            self.y_30_onehot[i][self.y_30[i]-1]=1        
    def Inception(self,x,nb_filter_para):
        (branch1,branch2,branch3,branch4)= nb_filter_para
        branch1x1 = Conv1D(branch1[0],1, padding='same',strides=1,name=None)(x)

        branch3x3 = Conv1D(branch2[0],1, padding='same',strides=1,name=None)(x)
        branch3x3 = Conv1D(branch2[1],3, padding='same',strides=1,name=None)(branch3x3)

        branch5x5 = Conv1D(branch3[0],1, padding='same',strides=1,name=None)(x)
        branch5x5 = Conv1D(branch3[1],1, padding='same',strides=1,name=None)(branch5x5)

        branchpool = MaxPooling1D(pool_size=3,strides=1,padding='same')(x)
        branchpool = Conv1D(branch4[0],1,padding='same',strides=1,name=None)(branchpool)

        x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=2)

        return x
    
    def Conv1d_BN(self,x, nb_filter,kernel_size, padding='same',strides=1,name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv1D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
        x = BatchNormalization(name=bn_name)(x)
        return x
    def InceptionV1(self):
        
        inpt = Input(shape=(150,1))

        x = self.Conv1d_BN(inpt,64,7,strides=2,padding='same')
        x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)
        x = self.Conv1d_BN(x,192,3,strides=1,padding='same')
        x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)

        x = self.Inception(x,[(64,),(96,128),(16,32),(32,)]) #Inception 3a 28x28x256
        x = self.Inception(x,[(128,),(128,192),(32,96),(64,)]) #Inception 3b 28x28x480
        x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x) #14x14x480

        x = self.Inception(x,[(192,),(96,208),(16,48),(64,)]) #Inception 4a 14x14x512
        x = self.Inception(x,[(160,),(112,224),(24,64),(64,)]) #Inception 4a 14x14x512
        x = self.Inception(x,[(128,),(128,256),(24,64),(64,)]) #Inception 4a 14x14x512
        x = self.Inception(x,[(112,),(144,288),(32,64),(64,)]) #Inception 4a 14x14x528
        x = self.Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 4a 14x14x832
        x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x) #7x7x832

        x = self.Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 5a 7x7x832
        x = self.Inception(x,[(384,),(192,384),(48,128),(128,)]) #Inception 5b 7x7x1024

        #Using AveragePooling replace flatten
        x = AveragePooling1D(pool_size=7,strides=7,padding='same')(x)
        x = Flatten()(x)
        x = Dropout(0.95)(x)
        x = Dense(1000,activation='relu')(x)
        x = Dense(150,activation='softmax')(x)
        
        model=Model(inputs=inpt,outputs=x)
        
        return model    


    def mse(self,y_ture, y_pred):
        return -K.sum((y_pred * y_ture/K.sum(y_pred) ))
        
    def cnn2d(self,unit=32,dropout=0.5):
    
        loss = CategoricalCrossentropy()
        self.model = Sequential()
        self.model.add(Conv2D(unit, (3, 3), input_shape=(300,300,1), padding='same',activation='relu'))
        self.model.add(Conv2D(unit, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(unit, (3, 3), activation='relu', padding='same'))
        #self.model.add(Dropout(dropout))
        self.model.add(Conv2D(unit, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(unit*2, (3, 3), activation='relu', padding='same'))
        #self.model.add(Dropout(dropout))
        self.model.add(Conv2D(unit*2, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(unit*2, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(unit*2, (3, 3), activation='relu', padding='same'))
        #self.model.add(Dropout(dropout))
        self.model.add(Conv2D(unit*2, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(unit*3, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(unit*3, (3, 3), activation='relu', padding="same"))
        #self.model.add(Dropout(dropout))
        self.model.add(Conv2D(unit*3, (3, 3), activation='relu', padding="same"))
        self.model.add(Conv2D(unit*3, (3, 3), activation='relu', padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
        #self.model.add(GlobalAveragePooling2D())
        self.model.add(Flatten())
        self.model.add(Reshape((-1,unit*3)))
        self.model.add(LSTM(unit*6))
        self.model.add(Dense(unit*6, activation='relu'))
        self.model.add(Dense(30, activation='softmax'))
        self.model.compile(loss=loss, optimizer="adam",metrics=['categorical_accuracy'])
        print(self.model.summary())        
        
        #mean_absolute_error
        #categorical_crossentropy categorical_accuracy
        callback = EarlyStopping(monitor="val_loss", patience=80, verbose=1, mode="auto")
        history = self.model.fit(self.xx, self.y_30_onehot, epochs=400, batch_size=64, validation_split=0.2, 
            callbacks=[callback],
            shuffle=True)



        self.pp=[]
        a=self.model.predict(self.mxx)
        for i in tqdm(range(len(self.lstmyy))):
            self.pp.append(np.argmax(a[i])*5+3)

        self.loss=history1.history['loss']
        self.valloss=history1.history['val_loss']
        print("validation:performance (Acc rate(AP3):" , pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.8):], self.lstmy[int(len(self.lstmyy)*0.8):], bpm_error_tolerant=3)[0])
        print("validation:performance (Acc rate(AP5):" , pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.8):], self.lstmy[int(len(self.lstmyy)*0.8):], bpm_error_tolerant=5)[0])
        print("validation:performance (Acc rate(AP10):", pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.8):], self.lstmy[int(len(self.lstmyy)*0.8):], bpm_error_tolerant=10)[0])
        # px.line(y=[history1.history['loss'],history1.history['val_loss']])
        # px.line(y=[np.array(self.pp),self.lstmy])
    def cnn1d(self,unit=128,kernel=10,dropout=0.5,epoch=80,early=50):
        
        loss = CategoricalCrossentropy()
        self.model_m = Sequential()
        self.model_m.add(Conv1D(unit, kernel, activation='relu', padding='same', input_shape=(300, 1)))
        self.model_m.add(Conv1D(unit, kernel, activation='relu' ))
        #self.model_m.add(GRU(64,return_sequences= True))
        #self.model_m.add(GRU()64,return_sequences= True))
        self.model_m.add(MaxPooling1D(3))
        self.model_m.add(Dropout(dropout))
        self.model_m.add(Conv1D(unit, kernel, activation='relu' ))
        self.model_m.add(Conv1D(unit*2, kernel, activation='relu' ))
        self.model_m.add(MaxPooling1D(3))
        self.model_m.add(Dropout(dropout))
        # self.model_m.add(Dropout(dropout))
        # self.model_m.add(Conv1D(unit*2, kernel, activation='relu' ))
        #self.model_m.add(Conv1D(unit*2, kernel, activation='relu'))
        #self.model_m.add(MaxPooling1D(3))
        #self.model_m.add(Conv1D(unit*2, kernel, activation='relu'))
        #self.model_m.add(Conv1D(unit*2, kernel, activation='relu'))
        #self.model_m.add(GRU(64))
        self.model_m.add(GlobalAveragePooling1D())
        
        
        self.model_m.add(Dense(300, activation='softmax'))
        self.model_m.compile(loss=loss, optimizer='adam',metrics=['categorical_accuracy'])
        print(self.model_m.summary())


        callback = EarlyStopping(monitor="val_loss", patience=early, verbose=1, mode="auto")
        history1 = self.model_m.fit(self.mxx, self.lstmyy, epochs=epoch, batch_size=64, validation_split=0.1, 
            callbacks=[callback],
            shuffle=True)
        print(self.model_m.summary())
        self.pp=[]
        a=self.model_m.predict(self.mxx)
        for i in tqdm(range(len(self.lstmyy))):
            self.pp.append(np.argmax(a[i])*5+3)

        self.loss=history1.history['loss']
        self.valloss=history1.history['val_loss']
        print("validation:performance (Acc rate(AP3):" , pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.9):], self.lstmy[int(len(self.lstmyy)*0.9):], bpm_error_tolerant=3)[0])
        print("validation:performance (Acc rate(AP5):" , pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.9):], self.lstmy[int(len(self.lstmyy)*0.9):], bpm_error_tolerant=5)[0])
        print("validation:performance (Acc rate(AP10):", pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.9):], self.lstmy[int(len(self.lstmyy)*0.9):], bpm_error_tolerant=10)[0])
        # px.line(y=[history1.history['loss'],history1.history['val_loss']])
        # px.line(y=[np.array(self.pp),self.lstmy])

    def cnn_googlenet(self,epoch=100,early=30):
        self.V1 = self.InceptionV1()
        self.V1.summary()
        self.V1.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])
        callback = EarlyStopping(monitor="val_loss", patience=early, verbose=1, mode="auto")

        History = self.V1.fit(self.mxx, self.lstmyy, epochs=epoch, batch_size=32, validation_split=0.1, 
            callbacks=[callback],
            shuffle=True)

        self.pp=[]
        a=self.V1.predict(self.mxx)
        for i in tqdm(range(len(self.lstmyy))):
            self.pp.append(np.argmax(a[i]))

        self.loss=History.history['loss']
        self.valloss=History.history['val_loss']
        print("validation:performance (Acc rate(AP3):" , pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.9):], self.lstmy[int(len(self.lstmyy)*0.9):], bpm_error_tolerant=3)[0])
        print("validation:performance (Acc rate(AP5):" , pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.9):], self.lstmy[int(len(self.lstmyy)*0.9):], bpm_error_tolerant=5)[0])
        print("validation:performance (Acc rate(AP10):", pf.performance( np.array(self.pp)[int(len(self.lstmyy)*0.9):], self.lstmy[int(len(self.lstmyy)*0.9):], bpm_error_tolerant=10)[0])
        






#%%
y_cl3=np.zeros(len(y))
for i in range(len(y)):
    if y[i]>90:
        y_cl3[i]=1
    elif y[i]<70:
        y_cl3[i]=3
    else:
        y_cl3[i]=2

train_x=x.T[:16000,:]
test_x=x.T[16000:,:]
train_y=y_cl3[:16000]
test_y=y_cl3[16000:]
#%%
from sklearn.neighbors import KNeighborsClassifier
# 建立 KNN 模型
knnModel = KNeighborsClassifier(n_neighbors=100)
# 使用訓練資料訓練模型
knnModel.fit(train_x,train_y)

# 使用訓練資料預測分類
predicted = knnModel.predict(test_x)
ind_predicted= knnModel.predict(x.T)
accuracy_score(test_y,predicted)

#%%
x_H=x.T[ind_predicted==1,:]
y_H=y[ind_predicted==1]

x_M=x.T[ind_predicted==2,:]
y_M=y[ind_predicted==2]

x_L=x.T[ind_predicted==3,:]
y_L=y[ind_predicted==3]

#%%
tt_H=spec_NN(x_H.T[:,:650], y_H[:650])
tt_H.prepro()

tt_M=spec_NN(x_M.T[:,:16000], y_M[:16000])
tt_M.prepro()

tt_L=spec_NN(x_L.T[:,:650], y_L[:650])
tt_L.prepro()
#%%
tt_M.cnn1d(128,20,0.6,200,80)

#%%
p=tt_H.model_m.predict(x_M.T[:,16000:].reshape(-1,300,1))
pp=[]
for i in tqdm(range(len(p))):
    pp.append(np.argmax(p[i]))
pp=np.array(pp)
px.line(y=[pp,y_H[16000:].reshape(-1,)])