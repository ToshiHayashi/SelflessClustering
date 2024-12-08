#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import sys
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

from PIL import Image
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score,log_loss, mean_absolute_error,median_absolute_error
#from tqdm import tqdm
import pickle
def makefile(what,filename):
    with open(filename,"wb") as f3:
        pickle.dump(what,f3)

def readfile(filename):
    with open(filename,"rb") as f4:
        ans=pickle.load(f4)
    return ans
def fastcopy(target):
    return pickle.loads(pickle.dumps(target,-1))
from math import log2
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.backend import clear_session
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, UpSampling1D, UpSampling2D, Conv2DTranspose,Conv1DTranspose
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.python.keras import losses
from tensorflow.keras.datasets import cifar10
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
#from keras import backend as K

def edit(images, rot):
    size=images.shape[1]
    edit_images=np.zeros(images.shape)
    if rot == 1:
        for n in range(size):
            for m in range(size):
                edit_images[:,m,n]=images[:,size-n-1,m]
    elif rot == 2:
        for m in range(size):
            for n in range(size):
                edit_images[:,m,n]=images[:,size-m-1,size-n-1]
    elif rot==3:
        for n in range(size):
            for m in range(size):
                edit_images[:,m,n]=images[:,n,size-m-1]
    return edit_images
def edit1(images, hide):
    size=images.shape[1]
    #print(size/2)
    edit_images=np.copy(fastcopy(images))
    index=int(size/2)
    
    if hide==1:
        edit_images[:,:index,:index]=0
    elif hide==2:
        edit_images[:,:index,index:]=0
    elif hide==3:
        edit_images[:,index:,:index]=0
    else:
        edit_images[:,index:,index:]=0
    return fastcopy(edit_images)
def edit2(images, hide):
    num=images.shape[0]
    size=images.shape[1]
    #print(size/2)
    edit_images=np.copy(fastcopy(images))
    
    resize=np.zeros([len(edit_images),size//2,size//2,3])
    for n in range(num):
        resize[n]=Image.fromarray(edit_images[n]).resize((size//2,size//2))
    if hide==1:
        edit_images[:,:int(size/2),:int(size/2),:]=resize
    elif hide==2:
        edit_images[:,:int(size/2),int(size/2):,:]=resize
    elif hide==3:
        edit_images[:,int(size/2):,:int(size/2),:]=resize
    else:
        edit_images[:,int(size/2):,int(size/2):,:]=resize
    return edit_images
def own_model():
    
    model = Sequential()

    model.add(Conv2D(filters = 48, kernel_size = (5,5), activation ='relu',input_shape=(32,32,3)))
    #model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters = 48, kernel_size = (3,3), activation ='relu'))
    #model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.2))

    
    model.add(Conv2D(filters = 48, kernel_size = (5,5), activation ='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = 48, kernel_size = (5,5), activation ='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 48, kernel_size = (6,6), activation ='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 48, kernel_size = (4,4), activation ='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 48, kernel_size = (4,4), activation ='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation = "relu")) #Fully connected layer
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(32, activation = "relu")) #Fully connected layer
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(9, activation = "softmax")) #Classification layer or output layer

    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


#for label in range(10):
#train=cifar100.load_data(label_mode="coarse")[0][0]
#X_test=cifar100.load_data(label_mode="coarse")[1][0]
#train_label=cifar100.load_data(label_mode="coarse")[0][1]
#y_test=cifar100.load_data(label_mode="coarse")[1][1]
#train=cifar100.load_data(label_mode="coarse")[0][0]
train=cifar10.load_data()[0][0]
x_test=cifar10.load_data()[1][0]
train_label=cifar10.load_data()[0][1]
y_test=cifar10.load_data()[1][1]
train_label=train_label.reshape(train_label.shape[0])
y_test=y_test.reshape(y_test.shape[0])
#train_label=(train_label/5).astype(int)
#y_test=(y_test/5).astype(int)
M=32
N=32
C=3
#x_test=readfile("CIFAR10_edit/x_test.pkl")
#x_test=((readfile("CIFAR10/x_test.pkl")/255)-0.5)*2
#y_test=readfile("CIFAR10/y_test.pkl")
#train_label=train_label.reshape(train_label.shape[0])
#y_test=y_test.reshape(y_test.shape[0])

#images=["Lenna.png"]
M=train.shape[1]
N=train.shape[2]
#c=0
size=32
C=3


all_results=[]
for e in range(5):
    result=[]
    for c in range(10):    
        clear_session()
        x_train=train[train_label==c]
        self_data=train[train_label!=c]
        self_label=train_label[train_label!=c]
        self_label[self_label>c]-=1
        if K.image_data_format() == 'channels_last':
            self_data = self_data.reshape(self_data.shape[0],
                                      M, N,C)
            
            input_shape = (M, N,C)
        else:
            self_data = self_data.reshape(self_data.shape[0],
                                      C, M,N)
            input_shape = (C, M, N)
        e=0
        #self_true=self_data[self_label==0]
        self_data, x_valid, self_label, y_valid = train_test_split(self_data, self_label, test_size=0.3, random_state=e,stratify=self_label)  
        model = own_model()
        fit_callbacks = [ callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=5,
                                                    mode='max')]
        # Train model
        #model.summary()
        #exit()
        #FE=Model(inputs=model.input,outputs=[model.layers[21].output])
        model.fit(self_data, self_label,
                  epochs=100,
                  batch_size=1000,
                  shuffle=True,
                  validation_data=(x_valid, y_valid),callbacks=fit_callbacks, verbose=0)
        
        #exit()
        
        if K.image_data_format() == 'channels_last':
            x_test = x_test.reshape(x_test.shape[0],
                                      M, N,C)
            x_train = x_train.reshape(x_train.shape[0],M, N,C)
            input_shape = (M, N,C)
        else:
            x_test = x_test.reshape(x_test.shape[0],
                                      C, M,N)
            input_shape = (C, M, N)
            
        
        
        
        #likeli_train=FE.predict(x_train)
        likeli_train=model.predict(x_train)
        result.append(likeli_train.sum(axis=0))
    all_results.append(result)
print(np.round(np.array(all_results).mean(axis=0),1))
#makefile(OCSVM_results,"CIFAR10-selflessOCSVM.pkl")
#makefile(LOF_results,"CIFAR10-selflessLOF.pkl")
#makefile(IF_results,"CIFAR10-selflessIF.pkl")
#makefile(GMM_results,"CIFAR10-selflessGMM.pkl")
    