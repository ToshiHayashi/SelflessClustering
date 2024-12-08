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
from tensorflow.keras.datasets import fashion_mnist as fmnist
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
    
def own_model():
    
    model = Sequential()

    model.add(Conv2D(filters = 48, kernel_size = (5,5), activation ='relu',input_shape=(28,28,1)))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters = 48, kernel_size = (3,3), activation ='relu'))
    model.add(BatchNormalization(axis=3))
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
train=mnist.load_data()[0][0]
train_label=mnist.load_data()[0][1]
x_test=mnist.load_data()[1][0]
y_test=mnist.load_data()[1][1]
train_label=train_label.reshape(len(train_label))
y_test=y_test.reshape(len(y_test))


#x_test=readfile("CIFAR10_edit/x_test.pkl")
#x_test=((readfile("CIFAR10/x_test.pkl")/255)-0.5)*2
#y_test=readfile("CIFAR10/y_test.pkl")
#train_label=train_label.reshape(train_label.shape[0])
#y_test=y_test.reshape(y_test.shape[0])
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
    
    resize=np.zeros([len(edit_images),size//2,size//2])
    for n in range(num):
        resize[n]=Image.fromarray(edit_images[n]).resize((size//2,size//2))
    if hide==1:
        edit_images[:,:int(size/2),:int(size/2)]=resize
    elif hide==2:
        edit_images[:,:int(size/2),int(size/2):]=resize
    elif hide==3:
        edit_images[:,int(size/2):,:int(size/2)]=resize
    else:
        edit_images[:,int(size/2):,int(size/2):]=resize
    return edit_images
#images=["Lenna.png"]
M=train.shape[1]
N=train.shape[2]
#C=train.shape[3]
C=1

#OCSVM_results=[]
#LOF_results=[]
#IF_results=[]
all_results=[]
for e in range(5):
    #OCSVM_result=np.zeros(10)
    #LOF_result=np.zeros(10)
    #IF_result=np.zeros(10)
    #GMM_result=np.zeros(10)
    result=[]
    for c in range(10):
        clear_session()
        x_train=train[train_label==c]
        self_data=train[train_label!=c]
        self_label=train_label[train_label!=c]
        self_label[self_label>c]-=1
        #train1=edit(x_train,1)
        #train2=edit(x_train,2)
        #train3=edit(x_train,3)
        #train1=edit2(x_train,1)
        #train2=edit2(x_train,2)
        #train3=edit2(x_train,3)
        #train4=edit2(x_train,4)
        #train4=edit1(x_train,1)
        #train5=edit1(x_train,2)
        #train6=edit1(x_train,3)
        #train7=edit1(x_train,4)
        #train1=edit2(x_train,1)
        #train2=edit2(x_train,2)
        #train3=edit2(x_train,3)
        #train4=edit2(x_train,4)
        #train8=edit2(x_train,1)
        #train9=edit2(x_train,2)
        #train10=edit2(x_train,3)
        #train11=edit2(x_train,4)
        #x_test1=edit(x_test,1)
        #x_test2=edit(x_test,2)
        #x_test3=edit(x_test,3)
       
        #x_train=train[train_label==c]
        
        #self_data=train[train_label==c]
        #self_label=np.zeros(len(self_data))
        #self_data=train1[train_label==c]
        #self_data=train1
        #self_label=np.zeros(len(self_data))
        #edit_windows=train1[train_label==c]
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+1])
        #edit_windows=train2[train_label==c]
        #edit_windows=train2
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+1])
        
        #edit_windows=train3[train_label==c]
        #edit_windows=train3
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+2])
        
        #edit_windows=train4
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+3])
        #edit_windows=train5
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+4])
        #edit_windows=train6
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+5])
        #edit_windows=train7
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+6])
        #edit_windows=train8
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+7])
        #edit_windows=train9
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+8])
        #edit_windows=train10
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+9])
        #edit_windows=train11
        #self_data=np.concatenate([self_data,edit_windows],axis=0)
        #self_label=np.concatenate([self_label,np.zeros(len(edit_windows))+10])
        #C=3
        if K.image_data_format() == 'channels_last':
            self_data = self_data.reshape(self_data.shape[0],
                                      M, N,C)
            
            input_shape = (M, N,C)
        else:
            self_data = self_data.reshape(self_data.shape[0],
                                      C, M,N)
            input_shape = (C, M, N)
        #e=0
        #self_true=self_data[self_label==0]
        #e=0
        self_data, x_valid, self_label, y_valid = train_test_split(self_data, self_label, test_size=0.3, random_state=e,stratify=self_label)  
        model = own_model()
        fit_callbacks = [ callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=5,
                                                    mode='max')]
        #FE=Model(inputs=model.input,outputs=[model.layers[21].output])
        # Train model
        model.fit(self_data, self_label,
                  epochs=100,
                  batch_size=1000,
                  shuffle=True,
                  validation_data=(x_valid, y_valid),callbacks=fit_callbacks, verbose=0)
        
        
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
        #OCSVM=OneClassSVM()
        #OCSVM.fit(likeli_train)
        #LOF=LocalOutlierFactor(novelty=True)
        #LOF.fit(likeli_train)
        #IF=IsolationForest()
        #IF.fit(likeli_train)
        #GMM=GaussianMixture(n_components=1)
        #GMM.fit(likeli_train)
        #cluster_train=likeli_train.argmax(axis=1)
        #likeli_test=FE.predict(x_test)
        #likeli_test=model.predict(x_test)
        #score=OCSVM.score_samples(likeli_test)
        #score2=LOF.score_samples(likeli_test)
        #score3=IF.score_samples(likeli_test)
        #score4=GMM.score_samples(likeli_test)
        #y_eval=np.zeros(len(y_test))
        #y_eval[y_test==c]=1
        #cluster=likeli_test.argmax(axis=1)
        #for i in range(3):
        #vector_test=edit(x_test,int(i+1))
        #cluster_test=model.predict(vector_test)
        #prob_acc=cluster_test[:,int(i+1)]
        #prob_all=prob_all+prob_acc
        #print("OCSVM:",roc_auc_score(y_eval,score))
        #print("LOF:",roc_auc_score(y_eval,score2))
        #print("IF:",roc_auc_score(y_eval,score3))
        #print("GMM",roc_auc_score(y_eval,score4))
        #OCSVM_result[c]=np.round(roc_auc_score(y_eval,score)*100,1)
        #LOF_result[c]=np.round(roc_auc_score(y_eval,score2)*100,1)
        #IF_result[c]=np.round(roc_auc_score(y_eval,score3)*100,1)
        #GMM_result[c]=np.round(roc_auc_score(y_eval,score4)*100,1)
    #OCSVM_results.append(OCSVM_result)
    #LOF_results.append(LOF_result)
    #IF_results.append(IF_result)
    #GMM_results.append(GMM_result)
#print(soft_train)
#print(hard_test)
#print(soft_test)
#print(np.array(OCSVM_results).mean(axis=0))
#print(np.array(LOF_results).mean(axis=0))
#print(np.array(IF_results).mean(axis=0))
#print(np.array(GMM_results).mean(axis=0))
#makefile(OCSVM_results,"FMNIST-selflessOCSVM.pkl")
#makefile(LOF_results,"FMNIST-selflessLOF.pkl")
#makefile(IF_results,"FMNIST-selflessIF.pkl")
#makefile(GMM_results,"FMNIST-selflessGMM.pkl")
#print(class_hard_train)
#print(class_soft_train)
#print(class_hard_test)
#print(class_soft_test)
#makefile(hard_train,"hard_FMNIST_rot_train.pkl")
#makefile(soft_train,"soft_FMNIST_rot_train.pkl")
#makefile(hard_test,"hard_FMNIST_rot_test.pkl")
#makefile(soft_test,"soft_FMNIST_rot_test.pkl")

#makefile(class_hard_train,"class_hard_FMNIST_rot_train.pkl")
#makefile(class_soft_train,"class_soft_FMNIST_rot_train.pkl")
#makefile(class_hard_test,"class_hard_FMNIST_rot_test.pkl")
#makefile(class_soft_test,"class_soft_FMNIST_rot_test.pkl")