#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
To construct a CNN and train it with the dataset which is .h5 file

Author:xyt
2019.8.8
'''
import tensorflow as tf 
import numpy as np
import keras
import keras.backend as K 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, GlobalMaxPooling1D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model


# In[2]:


import numpy
import csv
import h5py
import gc 
import argparse
import os 
import psutil


# In[3]:


#Set how many GPUs are used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# In[4]:


n_hidden_units = 50
n_convpool_layers = 1
n_convlayers = 2
n_reglayers = 2
max_poolsize = (2,2)
train_set_ratio = 0.7
valid_set_ratio = 0.15
set_split_ratio34 = 0.5
drop_rate = 0.0
n_filters = 130 #130 before 
reg_factor = 0.0
kernel_size = (2,2)
input_strides = 1
kernel_init = 'uniform'
cost_function = 'mean_squared_error'
batch_size = 64#128
feed_batch_size = 256
input_sample_shape = [128,128,5]
#target_sample_shape = [50]
n_epochs = 35
n_output = [0,50]
n_set1 = 10000
n_set2 = None
n_set100 = 10000 
n_post_test = 5000
xnorm_axis = (0,1,2,3)
early_stop_delta = 0.01 # 0.01 change or above is considered improvement
early_stop_patience = 10 


# In[27]:


#x_mean and x_range are for normalization of input data
def get_x_mean_n_range():
    ds = h5py.File("../data/data2D_gzipped_famfiles_%d.h5","r",driver='family', memb_size=2500*10**6)
    x_min = np.zeros((1,5),dtype = ds["random"]["x0"].dtype)
    x_max = np.zeros((1,5),dtype = ds["random"]["x0"].dtype)
    x_mean = np.zeros((1,5),dtype = ds["random"]["x0"].dtype)
    count = 0
    for key1 in list(ds.keys()):
        for key2 in ds[key1].keys():
            if key2.find("x")>=0 :
                temp_x = ds[key1][key2][:]
                if temp_x.shape[0] != 64:
                    continue
                    
                #handle mean
                count+=1
                x_mean += np.mean(temp_x,axis=(0,1,2))
                
                #handle max
                temp_max = np.amax(temp_x,axis=(0,1,2))
                temp_batch = np.vstack([temp_max,x_max])
                x_max = np.amax(temp_batch,axis=(0))
                
                #handle min
                temp_min = np.amin(temp_x,axis =(0,1,2))
                temp_batch = np.vstack([temp_min,x_min])
                x_min = np.amin(temp_batch,axis=(0))
                
    x_mean = x_mean/count     
    x_range = x_max-x_min
    return x_mean,x_range

#y_mean and y_range are for normalization of output data
def get_y_mean_n_range():
    ds = ds = h5py.File("../data/data2D_gzipped_famfiles_%d.h5","r",driver='family', memb_size=2500*10**6)
    y_min = 0 
    y_max = 0
    y_mean = 0
    count = 0
    for key1 in list(ds.keys()):
        for key2 in ds[key1].keys():
            if key2.find("y")>=0 :
                temp_y = np.real(ds[key1][key2][:])
                if temp_y.shape[0] != 64:
                    continue
                    
                #handle mean
                count+=1
                y_mean += np.mean(temp_y)
                
                #handle max
                temp_max = np.max(temp_y)
                y_max = temp_max if temp_max>y_max else y_max
                
                #handle min
                temp_min = np.min(temp_y)
                y_min = temp_min if temp_min<y_min else y_min
                
    y_mean = y_mean/count     
    y_range = y_max-y_min
    return y_mean,y_range


#Since the size of dataset is too large, training has to be done in batches
def generate_data_from_file(x_mean,x_range,y_mean,y_range):
    while 1:
        ds = h5py.File("../data/data2D_gzipped_famfiles_%d.h5","r",driver='family', memb_size=2500*10**6)
        for key1 in list(ds.keys()):
            x_count = 0
            y_count = 0
            for key2 in list(ds[key1].keys()):   #0-399 or 0-199
                if key2[0]!="x" or ds[key1][key2].shape[0]>64:
                    continue
                
                x_raw = ds[key1][key2][:]
                y_key = "y"+key2[1:]
                y_raw = ds[key1][y_key][:]
                x_normed = (x_raw-x_mean)/x_range
                y_normed = (y_raw-y_mean)/y_range
                
                yield(x_normed,y_normed)

#15360 pair of in/out datas in ds["random"]["x0"] and ds["random"]["y0"] are used as validation set
def get_validation_data(x_mean,x_range,y_mean,y_range):
    ds = h5py.File("../data/data2D%d.h5","r",driver='family', memb_size=2500*10**6)
    x_val_raw = ds["random"]["x0"][:]
    y_val_raw = ds["random"]["y0"][:]
    
    x_val_normed = (x_val_raw-x_mean)/x_range
    
    y_val_normed = (y_val_raw-y_mean)/y_range
    
    return x_val_normed,y_val_normed

#load test data
def get_test_data(x_mean,x_range,y_mean,y_range):
    x_test_raw = np.load("../data/inputMatrix.npz")["inputMatrix"][:]
    y_test_raw = np.load("../data/Solution.npz")["frequency1"][:,:50]
    x_test = (x_test_raw-x_mean)/x_range
    y_test = (y_test_raw-y_mean)/y_range
    return x_test,y_test,x_test_raw,y_test_raw
    


# In[6]:


x_mean,x_range = get_x_mean_n_range()
print(x_mean)
print(x_range)


# In[7]:


y_mean,y_range = get_y_mean_n_range()
print(y_mean)
print(y_range)


# In[9]:


def create_model():
    model=Sequential()

    model.add(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                     input_shape=(128,128,5),kernel_initializer= kernel_init))
    model.add(Activation('relu'))



    model.add(MaxPooling2D(pool_size=max_poolsize))


    model.add(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                     kernel_initializer= kernel_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=max_poolsize))


    model.add(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                     kernel_initializer= kernel_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=max_poolsize))
    

    model.add(Flatten())
    model.add(Dense(n_hidden_units))
    model.add(Activation('relu'))
    model.add(Dense(n_output[1]-n_output[0], activation='linear'))

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=early_stop_delta, patience=early_stop_patience, verbose=2, mode='auto')

    model.summary()
    return model


# In[24]:


#if 1 gpu is used:
#model = create_model()

#if >=2 gpus are used:
parallel_model = multi_gpu_model(model, gpus=2)


# In[25]:


parallel_model.compile(loss=cost_function, optimizer='adam', metrics=["MAE"])


# In[12]:


#load validation datas (15360)
x_val,y_val = get_validation_data(x_mean,x_range,y_mean,y_range)
# print(x_val.shape,y_val.shape)
# print(x_val[0][0][0])
# print(y_val[0])


# In[28]:


#train the model in batches using generator
parallel_model.fit_generator(generate_data_from_file(x_mean,x_range,y_mean,y_range),
                    steps_per_epoch=999,epochs=12,validation_data=(x_val,y_val))


# In[29]:


#models are all saved in ../model
model.save("../model/model_wholeSet_normed_c3_f130_e12.h5")


# In[49]:


x_test,y_test,x_test_raw,y_test_raw = get_test_data(x_mean,x_range,y_mean,y_range)


# In[50]:


loss,mae=model.evaluate(x_test,y_test)

#loss here is MSE
print("loss:",loss)

print("mae:",mae)


# In[ ]:




