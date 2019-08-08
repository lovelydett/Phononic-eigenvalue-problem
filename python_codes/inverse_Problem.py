#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
This is to approximate the inverse problem by looking for the largest band-width gap between e2 and e3
and its corresponding input

Author: xyt
2019.8.8
'''
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import numpy as np
import h5py
import math
import keras.backend as K

import keras.backend.tensorflow_backend as KTF

#this line sets the use of gpu or cpu
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))


# In[2]:


#These are the mean and range which are alredy calculated
x_mean = [[1.49880835e+11, 4.40176392e+03 ,2.74842257e-01 ,2.49137578e-01,2.49167977e-01]]
x_range =[2.99999999e+11, 8.00000000e+03 ,4.49999999e-01 ,4.99999586e-01,4.99985816e-01]
y_mean = 1096210.1699824824
y_range = 6727841.0

#load the test data, if neccessary normalize them
def get_test_data(isNormed):
    x_test = np.load("../data/inputMatrix.npz")["inputMatrix"][:]
    y_test = np.load("../data/Solution.npz")["frequency1"][:,:50]
    if(isNormed):
        x_test = (x_test-x_mean)/x_range
        y_test = (y_test-y_mean)/y_range
    return x_test,y_test


# In[6]:


x_test,y_test = get_test_data(isNormed = True)
x_test_raw,y_test_raw = get_test_data(isNormed = True)


# In[35]:


#To evaluate a particular model, use this cell to generate corresponding y_pred_ds and save to file data/y_pred_ds.h5

#if already got one, comment this cell

def generate_y_pred_ds(model):
    y_pred_ds = h5py.File("../data/y_pred_ds.h5","w")
    #ds = h5py.File("../data/data2D_gzipped_famfiles_%d.h5","r",driver='family', memb_size=2500*10**6)
    ds = h5py.File("../data/data2D_gzipped_famfiles_%d.h5","r",driver='family', memb_size=2500*10**6)
    for key1 in ds.keys():
        y_pred_ds.create_group(key1)
    for key1 in ds.keys():
        for key2 in ds[key1].keys():
            if key2[0]=="x":
                x_raw = ds[key1][key2][:]
                #norm
                x_normed = (x_raw-x_mean)/x_range
                #predict
                y_normed =model.predict(x_normed)
                y_pred = y_normed*y_range+y_mean
                y_pred_ds[key1].create_dataset("y"+key2[1:],data=y_pred)
                print(key1,key2,"successfully predicted")
                
#this is the particular model you wish to use to predict result
#all models should be in ../model
model = keras.models.load_model("../model/model_12000_all_normed_c3_f130.h5")
generate_y_pred_ds(model)
# after this line, y_pred_ds.h5 has been saved in ../data
# which is the prediction result of the model above


# In[3]:


#Use this cell to seperate(extract) y_true_value from original dataset and save them to file y_true_ds.h5
#if already got one, comment this cell
def generate_y_true_ds():
    y_true_ds = h5py.File("../data/y_true_ds.h5","w")
    #ds = h5py.File("../data/data2D_gzipped_famfiles_%d.h5","r",driver='family', memb_size=2500*10**6)
    ds = h5py.File("../data/data2D_gzipped_famfiles_%d.h5","r",driver='family', memb_size=2500*10**6)
    for key1 in ds.keys():
        y_true_ds.create_group(key1)
    for key1 in ds.keys():
        for key2 in ds[key1].keys():
            if key2[0]=="y":
                y_raw = ds[key1][key2][:]
                y_true_ds[key1].create_dataset(key2,data=y_raw)
                print(key1,key2,"successfully seperated")
generate_y_true_ds()


# In[9]:


#This cell gets each |e2-e3| in both y_pred and y_true, and error between them.
#and looks up for the largest band width

#load predict eigenvalues data and true eigenvalues data
y_pred_ds = h5py.File("../data/y_pred_ds2.h5","r")
y_true_ds = h5py.File("../data/y_true_ds.h5","r")


y_pred_e2_e3_array = np.zeros((79296,),dtype="float32")
y_true_e2_e3_array = np.zeros((79296,),dtype="float32")
largest_bw_pred = 0
key1_pred = "no"
key2_pred = "no"
index_pred = -1
largest_bw_true = 0
key1_true = "no"
key2_true = "no"
index_true = -1

#Travelsal all data and find the largest bw-gap
for key1 in y_true_ds.keys():
    for key2 in y_true_ds[key1].keys():
        y_pred_e2_e3 = abs(y_pred_ds[key1][key2][:,47]-y_pred_ds[key1][key2][:,48])
        y_true_e2_e3 = abs(y_true_ds[key1][key2][:,47]-y_true_ds[key1][key2][:,48])
        e2e3_error = abs(y_pred_e2_e3 - y_true_e2_e3)
        #e2e3_error_normed = (e2e3_error-y_mean)/y_range
        
        for i in range(y_pred_e2_e3.shape[0]):
            y_pred_e2_e3_array[i] = y_pred_e2_e3[i]
            y_true_e2_e3_array[i] = y_true_e2_e3[i]
            print(key1,",",key2,":",y_pred_e2_e3[i])
            if y_pred_e2_e3[i]>largest_bw_pred:
                largest_bw_pred = y_pred_e2_e3[i]
                key1_pred = str(key1)
                key2_pred = str(key2)
                index_pred = i
            if y_true_e2_e3[i]>largest_bw_true:
                largest_bw_true = y_true_e2_e3[i]
                key1_true = str(key1)
                key2_true = str(key2)
                index_true = i
print("largest bw in y_pred:",largest_bw_pred,"at:[",key1_pred,"][",key2_pred,"][",index_pred,']')     
print("largest bw in y_true:",largest_bw_true,"at:[",key1_true,"][",key2_true,"][",index_true,']')


# In[13]:


#This cell visualizes the band width across the y_pred, y_true and the error between them
from matplotlib import pyplot as plt
error_e2_e3_array = (y_true_e2_e3_array-y_pred_e2_e3_array)
x =  np.array(range(y_true_e2_e3_array.shape[0])) 
y =  y_pred_e2_e3_array
plt.title("Predicted band-width") 
plt.xlabel("Prediction") 
plt.ylabel("Error") 
plt.ylim((0,100000))
plt.plot(x,y) 
plt.show()
# plt.savefig('e2e3_true.png')


# In[ ]:




