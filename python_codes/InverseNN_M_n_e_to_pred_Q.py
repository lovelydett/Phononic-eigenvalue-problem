#!/usr/bin/env python
# coding: utf-8

# In[100]:


'''
This is just a little try on predicting wave numbers with material filed and eigenvalues
and is not yet trained and tested due to limited time.

Author:xyt
2019.8.8
'''
import tensorflow as tf 
import numpy as np
import keras
import keras.backend as K 
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, GlobalMaxPooling1D, MaxPooling2D,Input,Conv1D,MaxPooling1D,MaxPool2D
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
import scipy as sp
import keras.backend as K

import keras.backend.tensorflow_backend as KTF
 
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))


# In[98]:


def multi_input_model():
    input1_= Input(shape=(128,128,3), name='material',dtype = "float32")
    input2_ = Input(shape=(50,1), name='eigenvalues',dtype = "float32")
 
    x1 = Conv2D(60, kernel_size=(2,2), strides=1, activation='relu', padding='same')(input1_)
    x1 = MaxPool2D(pool_size=(2,2), strides=4)(x1)
    x1 = Conv2D(60, kernel_size=(2,2), strides=1, activation='relu', padding='same')(x1)
    x1 = MaxPool2D(pool_size=(2,2), strides=4)(x1)
    x1 = Conv2D(60, kernel_size=(2,2), strides=1, activation='relu', padding='same')(x1)
    x1 = MaxPool2D(pool_size=(2,2), strides=2)(x1)    
    x1 = Flatten()(x1)
    
    x2 = Conv1D(16, kernel_size=3, strides=1, activation='relu', padding='same')(input2_)
    x2 = MaxPooling1D(pool_size=2, strides=2)(x2)
    
    x2 = Flatten()(x2)
    #x2 = K.transpose(x2)
    x = keras.layers.concatenate([x1, x2])
    #x = Flatten()(x)
    
    x = Dense(10, activation='relu')(x)
    output_ = Dense(2, activation='relu', name='output')(x)
 
    model = Model(inputs=[input1_, input2_], outputs=[x])
    model.summary()
 
    return model


# In[99]:


model = multi_input_model()


# In[ ]:




