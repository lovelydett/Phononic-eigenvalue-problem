#!/usr/bin/env python
# coding: utf-8

# In[19]:


'''
To load, summary and evaluate a particular model with test set.

Author:xyt
2019.8.8
'''
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import numpy as np


# In[20]:


import h5py


# In[21]:


model = keras.models.load_model("../model/model_wholeSet_normed_c3_f130_e12.h5")


# In[22]:


model.summary()


# In[10]:


#keras.utils.plot_model(model,to_file='test.png')


# In[23]:


x_mean = [[1.49880835e+11, 4.40176392e+03 ,2.74842257e-01 ,2.49137578e-01,2.49167977e-01]]
x_range =[2.99999999e+11, 8.00000000e+03 ,4.49999999e-01 ,4.99999586e-01,4.99985816e-01]
y_mean = 1096210.1699824824
y_range = 6727841.0

#load test data and normalize them
def get_test_data(x_mean,x_range,y_mean,y_range):
    x_test_raw = np.load("../data/inputMatrix.npz")["inputMatrix"][:]
    y_test_raw = np.load("../data/Solution.npz")["frequency1"][:,:50]
    x_test = (x_test_raw-x_mean)/x_range
    y_test = (y_test_raw-y_mean)/y_range
    return x_test,y_test,x_test_raw,y_test_raw

x_test,y_test,x_test_raw,y_test_raw = get_test_data(x_mean,x_range,y_mean,y_range)


# In[24]:


print(x_test.shape)
print(y_test.shape)


# In[25]:


loss,mae=model.evaluate(x_test,y_test)

#loss here is MSE
print("loss:",loss)
print("MAE:",mae)


# In[12]:


y_pred_normed = model.predict(x_test)


# In[13]:


y_pred_raw = y_pred_normed*y_range+y_mean


# In[14]:


y_pred_raw.shape


# In[48]:


y_error = y_pred_raw-y_test_raw
np.set_printoptions(suppress=True)
# print(y_pred_raw[0])
# print(np.real(y_test_raw[
np.unique(x_test_raw[0,:,:,3])


# In[16]:


np.sum(np.absolute(y_pred_raw[0] - y_test_raw[0]))/len(y_test[0])


# In[20]:


import matplotlib.pyplot as plt


# In[26]:


plt.plot(y_pred_raw[21])


# In[ ]:




