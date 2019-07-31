#### This code pertains to the 2D machine learning results presented in Deep Convolutional Neural Networks for Eigenvalue Problems in Mechanics 
#### There is an associated dataset publicly available in the lab's Harvard Dataverse. Any questions? Contact David at dfinolbe@gmail.com 

#################################################################################################################################################
'''
Copyright 2019 David Finol, Yan Lu, Vijay Mahadevan, Ankit Srivastava

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
################################################################################################################################################

####### BASIC NOTES #########

### Code is configured for multi-GPU computing(single CPU node with multiple GPUs) and Online Learning (batch-based generator data pipiline feeding) 
### A functions python file complement this script. The data pipeline (generator) and other online normalization stats funcitons
### The data generators need to be modified to read the HDF5 family of files of datasets provided. The 'family' driver and sprintf format (..._%d.h5) must be used when calling H5PY.
### There is also a shell filed used to execute this file
### This file was created on Python 2.7.6. Will need adjustment for newer versions. In particular printing format


import numpy as np
import keras
import keras.backend as K 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, GlobalMaxPooling1D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model

import functions_main as fn
import numpy
import csv
import h5py
import gc 
import tensorflow as tf 
import argparse
import os 
import psutil


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
        help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
        help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]


### Parameter Setup ###

dataset_title = '-2D Bandgap Ratio, Unconstrained Mat Properties-'
np.random.seed(7)
file1 = '/share/apps/Phononics/Codes/Neural_networks/TwoDRandMat_fourier/DataNN1.npz' # ignore
file2 = '/share/apps/Phononics/Codes/Neural_networks/TwoDRandMat_fourier/Data/data2D.h5'# INPUT file, adjust to your directory
#file3 = '/share/apps/Phononics/Codes/Neural_networks/TwoDRandMat_fourier/Data/random'
file_real_inp = '/home/ylu50/workingcopy/share/Neural_networks/TwoDRandMat_fourier_test/inputMatrix.npz'# Dataset for testing, which contains real optimized materials
file_real_out = '/home/ylu50/workingcopy/share/Neural_networks/TwoDRandMat_fourier_test/Solution.npz' # Same as before but outputs
filenumber = 'GenTest'
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
early_stop_patience = 10 # keep optimizing for 10 iterations under "no improvement"
out_filepath = 'model_store/file_003.h5'


## Print Key Parameters for record keeping 

print('Network Params:')
print(file2)
print ('cost_function:', cost_function)
print ('n_hidden_units:',n_hidden_units)
print ('batch_size:', batch_size)
print ('kernel_size:', kernel_size)
print ('n_filters:', n_filters)
print ('drop_rate:', drop_rate)
print ('n_convpool_layers', n_convpool_layers)
print ('n_convlayers:', n_convlayers)
print ('n_reglayers:', n_reglayers)
print ('train_set_ratio:', train_set_ratio)
print ('valid_set_ratio:', valid_set_ratio)
print ('n_post_test', n_post_test)
print ('early_stop_delta:', early_stop_delta)
print ('early_stop_patience:', early_stop_patience)

## Loading Data

## Input Data Check by-hand 

real_inp = np.load(file_real_inp)
real_inp = real_inp['inputMatrix']
real_out = np.load(file_real_out)
real_out = np.real(real_out[real_out.keys()[1]][:,0:50])
print('real material info')
print(real_inp.shape)
print(real_out.shape)
print('inp mean and max')
print(np.mean(real_inp))
print(np.amax(real_inp))
print('out mean and max')
print(np.mean(real_out))
print(np.amax(real_out))

gc.collect()

### Pipilining Large Datasets ###

## Building Model

model=Sequential()

model.add(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                 input_shape=(128,128,5),kernel_initializer= kernel_init))
model.add(Activation('relu'))

#model.add(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
#                 kernel_initializer= kernel_init))
#model.add(Activation('relu'))

#model.add(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
#                 kernel_initializer= kernel_init))
#model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=max_poolsize))

#model.add(BatchNormalization())

model.add(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                 kernel_initializer= kernel_init))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_poolsize))

#model.add(BatchNormalization())

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

# check to see if we are compiling using just a single GPU
if G <= 1:
        print("[INFO] training with 1 GPU...")
        model = model

# otherwise, we are compiling using multiple GPUs
else:
        print("[INFO] training with {} GPUs...".format(G))

        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
                # initialize the model
                model = model#MiniGoogLeNet.build(width=32, height=32, depth=3,
                        #classes=10)

        # make the model parallel
        model = multi_gpu_model(model, gpus=G)


def MAE(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true)/K.abs(y_true))

def MAE_np(y_true, y_pred):
    return np.mean(abs(y_pred-y_true)/abs(y_true))


model.compile(loss=cost_function, optimizer='adam', metrics=[MAE])




## Data Normalization
xnsetlist = range(0,1)

### Datasets Info

groupname = 'random'
gname_train2 = 'random2'
gname_train3 = 'random3'
gname_train4 = 'random5'
gname_test2 = 'random2'
gname_test3 = 'random3'
gname_test4 = 'random5'

nset_train2 = 0
nset_train1 = 782 #8*60/4
nset_trainranL = 782#8*40
nset_train3 = 0
nset_test1 = 1
nset_test2 = 50
nset_test3 = 50
nset_test4 = 50
setlen = 64
setlent1 = 64*78


datasets = h5py.File(file2, 'r')
dset1 = datasets[groupname]
dset2 = datasets[gname_test2]
dset3 = datasets[gname_test2]
dset4 = datasets[gname_test3]
dsetnames = dset1.keys()
dsetnames2 = dset2.keys()
dsetnames3 = dset3.keys()
dsetnames4 = dset4.keys()
ynsetlist = range(len(dsetnames)/2,1+len(dsetnames)/2)
groups = datasets.values()
print('Input Shapes:')
print(dset1[dsetnames[nset_train1]].shape)
print(dset2[dsetnames2[nset_test2]].shape)
print(dset3[dsetnames3[nset_test3]].shape)
print(dset4[dsetnames4[nset_test4]].shape)
gc.collect()

## Getting Data moments
xdind = 0#nset_train1
ydind = (len(dsetnames)/2) + xdind
xaxis = (0,1,2,3)
axis = (0,1,2)
yaxis = (0,1)
yaxis2 = (0)
xn_output =[0,-1]#dset1[dsetnames[nset_train1]].shape[1]]
x_mean = fn.get_mean_batch(file2,groupname,xdind,axis,xn_output)
x_var,x_max = fn.get_variance_batch(file2,groupname,xdind,axis,x_mean,xn_output)
x_max = np.amax(dset1[dsetnames[xdind]],axis=axis)
x_var2 = np.var(dset1[dsetnames[xdind]],axis=axis)


x_m0 = np.mean(dset1[dsetnames[xdind]][:,:,:,0])
x_m1 = np.mean(dset1[dsetnames[xdind]][:,:,:,1])
x_m2 = np.mean(dset1[dsetnames[xdind]][:,:,:,2])
x_m3 = np.mean(dset1[dsetnames[xdind]][:,:,:,3])
x_m4 = np.mean(dset1[dsetnames[xdind]][:,:,:,4])

y_mean = fn.get_mean_batch(file2,groupname,ydind,yaxis,n_output)
y_var,y_max = fn.get_variance_batch(file2,groupname,ydind,yaxis,y_mean,n_output)
yaltmax = np.amax(np.real(dset1[dsetnames[ydind]][:,n_output[0]:n_output[1]]),axis=yaxis2)
y_mean2 = fn.get_mean_batch(file2,groupname,ydind,yaxis2,n_output)
y_var2,y_max2 = fn.get_variance_batch(file2,groupname,ydind,yaxis2,y_mean,n_output)
y_max = 1.0e+04

print('Input: Global Mean and Variance, Max')
print(x_mean)
print(x_var)
print('var2:')
print(x_var2)
print(x_max)
print(dsetnames[xdind])
print(x_m0)
print(x_m1)
print(x_m2)
print(x_m3)
print(x_m4)

print('Output: Global Mean and Variance, Max')
print(y_mean)
#print(y_var)
print(y_max)
print(dsetnames[ydind])
print('mean on vector form:')
print(y_mean2)
#print(y_var2)
#print(y_max2)
print(yaltmax)


## This part allows you to try distinct 
# normalization approaches, x_stat and y_stat are the normalization statistics for all batches
y_mean2 = y_mean2*0
mat_n = (1e+11,1e+3,0.0001,0.0001,0.0001)
mat_n = (1.0,1.0,1.0,1.0,1.0)
x_mean = x_mean/mat_n
xstat = [x_mean,x_var,x_max]
#ystat = [y_mean,y_max]
ystat = [y_mean2,yaltmax]
 
print('Training started..')


### Parameters for data generator/pipiline ###
feed_batch_size = 16 # Number of batches to feed into a single GPU (dependent on mememory size and overall batch size needed for training)
n = 1
mix_factor = 2
samplevec_train = [0,feed_batch_size * G*100*n] # starting and ending index for training examples batches to used in the generator (training)
samplevec_val = [feed_batch_size * G*100*n,feed_batch_size * G*150*n] # same as above but for validation
#set3 samplevec_val = [feed_batch_size * G*600/2,feed_batch_size * G*800/2]
samplevec_test = [feed_batch_size * G*0*n,feed_batch_size * G*20*n]
#samplevec_test_pred = [feed_batch_size * G*120*n,feed_batch_size * G*121*n]
samplevec_train2 = [0,setlen*nset_train2]
samplevec_train_mixed = [0,setlen*nset_train1]
samplevec_train_mixed_v3 = [0,setlen*(nset_train1-1)]
samplevec_test1 = [0,setlent1]
samplevec_test2 = [0,setlen*nset_test2]
samplevec_trainranL = [0,setlen*(nset_trainranL-1)]
samplevec_valran = [0,feed_batch_size * G*35*n]

samplevec_test_pred = [0,feed_batch_size * G*1]

filesize = samplevec_train[1]
#filesize = dset1[dsetnames[dind]].shape[0]
print('File Size')
print(filesize)

# Batch Training 
perform_shuffle = False 
repeat_count = 1
model_dir = '/home/dfinolbe/DeepBand/2D_Case/trained_nets'# Directory to store output model

## As it is explained in the Paper, training with a combination of distinct material phases yields the models with the best generalization results
## Below you will find distinct generators (data pipilines) that will feed batches with different ratios of these phases

### TRAIN SET Random Large (Single Phase Material)
#model.fit_generator(generator = fn.generate_batches_from_hdf5_file_randomL_v2(file2,feed_batch_size*G,samplevec_trainranL,groupname,xstat,ystat,axis,nset_trainranL,setlen,n_output),
#      steps_per_epoch = samplevec_trainranL[1] // (feed_batch_size * G),
##      use_multiprocessing=True,
#      validation_data = fn.generate_batches_from_hdf5_file_v3(file2,feed_batch_size*G,samplevec_valran,groupname,xstat,ystat,axis,nset_train2,n_output),
#      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
#      verbose=2,epochs = n_epochs)


### Mixed Datasets Training 

### TRAIN SET 4 Mixed v3 
model.fit_generator(generator = fn.generate_batches_from_hdf5_file_v3_4setmixed(file2,feed_batch_size*G,samplevec_train_mixed,gname_train2,groupname,gname_train3,gname_train4,xstat,ystat,axis,setlen,nset_train1,mix_factor),
      steps_per_epoch = samplevec_train_mixed_v3[1] // (feed_batch_size * G),
      validation_data = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_val,groupname,xstat,ystat,axis,nset_train2),
      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
      verbose=2,epochs = n_epochs)


### TRAIN SET 3 Mixed v3 (64 batch files)
#model.fit_generator(generator = fn.generate_batches_from_hdf5_file_v3_3setmixed(file2,feed_batch_size*G,samplevec_train_mixed,gname_train2,groupname,gname_train3,xstat,ystat,axis,setlen,nset_train1,mix_factor,n_output),
#      steps_per_epoch = samplevec_train_mixed_v3[1] // (feed_batch_size * G),
#      validation_data = fn.generate_batches_from_hdf5_file_v3(file2,feed_batch_size*G,samplevec_valran,groupname,xstat,ystat,axis,nset_train2,n_output),
#      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
#      verbose=2,epochs = n_epochs)
#print('test case: 50 random2, 25 r3 + 25 r')


### TRAIN SET 3 Sets/Phases Mixed
#model.fit_generator(generator = fn.generate_batches_from_hdf5_file_v2_3setmixed(file2,feed_batch_size*G,samplevec_train_mixed,gname_train2,groupname,gname_train3,xstat,ystat,axis,setlen,nset_train1,nset_train2,mix_factor),
#      steps_per_epoch = samplevec_train_mixed[1] // (feed_batch_size * G),
#      validation_data = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_val,groupname,xstat,ystat,axis,nset_train2),
#      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
#      verbose=2,epochs = n_epochs)



### TRAIN SET 2 Sets/Phaes Mixed
#model.fit_generator(generator = fn.generate_batches_from_hdf5_file_v2_2setmixed(file2,feed_batch_size*G,samplevec_train_mixed,gname_train2,groupname,xstat,ystat,axis,setlen,nset_train1,nset_train2,mix_factor),
#      steps_per_epoch = samplevec_train_mixed[1] // (feed_batch_size * G),
#      use_multiprocessing=True,
#      validation_data = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_val,groupname,xstat,ystat,axis,nset_train2),
#      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
#      verbose=2,epochs = n_epochs)


### Single Dataset Training ###


### TRAIN SET 1
#model.fit_generator(generator = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_train,groupname,xstat,ystat,axis,nset_train1),
#      steps_per_epoch = filesize // (feed_batch_size * G),
##      use_multiprocessing=True,
#      validation_data = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_val,groupname,xstat,ystat,axis,nset_train1),
#      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
#      verbose=2,epochs = n_epochs*2,shuffle = True,initial_epoch = n_epochs)

### TRAIN SET 2
#model.fit_generator(generator = fn.generate_batches_from_hdf5_file_test_v2(file2,feed_batch_size*G,samplevec_train2,gname_test2,xstat,ystat,axis,nset_train2,setlen),
#      steps_per_epoch = setlen*nset_train2 // (feed_batch_size * G),
#      use_multiprocessing=True,
#      validation_data = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_val,groupname,xstat,ystat,axis,nset_train1),
#      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
#      verbose=2,epochs = n_epochs)

### TRAIN SET 3
#model.fit_generator(generator = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_train,gname_train3,xstat,ystat,axis,nset_train3),
#      steps_per_epoch = filesize // (feed_batch_size * G),
#      validation_data = fn.generate_batches_from_hdf5_file_v2(file2,feed_batch_size*G,samplevec_val,groupname,xstat,ystat,axis,nset_train1),
#      validation_steps = abs(samplevec_val[0]-samplevec_val[1]) // (feed_batch_size * G),
#      verbose=2,epochs = n_epochs)

### Metrics on Test Sets
score = model.evaluate_generator(generator = fn.generate_batches_from_hdf5_file_test_v3(file2,feed_batch_size*G,samplevec_test1,groupname,xstat,ystat,axis,nset_test1,setlent1,n_output),
      steps = abs(samplevec_test2[0]-samplevec_test2[1]) // (feed_batch_size * G))
print('Test Set 1')
print ('Test loss:', score[0])
print ('Test error:', score[1])

score2 = model.evaluate_generator(generator = fn.generate_batches_from_hdf5_file_test_v3(file2,feed_batch_size*G,samplevec_test2,gname_test2,xstat,ystat,axis,nset_test2,setlen,n_output),
      steps = abs(samplevec_test2[0]-samplevec_test2[1]) // (feed_batch_size * G))

print('Test Set 2')
print(gname_test2)
print ('Test loss:', score2[0])
print ('Test error:', score2[1])

score3 = model.evaluate_generator(generator = fn.generate_batches_from_hdf5_file_test_v3(file2,feed_batch_size*G,samplevec_test2,gname_test3,xstat,ystat,axis,nset_test3,setlen,n_output),
      steps = abs(samplevec_test2[0]-samplevec_test2[1]) // (feed_batch_size * G))
print('Test Set 3')
print(gname_test3)
print ('Test loss:', score3[0])
print ('Test error:', score3[1])

score4 = model.evaluate_generator(generator = fn.generate_batches_from_hdf5_file_test_v3(file2,feed_batch_size*G,samplevec_test2,gname_test4,xstat,ystat,axis,nset_test4,setlen,n_output),
      steps = abs(samplevec_test2[0]-samplevec_test2[1]) // (feed_batch_size * G))
print('Test Set 4')
print(gname_test4)
print ('Test loss:', score4[0])
print ('Test error:', score4[1])

### Real Material Test Set Predictions
print('Test Set 5: Real Mat')
mat_norm = (1e+11,1e+3,0.0001)
mat_norm = (1.0,1.0,1.0)
real_inp_norm,xm,xr = fn.data_normalize_v2_modulus(real_inp,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[0],norm_axis=axis,mat_axis=0)
real_inp_norm,xm,xr = fn.data_normalize_v2_modulus(real_inp_norm,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[1],norm_axis=axis,mat_axis=1)
real_inp_norm,xm,xr = fn.data_normalize_v2_modulus(real_inp_norm,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis,mat_axis=2)
real_inp_norm,xm,xr = fn.data_normalize_v2_modulus(real_inp_norm,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis,mat_axis=3)
real_inp_norm,xm,xr = fn.data_normalize_v2_modulus(real_inp_norm,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis,mat_axis=4)

real_inp_norm,xmean,xxrange = fn.data_normalize_v2(real_inp_norm,norm_type = 'mean',data_mean= x_mean,data_range= x_max,norm_axis=axis)

real_out_norm,ymean,yrange = fn.data_normalize(real_out[:,n_output[0]:n_output[1]],norm_type = 'mean',data_mean=y_mean2,data_range=yaltmax)
score5 = model.evaluate(real_inp_norm,real_out_norm)

### By Hand Prediction Check 
print ('Test loss:', score5[0])
print ('Test error:', score5[1])

y_pred_real = model.predict(real_inp_norm)
print('Predictions')
print(y_pred_real[10:12])
print('actual outputs')
print(real_out_norm[10:12])
print('Alt np MAE')
print(MAE_np(real_out_norm[:11],y_pred_real[:11]))
print(MAE_np(real_out_norm[11:],y_pred_real[11:]))
print(MAE_np(real_out_norm[0:2],y_pred_real[0:2]))

 


