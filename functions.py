#### This functions are to be coupled with the main code that pertains to the 2D machine learning results presented in
#### Deep Convolutional Neural Networks for Eigenvalue Problems in Mechanics
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

### These are functions that are coupled with the main code provided
### There are various versions and other functions not needed for the 2D Eigenvalue problem in this File.


import numpy
import numpy as np
import scipy.io as sio
import tensorflow as tf
import h5py
import gc
import keras.backend as K
global graph
import keras
import os
import psutil

def get_si(groups):
    x = os.path.getsize(data2D_gzipped_famfiles_0.h5") / len(groups)
    x = x + os.path.getsize("data2D1.h5") / len(groups)
    x = x + os.path.getsize("data2D2.h5") / len(groups)
    x = x + os.path.getsize("data2D3.h5") / len(groups)
    x = x + os.path.getsize("data2D4.h5") / len(groups)
    x = x + os.path.getsize("data2D5.h5") / len(groups)
    x = x + os.path.getsize("data2D6.h5") / len(groups)
    return x


def get_variance_batch(filepath,groupname,dsetindex,axis,mean,n_output):
    """
    Returns the variance and maximum values of a dataset whenever python functions are unable due to memory restrictions.
    :param str filepath: Filepath of the data upon which the mean should be calculated.
    dsetindex: dataset index: represents the index of the dataset within the selected groupname
    :return: ndarray xs_var and xs_max: variance and max of the dataset.
    """

    datasets = h5py.File(filepath, 'r', driver='family', memb_size=2500*10**6)
    f = datasets[groupname]
    groups = datasets.values()
    dsetnames = list(f.keys())
    datatype = np.float32
    # check available memory and divide the mean calculation in steps with psutil
    # print(psutil.virtual_memory()[1])
    total_memory = 0.5 * psutil.virtual_memory()[1]  # In bytes. Take 1/2 of what is available, just to make sure.
    filesize = get_si(groups)/len(groups)
    steps = int(np.ceil(filesize/total_memory))
    n_rows = f[dsetnames[dsetindex]].shape[0]
    stepsize = int(n_rows / float(steps))

    xs_max_arr = None
    xs_var_arr = None
    for i in range(steps):
        if xs_var_arr is None: # create xs_mean_arr that stores intermediate mean_temp results
            xs_var_arr = np.zeros((steps, ) + f[dsetnames[dsetindex]][:,n_output[0]:n_output[1]].shape[1:], dtype=datatype)
            xs_max_arr = np.zeros((steps, ) + f[dsetnames[dsetindex]][:,n_output[0]:n_output[1]].shape[1:], dtype=datatype)

        if i == steps-1: # for the last step, calculate mean till the end of the file
            data_temp = np.real(f[dsetnames[dsetindex]][i * stepsize: n_rows,n_output[0]:n_output[1]])
            max_temp = np.amax(data_temp,axis=axis)
            xs_var_temp = np.var(data_temp, axis=axis, dtype=datatype)
            # xs_var_temp = np.sum(np.square(f['matrices'][i * stepsize: n_rows] - mean),axis=axis)
        else:
            data_temp = np.real(f[dsetnames[dsetindex]][i * stepsize: (i+1) * stepsize,n_output[0]:n_output[1]])
            max_temp = np.amax(data_temp,axis=axis)
            xs_var_temp = np.var(data_temp, axis=axis, dtype=datatype)
            # xs_var_temp = np.sum(np.square(f['matrices'][i * stepsize: (i+1) * stepsize]- mean),axis=axis)
        xs_var_arr[i] = xs_var_temp
        xs_max_arr[i] = max_temp
    # xs_var = np.sum((1.0/n_rows)*xs_var_arr)
    # xs_var = np.var(xs_var_arr, axis=axis, dtype=np.float64).astype(np.float32)

    xs_var = np.mean(xs_var_arr)
    xs_max = np.amax(xs_max_arr)
    datasets.close()
    return xs_var,xs_max



def get_mean_batch(filepath,groupname,dsetindex,axis,n_output):
    """
    Returns the mean of a dataset whenever python functions are unable due to memory restrictions.
    :param str filepath: Filepath of the data upon which the mean should be calculated.
    dsetindex: represents the index of the dataset within the selected groupname
    :return: ndarray xs_mean: mean of the dataset.
    """

    datasets = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**6)
    f = datasets[groupname]
    groups = datasets.values()
    dsetnames = list(f.keys())
    datatype = np.float32


    # check available memory and divide the mean calculation in steps
#    print(psutil.virtual_memory()[1])
    total_memory = 0.5 * psutil.virtual_memory()[1]  # In bytes. Take 1/2 of what is available, just to make sure.
    filesize = get_si(groups)/len(groups)
    steps = int(np.ceil(filesize/total_memory))
    n_rows = f[dsetnames[dsetindex]].shape[0]
    stepsize = int(n_rows / float(steps))
#    print('info')
#    print(n_rows)
#    print(stepsize)
    xs_mean_arr = None
    # for i in range(steps):
    #     if xs_mean_arr is None:  # create xs_mean_arr that stores intermediate mean_temp results
    #         xs_mean_arr = np.zeros((steps, ) + f[dsetnames[dsetindex]].shape[1:], dtype=datatype)
    #
    #     if i == steps-1:  # for the last step, calculate mean till the end of the file
    #         xs_mean_temp = np.mean(np.real(f[dsetnames[dsetindex]][i * stepsize: n_rows]), axis=axis, dtype= datatype)
    #         # print(i)
    #     else:
    #         xs_mean_temp = np.mean(np.real(f[dsetnames[dsetindex]][i*stepsize: (i+1) * stepsize]), axis=axis, dtype=datatype)
    #     xs_mean_arr[i] = xs_mean_temp

    # xs_mean_arr = np.zeros((steps,) + f[dsetnames[dsetindex]].shape[1:], dtype=datatype)
    # xs_mean_temp = np.mean(np.real(f[dsetnames[dsetindex]], axis=axis, dtype=datatype)

    xs_mean = np.mean(np.real(f[dsetnames[dsetindex]]), axis=axis, dtype= datatype)
    # print(stepsize)
    datasets.close()
    return xs_mean

def generate_batches_from_hdf5_file_randomL_v2(filepath, batchsize,samplevec,groupname,xstats,ystats,axis,nsetvec,nsampset,n_output):
    """
    Generator that returns batches of inputs/features ('xs') and outputs/targets ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of inputs and outputs.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    mat_norm = (1e+11,1e+3,0.0001)
    mat_norm = (1.0,1.0,1.0)

    dimensions = (batchsize, 128, 128, 5)
    nsetvec =  np.arange(0,nsetvec)
    while 1:
        fil = h5py.File(filepath,'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        dsnames = list(f.keys())
        # filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries_total = 0
        n_entries = 0
        nset = 1
        # as long as we haven't read all entries from the file: keep reading
        while n_entries_total <  abs(samplevec[0]-samplevec[1]):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs = f[dsnames[nsetvec[nset]]][n_entries : n_entries + batchsize ]
            # Old  Modulus norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)

           # New Modulus norm factor
#old         xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[0],norm_axis=axis, mat_axis=0)

            # Density norm factor
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[1],norm_axis=axis, mat_axis=1)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=2)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=3)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=4)

            # Overall norm
            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_max,norm_axis=axis)

            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nsetvec[nset]]][n_entries:n_entries+batchsize,n_output[0]:n_output[1]])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'max',data_mean=0,data_range=y_max)
           # we have read one more batch from this file
            n_entries_total += batchsize
            n_entries += batchsize
            if n_entries >= nsampset:
                nset += 1
                n_entries = 0
            yield (xs, ys)
        fil.close()


def generate_batches_from_hdf5_file_v3_4setmixed(filepath, batchsize,samplevec,groupname,gname2,gname3,gname4,xstats,ystats,axis,nsampset,nsetvec,mix_factor):
    """
    Generator that returns batches of inputs ('xs') and outputs ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of inputs and outputs.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    nsetvec =  np.arange(0,nsetvec)
    feed_count = 0
    dimensions = (batchsize, 128, 128, 5)

    while 1:
        fil = h5py.File(filepath,'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        f2 = fil[gname2]
        f3 = fil[gname3]
        f4 = fil[gname4]

        dsnames = list(f.keys())
        dsnames2 = list(f2.keys())
        dsnames3 = list(f3.keys())
        dsnames4 = list(f4.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries_total = 0
        n_entries = 0
        n_entries_set2 = 0
        nset = 1
        nset2  = 1
        #print('inloop')
        #print(feed_count)
        #mix_factor = 2
        # as long as we haven't read all entries from the file: keep reading
        while n_entries_total <  abs(samplevec[0]-samplevec[1]):
            #print('n_entries_total')
            #print(n_entries_total)
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs2 = f2[dsnames2[nsetvec[nset2]]][n_entries_set2 : n_entries_set2 + int(batchsize/mix_factor/2) ]
            xs = f[dsnames[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor/2) ]
            xs3 = f3[dsnames3[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor/2) ]
            xs4 = f4[dsnames4[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor/2 )]

            xs = np.vstack([xs,xs2,xs3,xs4])
           # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)
            #print(xs.shape)
            # Targets
            ys2 =  np.real(f2[dsnames2[int(len(dsnames2)/2)+nsetvec[nset2]]][n_entries_set2:n_entries_set2+int(batchsize/mix_factor/2)])
            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor/2)])
            ys3 =  np.real(f3[dsnames3[int(len(dsnames3)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor/2)])
            ys4 =  np.real(f4[dsnames4[int(len(dsnames4)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor/2)])

            ys = np.vstack([ys,ys2,ys3,ys4])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'max',data_mean=0,data_range=y_max)

            # we have read one more batch from this file
            #print(ys.shape)
            n_entries_total += batchsize
            n_entries += int(batchsize/mix_factor/2)
            n_entries_set2 += int(batchsize/mix_factor/2)

            if n_entries_set2 >= nsampset:
                nset2 += 1
                n_entries_set2 = 0

            if n_entries >= nsampset:
                nset += 1
                n_entries = 0

            yield (xs, ys)
        fil.close()


def generate_batches_from_hdf5_file_v3_3setmixed(filepath, batchsize,samplevec,groupname,gname2,gname3,xstats,ystats,axis,nsampset,nsetvec,mix_factor,n_output):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    mat_norm = (1e+11,1e+3,0.0001)
    mat_norm = (1.0,1.0,1.0)
    nsetvec =  np.arange(0,nsetvec)
    feed_count = 0
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel

    while 1:
        fil = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        f2 = fil[gname2]
        f3 = fil[gname3]
        dsnames = list(f.keys())
        dsnames2 = list(f2.keys())
        dsnames3 = list(f3.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries_total = 0
        n_entries = 0
        n_entries_set2 = 0
        nset = 1
        nset2  = 1
        # print('inloop')
        # print(feed_count)
        # mix_factor = 2
        # as long as we haven't read all entries from the file: keep reading
        while n_entries_total <  abs(samplevec[0]-samplevec[1]):
            #print('n_entries_total')
            #print(n_entries_total)
            # start the next batch at index 0
            # create numpy arrays of input data (features)
#            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)

            xs2 = f2[dsnames2[nsetvec[nset2]]][n_entries_set2 : n_entries_set2 + int(batchsize/mix_factor) ]
            xs = f[dsnames[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor/2) ]
            xs3 = f3[dsnames3[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor/2) ]
            # print(xs3.shape)
            # print(xs2.shape)
            # print(xs.shape)
            xs = np.vstack([xs,xs2,xs3])
           # Modulus norm factor
# old         xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[0],norm_axis=axis, mat_axis=0)

            # Density norm factor
            # xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)
            xs, xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[1],norm_axis=axis, mat_axis=1)
            xs, xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=2)
            xs, xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=3)
            xs, xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=4)

            # Overall norm
            xs, xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_max,norm_axis=axis)

            # xs,xm,xr = data_normalize_v2(xs,norm_type='max',data_mean = 0 ,data_range = x_max/mat_norm[0] ,norm_axis=axis)
            # print(xs.shape)
            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...

            ys2 = np.real(f2[dsnames2[int(len(dsnames2)/2)+nsetvec[nset2]]][n_entries_set2:n_entries_set2+int(batchsize/mix_factor),n_output[0]:n_output[1]])
            ys = np.real(f[dsnames[int(len(dsnames)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor/2),n_output[0]:n_output[1]])
            ys3 = np.real(f3[dsnames3[int(len(dsnames3)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor/2),n_output[0]:n_output[1]])

            ys = np.vstack([ys,ys2,ys3])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'max',data_mean=y_mean,data_range=y_max)
#            ys = keras.utils.normalize(ys, axis=-1, order=2)

            # we have read one more batch from this file
            # print(ys.shape)
            n_entries_total += batchsize
            n_entries += int(batchsize/mix_factor/2)
            n_entries_set2 += int(batchsize/mix_factor)
#            print('check5')
            # print('n_entries+n_entries_set2+nsampset')
            # print(n_entries)
            # print(n_entries_set2)
            # print(nsampset)
            if n_entries_set2 >= nsampset:
                nset2 += 1
                n_entries_set2 = 0

            if n_entries >= nsampset:
                nset += 1
                n_entries = 0
             #   print('nset')
             #   print(nset)
#            print('check6')
            yield (xs, ys)
        fil.close()

def generate_batches_from_hdf5_file_v2_3setmixed(filepath, batchsize,samplevec,groupname,gname2,gname3,xstats,ystats,axis,nsampset,nsetvec,nset2,mix_factor):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    nsetvec =  np.arange(0,nsetvec)
    feed_count = 0
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel

    while 1:
        fil = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        f2 = fil[gname2]
        f3 = fil[gname3]
        dsnames = list(f.keys())
        dsnames2 = list(f2.keys())
        dsnames3 = list(f3.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries_total = 0
        n_entries = 0
        n_entries_set2 = 0
        nset = 0
        #print('inloop')
        #print(feed_count)
        #mix_factor = 2
        # as long as we haven't read all entries from the file: keep reading
        while n_entries_total <  abs(samplevec[0]-samplevec[1]):
            #print('n_entries_total')
            #print(n_entries_total)
            # start the next batch at index 0
            # create numpy arrays of input data (features)
#            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)

            xs2 = f2[dsnames2[nset2]][n_entries_set2+samplevec[0] : n_entries_set2 + int(batchsize/mix_factor) + samplevec[0]]
            xs = f[dsnames[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor/2) ]
            xs3 = f3[dsnames3[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor/2) ]

            xs = np.vstack([xs,xs2,xs3])
           # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)

            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= 0,data_range= x_max/mat_norm[0],norm_axis=axis)

            #print(xs.shape)
            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
 #norm           ys =  keras.utils.normalize(np.real(f[dsnames[len(dsnames)/2]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]]), axis=-1, order=2)

            ys2 =  np.real(f2[dsnames2[int(len(dsnames2)/2) + nset2]][n_entries_set2 + samplevec[0]:n_entries_set2+int(batchsize/mix_factor)+samplevec[0]])
            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor/2)])
            ys3 =  np.real(f3[dsnames3[int(len(dsnames3)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor/2)])

            ys = np.vstack([ys,ys2,ys3])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'mean',data_mean=y_mean,data_range=y_max)
#            ys = keras.utils.normalize(ys, axis=-1, order=2)

            # we have read one more batch from this file
            #print(ys.shape)
            n_entries_total += batchsize
            n_entries += int(batchsize/mix_factor)
            n_entries_set2 += int(batchsize/mix_factor)
#            print('check5')
            #print('n_entries+n_entries_set2+nsampset')
            #print(n_entries)
            #print(n_entries_set2)
            #print(nsampset)
            if n_entries >= nsampset:
                nset += 1
                n_entries = 0
             #   print('nset')
             #   print(nset)
#            print('check6')
            yield (xs, ys)
        fil.close()
#        feed_count +=1
        #print(ys[0:5])


def generate_batches_from_hdf5_file_v2_2setmixed(filepath, batchsize,samplevec,groupname,gname2,xstats,ystats,axis,nsampset,nsetvec,nset2,mix_factor):
    """
    Generator that returns batches of inputs ('xs') and outputs ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of inputs and outputs.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    nsetvec =  np.arange(0,nsetvec)
    feed_count = 0
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel

    while 1:
        fil = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        f2 = fil[gname2]
        dsnames = list(f.keys())
        dsnames2 = list(f2.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries_total = 0
        n_entries = 0
        n_entries_set2 = 0
        nset = 0
        #print('inloop')
        #print(feed_count)
        #mix_factor = 2
        # as long as we haven't read all entries from the file: keep reading
        while n_entries_total <  abs(samplevec[0]-samplevec[1]):
            #print('n_entries_total')
            #print(n_entries_total)
            # start the next batch at index 0
            # create numpy arrays of input data (features)
#            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)
            xs2 = f2[dsnames2[nset2]][n_entries_set2+samplevec[0] : n_entries_set2 + int(batchsize/mix_factor) + samplevec[0]]
#            xs2 = f2[dsnames[nset2]][n_entries+samplevec[0] : n_entries + batchsize/mix_ratio + samplevec[0]]
#            print('check2')
            xs = f[dsnames[nsetvec[nset]]][n_entries : n_entries + int(batchsize/mix_factor) ]
#            print('check3')
            xs = np.vstack([xs,xs2])
#            print('check3')

#            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_var,norm_axis=axis)
#           xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max,norm_axis=axis)
            # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)
            #print(xs.shape)
            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
 #norm           ys =  keras.utils.normalize(np.real(f[dsnames[len(dsnames)/2]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]]), axis=-1, order=2)

            ys2 =  np.real(f2[dsnames2[int(len(dsnames2)/2) + nset2]][n_entries_set2 + samplevec[0]:n_entries_set2+int(batchsize/mix_factor)+samplevec[0]])
            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nsetvec[nset]]][n_entries:n_entries+int(batchsize/mix_factor)])
            #print(ys2.shape)
            #print(ys.shape)
            ys = np.vstack([ys,ys2])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'max',data_mean=0,data_range=y_max)
#            ys = keras.utils.normalize(ys, axis=-1, order=2)

            # we have read one more batch from this file
            #print(ys.shape)
            n_entries_total += batchsize
            n_entries += int(batchsize/mix_factor)
            n_entries_set2 += int(batchsize/mix_factor)
#            print('check5')
            #print('n_entries+n_entries_set2+nsampset')
            #print(n_entries)
            #print(n_entries_set2)
            #print(nsampset)
            if n_entries >= nsampset:
                nset += 1
                n_entries = 0
             #   print('nset')
             #   print(nset)
#            print('check6')
            yield (xs, ys)
        fil.close()
        feed_count +=1
        #print(ys[0:5])

def generate_batches_from_hdf5_file_pred_v2(filepath, batchsize,samplevec,groupname,xstats,ystats,axis,nset):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats

    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel

    while 1:
        fil = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        dsnames = list(f.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries <  abs(samplevec[0]-samplevec[1]):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
#            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)
            xs = f[dsnames[nset]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]]
#            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_var,norm_axis=axis)
#           xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max,norm_axis=axis)
            # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)

#            xs = keras.utils.normalize(xs, axis=-1, order=2)
#            xs = np.reshape(xs, dimensions).astype('float32')

            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
 #norm           ys =  keras.utils.normalize(np.real(f[dsnames[len(dsnames)/2]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]]), axis=-1, order=2)
            ys =  np.real(f[dsnames[int(len(dsnames)/2) + nset]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'max',data_mean=0,data_range=y_max)
#            ys = keras.utils.normalize(ys, axis=-1, order=2)

#            ys = np.array(np.zeros((batchsize, 2))) # data with 2 different classes (e.g. dog or cat)

            # Select the labels that we want to use, e.g. is dog/cat
#            for c, y_val in enumerate(y_values):
#                ys[c] = encode_targets(y_val, class_type='dog_vs_cat') # returns categorical labels [0,1], [1,0]

            # we have read one more batch from this file

 #           n_entries_total += batchsize
            n_entries += batchsize
#            if n_entries >= nsampset:
#                nset += 1
#                n_entries = 0


#            print(ys[0])
            yield (xs, ys)
        fil.close()
        print(ys[0:5])

def generate_batches_from_hdf5_file_test_v3(filepath, batchsize,samplevec,groupname,xstats,ystats,axis,nsetvec,nsampset,n_output):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel
    nsetvec =  np.arange(0,nsetvec)
    mat_norm = (1e+11,1e+3,0.0001)
    mat_norm = (1.0,1.0,1.0)

    while 1:
        fil = h5py.File(filepath,'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        dsnames = list(f.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries_total = 0
        n_entries = 0
        nset = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries_total <  abs(samplevec[0]-samplevec[1]):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
# norm            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)
            xs = f[dsnames[nsetvec[nset]]][n_entries : n_entries + batchsize ]
#            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_var,norm_axis=axis)
            #xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max,norm_axis=axis)
            # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[0],norm_axis=axis, mat_axis=0)
            # Density norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[1],norm_axis=axis, mat_axis=1)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=2)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=3)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=4)

            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_max,norm_axis=axis)

#            xs = keras.utils.normalize(xs, axis=-1, order=2)
#            ys =  keras.utils.normalize(np.real(f[dsnames[len(dsnames)/2]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]]), axis=-1, order=2)

            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nsetvec[nset]]][n_entries:n_entries+batchsize,n_output[0]:n_output[1]])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'mean',data_mean=y_mean,data_range=y_max)
           # we have read one more batch from this file
            n_entries_total += batchsize
            n_entries += batchsize
            if n_entries >= nsampset:
                nset += 1
                n_entries = 0
                #print(nset)
                #print(n_entries_total)
            yield (xs, ys)
        fil.close()


def generate_batches_from_hdf5_file_v3(filepath, batchsize,samplevec,groupname,xstats,ystats,axis,nset,n_output):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel
    mat_norm = (1e+11,1e+3,0.0001)
    mat_norm = (1.0,1.0,1.0)

    while 1:
        fil = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**66)
        f = fil[groupname]
        dsnames = list(f.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries <  abs(samplevec[0]-samplevec[1]):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
# norm            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)
            xs = f[dsnames[nset]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]]
            #xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_var,norm_axis=axis)
            # Modulus norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)

            # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[0],norm_axis=axis, mat_axis=0)
            # Density norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[1],norm_axis=axis, mat_axis=1)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=2)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=3)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=4)

            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_max,norm_axis=axis)


            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nset]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0],n_output[0]:n_output[1]])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'mean',data_mean=y_mean,data_range=y_max)
            n_entries += batchsize
            yield (xs, ys)
        fil.close()


def generate_batches_from_hdf5_file_v2(filepath, batchsize,samplevec,groupname,xstats,ystats,axis,nset):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel

    while 1:
        fil = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        dsnames = list(f.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries <  abs(samplevec[0]-samplevec[1]):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
# norm            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)
            xs = f[dsnames[nset]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]]
            #xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_var,norm_axis=axis)
            # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)

#            xs = keras.utils.normalize(xs, axis=-1, order=2)
#            xs = np.reshape(xs, dimensions).astype('float32')

            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
#            ys =  keras.utils.normalize(np.real(f[dsnames[len(dsnames)/2]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]]), axis=-1, order=2)
            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nset]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'max',data_mean=0,data_range=y_max)
#            ys = keras.utils.normalize(ys, axis=-1, order=2)

#            ys = np.array(np.zeros((batchsize, 2))) # data with 2 different classes (e.g. dog or cat)

            # Select the labels that we want to use, e.g. is dog/cat
#            for c, y_val in enumerate(y_values):
#                ys[c] = encode_targets(y_val, class_type='dog_vs_cat') # returns categorical labels [0,1], [1,0]

            # we have read one more batch from this file
            n_entries += batchsize
            yield (xs, ys)
        fil.close()

def generate_batches_from_hdf5_file_v3(filepath, batchsize,samplevec,groupname,xstats,ystats,axis,nset,n_output):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel
    mat_norm = (1e+11,1e+3,0.0001)
    mat_norm = (1.0,1.0,1.0)

    while 1:
        fil = h5py.File(filepath,'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        dsnames = list(f.keys())
        # filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries <  abs(samplevec[0]-samplevec[1]):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
# norm            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)
            xs = f[dsnames[nset]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]]
            #xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_var,norm_axis=axis)
            # Modulus norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)

            # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[0],norm_axis=axis, mat_axis=0)
            # Density norm
#            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[1],norm_axis=axis, mat_axis=1)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=2)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=3)
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= mat_norm[2],norm_axis=axis, mat_axis=4)

            xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_max,norm_axis=axis)


            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nset]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0],n_output[0]:n_output[1]])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'mean',data_mean=y_mean,data_range=y_max)
            n_entries += batchsize
            yield (xs, ys)
        fil.close()


def generate_batches_from_hdf5_file_v2(filepath, batchsize,samplevec,groupname,xstats,ystats,axis,nset):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    [x_mean,x_var,x_max] = xstats
    [y_mean,y_max] = ystats
    dimensions = (batchsize, 128, 128, 5) # 28x28 pixel, one channel

    while 1:
        fil = h5py.File(filepath, 'r',driver='family',memb_size=2500*10**6)
        f = fil[groupname]
        dsnames = list(f.keys())
#        filesize = len(f[dsnames[0]])
        filesize = samplevec[1]

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries <  abs(samplevec[0]-samplevec[1]):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
# norm            xs = keras.utils.normalize(f[dsnames[0]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]], axis=-1, order=2)
            xs = f[dsnames[nset]][n_entries+samplevec[0] : n_entries + batchsize + samplevec[0]]
            #xs,xmean,xxrange = data_normalize_v2(xs,norm_type = 'mean',data_mean= x_mean,data_range= x_var,norm_axis=axis)
            # Modulus norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[0],norm_axis=axis, mat_axis=0)
            # Density norm
            xs,xmean,xxrange = data_normalize_v2_modulus(xs,norm_type = 'max',data_mean= x_mean,data_range= x_max[1],norm_axis=axis, mat_axis=1)

#            xs = keras.utils.normalize(xs, axis=-1, order=2)
#            xs = np.reshape(xs, dimensions).astype('float32')

            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
#            ys =  keras.utils.normalize(np.real(f[dsnames[len(dsnames)/2]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]]), axis=-1, order=2)
            ys =  np.real(f[dsnames[int(len(dsnames)/2)+nset]][n_entries+samplevec[0]:n_entries+batchsize+samplevec[0]])
            ys,ymean,yrange = data_normalize(ys,norm_type = 'max',data_mean=0,data_range=y_max)
#            ys = keras.utils.normalize(ys, axis=-1, order=2)

#            ys = np.array(np.zeros((batchsize, 2))) # data with 2 different classes (e.g. dog or cat)

            # Select the labels that we want to use, e.g. is dog/cat
#            for c, y_val in enumerate(y_values):
#                ys[c] = encode_targets(y_val, class_type='dog_vs_cat') # returns categorical labels [0,1], [1,0]

            # we have read one more batch from this file
            n_entries += batchsize
            yield (xs, ys)
        fil.close()

def data_normalize(x_data,norm_type,data_mean,data_range):

 #print('Pre-norm Stats')
 #print 'max:', np.amax(x_data)
 #print 'min:', np.amin(x_data)
 #print 'mean:', np.mean(x_data)

 if type(data_mean) == str:
  data_mean = numpy.mean(x_data)

 if type(data_range) == str:
  data_range = numpy.max(x_data) - numpy.min(x_data)

 if norm_type =='mean':
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'STD':
  data_range = numpy.std(x_data)
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'max':
#   data_range = numpy.amax(x_data)
   data_mean =  0
   x_data = (x_data-data_mean)/data_range

# print('Post-norm Stats')
# print 'max:', np.amax(x_data)
# print 'min:', np.amin(x_data)
# print 'mean:', np.mean(x_data)

 return x_data, data_mean, data_range

def data_normalize_v2(x_data,norm_type,data_mean,data_range,norm_axis):

# print('Pre-norm Stats')
# print 'max:', np.amax(x_data)
# print 'min:', np.amin(x_data)
# print 'mean:', np.mean(x_data)

 if type(data_mean) == str:
  data_mean = numpy.mean(x_data,axis=norm_axis)

 if type(data_range) == str:
  data_range = numpy.max(x_data,axis=norm_axis) - numpy.min(x_data,axis=norm_axis)
  if norm_type == 'var':
   data_range = numpy.var(x_data,axis=norm_axis)
  if norm_type == 'max':
   data_range = numpy.max(x_data,axis=norm_axis) - numpy.min(x_data,axis=norm_axis)


 if norm_type =='mean':
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'var':
#  data_range = numpy.var(x_data)
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'max':
#   data_range = numpy.amax(x_data)
   data_mean =  0
   x_data = (x_data-data_mean)/data_range

 #print('Post-norm Stats')
 #print 'max:', np.amax(x_data)
 #print 'min:', np.amin(x_data)
 #print 'mean:', np.mean(x_data)

 return x_data, data_mean, data_range

def data_normalize_v2_modulus(x_data,norm_type,data_mean,data_range,norm_axis,mat_axis):

 if type(data_mean) == str:
  data_mean = numpy.mean(x_data,axis=norm_axis)

 if type(data_range) == str:
  data_range = numpy.max(x_data,axis=norm_axis) - numpy.min(x_data,axis=norm_axis)
  if norm_type == 'var':
   data_range = numpy.var(x_data,axis=norm_axis)
  if norm_type == 'max':
   data_range = numpy.max(x_data,axis=norm_axis) - numpy.min(x_data,axis=norm_axis)


 if norm_type =='mean':
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'var':
#  data_range = numpy.var(x_data)
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'max':
#   data_range = numpy.amax(x_data)
   data_mean =  0
   x_data[:,:,:,mat_axis] = (x_data[:,:,:,mat_axis]-data_mean)/data_range

 return x_data, data_mean, data_range

def data_normalize_vold(x_data,norm_type,data_mean,data_range):

 print('Pre-norm Stats')
 print ('max:', np.amax(x_data))
 print ('min:', np.amin(x_data) )
 print ('mean:', np.mean(x_data))

 if data_mean == None:
  data_mean = numpy.mean(x_data)

 if data_range==None:
  data_range = numpy.max(x_data) - numpy.min(x_data)

 if norm_type =='mean':
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'STD':
  data_range = numpy.std(x_data)
  x_data = (x_data-data_mean)/data_range

 if norm_type == 'max':
   data_range = numpy.amax(x_data)
   data_mean =  0
   x_data = (x_data-data_mean)/data_range

 print('Post-norm Stats')
 print ('max:', np.amax(x_data))
 print ('min:', np.amin(x_data))
 print ('mean:', np.mean(x_data))

 return x_data, data_mean, data_range

