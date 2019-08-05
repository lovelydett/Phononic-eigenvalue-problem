import h5py
import os

def get_mean(dataSet,datasetName,axis):
    return np.mean(dataSet[datasetName],axis=axis,dtype = dataSet[satasetName].dtype)


def get_variance(dataSet,datasetName,axis):
    datatype = dataSet[datasetName].dtype
    xs_max = np.amax(np.real(dataSet[datasetName]),axis=axis)
    xs_var = np.var(np.real(dataSet[datasetName]), axis=axis, dtype=datatype)
    return xs_var,xs_max