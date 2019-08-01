import h5py

datasets = h5py.File('data/data2D_gzipped_famfiles_%d.h5', 'r',driver='family',memb_size=2500*10**6)
print(list(datasets.keys()))