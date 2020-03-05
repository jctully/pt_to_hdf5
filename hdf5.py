import torch
from os import listdir
from os.path import isfile, join
import numpy
import h5py

ptName = "norm_rcp26_r4i1p1.pt"

mypath = '/research/hutchinson/data/ml_climate/MIROC5_Tensors_norm'
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-3:]=='.pt']

#/research/hutchinson/data/ml_climate/MIROC5_Tensors_norm

x = torch.load("/research/hutchinson/data/ml_climate/MIROC5_Tensors_norm/" + ptName)  # Open pt file
x = x.detach()  # Get rid of grads
x = x.numpy()  # To numpy

f = h5py.File("data.hdf5", "a")
for p in range(len(files)):
    grp = f.create_group(files[p])
    for i in range(x.shape[3]):
        ds = grp.create_dataset('day_'+str(i), (7,128,256), dtype='Float32')
        ds[...] = x[:,:,:,i]

    print(files[p], ": len ", len(f.keys()))

