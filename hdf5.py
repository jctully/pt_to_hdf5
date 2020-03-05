import torch
import os
import numpy
import h5py

ptName = "norm_rcp26_r4i1p1.pt"

#/research/hutchinson/data/ml_climate/MIROC5_Tensors_norm

x = torch.load("/research/hutchinson/data/ml_climate/MIROC5_Tensors_norm/" + ptName)  # Open pt file
x = x.detach()  # Get rid of grads
x = x.numpy()  # To numpy

f = h5py.File(ptName+".hdf5", "a")
for i in range(x.shape[3]):
    ds = f.create_dataset('day_'+str(i), (7,128,256), dtype='Float32')
    ds[...] = x[:,:,:,i]

print(f['/day_0'])

