#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:56:36 2020

@author: tnye
"""

# Imports
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data_types = ['disp','sm']

runs = np.array(['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
        'run.000005', 'run.000006', 'run.000007', 'run.000008', 'run.000009',
        'run.000010', 'run.000011'])


if rank == 0:
    # make data
    full_data = np.arange(len(runs), dtype='d')
    print("I'm {0} and fulldata is: {1}".format(rank,full_data))
    # data=np.arange(100, dtype='<U10')
else:
    # initialize variables
    full_data=None

count = 3
# mydata = np.empty(count, dtype='<U10')
mydata = np.empty(count, dtype='d')
# comm.Scatter([full_data, count, MPI.INT],[mydata, count, MPI.INT],root=0)
comm.Scatter(full_data,mydata,root=0)
# comm.Scatter([full_data, count],[mydata, count],root=0)
print("After Scatter, I'm {0} and mydata is: {1}".format(rank,mydata))

# mydata = np.empty(numdata//4, dtype='<U10')
# comm.Scatter(full_data,recvbuf,root=0)
# print("After Scatter, I'm {0} and mydata is: {1}".format(rank,recvbuf))

# if rank == 0:
#     recvbuf = np.empty(numdata, dtype='<U10')

# comm.Gather(mydata, recvbuf, root=0)
# # comm.Gather(recvbuf, mydata, root=0)

# if rank ==0:
#     print(f'Rank: {rank}, recvbuf received: {recvbuf}')

sendbuf = mydata
print(f'Rank: {rank}, sendbuf: {sendbuf}')

recvbuf=None
if rank == 0:
    recvbuf = np.empty(count*size, dtype='d')

comm.Gather(sendbuf, recvbuf, root=0)
print(f'Rank: {rank}, recvbuf received: {recvbuf}')