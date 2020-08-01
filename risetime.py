#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:14:44 2020

@author: tnye
"""

# Imports
import numpy as np
from numpy import genfromtxt
from os import makedirs, path, rename
from glob import glob

# home = '/Users/tnye/FakeQuakes/parameters/stress_drop/sd0.1'
home = '/home/tnye/fakequakes/parameters/rise_time/rt2x'
rupt_dir = f'{home}/disp/output/ruptures'
orig_rupt_dir = f'{home}/disp/output/orig_ruptures'

# Rise time multiplication factor 
mf = 2

# Rename original ruptures folder 
if not path.exists(orig_rupt_dir):
    # Rename original ruptures folder 
    rename(rupt_dir, orig_rupt_dir)

# Create new folder called 'ruptures' to store new rise time .rupt files in
if not path.exists(rupt_dir):
    makedirs(rupt_dir)

# Gather original rupture files
rupt_files = np.array(sorted(glob(f'{orig_rupt_dir}/*.rupt')))

# Loop through rupture files and multiple rise time (dura column) by 
# multiplication factor
for r in rupt_files:
    
    f = open(r, 'r')
    header = f.readline().rstrip('\n')
    
    rupt = genfromtxt(r)
    dura = rupt[:,7]
    
    # Create empty array to store new rise times in
    new_dura = np.array([])
    
    for i, rt in enumerate(dura):
        new_dura = np.append(new_dura, mf*rt)
    
    # Change dura column to be new_dura column
    rupt[:,7] = new_dura

    # Save new .rupt file
    rupt_name = r.split('/')[-1]
    np.savetxt(f'{rupt_dir}/{rupt_name}', rupt, delimiter='\t',
               header=header, comments='', fmt='%s')

    # np.savetxt(f'/Users/tnye/FakeQuakes/{rupt_name}',rupt,delimiter='\t',
    #            header=header,comments='',fmt='%s')    
