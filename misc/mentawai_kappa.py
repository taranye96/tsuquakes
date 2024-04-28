#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:51:38 2020

@author: tnye
"""

###############################################################################
# Script that gets Vs30 and Kappa for the Mentawai strong motion stations and 
# saves it to a csv file. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
import subprocess
from shlex import split
import os
from shlex import split
import tsueqs_main_fns as tmf


############################### Define Parameters #############################

# Path to GMT executeable
gmt_path = '/usr/local/bin/'

# Vs30 global proxy grd file to use to extract vs30 values
vs30grdfile = '/Users/tnye/tsuquakes/data/vs30/global_vs30.grd'

station_types = ['gnss', 'sm']

# Strong motion stations data
station_data = pd.read_csv('/Users/tnye/tsuquakes/data/stations/sm_stations.csv')
stations = station_data.Station.values
stlons = station_data.Longitude.values
stlats = station_data.Latitude.values


################################### Get Vs30 ##################################

# Initialize a text file with strong motion station coordinates 
np.savetxt('/Users/tnye/tsuquakes/data/vs30/stn_coords.txt',np.c_[stlons,stlats],fmt='%.8f,%.8f',delimiter=',')

# Call grdtrack with the vs30 file:
calltext = "%sgmt grdtrack /Users/tnye/tsuquakes/data/vs30/stn_coords.txt -G%s > /Users/tnye/tsuquakes/data/vs30/sm_vs30.txt" %(gmt_path,vs30grdfile)

# Make system call
command = split(calltext)
p = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out,err = p.communicate()

# Print output
print(f'out={out}')
print(f'err={err}')


################################### Get Kappa #################################

# Read in the Vs30 output file:
vs30_data = np.genfromtxt('/Users/tnye/tsuquakes/data/vs30/sm_vs30.txt',usecols=2)

# Initialize list for Kappa values
kappa = np.array([])

# Kappa-Vs30 relation
for vs30 in vs30_data:
    k0 = np.exp(3.490 - (1.062 * np.log(vs30)))
    kappa = np.append(kappa,k0)
    
# Crete dictionary 
dataset_dict = {'Station':stations,'Longitude':stlons, 'Latitude':stlats,
                'Vs30':vs30_data, 'Kappa':kappa}
df = pd.DataFrame(data=dataset_dict)

# Save df to csv 
df.to_csv('/Users/tnye/tsuquakes/flatfiles/vs30_kappa.csv',index=False)


