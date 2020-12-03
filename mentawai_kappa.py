#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:51:38 2020

@author: tnye
"""

###############################################################################
# Script that gets Vs30 and Kappa for the Mentawai strong motion stations. 
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

# # CSV file with event and station information 
# threshold0_path = '/Users/tnye/tsuquakes/flatfiles/threshold_0cm_ia_cav.csv'

# # Read in dataframe
# thresh0_df = pd.read_csv(threshold0_path)

# # Select out Mentawai data
# Mentawai_df = thresh0_df[thresh0_df['eventname'] == 'Mentawai2010']

# # Get station information 
# stations = Mentawai_df.station.values
# stlons = Mentawai_df.stlon.values
# stlats = Mentawai_df.stlat.values

# Strong motion stations data
sm_data = pd.read_csv('/Users/tnye/tsuquakes/data/stations/sm_stations.csv')
stations = sm_data.Station.values
stlons = sm_data.Longitude.values
stlats = sm_data.Latitude.values


################################### Get Vs30 ##################################

# Initialize a text file with strong motion station coordinates 
np.savetxt('/Users/tnye/tsuquakes/data/vs30/stn_coords.txt',np.c_[stlons,stlats],fmt='%.8f,%.8f',delimiter=',')

# Call grdtrack with the vs30 file:
calltext="%sgmt grdtrack /Users/tnye/tsuquakes/data/vs30/stn_coords.txt -G%s > /Users/tnye/tsuquakes/data/vs30/sm_vs30.txt" %(gmt_path,vs30grdfile)

# Make system call
command=split(calltext)
p = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out,err = p.communicate()

# Print output
print(f'out={out}')
print(f'err={err}')


################################### Get Kappa #################################

# Read in the Vs30 output file:
vs30_data = np.genfromtxt('sm_vs30.txt',usecols=2)

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
df.to_csv('/Users/tnye/tsuquakes/data/vs30/vs30_kapp.csv',index=False)


