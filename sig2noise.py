#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:50:49 2020

@author: tnye
"""

# Standard Library Imports 
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import sig2noise_fns as snr

# Read in dataframes
# df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/pga_pgv.csv')
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/pgd_obs.csv')
# sm_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/sm_stations.csv'))
gnss_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/disp_stations.csv'))

flatfile_path = '/Users/tnye/tsuquakes/flatfiles/gnss_SNR.csv'

# Corrected Miniseed files directories
# avg_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_acc'
avg_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_disp'

# Get corrected mseed files
avg_files = np.array(sorted((glob(avg_dir + '/*'))))

# Get arrays of hypocentral distance 
hypdist_list = np.array(df['hypdist'])
indexes = np.unique(hypdist_list, return_index=True)[1]
hypdists = [hypdist_list[index] for index in sorted(indexes)]

origintime = pd.to_datetime('2010-10-25T14:42:22')

epi_lon = 100.082
epi_lat = -3.487
hypdepth = df['hypdepth'][0]

## Gather all streams
streams = []
stream_counter = []

## Loop through stations 

# Read file
SNR_list = []
for file in avg_files:
    st = read(file)
    tr = st[0]

    # Get station coords
    st_index = np.where(gnss_stations==tr.stats.station)[0][0]
    st_lon = gnss_stations[st_index][1]
    st_lat = gnss_stations[st_index][2]

    # Get UTC datetime of P-wave arrival
    p_datetime = snr.get_P_arrival(epi_lon,epi_lat,hypdepth,st_lon,st_lat,6.5,origintime)
    
    # Convert to matplotlib number for numerical analysis 
    p = date2num(p_datetime)

    # Calculate SNR
    SNR = snr.get_SNR(st,p)
    SNR_list.append(SNR)

# Create an expanded list for SNR because dataframe has a row for each component
exp_SNR_list = []
for val in SNR_list:
    exp_SNR_list.append(val)
    exp_SNR_list.append(val)
    exp_SNR_list.append(val)

## Add SNR to dataframe
df['SNR'] = exp_SNR_list
df.to_csv(flatfile_path,index=False)

## Plot results
figpath = '/Users/tnye/tsuquakes/plots/SNR/mentawai_disp_max.png'
plt.scatter(hypdists,np.log10(SNR_list))
plt.xlabel('Hyocentral Dist (km)')
plt.ylabel('log10 SNR')
plt.title('Acceleration SNR')
plt.savefig(figpath, bbox_inches='tight', dpi=300)

plt.close()
    
