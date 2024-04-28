#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:50:49 2020

@author: tnye
"""

###############################################################################
# Script that calculates the signal to noise ratio for the 2010 M7.8 Mentawai
# data at various GNSS and sm stations and stores it in a dataframe and plots
# it against hypdist. 
###############################################################################

# Standard Library Imports 
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import sig2noise_fns as snr
import signal_average_fns as avg

# Read in dataframe
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')

# Event parameters
origintime = pd.to_datetime('2010-10-25T14:42:22')
epi_lon = 100.082
epi_lat = -3.487
hypdepth = df['hypdepth (km)'][0]

# Corrected horizontal average mseed file directories 
disp_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_disp'
acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_acc'

data_types = ['disp', 'sm']

for data in data_types:
    
    # Corrected horizontal average mseed file directories 
    mseed_dir = f'/Users/tnye/tsuquakes/data/Mentawai2010/avg_{data}'
    
    # Corrected mseed files
    files = np.array(sorted((glob(mseed_dir + '/*'))))

    # Get arrays of hypocentral distance 
    if data == 'disp':
        hypdists = np.array(df['hypdist'][:13])
        stations = df['station'][:13]
        flatfile_path = '/Users/tnye/tsuquakes/flatfiles/gnss_SNR.csv'
    else:
        hypdists = np.array(df['hypdist'][13:])
        stations = df['station'][13:]
        flatfile_path = '/Users/tnye/tsuquakes/flatfiles/sm_SNR.csv'
    
    # Loop through stations 
    SNR_list = []
    for file in files:
        st = read(file)
        tr = st[0]
    
        # Get station coords
        st_index = np.where(np.array(df['station'])==tr.stats.station)[0][0]
        st_lon = df['stlon'][st_index]
        st_lat = df['stlat'][st_index]
    
        # Get UTC datetime of P-wave arrival
        p_datetime = snr.get_P_arrival(epi_lon,epi_lat,hypdepth,st_lon,st_lat,6.5,origintime)
        
        # Convert to matplotlib number for numerical analysis 
        p = date2num(p_datetime)
    
        # Calculate SNR
        SNR = snr.get_SNR(st,p)
        SNR_list.append(SNR)
    
    # # Add SNR to dataframe
    # df['SNR'] = SNR_list
    # df.to_csv(flatfile_path,index=False)
    
    # Make Figure
    if data == 'disp':
        title = 'GNSS SNR'
    else:
        title = 'Strong Motion SNR'
    
    figpath = '/Users/tnye/tsuquakes/plots/SNR/mentawai_{data}.png'
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(hypdists,SNR_list)
    for i, label in enumerate(stations):
        ax.annotate(label, (hypdists[i], SNR_list[i]))
    ax.set_xlabel('Hyocentral Dist (km)')
    ax.set_ylabel('SNR')
    ax.set_title(f'{title}')
    ax.set_yscale('log')
    plt.show()
    
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    
    plt.close()


############### Get SNR using all 3 components for GNSS stations ##############

mseed_dir = f'/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'
stations = df['station'][:13]

SNR_list = []
for stn in stations:
    files = sorted(glob(f'{mseed_dir}/{stn}*'))
    tr = read(files[2])[0]
    # avg_data = avg.get_eucl_norm_3comp(read(files[0])[0].data, read(files[1])[0].data, read(files[2])[0].data)
    # avg_data = avg.get_geom_avg_3comp(read(files[0])[0].data, read(files[1])[0].data, read(files[2])[0].data)
    avg_data = avg.get_eucl_norm_2comp(read(files[2])[0].data, read(files[2])[0].data)
    
     # Get station coords
    st_index = np.where(np.array(df['station'])==stn)[0][0]
    st_lon = df['stlon'][st_index]
    st_lat = df['stlat'][st_index]

    # Get UTC datetime of P-wave arrival
    p_datetime = snr.get_P_arrival(epi_lon,epi_lat,hypdepth,st_lon,st_lat,6.5,origintime)
    
    # Convert to matplotlib number for numerical analysis 
    p_arrival = date2num(p_datetime)
    
    # Compute signal
    absolute_difference_function = lambda list_value : abs(list_value - p_arrival)
    sig_start = min(tr.times('matplotlib'), key=absolute_difference_function)

    # Crop data to just include signal 
    sig_data = []
    for i, time in enumerate(tr.times('matplotlib')):
        if time >= sig_start:
            sig_data.append(avg_data[i])
    
    # Crop data to just include noise
    noise_data = []
    for i, time in enumerate(tr.times('matplotlib')):
        if time < sig_start:
            noise_data.append(avg_data)

    # Calculate SNR
    SNR = np.var(sig_data)/np.var(noise_data)
    SNR_list.append(SNR)
    
# Make Figure
title = 'GNSS SNR'

figpath = '/Users/tnye/tsuquakes/plots/SNR/mentawai_{data}.png'
fig = plt.figure()
ax = plt.gca()
ax.scatter(hypdists,SNR_list)
for i, label in enumerate(stations):
    ax.annotate(label, (hypdists[i], SNR_list[i]))
ax.set_xlabel('Hyocentral Dist (km)')
ax.set_ylabel('SNR')
ax.set_title(f'{title} Z-component')
ax.set_yscale('log')
plt.show()

plt.savefig(figpath, bbox_inches='tight', dpi=300)
plt.close()
