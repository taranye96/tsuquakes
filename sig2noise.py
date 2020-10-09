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

# Read in dataframe
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')

# Event parameters
origintime = pd.to_datetime('2010-10-25T14:42:22')
epi_lon = 100.082
epi_lat = -3.487
hypdepth = df['hypdepth'][0]

# Path to send flatfile
flatfile_path = '/Users/tnye/tsuquakes/flatfiles/gnss_SNR.csv'

# Corrected horizontal average mseed file directories 
disp_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_disp'
acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_acc'

data_types = ['disp', 'sm']

for data in data_types:
    
    # List of stations
    stations = np.array(pd.read_csv(f'/Users/tnye/tsuquakes/data/stations/{data}_stations.csv'))
    
    # Corrected horizontal average mseed file directories 
    mseed_dir = f'/Users/tnye/tsuquakes/data/Mentawai2010/avg_{data}'
    
    # Corrected mseed files
    files = np.array(sorted((glob(mseed_dir + '/*'))))

    # Get arrays of hypocentral distance 
    if data == 'disp':
        hypdists = np.array(df['hypdist'][:13])
    else:
        hypdists = np.array(df['hypdist'][13:])
    
    # Loop through stations 
    SNR_list = []
    for file in files:
        st = read(file)
        tr = st[0]
    
        # Get station coords
        st_index = np.where(stations==tr.stats.station)[0][0]
        st_lon = stations[st_index][1]
        st_lat = stations[st_index][2]
    
        # Get UTC datetime of P-wave arrival
        p_datetime = snr.get_P_arrival(epi_lon,epi_lat,hypdepth,st_lon,st_lat,6.5,origintime)
        
        # Convert to matplotlib number for numerical analysis 
        p = date2num(p_datetime)
    
        # Calculate SNR
        SNR = snr.get_SNR(st,p)
        SNR_list.append(SNR)
    
    # Add SNR to dataframe
    df['SNR'] = SNR_list
    df.to_csv(flatfile_path,index=False)
    
    # Make Figure
    if data == 'disp':
        title = 'Displacement SNR'
    else:
        title = 'Acceleration SNR'
    
    figpath = '/Users/tnye/tsuquakes/plots/SNR/mentawai_{data}.png'
    plt.scatter(hypdists,np.log10(SNR_list))
    plt.xlabel('Hyocentral Dist (km)')
    plt.ylabel('log10 SNR')
    plt.title(f'{title}')
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    
    plt.close()
        
