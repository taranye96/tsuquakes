#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:20:59 2023

@author: tnye
"""

# Imports
import numpy as np
from obspy import read
from glob import glob
from obspy import UTCDateTime
import tsueqs_main_fns as tmf

fcorner_low = 0.04
order = 4

full_files = sorted(glob('/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/full_files/*.SAC'))

for file in full_files:
    st = read(file)
    
    stn = file.split('/')[-1].split('.')[0]
    channel = file.split('/')[-1].split('.')[1]
    
    start = '2010-10-25T14:42:12.000000Z'
    end = '2010-10-25T14:50:42.000000Z'
    
    # st_filt = tmf.highpass(st,fcorner_high,st[0].stats.sampling_rate,order,zerophase=True)
    
    st[0] = st[0].trim(UTCDateTime(start),UTCDateTime(end))
    
    filepath = '/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/events'
    st.write(f'{filepath}/{stn.upper()}.{channel}.mseed', format='MSEED')


for file in full_files:
    
    st = read(file)
    
    stn = file.split('/')[-1].split('.')[0]
    channel = file.split('/')[-1].split('.')[1]

    st[0] = st[0].trim(endtime=UTCDateTime('2010-10-25T14:39:53.000000Z'))
    
    baseline = tmf.compute_baseline(st,10)
    
    # Get the baseline corrected stream object
    basecorr = tmf.correct_for_baseline(st,baseline)
    
    # filt = tmf.lowpass(basecorr,fcorner_low,st[0].stats.sampling_rate,order,zerophase=True)
    filepath = '/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC'
    basecorr.write(f'{filepath}/noise/{stn.upper()}.{channel}.mseed', format='MSEED')
    
    basecorr.plot(outfile=f'{filepath}/noise_plots/{stn.upper()}.{channel}.png')
    