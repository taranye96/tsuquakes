#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 17:42:11 2023

@author: tnye
"""

# Imports
import numpy as np
from glob import glob
from obspy import read, UTCDateTime
from mtspec import mtspec
import matplotlib.pyplot as plt

exclusions = ['/CGJI','/CNJI','/LASI','/MLSI','/PPBI','/PSI','/TSI'] # far stations containing surface waves
obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/average/rotd50/vel/*')) \
         if not any(exclude in file for exclude in exclusions)]

for file in obs_files:
    st = read(file)
    tr = st[0].trim(starttime=UTCDateTime("2010-10-25T14:47:02"))
    
    amp_squared, freq =  mtspec(tr.data, delta=tr.stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=tr.stats.npts, quadratic=True)
    amp = np.sqrt(amp_squared)
    
    # Plot
    plt.figure()
    plt.loglog(freq, amp)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(tr.stats.station)
    plt.xlim(xmax=10)