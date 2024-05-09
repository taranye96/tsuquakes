#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:29:25 2024

@author: tnye
"""

import numpy as np
from glob import glob
from obspy import read, UTCDateTime
from mtspec import mtspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

fig, ax = plt.subplots(1,1,figsize=(5.5,4))

acc_files = sorted(glob('/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr/*.HNE.mseed'))


for file in acc_files:
    st = read(file)
    st[0] = st[0].trim(starttime=UTCDateTime("2010-10-25T14:47:02"))
    data = st[0].data
    
    amp2, freq =  mtspec(data, delta=st[0].stats.delta, time_bandwidth=4, number_of_tapers=7, quadratic=True)
    amp = np.sqrt(amp2)

    ax.loglog(freq,amp,c='k',alpha=0.5)


disp_files = sorted(glob('/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/events/*.LXE.mseed'))

for file in disp_files:
    st = read(file)
    st[0] = st[0].trim(starttime=UTCDateTime("2010-10-25T14:47:02"))
    data = st[0].data
    
    amp2, freq =  mtspec(data, delta=st[0].stats.delta, time_bandwidth=4, number_of_tapers=7, quadratic=True)
    amp = np.sqrt(amp2)

    ax.loglog(freq,amp,c='blue',alpha=0.5)

ax.set_xlim(0.02,1)
ax.set_ylim(ymin=1e-9)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Fourier Ampltidue')
ax.grid(alpha=0.5)
acc = Line2D([0],[0], color='k')
disp = Line2D([0],[0], color='blue')
ax.legend([acc, disp],
          ['Strong Motion Acceleration','GNSS Displacement'])

plt.subplots_adjust(bottom=0.125,right=0.97,top=0.97)
plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/S_ringing-FAS.png',dpi=300)

