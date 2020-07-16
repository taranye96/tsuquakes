#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:43:30 2020

@author: tnye
"""

# Standard Library Imports 
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
import geopy.distance
import datetime
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.font_manager import FontProperties

# Read in dataframe and get stats
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')
origintime = pd.to_datetime('2010-10-25T14:42:22')
epi_lon = 100.082
epi_lat = -3.487
hypdepth = df['hypdepth (km)'][0]

################# Observed 
# Stations
disp_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/disp_stations.csv'))
sm_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/sm_stations.csv'))

# GPS files directory 
gps_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'

# Get GPS data
gps_files = np.array(sorted((glob(gps_dir + '/LNNG*'))))

# Get station stream
obs_stream = []
stream_counter = []
for i, file in enumerate(gps_files):
    st = read(file)
    tr = st[0]

    if i == 0:
        stream_counter.append(st)
    else:
        if tr.stats.station == read(gps_files[i-1])[0].stats.station:
            stream_counter[0] = stream_counter[0] + st
        else:
            obs_stream.append(stream_counter)
            stream_counter = []
            stream_counter.append(st)
        
        if i == (len(gps_files) - 1):
            obs_stream.append(stream_counter)

obs_stream = obs_stream[0]

################ Synthetics

# run number
runs = ['run.000000', 'run.000001', 'run.000002'] 
        #'run.000003', 'run.000004',
        # 'run.000005', 'run.000006', 'run.000007', 'run.000008', 'run.000009',
        # 'run.000010', 'run.000011']


for run in runs:
    # GPS files directory 
    disp_dir = '/Users/tnye/FakeQuakes/8min_16cores/disp/waveforms/run.' + run + '/'
    
    # Get disp data
    # disp_files = np.array(sorted((glob(disp_dir + '/SLBU*.sac'))))
    disp_files = np.array(sorted((glob(disp_dir + '*.sac'))))
    
    TODO: fix here on..........

    
    syn_stream = []
    stream_counter = []
    for i, file in enumerate(disp_files):
        st = read(file)
        tr = st[0]
    
        if i == 0:
            stream_counter.append(st)
        else:
            if tr.stats.station == read(disp_files[i-1])[0].stats.station:
                stream_counter[0] = stream_counter[0] + st
            else:
                syn_stream.append(stream_counter)
                stream_counter = []
                stream_counter.append(st)
            
            if i == (len(disp_files) - 1):
                syn_stream.append(stream_counter)
    
    syn_stream = syn_stream[0]
    
    ################## Combine 
    
    # Combine observed and synthetic streams 
    st = []
    for i, tr in enumerate(obs_stream):
        comp = tr + syn_stream[i]
        st.append(comp)
    
    # Station stats
    station = st[0][0].stats.station
    
    # Get traces ready to plot
    obs1 = st[0][0]
    otimes1 = [i for i in obs1.times('matplotlib')]
    odisp1 = obs1.data[:len(otimes1)]
    
    obs2 = st[0][1]
    otimes2 = [i for i in obs2.times('matplotlib')]
    odisp2 = obs2.data[:len(otimes2)]
    
    obs3 = st[0][2]
    otimes3 = [i for i in obs3.times('matplotlib')]
    odisp3 = obs3.data[:len(otimes2)]
    
    syn1 = st[0][3]
    stimes1 = syn1.times('matplotlib')
    sdisp1 = syn1.data[:len(stimes1)]
    
    syn2 = st[0][4]
    stimes2 = syn2.times('matplotlib')
    sdisp2 = syn2.data[:len(stimes2)]
    
    syn3 = st[0][5]
    stimes3 = syn3.times('matplotlib')
    sdisp3 = syn3.data[:len(stimes3)]
    
    # Calc hypdist
    st_index = np.where(disp_stations==obs1.stats.station)[0][0]
    st_lon = disp_stations[st_index][1]
    st_lat = disp_stations[st_index][2]
    epdist = geopy.distance.geodesic((epi_lat, epi_lon), (st_lat, st_lon)).km
    hypdist = np.sqrt(hypdepth**2 + epdist**2)
    
    # Calc P and S wave arrival times in seconds
    p_time = hypdist/6.5
    s_time = hypdist/4.15
    
    # Convert arrival times to UTC DateTime objects
    dp = datetime.timedelta(seconds=p_time)
    ds = datetime.timedelta(seconds=s_time)
    p_arrival = origintime+dp
    s_arrival = origintime+ds
    
    # Set legend font properties
    fontP = FontProperties()
    fontP.set_size('x-small')
    fontP2 = FontProperties()
    fontP2.set_size('x-small')
    
    ######## Plot Waveforms ########
    
    title = 'LNNG Disp Waveforms'
    
    # Plot first channel
    fig, axs = plt.subplots(3)
    axs[0].plot(otimes1, odisp1, 'k-', lw=.8, label='Observed')
    axs[0].plot(stimes1, sdisp2, color='C1', lw=.8, label='Synthetic')
    axs[0].set_xticklabels([])
    axs[0].xaxis.set_major_locator(plt.NullLocator())
    axs[0].legend(bbox_to_anchor = [0.82, 1], labelspacing=.1, framealpha=.8, prop=fontP)
    
    # Plot origin and arrival times and create legend
    o = axs[0].axvline(origintime, color='m', linestyle='--', lw=.5)
    p = axs[0].axvline(p_arrival, color='b', linestyle='--', lw=.5)
    s = axs[0].axvline(s_arrival, color='g', linestyle='--', lw=.5)
    lines = []
    lines.append(o)
    lines.append(p)
    lines.append(s)
    leg = Legend(axs[0], lines, ['origin time', 'p arrival', 's arrival'],
                 loc='upper right', labelspacing=.1, borderaxespad = 0.2,
                 framealpha=.8, prop=fontP2)
    axs[0].add_artist(leg);
    
    # Plot second channel
    axs[1].plot(otimes2, odisp2, 'k-', lw=.8)
    axs[1].plot(stimes2, sdisp2, color='C1', lw=.8)
    axs[1].set_xticklabels([])
    axs[1].xaxis.set_major_locator(plt.NullLocator())
    axs[1].axvline(origintime, color='m', linestyle='--', lw=.5)
    axs[1].axvline(p_arrival, color='b', linestyle='--', lw=.5)
    axs[1].axvline(s_arrival, color='g', linestyle='--', lw=.5)
    
    # Plot third channel
    axs[2].plot(otimes3, odisp3, 'k-', lw=.8)
    axs[2].plot(stimes3, sdisp3, color='C1', lw=.8)
    axs[2].axvline(origintime, color='m', linestyle='--', lw=.5)
    axs[2].axvline(p_arrival, color='b', linestyle='--', lw=.5)
    axs[2].axvline(s_arrival, color='g', linestyle='--', lw=.5)
    plt.xticks(rotation=30)
    plt.xlabel('UTC Time(hr:min:sec)')
    
    # Format whole figure
    plt.subplots_adjust(hspace=0)
    fig.suptitle(title, fontsize=14)
    fig.text(-.03, 0.5, 'Amplitude (m)', va='center', rotation='vertical')
    
    # Save figure
    figpath = '/Users/tnye/tsuquakes/Figures/wf_comp/8min_16cores/gnss_LNNG'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
