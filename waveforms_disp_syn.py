#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:35:18 2020

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

# Read in dataframes
df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/pga_pgv.csv')
df_pgd = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/pgd_flatfile.csv')
gm_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/sm_stations.csv'))
disp_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/disp_stations.csv'))

# Corrected Miniseed files directories
acc_dir = ''
vel_dir = ''

# run number
run = '000000'

# GPS files directory 
disp_dir = '/Users/tnye/FakeQuakes/Mentawai_timefix/output/waveforms/run.' + run

# Define path to save waveforms
figpath = '/Users/tnye/FakeQuakes/Mentawai_timefix/plots/waveforms/disp/run.' + run + '/'

# Get disp data
disp_files = np.array(sorted((glob(disp_dir + '/*.sac'))))

origintime = pd.to_datetime('2010-10-25T14:42:22')

epi_lon = 100.082
epi_lat = -3.487
hypdepth = df['hypdepth'][0]

################################ Displacement #################################

##### Create stream of 3 traces for each station #####

streams = []
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
            streams.append(stream_counter)
            stream_counter = []
            stream_counter.append(st)
        
        if i == (len(disp_files) - 1):
            streams.append(stream_counter)

##### Start calculations #####
           
for st in streams:
    station = st[0][0].stats.station
    channel = st[0][0].stats.channel
    network = st[0][0].stats.network

    tr1 = st[0][0]
    times1 = tr1.times()
    acc1 = tr1.data
    lab1 = 'LXE'

    tr2 = st[1][0]
    times2 = tr2.times()
    acc2 = tr2.data
    lab2 = 'LXN'

    tr3 = st[2][0]
    times3 = tr3.times()
    acc3 = tr3.data
    lab3 = 'LXZ'

    title = 'Station P316' 

    # Calc hypdist
    st_index = np.where(disp_stations==tr1.stats.station)[0][0]
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
    fontP.set_size('small')
    fontP2 = FontProperties()
    fontP2.set_size('xx-small')

    ######## Plot Waveforms ########

    # Plot first channel
    fig, axs = plt.subplots(3)
    axs[0].plot(tr1.times("matplotlib"), tr1.data, 'k-', lw=.8, label=lab1)
    axs[0].set_xticklabels([])
    axs[0].xaxis.set_major_locator(plt.NullLocator())
    axs[0].legend(loc='upper left', labelspacing=.1, framealpha=.8, prop=fontP)

    # Plot origin and arrival times and create legend
    o = axs[0].axvline(origintime, color='r', linestyle='--', lw=1)
    p = axs[0].axvline(p_arrival, color='b', linestyle='--', lw=1)
    s = axs[0].axvline(s_arrival, color='g', linestyle='--', lw=1)
    lines = []
    lines.append(o)
    lines.append(p)
    lines.append(s)
    leg = Legend(axs[0], lines, ['origin time', 'p arrival', 's arrival'],
                 loc='upper right', labelspacing=.1, borderaxespad = 0.2,
                 framealpha=.8, prop=fontP2)
    axs[0].add_artist(leg);

    # Plot second channel
    axs[1].plot(tr2.times("matplotlib"), tr2.data, 'k-', lw=.8, label=lab2)
    axs[1].set_xticklabels([])
    axs[1].xaxis.set_major_locator(plt.NullLocator())
    axs[1].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[1].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[1].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[1].legend(loc='upper left', framealpha=.8, prop=fontP)

    # Plot third channel
    axs[2].plot(tr3.times("matplotlib"), tr3.data, 'k-', lw=.8, label=lab3)
    axs[2].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[2].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[2].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[2].legend(loc='upper left', framealpha=.8, prop=fontP)
    plt.xticks(rotation=30)
    plt.xlabel('Matplotlib Time')

    # Format whole figure
    plt.subplots_adjust(hspace=0)
    fig.suptitle(title, fontsize=14)
    fig.text(-.01, 0.5, 'Amplitude (m)', va='center', rotation='vertical')

    # Save figure
    figpath2 = figpath + station + '.wf.png'
    plt.savefig(figpath2, bbox_inches='tight', dpi=300)
    plt.close()

