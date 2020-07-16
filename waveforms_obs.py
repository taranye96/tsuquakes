#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:32:19 2019

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
sm_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/sm_stations.csv'))
disp_stations = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/disp_stations.csv'))

# Corrected Miniseed files directories
acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr'

# GPS files directory 
gps_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'

# Get corrected mseed files
corr_acc = np.array(sorted((glob(acc_dir + '/*'))))
filt_vel = np.array(sorted((glob(vel_dir + '/*'))))

# Get GPS data
gps_files = np.array(sorted((glob(gps_dir + '/*'))))

# Get arrays of pga, pgv, and hypocentral distance 
pga = np.array(df['pga'])
pgv = np.array(df['pgv'])
pgd = np.array(df_pgd['pgd_meters'])
hypdist = np.array(df['hypdist'])
hypdist_pgd = np.array(df_pgd['hypdist'])

origintime = pd.to_datetime('2010-10-25T14:42:22')

epi_lon = 100.082
epi_lat = -3.487
hypdepth = df['hypdepth'][0]

############################### Acceleration ##################################

##### Create stream of 3 traces for each station #####

streams = []
stream_counter = []
for i, file in enumerate(corr_acc):
    st = read(file)
    tr = st[0]

    if i == 0:
        stream_counter.append(st)
    else:
        if tr.stats.station == read(corr_acc[i-1])[0].stats.station:
            stream_counter[0] = stream_counter[0] + st
        else:
            streams.append(stream_counter)
            stream_counter = []
            stream_counter.append(st)
        
        if i == (len(corr_acc) - 1):
            streams.append(stream_counter)

##### Start calculations #####
           
for st in streams:
    station = st[0][0].stats.station
    channel = st[0][0].stats.channel
    network = st[0][0].stats.network

    tr1 = st[0][0]
    times1 = tr1.times()
    acc1 = tr1.data
    lab1 = tr1.stats.channel

    tr2 = st[0][1]
    times2 = tr2.times()
    acc2 = tr2.data
    lab2 = tr2.stats.channel

    tr3 = st[0][2]
    times3 = tr3.times()
    acc3 = tr3.data
    lab3 = tr3.stats.channel

    title = network + '.' + station + ' acc'

    # Calc hypdist
    st_index = np.where(sm_stations==tr1.stats.station)[0][0]
    st_lon = sm_stations[st_index][1]
    st_lat = sm_stations[st_index][2]
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
    axs[0].plot(tr1.times('matplotlib'), tr1.data, 'k-', lw=.8, label=lab1)
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
    axs[1].plot(tr2.times('matplotlib'), tr2.data, 'k-', lw=.8, label=lab2)
    axs[1].set_xticklabels([])
    axs[1].xaxis.set_major_locator(plt.NullLocator())
    axs[1].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[1].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[1].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[1].legend(loc='upper left', framealpha=.8, prop=fontP)

    # Plot third channel
    axs[2].plot(tr3.times('matplotlib'), tr3.data, 'k-', lw=.8, label=lab3)
    axs[2].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[2].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[2].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[2].legend(loc='upper left', framealpha=.8, prop=fontP)
    plt.xticks(rotation=30)
    plt.xlabel('UTC Time')

    # Format whole figure
    plt.subplots_adjust(hspace=0)
    fig.suptitle(title, fontsize=14)
    fig.text(-.03, 0.5, 'Amplitude (m/s/s)', va='center', rotation='vertical')

    # Save figure
    figpath = '/Users/tnye/tsuquakes/plots/waveforms/acc/' + station + '.wf.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()

################################ Velocity #####################################

##### Create stream of 3 traces for each station #####

streams = []
stream_counter = []
for i, file in enumerate(filt_vel):
    st = read(file)
    tr = st[0]

    if i == 0:
        stream_counter.append(st)
    else:
        if tr.stats.station == read(filt_vel[i-1])[0].stats.station:
            stream_counter[0] = stream_counter[0] + st
        else:
            streams.append(stream_counter)
            stream_counter = []
            stream_counter.append(st)
        
        if i == (len(filt_vel) - 1):
            streams.append(stream_counter)

##### Start calculations #####
           
for st in streams:
    station = st[0][0].stats.station
    channel = st[0][0].stats.channel
    network = st[0][0].stats.network

    tr1 = st[0][0]
    times1 = tr1.times()
    acc1 = tr1.data
    lab1 = tr1.stats.channel

    tr2 = st[0][1]
    times2 = tr2.times()
    acc2 = tr2.data
    lab2 = tr2.stats.channel

    tr3 = st[0][2]
    times3 = tr3.times()
    acc3 = tr3.data
    lab3 = tr3.stats.channel

    title = network + '.' + station + ' vel'

    # Calc hypdist
    st_index = np.where(sm_stations==tr1.stats.station)[0][0]
    st_lon = sm_stations[st_index][1]
    st_lat = sm_stations[st_index][2]
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
    axs[0].plot(tr1.times('matplotlib'), tr1.data, 'k-', lw=.8, label=lab1)
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
    axs[1].plot(tr2.times('matplotlib'), tr2.data, 'k-', lw=.8, label=lab2)
    axs[1].set_xticklabels([])
    axs[1].xaxis.set_major_locator(plt.NullLocator())
    axs[1].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[1].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[1].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[1].legend(loc='upper left', framealpha=.8, prop=fontP)

    # Plot third channel
    axs[2].plot(tr3.times('matplotlib'), tr3.data, 'k-', lw=.8, label=lab3)
    axs[2].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[2].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[2].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[2].legend(loc='upper left', framealpha=.8, prop=fontP)
    plt.xticks(rotation=30)
    plt.xlabel('UTC Time')

    # Format whole figure
    plt.subplots_adjust(hspace=0)
    fig.suptitle(title, fontsize=14)
    fig.text(-.03, 0.5, 'Amplitude (m/s)', va='center', rotation='vertical')

    # Save figure
    figpath = '/Users/tnye/tsuquakes/plots/waveforms/vel/' + station + '.wf.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()

################################ Displacement #################################

##### Create stream of 3 traces for each station #####

streams = []
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
            streams.append(stream_counter)
            stream_counter = []
            stream_counter.append(st)
        
        if i == (len(gps_files) - 1):
            streams.append(stream_counter)

##### Start calculations #####
           
for st in streams:
    station = st[0][0].stats.station
    channel = st[0][0].stats.channel
    network = st[0][0].stats.network

    tr1 = st[0][0]
    times1 = tr1.times('matplotlib')
    acc1 = tr1.data[:len(times1)]
    lab1 = tr1.stats.channel

    tr2 = st[0][1]
    times2 = tr2.times('matplotlib')
    acc2 = tr2.data[:len(times1)]
    lab2 = tr2.stats.channel

    tr3 = st[0][2]
    times3 = tr3.times('matplotlib')
    acc3 = tr3.data[:len(times1)]
    lab3 = tr3.stats.channel

    title = network + '.' + station + ' disp'

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
    axs[0].plot(times1, tr1.data, 'k-', lw=.8, label=lab1)
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
    axs[1].plot(times2, tr2.data, 'k-', lw=.8, label=lab2)
    axs[1].set_xticklabels([])
    axs[1].xaxis.set_major_locator(plt.NullLocator())
    axs[1].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[1].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[1].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[1].legend(loc='upper left', framealpha=.8, prop=fontP)

    # Plot third channel
    axs[2].plot(times3, tr3.data, 'k-', lw=.8, label=lab3)
    axs[2].axvline(origintime, color='r', linestyle='--', lw=1)
    axs[2].axvline(p_arrival, color='b', linestyle='--', lw=1)
    axs[2].axvline(s_arrival, color='g', linestyle='--', lw=1)
    axs[2].legend(loc='upper left', framealpha=.8, prop=fontP)
    plt.xticks(rotation=30)
    plt.xlabel('UTC Time')

    # Format whole figure
    plt.subplots_adjust(hspace=0)
    fig.suptitle(title, fontsize=14)
    fig.text(-.03, 0.5, 'Amplitude (m)', va='center', rotation='vertical')

    # Save figure
    figpath = '/Users/tnye/tsuquakes/plots/waveforms/disp/' + station + '.wf.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()