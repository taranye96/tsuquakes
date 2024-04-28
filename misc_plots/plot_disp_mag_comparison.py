#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:24:52 2021

@author: tnye
"""

###############################################################################
# Script that makes comparison plots between the observed displacement
# waveforms and synthetic displacement waveforms of various magnitudes. This
# is to help evaluate what magnitude earthquakes to be generating. 
###############################################################################

# Imports 
from obspy import read
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

station_list = ['BSAT', 'BTHL', 'KTET', 'LAIS', 'LNNG', 'MKMK', 'MNNA', 'NGNG',
                'PKRT', 'PRKB', 'SLBU', 'SMGY', 'TRTK']

mag_list = [7.5, 7.8, 7.85, 7.9]

obs_wfs = sorted(glob('/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/*.LXE*'))
waveforms = glob('/Users/tnye/FakeQuakes/parameters/mag_test/output/waveforms/*/*.LYE*')

m7 = []
m75 = []
m8 = []
m85 = []
m9 = []

m7_ruptures = ['mentawai.000000', 'mentawai.000001', 'mentawai.000002', 'mentawai.000003']
m75_ruptures = ['mentawai.000004', 'mentawai.000005', 'mentawai.000006', 'mentawai.000007']
m8_ruptures = ['mentawai.000008', 'mentawai.000009', 'mentawai.000010', 'mentawai.000011']
m85_ruptures = ['mentawai.000012', 'mentawai.000013', 'mentawai.000014', 'mentawai.000015']
m9_ruptures = ['mentawai.000016', 'mentawai.000017', 'mentawai.000008', 'mentawai.000019']
            
            
for w in waveforms:
    rupture = w.split('/')[8]
    if rupture in m7_ruptures:
        m7.append(w)
    elif rupture in m75_ruptures:
        m75.append(w)
    elif rupture in m8_ruptures:
        m8.append(w)
    elif rupture in m85_ruptures:
        m85.append(w)
    elif rupture in m9_ruptures:
        m9.append(w)


# Set up figure
for s, stn in enumerate(station_list):
    fig, axs = plt.subplots(2,3,figsize=(10,10))
    k = 0
    for i in range(2):
        for j in range(3):
            if k != 5:
                if i==0 and j==0:
                    files = m7
                    title = 'M7'
                elif i==0 and j==1:
                    files = m75
                    title = 'M7.5'
                elif i==0 and j==2:
                    files = m8
                    title = 'M8'
                elif i==1 and j==0:
                    files = m85
                    title = 'M8.5'
                elif i==1 and j==1:
                    files = m9
                    title = 'M9'
                # Plot observed
                axs[i][j].plot(read(obs_wfs[s])[0].times('matplotlib'),read(obs_wfs[s])[0].data,lw=1,c='mediumpurple',ls='-',label='observed')
                # Plot synthetic 
                for file in files:
                    if stn in file.split('/')[9].split('.')[0]:
                        axs[i][j].plot(read(file)[0].times('matplotlib'),read(file)[0].data,lw=1,c='darkturquoise',alpha=0.7,ls='-',label='synthetic')
                        axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    axs[i][j].set_title(title,fontsize=10)
                k+=1
    fig.delaxes(axs[1][2])
    fig.text(0.5, 0.005, 'Time', ha='center')
    fig.text(0.005, 0.5, 'Amp (m)', va='center', rotation='vertical')
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.72,0.075), framealpha=None)
    fig.suptitle('Magnitude PGD Comparison', fontsize=12, y=1)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.925, wspace=0.1, hspace=0.35)
    plt.savefig(f'/Users/tnye/tsuquakes/plots/Mw_static-offset/{stn}.png', dpi=300)
    
    plt.close()
    
    
    
    