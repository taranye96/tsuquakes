#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:54:02 2021

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from obspy import read
import datetime
import matplotlib.pyplot as plt

# Parameters
Vs = 3.1 
start = 2
end = 15
main_df = pd.read_csv('/Users/tnye/kappa/data/flatfiles/SNR_5_file.csv')

events = [['Event_2000_01_06_21_38_10', 'BDM', 'BKS', 'POTR'],
          ['Event_2004_05_04_15_04_42', 'JRSC', 'MHC', 'WENL'],
          ['Event_2012_02_06_16_44_58', 'HAST', 'PACP', 'SAO']]

# Loop through events and stations
for i in range(len(events)):
    event = events[i][0]
    stns = np.array(events[i][1:])
    
    # Get origin time
    yyyy, mth, dd, hr, mm, sec = event.split('_')[1:]
    eventid = f'{yyyy}_{mth}_{dd}_{hr}_{mm}_{sec}'
    orig = datetime.datetime(int(yyyy),int(mth),int(dd),int(hr),int(mm),int(sec))
    
    # Loop through stations
    for j, stn in enumerate(stns):
        
        # Set up figure
        fig, ax = plt.subplots()

        # Calculate theoretical S-wave arrival
        stn_ind = np.where((main_df['Name']==stn) & (main_df['OrgT']==f'{yyyy}-{mth}-{dd} {hr}:{mm}:{sec}'))[0][0] 
        network = main_df['Network'][stn_ind]
        rhyp = main_df['rhyp'].iloc[stn_ind]
        
        # Calc S-wave arrival time in seconds after origin time 
        stime = rhyp/Vs
        
        # Calc S-wave arrival time as a datetime object
        Sarriv = orig + datetime.timedelta(0,stime)
        
        # Read in waveforms 
        tr = read(f'/Users/tnye/kappa/data/waveforms/acc/filtered/{event}/{network}_{stn}_HHN_{eventid}.sac')[0]
        cut_record = f'/Users/tnye/kappa/data/waveforms/acc/Vs{Vs}/cut{start}_{end}/{event}/{network}_{stn}_HHN_{eventid}.sac'
        tr_cut = read(cut_record)[0]
        
        # Plot waveforms and arrival times
        ax.plot(tr.times('matplotlib'),tr.data, color='black')
        ax.plot(tr_cut.times('matplotlib'),tr_cut.data, color='C1')
        ax.axvline(x=orig, ls='--', color='red', label='Origin')
        ax.axvline(x=Sarriv, ls='--', color='green', label=f'S-arrival: Vs = {Vs} km/s')
        ax.text(0.9,0.9,f'Station {stn}',size=7, ha='center', va='center', transform=ax.transAxes)
        ax.legend(loc='lower right')
        ax.set_title(f'Event {event}')
        
        # Format to show date on x-axis
        ax.xaxis_date()
        fig.autofmt_xdate()

        plt.savefig(f'/Users/tnye/kappa/plots/cut_files/{event}_{stn}.png', dpi=300)

        plt.close()