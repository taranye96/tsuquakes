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
Vs1 = 3.1 
Vs2 = 3.5
Vs3 = 3.2
start = 5
end = 15
main_df = pd.read_csv('/Users/tnye/kappa/data/flatfiles/SNR_5_file.csv')

events = [['Event_2000_01_06_21_38_10', 'BDM', 'BKS', 'POTR'],
          ['Event_2004_05_04_15_04_42', 'JRSC', 'MHC', 'WENL'],
          ['Event_2012_02_06_16_44_58', 'HAST', 'PACP', 'SAO']]

# Set up figure
fig, axs = plt.subplots(3,3,figsize=(10,10))
plt.subplots_adjust(left=0.05, bottom=0.075, right=0.995, top=0.95, wspace=0.25, hspace=0.45)

# Loop through events and stations
for i in range(len(events)):
    event = events[i][0]
    # st1 = events[i][1]
    # st2 = events[i][2]
    # st3 = events[i][3]
    stns = np.array(events[i][1:])
    
    # Get origin time
    yyyy, mth, dd, hr, mm, sec = event.split('_')[1:]
    eventid = f'{yyyy}_{mth}_{dd}_{hr}_{mm}_{sec}'
    orig = datetime.datetime(int(yyyy),int(mth),int(dd),int(hr),int(mm),int(sec))
    
    # Loop through stations
    for j, stn in enumerate(stns):

        # Calculate theoretical S-wave arrival
        stn_ind = np.where((main_df['Name']==stn) & (main_df['OrgT']==f'{yyyy}-{mth}-{dd} {hr}:{mm}:{sec}'))[0][0]    
        rhyp = main_df['rhyp'].iloc[stn_ind]
        
        # Calc S-wave arrival time in seconds after origin time 
        stime1 = rhyp/Vs1
        stime2 = rhyp/Vs2
        stime3 = rhyp/Vs3
        
        # Calc S-wave arrival time as a datetime object
        Sarriv1 = orig + datetime.timedelta(0,stime1)
        Sarriv2 = orig + datetime.timedelta(0,stime2)
        Sarriv3 = orig + datetime.timedelta(0,stime3)
        trim_5_sec = orig + datetime.timedelta(0,stime1-2)
        
        # Read in waveforms 
        tr = read(f'/Users/tnye/kappa/data/waveforms/acc/filtered/{event}/{network}_{stn}_HHN_{eventid}.sac')[0]
        record1 = f'/Users/tnye/kappa/data/waveforms/acc/Vs{Vs1}/cut{start}_{end}/{event}/{network}_{stn}_HHN_{eventid}.sac'
        record2 = f'/Users/tnye/kappa/data/waveforms/acc/Vs{Vs1}/cut{start}_{end}/{event}/{network}_{stn}_HHN_{eventid}.sac'
        tr1 = read(record1)[0]
        tr2 = read(record2)[0]
        
        # Assign subplot number
        m = i
        n = j
        
        
        # Plot waveforms and arrival times
        axs[m][n].plot(tr.times('matplotlib'),tr.data, color='black')
        axs[m][n].axvline(x=orig, ls='--', lw=1, color='red', label='Origin')
        axs[m][n].axvline(x=Sarriv1, ls='--', lw=1, color='green', label=f'S-arrival: Vs = {Vs1} km/s')
        axs[m][n].axvline(x=Sarriv3, ls='--', lw=1, color='yellow', label=f'S-arrival: Vs = {Vs3} km/s')
        axs[m][n].axvline(x=Sarriv2, ls='--', lw=1, color='blue', label=f'S-arrival: Vs = {Vs2} km/s')
        axs[m][n].axvline(x=trim_5_sec, ls='-', lw=1.5, color='C1', label=f'Trim 5s before Vs = {Vs1} km/s')
        axs[m][n].text(0.9,0.9,f'{stn}',size=7, ha='center', va='center', transform=axs[m][n].transAxes)
        
        # Add legend to one subplot
        if [m,n] == [0,2]:
            axs[m][n].legend(prop={'size': 6})
        
        # Format to show date on x-axis
        axs[m][n].xaxis_date()
        
        # Format figure
        axs[m][n].tick_params(axis='x', labelrotation=30, labelsize=8)
        axs[m][n].tick_params(axis='y', labelsize=8)
        axs[m][n].set_title(f'Event {eventid}',fontsize=10)
        
# fig.autofmt_xdate()

plt.savefig('/Users/tnye/kappa/plots/cut_files/misc_events.png', dpi=300)

plt.close()
