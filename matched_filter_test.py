#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:04:49 2023

@author: tnye
"""

# Imports
from glob import glob
import numpy as np
import pandas as pd
from obspy import read
from os import path, makedirs
import matched_filter_fns as filt

home = '/Users/tnye/FakeQuakes/simulations/final_suite/'
project_name = 'standard'

rupture_logs = sorted(glob(f'{home}{project_name}/output/ruptures/*.log'))

order = 4

# Make folder to save broadband waveforms
if not path.exists(f'{home}{project_name}/matched_filter'):
    makedirs(f'{home}{project_name}/matched_filter')

for log in rupture_logs:
    
    run = log.split('/')[-1].strip('.log')
    
    # Gather low frequency and high frequency waveforms
    # lf_files = sorted(glob(f'{home}{project_name}/output/waveforms/{run}/*LY*',recursive=True))
    hf_files = sorted(glob(f'{home}{project_name}/output/waveforms/{run}/*HN*.mpi*',recursive=True))

    # Group all files by station
    N = 3
    # grouped_lf_files = [lf_files[n:n+N] for n in range(0, len(lf_files), N)]
    grouped_hf_files = [hf_files[n:n+N] for n in range(0, len(hf_files), N)]
    
    fc = 0.998
    
    # Loop over stations
    for i_stn in range(len(grouped_hf_files)):
        
        stn = grouped_hf_files[i_stn][0].split('/')[-1].split('.')[0]
        
        stn_lf_files = sorted(glob(f'{home}{project_name}/output/waveforms/{run}/{stn}.LY*'))
        stn_hf_files = grouped_hf_files[i_stn]
        
        # Loop over components
        for j_comp in range(len(stn_lf_files)):
            lf_st = read(stn_lf_files[j_comp])
            hf_st = read(stn_hf_files[j_comp])
            
            # Perform matched filter
            bb_st = filt.matched_filter(home, project_name, run, lf_st, hf_st, 0.998, fc, order, zero_phase=True)
            
            # Get record info
            stn = bb_st[0].stats.station
            comp = stn_lf_files[j_comp].split('/')[-1].split('.')[1][2] 
            
            # Make folder to save broadband waveforms
            if not path.exists(f'{home}{project_name}/matched_filter/{run}/'):
                makedirs(f'{home}{project_name}/matched_filter/{run}/')
            
            # Save broadband stream
            bb_st.write(f'{home}{project_name}/matched_filter/{run}/{stn}.bb.HN{comp}.mseed',format='MSEED')
  