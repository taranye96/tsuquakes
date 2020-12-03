#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 07:07:20 2020

@author: tnye
"""

###############################################################################
# Script that plots the acceleration spectra residuals for each strong motion 
# seismic staton. This is used to evaluate Kappa at these stations and modify
# FakeQuakes to match this.    
###############################################################################

# Imports
import numpy as np
import pandas as pd
from math import log10, floor
from itertools import repeat
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Which stress drop 
parameter = 'stress_drop'
project = 'sd0.1'

# Path to flatfile of residuals
flatfile_path = f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/residuals/{project}_lnres.csv'

# Path to save log file
log_file = f'/Users/tnye/tsuquakes/data/res_slope_logs/{project}_slopes.log'

# Acceleration spectra bin edges
bin_edges = [0.004, 0.00592, 0.00875, 0.01293, 0.01913, 0.02828,
                         0.04183, 0.06185, 0.09146, 0.13525, 0.20000, 0.29575,
                         0.43734, 0.64673, 0.95635, 1.41421, 2.09128, 3.09249,
                         4.57305, 6.76243, 10.00000]

# Initialize lists for all the bin means and higher frequency bin means 
bin_means = []
hf_bin_means = []

# Loop through bins to get means 
for i in range(len(bin_edges)):
    if i != 0:
        # Visual mean between two values in logspace 
        mean = np.exp((np.log10(bin_edges[i])+np.log10(bin_edges[i-1]))/2)
        bin_means.append(mean)
        
        # High frequency means used to calculate slope 
        if mean > 1.4:
            hf_bin_means.append(mean)

# Initialize lists for sm stations and spectra residuals from the main flatfile
sm_stns = []
spec_res = []

# Read in flatfile
flatfile = pd.read_csv(flatfile_path)
stn_types = np.array(flatfile['station_type'])
stns = np.array(flatfile['station'])
acc_bins = np.array(flatfile.iloc[:,84:104])

# Loop through station types
for i, stn_type in enumerate(stn_types):
    # Only append stations and residuals if it is a sm station
    if stn_type == 'SM':
        sm_stns.append(stns[i])
        spec_res.append(acc_bins[i])

# Get list of unique stations
unique_stns = np.unique(sm_stns)

# Set up log to save high frequency slopes to
try:
    f = open(log_file,'x')
except:
    f = open(log_file,'w')
finally:
    
    # Loop through stations
    for stn in unique_stns:
        stn_ind = np.where(np.array(sm_stns)==stn)[0]
       
        # Initialize lists for bin mean and residuals
        bin_list = []
        spec_res_list = []
        hf_spec_res = []
        
        # Loop through bins
        for i, mean in enumerate(bin_means):
            
            # List for residuals for specific bin and station 
            bin_res = np.array([])
            
            # Loop through instances of that station
            for j in stn_ind:
                    
                    # Get residual for that bin and station instance
                    bin_res = np.append(bin_res,spec_res[j][i])
            
            avg_bin_res = np.mean(bin_res)
                    
            # bin_list.extend(repeat(str(bin_means[i]),len(bin_res)))
            bin_list.extend(repeat(bin_means[i],len(bin_res)))
            spec_res_list.append(bin_res)
            
            if mean > 1.4:
                hf_spec_res.append(avg_bin_res)
        
        # Create dataframe for station
        spec_res_list = [val for sublist in spec_res_list for val in sublist]
        spec_data = {'x': bin_list, 'y':spec_res_list}
        spec_df = pd.DataFrame(data=spec_data)    
       
        ############################# Make Figure #################################
        
        # Set figsize
        plt.figure(figsize=(10, 10))
        
        # Set box colors 
        u = np.unique(spec_df['x'])
        color=plt.cm.Spectral(np.linspace(.1,.8, len(u)))
        
        # Determine width of boxes 
        w = 0.06
        width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
        
        # Make boxplot
        for c, (name, group) in zip(color,spec_df.groupby("x")):
            bp = plt.boxplot(group.y.values, positions=[name], widths=width(np.unique(group.x.values),w), showfliers=False, patch_artist=True)
            bp['boxes'][0].set_facecolor(c)
        
        # Plot slope of high frequency residuals 
        coefficients = np.polyfit(np.log10(hf_bin_means), hf_spec_res, 1)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(np.log10(hf_bin_means))
        plt.plot(hf_bin_means,y_fit)
        
        # Set up text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(1, 0, f'Slope = {coefficients[0]}', fontsize=8, bbox=props)
        
        # Figure Properties 
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ln Residuals (m/s)')
        plt.title(f'{stn} {project} acc residuals')
        plt.savefig(f'/Users/tnye/tsuquakes/plots/station_spec_res/{stn}_{project}.png', dpi=300)
        plt.close()
        
        # Save slopes to log file
        f.write(f'{stn} {coefficients[0]}\n')
    
    f.close()
    
