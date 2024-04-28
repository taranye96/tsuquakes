#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 07:07:20 2020

@author: tnye
"""

###############################################################################
# Script that plots the acceleration spectra residuals for each strong motion 
# seismic staton. The slopes of the high frequency residuals are also saved to
# a csv file.  This is used to evaluate Kappa at these stations and modify
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

# Define parameters 
# parameters = ['standard', 'stress_drop']
# projects = ['std', 'sd2.0', 'sd1.0', 'sd0.3', 'sd0.1']
parameters = ['test']
projects = ['kappa_test']

# Initialize list for high frequency slopes for all projects
all_slopes_list = []

# Loop through parameters:
for parameter in parameters:
    
    # Loop through projects:
    for project in projects:
        
        # See if parameter/project path exists
        try:
            
            # Initialize list for high frequency slopes for this project
            slopes_list = []
            
            # Path to flatfile of residuals
            flatfile_path = f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/residuals/{project}_lnres.csv'
            
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
                
            # Loop through stations
            for stn in unique_stns:
                
                # Indexes where station appears
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
               
                
                ######################## Make Figure ######################
                
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
                
                # Add slopes to list
                slopes_list.append(coefficients[0])
            
            # Append project's slopes to the full slopes list
            all_slopes_list.append(slopes_list)
                
        except:
            continue

# Sepearate out project lists
sd5_slopes,sd2_slopes,sd1_slopes,sd_3_slopes,sd_1_slopes = [i for i in all_slopes_list]
        
# Crete dictionary 
dataset_dict = {'Station':unique_stns,'5.0 MPa':sd5_slopes, '2.0 MPa':sd2_slopes,
                '1.0 MPa':sd1_slopes, '0.3 MPa':sd_3_slopes, '0.1 MPa':sd_1_slopes}
df = pd.DataFrame(data=dataset_dict)                    

# Save df to csv 
df.to_csv('/Users/tnye/tsuquakes/flatfiles/station_res_slopes.csv',index=False)
