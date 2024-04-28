#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:44:46 2020

@author: tnye
"""

###############################################################################
# Script that plots the synthetic acceleration spectra and compares Kappa to the
# slope of the high frequency spectra.  
###############################################################################

# Imports
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
from mtspec import mtspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tsueqs_main_fns as tmf


############################### Set up Data ###################################

# Kappa file
kappa_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/vs30_kappa.csv')

# Directory of acceleration mseed files
sm_dir = '/Users/tnye/FakeQuakes/parameters/test/kappa_test/sm/output/waveforms/mentawai.000000/'

# Gather files 
acc_files = np.array(sorted(glob(sm_dir + '*.bb.HNE.sac')))

# Get earthquake data
eq_table = pd.read_csv('/Users/tnye/tsuquakes/data/misc/events.csv' )
hyplon = eq_table['Longitude'][11]
hyplat = eq_table['Latitude'][11]
hypdepth = eq_table['Depth (km)'][11]

# Get metadata
metadata_file = '/Users/tnye/tsuquakes/data/Mentawai2010/Mentawai2010_sm.chan'
metadata = pd.read_csv(metadata_file, sep='\t', header=0, 
                       names=['net', 'sta', 'loc', 'chan', 'lat', 'lon', 'elev',
                              'samplerate', 'gain', 'units'])
    
# There might be white spaces in the station name, so remove those
metadata.sta = metadata.sta.astype(str)
metadata.sta = metadata.sta.str.replace(' ','')

# Initialize lists
amps = []
freqs = []
stations = []
hypdists = []
kappa = kappa_df.Kappa.values


############################ Loop through files ###############################

for file in acc_files:
    
    # Read in data
    st = read(file)
    tr = st[0]
    station = tr.stats.station
    
    # Get station data 
    station_metadata = metadata[metadata.sta == station].reset_index(drop=True)
    stlon = station_metadata.loc[0].lon
    stlat = station_metadata.loc[0].lat
    stelev = station_metadata.loc[0].elev
    
    # Calculate hypocentral distance 
    hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon,hyplat,hypdepth)
    
    # Compute power spectra
    power_spec, freq =  mtspec(tr.data, delta=tr.stats.delta, time_bandwidth=4, 
                              number_of_tapers=7, nfft=tr.stats.npts, quadratic=True)
    
    # Convert to amplitude spectra
    amp_spec = np.sqrt(power_spec)
    
    # Store data in lists
    amps.append(amp_spec)
    freqs.append(freq)
    stations.append(tr.stats.station)
    hypdists.append(hypdist)
    
    
################################# Make Figure #################################

# Set figure axes
units = 'm/s'
ylim = 7*10**-15, 6*10**-1
xlim = .002, 10
dim = 6,3

# What to sort by
sortby = ([hypdists,'hypdist'], [kappa,'kappa'])

for s in sortby:
    
    # Sort hypdist and get indices
    sort_id = np.argsort(np.argsort(s[0]))
    
    
    # Sort freq and amps based off hypdist
    def sort_list(list1, list2): 
        zipped_pairs = zip(list2, list1) 
        z = [x for _, x in sorted(zipped_pairs)] 
        return z
    
    sort_hypdists = sort_list(hypdists, sort_id)
    sort_freqs = sort_list(freqs, sort_id)
    sort_spec = sort_list(amps, sort_id)
    sort_stns = sort_list(stations, sort_id)
    sort_kappa = sort_list(kappa, sort_id)
    
    # Make lists for high frequencies and spectra 
    hf_freqs = []
    hf_spec = []
    for stn in range(len(sort_freqs)):
        stn_freqs = []
        stn_spec = []
        for i, freq in enumerate(sort_freqs[stn]):
            if freq >= 1.5 and freq <= 10:
                stn_freqs.append(freq)
                stn_spec.append(sort_spec[stn][i])
        hf_freqs.append(stn_freqs)
        hf_spec.append(stn_spec)
    
    # Initialize list for slopes
    sort_slopes = []
    
    # Set up figure
    fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,10))
    k = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            if k+1 <= len(stations):
                
                # Get line for high frequencies
                coefficients = np.polyfit(np.log10(hf_freqs[k]), np.log10(hf_spec[k]), 1)
                sort_slopes.append(coefficients[0])
                polynomial = np.poly1d(coefficients)
                x = np.logspace(sort_freqs[0][0],sort_freqs[0][-1])
                log10_y_fit = polynomial(np.log10(hf_freqs[k]))
    
                axs[i][j].loglog(sort_freqs[k],sort_spec[k],lw=1,c='steelblue',ls='-',label='observed')
                axs[i][j].plot(hf_freqs[k],10**log10_y_fit,c='black')
                axs[i][j].grid(linestyle='--')
                axs[i][j].text(0.6,5E-2,f'Kappa={round(sort_kappa[k],3)}s',
                               transform=axs[i][j].transAxes,size=7)
                axs[i][j].text(0.025,8E-1,'HNE',transform=axs[i][j].transAxes,size=7)
                axs[i][j].text(0.025,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                               transform=axs[i][j].transAxes,size=7)
                axs[i][j].text(0.6,16E-2,f'Slope={round(coefficients[0],4)}km',
                               transform=axs[i][j].transAxes,size=7)
                axs[i][j].set_xlim(xlim)
                axs[i][j].set_ylim(ylim)
                axs[i][j].set_title(sort_stns[k],fontsize=10)
                if i < dim[0]-2:
                    axs[i][j].set_xticklabels([])
                if i == dim[0]-2 and j == 0:
                    axs[i][j].set_xticklabels([])
                if j > 0:
                    axs[i][j].set_yticklabels([])
                k += 1
    fig.text(0.5, 0.005, 'Frequency (Hz)', ha='center')
    fig.text(0.005, 0.5, f'Amp ({units})', va='center', rotation='vertical')
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.delaxes(axs[5][1])
    fig.delaxes(axs[5][2])
    fig.suptitle('Synthetic Spectra: Mentawai.000000', fontsize=12, y=1)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.925, wspace=0.1, hspace=0.35)
    
    # Save figure
    plt.savefig(f'/Users/tnye/tsuquakes/plots/kappa/sortby_{s[1]}_syn.png', dpi=300)
    plt.close()
    

########################### Make Kappa-Slopes Figure ##########################

# Set up Figure
fig = plt.figure(figsize=(10,10))
ax = plt.gca()
color_map = plt.cm.get_cmap('plasma').reversed()
im = ax.scatter(sort_kappa, sort_slopes, c=sort_hypdists, cmap=color_map)
ax.set_xlabel('Kappa(s)')
ax.set_ylabel('High Frequency Spectra Slope')

# Set up colorbar
divider = make_axes_locatable(ax)
cax = divider.new_vertical(size='2.5%', pad=0.8, pack_start=True)
fig.add_axes(cax)
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Hypdist(km)')
cbar.ax.invert_yaxis()

# Save figure
plt.savefig('/Users/tnye/tsuquakes/plots/kappa/kappa_slopes_syn.png', dpi=300)
plt.close()

