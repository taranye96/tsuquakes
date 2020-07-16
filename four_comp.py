#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:15:48 2020

@author: tnye
"""

# Imports
from glob import glob
import numpy as np
import pandas as pd
from obspy import read
import tsueqs_main_fns as tmf
import IM_fns
import matplotlib.pyplot as plt

parameter = 'stress_drop_runs'

projects = ['sd0.3_etal_standard', 'sd1.0_etal_standard', 'sd2.0_etal_standard']
            
# runs = ['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
#         'run.000005', 'run.000006', 'run.000007']
runs = ['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
        'run.000005', 'run.000006', 'run.000007', 'run.000008', 'run.000009',
        'run.000010', 'run.000011', 'run.000012', 'run.000013', 'run.000014',
        'run.000015']

# data_types = ['disp', 'acc', 'vel']
data_types = ['acc', 'vel']

for project in projects:

    for run in runs:
        
        for data in data_types:
    
            param_dir = f'/Users/tnye/FakeQuakes/{parameter}/{project}/' 
            
            ### Set paths and parameters #### 
            
            # Data directory                          
            data_dir = '/Users/tnye/tsuquakes/data'
            
            # Table of earthquake data
            eq_table_path = '/Users/tnye/tsuquakes/data/misc/events.csv'   
            eq_table = pd.read_csv(eq_table_path)
            
            ### Get event data ###
            eventname = 'Mentawai2010'
            country = eq_table['Country'][11]
            hyplon = eq_table['Longitude'][11]
            hyplat = eq_table['Latitude'][11]
            hypdepth = eq_table['Depth (km)'][11]
                
            # Synthetic miniseed dir
            disp_syn_dir = param_dir + f'disp/output/waveforms/{run}/'
            sm_syn_dir = param_dir + f'sm/output/waveforms/{run}/'
            disp_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr'
            acc_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr'
            vel_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr'
            
            # Filtering
            threshold = 0.0
            fcorner = 1/15.                          # Frequency at which to high pass filter
            order = 2                                # Number of poles for filter  
            
            ##################### Data Processing and Calculations ####################
            
            if data == 'disp':
                metadata_file = data_dir + '/' + eventname + '/' + eventname + '_disp.chan'
                syn_files = np.array(sorted(glob(disp_syn_dir + '*.sac')))
                obs_files = np.array(sorted((glob(disp_dir + '/*'))))
                filtering = False
                dtype = 'disp'
            elif data == 'acc':
                metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
                syn_files = np.array(sorted(glob(sm_syn_dir + '*.bb*.sac')))
                obs_files = np.array(sorted((glob(acc_dir + '/*'))))
                filtering = True
                dtype = 'sm'
            elif data == 'vel':
                metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
                syn_files = np.array(sorted(glob(sm_syn_dir + '*.bb*.sac')))
                obs_files = np.array(sorted((glob(vel_dir + '/*'))))
                filtering = False
                dtype = 'sm'
            
            metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                                  names=['net', 'sta', 'loc', 'chan', 'lat',
                                         'lon', 'elev', 'samplerate', 'gain', 'units'])
            
            # There might be white spaces in the station name, so remove those
            metadata.sta = metadata.sta.astype(str)
            metadata.sta = metadata.sta.str.replace(' ','')
            
            syn_freqs = []
            syn_amps = []
            obs_freqs = []
            obs_amps = []
            hypdists = []
            
            
            ############################# Synthetic Spectra ###############################
            
            # Create lists to add station names, channels, and miniseed files to 
            stn_name_list = []
            channel_list = []
            mseed_list = []
            
            # Group all files by station
            N = 3
            stn_files = [syn_files[n:n+N] for n in range(0, len(syn_files), N)]
            
            # Loop over files to get the list of station names, channels, and mseed files 
            for station in stn_files:
                components = []
                mseeds = []
            
                stn_name = station[0].split('/')[-1].split('.')[0]
                stn_name_list.append(stn_name)
                
                for mseed_file in station:
                    if data == 'disp':
                        channel_code = mseed_file.split('/')[-1].split('.')[1]
                    elif data == 'acc':
                        channel_code = mseed_file.split('/')[-1].split('.')[2]
                    elif data == 'vel':
                        channel_code = mseed_file.split('/')[-1].split('.')[2]
            
                    components.append(channel_code)
                    mseeds.append(mseed_file)
            
                channel_list.append(components)
                mseed_list.append(mseeds)
            
            # Loop over the stations for this earthquake, and start to run the computations:
            for i, station in enumerate(stn_name_list):
                
                # Get the instrument (HN or LX) and component (E,N,Z) for this station
                components = []
                for channel in channel_list[i]:
                    components.append(channel[2])
            
                    
                    # Get the metadata for this station from the chan file - put it into
                #     a new dataframe and reset the index so it starts at 0
                if country == 'Japan':
                    station_metadata = metadata[(metadata.net == station[0:2]) &
                                                (metadata.sta == station[2:])].reset_index(drop=True)
                    
                else:
                    station_metadata = metadata[metadata.sta == station].reset_index(drop=True)       # what is going on here
                      
              
                # Pull out the data. Take the first row of the subset dataframe, 
                #    assuming that the gain, etc. is always the same:
                stlon = station_metadata.loc[0].lon
                stlat = station_metadata.loc[0].lat
                stelev = station_metadata.loc[0].elev
                stsamprate = station_metadata.loc[0].samplerate
              
            
                ##################### Start computations ######################        
            
                # Compute the hypocentral distance
                hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon,hyplat,hypdepth)
            
                # List for all spectra at station
                station_spec = []
            
                # Get the components
                components = np.asarray(components)
                
                ## East component 
            
                # Find and read in East component
                E_index = np.where(components=='E')[0][0]
                E_raw = read(mseed_list[i][E_index])
            
                # High pass filter strong motion data at fcorner specified above
                if filtering == True:
                    E_filt = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)
                    E_record = E_filt
                else:
                    E_record = E_raw
            
                
                ### Velocity 
                
                if data == 'vel':
                    # Convert acceleration record to velocity 
                    E_raw = tmf.accel_to_veloc(E_raw)
            
                    # High pass filter velocity 
                    E_record = tmf.highpass(E_raw,fcorner,stsamprate,order,zerophase=True)
            
            
                ####################### Intensity Measures ###########W########
            
                # Calc Spectra
                E_spec_data, freqE, ampE = IM_fns.calc_spectra(E_record, dtype)
                
                # Plot spectra
                tr = E_record[0]
                station = tr.stats.station
                
                        
                # Define label 
                component = 'E'
                label = channel + component 
                
                syn_freqs.append(freqE.tolist())
                syn_amps.append(ampE.tolist())
                hypdists.append(hypdist)
                
            
            ################################### Observed ###########3######################
            
            # Create lists to add station names, channels, and miniseed files to 
            stn_name_list = []
            channel_list = []
            mseed_list = []
            
            # Group all files by station
            N = 3
            stn_files = [obs_files[n:n+N] for n in range(0, len(obs_files), N)]
            
            # Loop over files to get the list of station names, channels, and mseed files 
            for station in stn_files:
                components = []
                mseeds = []
            
                stn_name = station[0].split('.')[0].split('/')[-1]
                if stn_name != 'SISI':
                    stn_name_list.append(stn_name)
                    
                    for mseed_file in station:
                        channel_code = mseed_file.split('/')[-1].split('.')[1]
                        components.append(channel_code)
                        mseeds.append(mseed_file)
                
                    channel_list.append(components)
                    mseed_list.append(mseeds)
            
            # Loop over the stations for this earthquake, and start to run the computations:
            for i, station in enumerate(stn_name_list):
                    
                # Get the components for this station (E, N, and Z):
                components = []
                for channel in channel_list[i]:
                    components.append(channel[2])
                    
                # Get the metadata for this station from the chan file - put it into
                #     a new dataframe and reset the index so it starts at 0
                if country == 'Japan':
                    station_metadata = metadata[(metadata.net == station[0:2]) & (metadata.sta == station[2:])].reset_index(drop=True)
                    
                else:
                    station_metadata = metadata[metadata.sta == station].reset_index(drop=True)       # what is going on here
            
                            
                # Pull out the data. Take the first row of the subset dataframe, 
                #    assuming that the gain, etc. is always the same:
                stnetwork = station_metadata.loc[0].net
                stlon = station_metadata.loc[0].lon
                stlat = station_metadata.loc[0].lat
                stelev = station_metadata.loc[0].elev
                stsamprate = station_metadata.loc[0].samplerate
                stgain = station_metadata.loc[0].gain
            
            
                ######################### Start computations ##########################       
                
                # List for all spectra at station
                station_spec = []
            
                # Get the components
                components = np.asarray(components)
                    
                # Get index for E component 
                E_index = np.where(components=='E')[0][0]
                # Read file into a stream object
                E_record = read(mseed_list[i][E_index])
            
            
                ####################### IMs ########################
            
                # Calc Spectra
                E_spec_data, freqE, ampE = IM_fns.calc_spectra(E_record, dtype)
                
                # Append spectra to observed list
                obs_freqs.append(freqE.tolist())
                obs_amps.append(ampE.tolist())
                
            
            ################################# Make Figure #################################
            
            # Set figure axes
            if data == 'disp':
                   units = 'm*s'
                   channel = 'LX' 
                   ylim = 10**-4, 6*10**-1
                   xlim = 2*10**-3, 5*10**-1
                   dim = 5,3
                   figsize = 10,20
            elif data == 'acc':
                   units = 'm/s'
                   channel = 'HN'
                   ylim = 6*10**-15, 6*10**-1
                   xlim = .002, 10
                   dim = 6,3
                   figsize = 10,30
            elif data == 'vel':
                   units = 'm'
                   channel = 'HN'
                   ylim = 6*10**-15, 8*10**-2
                   xlim = .002, 10
                   dim = 6,3
                   figsize = 10,30
            
            # Sort hypdist and get indices
            sort_id = np.argsort(hypdists)
            sort_hypdists = np.sort(hypdists)
            
            # Sort freq and amps based off hypdist
            def sort_list(list1, list2): 
              
                zipped_pairs = zip(list2, list1) 
              
                z = [x for _, x in sorted(zipped_pairs)] 
                  
                return z 
            sort_syn_freqs = sort_list(syn_freqs, sort_id)
            sort_syn_amps = sort_list(syn_amps, sort_id)
            sort_obs_freqs = sort_list(obs_freqs, sort_id)
            sort_obs_amps = sort_list(obs_amps, sort_id)
            sort_stn_name = sort_list(stn_name_list, sort_id)
            
            if data == 'disp':
                # Set up figure
                fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,12))
                k = 0
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        if k+1 <= len(stn_name_list):
                            axs[i][j].loglog(sort_syn_freqs[k],sort_syn_amps[k],lw=1,ls='--',label='synthetic')
                            axs[i][j].loglog(sort_obs_freqs[k],sort_obs_amps[k],lw=1,ls='-',label='observed')
                            axs[i][j].grid(linestyle='--')
                            axs[i][j].set_xlim(xlim)
                            axs[i][j].set_ylim(ylim)
                            axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                            axs[i][j].text(0.65,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                                           transform=axs[i][j].transAxes,size=7)
                            if i < dim[0]-2:
                                # plt.setp(axs[i][j], xticks=[])
                                axs[i][j].set_xticklabels([])
                            if i == dim[0]-2 and j == 0:
                                # plt.setp(axs[i][j], xticks=[])
                                axs[i][j].set_xticklabels([])
                            if j > 0:
                                # plt.setp(axs[i][j], yticks=[])
                                axs[i][j].set_yticklabels([])
                            k += 1
                fig.text(0.51, 0.005, 'Frequency (Hz)', ha='center')
                fig.text(0.005, 0.5, f'Amp ({units})', va='center', rotation='vertical')
                axs_list = []
                handles, labels = axs[0][0].get_legend_handles_labels()
                fig.legend(handles, labels, loc=(0.8,0.075), framealpha=None)
                if data == 'disp':
                        fig.delaxes(axs[4][1])
                        fig.delaxes(axs[4][2])
                else:
                        fig.delaxes(axs[5][1])
                        fig.delaxes(axs[5][2])
                fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=1)
                fig.text(0.45, 0.125, (r"$\bf{" + 'Project:' + "}$" + '' + project))
                fig.text(0.45, 0.1, (r'$\bf{' + 'Run:' + '}$' + '' + run))
                fig.text(0.45, 0.075, (r'$\bf{' + 'DataType:' '}$' + '' + data))
                plt.tight_layout()
        
                plt.savefig(f'/Users/tnye/tsuquakes/plots/fourier_comp/{project}/{run}/{data}.png', dpi=300)
                plt.close()
                
            else:
                # Set up figure
                fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,15))
                k = 0
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        if k+1 <= len(stn_name_list):
                            axs[i][j].loglog(sort_syn_freqs[k],sort_syn_amps[k],lw=1,ls='--',label='synthetic')
                            axs[i][j].loglog(sort_obs_freqs[k],sort_obs_amps[k],lw=1,ls='-',label='observed')
                            axs[i][j].grid(linestyle='--')
                            axs[i][j].text(0.7,5E-13,f'Hypdist={int(sort_hypdists[k])}km',size=7)
                            axs[i][j].set_xlim(xlim)
                            axs[i][j].set_ylim(ylim)
                            axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                            if i < dim[0]-2:
                                # plt.setp(axs[i][j], xticks=[])
                                axs[i][j].set_xticklabels([])
                            if i == dim[0]-2 and j == 0:
                                # plt.setp(axs[i][j], xticks=[])
                                axs[i][j].set_xticklabels([])
                            if j > 0:
                                # plt.setp(axs[i][j], yticks=[])
                                axs[i][j].set_yticklabels([])
                            k += 1
                fig.text(0.51, 0.005, 'Frequency (Hz)', ha='center')
                fig.text(0.005, 0.5, f'Amp ({units})', va='center', rotation='vertical')
                axs_list = []
                handles, labels = axs[0][0].get_legend_handles_labels()
                fig.legend(handles, labels, loc=(0.8,0.06), framealpha=None)
                if data == 'disp':
                        fig.delaxes(axs[4][1])
                        fig.delaxes(axs[4][2])
                else:
                        fig.delaxes(axs[5][1])
                        fig.delaxes(axs[5][2])
                fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=1)
                fig.text(0.45, 0.115, (r"$\bf{" + 'Project:' + "}$" + '' + project))
                fig.text(0.45, 0.09, (r'$\bf{' + 'Run:' + '}$' + '' + run))
                fig.text(0.45, 0.065, (r'$\bf{' + 'DataType:' '}$' + '' + data))
                plt.tight_layout()
        
                plt.savefig(f'/Users/tnye/tsuquakes/plots/fourier_comp/{parameter}/{project}/{run}/{data}.png', dpi=300)
                plt.close()