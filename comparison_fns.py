#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:52:17 2020

@author: tnye
"""

###############################################################################
# Module with functions used to make comparison plots between observed and 
# synthetic data. 
###############################################################################


def plot_spec_comp(syn_freqs, syn_spec, obs_freqs, obs_spec, stn_list, hypdists, data_type, home, parameter, project, run):
    """
    Makes a figure comparing observed spectra to synthetic spectra with
    subplots for each station. 

    Inputs:
        syn_freqs(list): Array of list of frequencies obtained when computing
                         Fourier spectra of the synthetics for each station
        syn_amps(list): Array of list of amplitudes obtained when computing
                        Fourier spectra of the synthetics for each station
        obs_freqs(list): Array of list of frequencies obtained when computing
                         Fourier spectra of the observed data for each station
        obs_amps(list): Array of list of amplitudes obtained when computing
        stn_list(list): List of station names
                        Fourier spectra of the observed data for each station
        hypdists(list): List of hypocentral distances correlating with the 
                        stations used to get spectra
        data_type(str): Type of data
                            Options:
                                disp
                                acc
                                vel
        home(str): Base of path to save plots.
        parameter(str): Name of parameter folder.
        project(str): Name of simulation project.
        run(str): Synthetics run number.

    Output:
            Just saves the plots to ther respective directories. 
            
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from os import path, makedirs
            
    # Set figure axes
    if data_type == 'disp':
       units = 'm*s'
       ylim = 10**-4, 7*10**-1
       xlim = 2*10**-3, 5*10**-1
       dim = 5,3
    elif data_type == 'acc':
       units = 'm/s'
       ylim = 7*10**-15, 6*10**-1
       xlim = .002, 10
       dim = 6,3
    elif data_type == 'vel':
       units = 'm'
       ylim = 6*10**-15, 8*10**-2
       xlim = .002, 10
       dim = 6,3
    
    # Sort hypdist and get indices
    sort_id = np.argsort(hypdists)
    sort_hypdists = np.sort(hypdists)
    
    # Sort freq and amps based off hypdist
    def sort_list(list1, list2): 
        zipped_pairs = zip(list2, list1) 
        z = [x for _, x in sorted(zipped_pairs)] 
        return z
    
    sort_syn_freqs = sort_list(syn_freqs, sort_id)
    sort_syn_spec = sort_list(syn_spec, sort_id)
    sort_obs_freqs = sort_list(obs_freqs, sort_id)
    sort_obs_spec = sort_list(obs_spec, sort_id)
    sort_stn_name = sort_list(stn_list, sort_id)
    
    if data_type == 'disp':
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,10))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=1,c='C1',ls='-',label='synthetic')
                    axs[i][j].loglog(sort_obs_freqs[k],sort_obs_spec[k],lw=1,c='steelblue',ls='-',label='observed')
                    axs[i][j].grid(linestyle='--')
                    axs[i][j].set_xlim(xlim)
                    axs[i][j].set_ylim(ylim)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    axs[i][j].text(0.025,5E-2,'E',transform=axs[i][j].transAxes,size=7)
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
        fig.text(0.5, 0.005, 'Frequency (Hz)', ha='center')
        fig.text(0.005, 0.5, f'Amp ({units})', va='center', rotation='vertical')
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.72,0.075), framealpha=None)
        if data_type == 'disp':
                fig.delaxes(axs[4][1])
                fig.delaxes(axs[4][2])
        else:
                fig.delaxes(axs[5][1])
                fig.delaxes(axs[5][2])
        fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=1)
        fig.text(0.445, 0.125, (r"$\bf{" + 'Project:' + "}$" + '' + project))
        fig.text(0.445, 0.1, (r'$\bf{' + 'Run:' + '}$' + '' + run))
        fig.text(0.445, 0.075, (r'$\bf{' + 'DataType:' '}$' + '' + data_type))
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.925, wspace=0.1, hspace=0.35)
        
        if not path.exists(f'{home}/parameters/{parameter}/{project}/plots/comparison/spectra'):
            makedirs(f'{home}/parameters/{parameter}/{project}/plots/comparison/spectra')
            
        plt.savefig(f'{home}/parameters/{parameter}/{project}/plots/comparison/spectra/{run}_{data_type}.png', dpi=300)
        plt.close()
    
    else:
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,10))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=1,c='C1',ls='-',label='synthetic')
                    axs[i][j].loglog(sort_obs_freqs[k],sort_obs_spec[k],lw=1,c='steelblue',ls='-',label='observed')
                    axs[i][j].grid(linestyle='--')
                    axs[i][j].text(0.025,5E-2,'E',transform=axs[i][j].transAxes,size=7)
                    axs[i][j].text(0.6,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                                   transform=axs[i][j].transAxes,size=7)
                    axs[i][j].set_xlim(xlim)
                    axs[i][j].set_ylim(ylim)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
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
        fig.legend(handles, labels, loc=(0.725,0.06), framealpha=None)
        if data_type == 'disp':
                fig.delaxes(axs[4][1])
                fig.delaxes(axs[4][2])
        else:
                fig.delaxes(axs[5][1])
                fig.delaxes(axs[5][2])
        fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=1)
        fig.text(0.445, 0.115, (r"$\bf{" + 'Project:' + "}$" + '' + project))
        fig.text(0.445, 0.09, (r'$\bf{' + 'Run:' + '}$' + '' + run))
        fig.text(0.445, 0.065, (r'$\bf{' + 'DataType:' '}$' + '' + data_type))
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.925, wspace=0.1, hspace=0.35)
        
        if not path.exists(f'{home}/parameters/{parameter}/{project}/plots/comparison/spectra'):
            makedirs(f'{home}/parameters/{parameter}/{project}/plots/comparison/spectra')
        
        plt.savefig(f'{home}/parameters/{parameter}/{project}/plots/comparison/spectra/{run}_{data_type}.png', dpi=300)
        plt.close()
        
        
def plot_wf_comp(syn_times, syn_amps, obs_times, obs_amps, stn_list, hypdists, data_type, home, parameter, project, run):
    """
    Makes a figure comparing observed spectra to synthetic spectra with
    subplots for each station. 

    Inputs:
        syn_freqs(list): Array of list of times for synthetic waveforms
        syn_amps(list): Array of list of amplitudes for synthetic waveforms
        obs_freqs(list): Array of list of times for observed waveforms
        obs_amps(list): Array of list of amplitudes for observed waveforms
        stn_list(list): List of station names
                        Fourier spectra of the observed data for each station
        hypdists(list): List of hypocentral distances correlating with the 
                        stations used to get spectra
        data_type(str): Type of data
                            Options:
                                disp
                                acc
                                vel
        home(str): Base of path to save plots.
        parameter(str): Name of parameter folder.
        project(str): Name of simulation project.
        run(str): Synthetics run number.

    Output:
            Just saves the plots to ther respective directories. 
            
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from os import path, makedirs
      
    # Set figure parameters based on data type
    if data_type == 'disp':
           units = 'm'
           dim = 5,3
    elif data_type == 'acc':
           units = 'm/s/s'
           dim = 6,3
    elif data_type == 'vel':
           units = 'm/s'
           dim = 6,3
    
    # Sort hypdist and get sorted indices
    sort_id = np.argsort(hypdists)
    sort_hypdists = np.sort(hypdists)
    
    # Function to sort list based on list of indices 
    def sort_list(list1, list2): 
        zipped_pairs = zip(list2, list1) 
        z = [x for _, x in sorted(zipped_pairs)] 
        return z 
    
    # Sort times and amps based off hypdist
    sort_syn_times = sort_list(syn_times, sort_id)
    sort_syn_amps = sort_list(syn_amps, sort_id)
    sort_obs_times = sort_list(obs_times, sort_id)
    sort_obs_amps = sort_list(obs_amps, sort_id)
    sort_stn_name = sort_list(stn_list, sort_id)
    
    # Make figure and subplots
    if data_type == 'disp':
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,10))
        k = 0     # subplot index
        # Loop rhough rows
        for i in range(dim[0]):
            # Loop through columns
            for j in range(dim[1]):
                # Only make enough subplots for length of station list
                if k+1 <= len(stn_list):
                    axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                     color='C1',lw=0.4,label='synthetic')
                    axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                     'k-',lw=0.4,label='observed')
                    axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    axs[i][j].text(0.625,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                                    transform=axs[i][j].transAxes,size=7)
                    axs[i][j].text(0.025,5E-2,'E',transform=axs[i][j].transAxes,size=7)
                    if i < dim[0]-2:
                        axs[i][j].set_xticklabels([])
                    if i == dim[0]-2 and j == 0:
                        axs[i][j].set_xticklabels([])
                    k += 1
        fig.text(0.5, 0.005, 'UTC Time(hr:min:sec)', ha='center')
        fig.text(0.005, 0.5, f'Amplitude ({units})', va='center', rotation='vertical')
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.74,0.09), framealpha=None)
        if data_type == 'disp':
                fig.delaxes(axs[4][1])
                fig.delaxes(axs[4][2])
        else:
                fig.delaxes(axs[5][1])
                fig.delaxes(axs[5][2])
        fig.suptitle('Waveform Comparison', fontsize=12, y=1)
        fig.text(0.43, 0.135, (r"$\bf{" + 'Project:' + "}$" + '' + project))
        fig.text(0.43, 0.115, (r'$\bf{' + 'Run:' + '}$' + '' + run))
        fig.text(0.43, 0.09, (r'$\bf{' + 'DataType:' '}$' + '' + data_type))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.075, top=0.925,
                            wspace=0.325, hspace=0.4)
    
        if not path.exists(f'{home}/parameters/{parameter}/{project}/plots/comparison/wf/'):
          makedirs(f'{home}/parameters/{parameter}/{project}/plots/comparison/wf')
      
        plt.savefig(f'{home}/parameters/{parameter}/{project}/plots/comparison/wf/{run}_{data_type}.png', dpi=300)
        plt.close()
        
        
    else:
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,10))
        k = 0     # subplot index
        # Loop through rows
        for i in range(dim[0]):
            # Loop through columns 
            for j in range(dim[1]):
                # Only make enough subplots for length of station list
                if k+1 <= len(stn_list):
                    axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                     color='C1',lw=0.4,label='synthetic')
                    axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                    'k-',lw=0.4,label='observed')
                    axs[i][j].text(0.025,5E-2,'E',transform=axs[i][j].transAxes,size=7)
                    axs[i][j].text(0.6,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                                   transform=axs[i][j].transAxes,size=7)
                    axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    if i < dim[0]-2:
                        axs[i][j].set_xticklabels([])
                    if i == dim[0]-2 and j == 0:
                        axs[i][j].set_xticklabels([])
                    k += 1
        fig.text(0.5, 0.005, 'UTC Time(hr:min:sec))', ha='center')
        fig.text(0.005, 0.5, f'Amplitude ({units})', va='center', rotation='vertical')
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.74,0.08), framealpha=None)
        if data_type == 'disp':
                fig.delaxes(axs[4][1])
                fig.delaxes(axs[4][2])
        else:
                fig.delaxes(axs[5][1])
                fig.delaxes(axs[5][2])
        fig.suptitle('Waveform Comparison', fontsize=12, y=1)
        fig.text(0.435, 0.125, (r"$\bf{" + 'Project:' + "}$" + '' + project))
        fig.text(0.435, 0.105, (r'$\bf{' + 'Run:' + '}$' + '' + run))
        fig.text(0.435, 0.08, (r'$\bf{' + 'DataType:' '}$' + '' + data_type))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.075, top=0.925,
                            wspace=0.325, hspace=0.4)
    
        if not path.exists(f'{home}/parameters/{parameter}/{project}/plots/comparison/wf/'):
            makedirs(f'{home}/parameters/{parameter}/{project}/plots/comparison/wf')
        
        plt.savefig(f'{home}/parameters/{parameter}/{project}/plots/comparison/wf/{run}_{data_type}.png', dpi=300)
        plt.close()
        