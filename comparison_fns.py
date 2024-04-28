#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:52:17 2020

@author: tnye
"""

###############################################################################
# Module with functions used to make comparison plots between observed and 
# synthetic data. These functions are imported and used in synthetic_calc_mpi.py
# and synthetic_calc.py. 
###############################################################################


def plot_spec_comp(plot_dir,syn_freqs, syn_spec, obs_freqs, obs_spec, stn_list, hypdists, data_type, home, parameter, project, run, spec_type):
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
       ylim = 10**-5, 9*10**-1
       xlim = 0.004, 5*10**-1
       dim = 3,3
    elif data_type == 'acc':
       units = 'm/s'
       ylim = 7*10**-15, 6*10**-1
       xlim = .002, 10
       dim = 3,3
    elif data_type == 'vel':
       units = 'm'
       ylim = 6*10**-15, 8*10**-2
       xlim = .002, 10
       dim = 3,3
    
    # Sort hypdist and get indices
    sort_id = np.argsort(np.argsort(hypdists))
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
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,8))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=1,c='C1',ls='-',label='synthetic')
                    axs[i][j].loglog(sort_obs_freqs[k],sort_obs_spec[k],lw=1,c='steelblue',ls='-',label='observed')
                    axs[i][j].grid(linestyle='--')
                    axs[i][j].text(0.025,5E-2,'2-comp eucnorm',transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].set_xlim(xlim)
                    axs[i][j].set_ylim(ylim)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    if i < 1:
                        axs[i][j].set_xticklabels([])
                    if i == 1 and j == 0:
                        axs[i][j].set_xticklabels([])
                    if j > 0:
                        axs[i][j].set_yticklabels([])
                    k += 1
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.delaxes(axs[2][1])
        fig.delaxes(axs[2][2])
        fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('Frequency (Hz)',fontsize=12)
        fig.legend(handles, labels, loc=(0.72,0.25), framealpha=None, frameon=False)
        fig.text(0.72, 0.2, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.175, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.15, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.09, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
        
        if not path.exists(f'{plot_dir}/comparison/spectra/{data_type}'):
            makedirs(f'{plot_dir}/comparison/spectra/{data_type}')
            
        plt.savefig(f'{plot_dir}/comparison/spectra/{data_type}/{run}_{data_type}_{spec_type}.png', dpi=300)
        plt.close()
    
    else:
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,9.5))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=1,c='C1',ls='-',label='synthetic')
                    axs[i][j].loglog(sort_obs_freqs[k],sort_obs_spec[k],lw=1,c='steelblue',ls='-',label='observed')
                    axs[i][j].grid(linestyle='--')
                    axs[i][j].text(0.025,5E-2,'rotd50',transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].set_xlim(xlim)
                    axs[i][j].set_ylim(ylim)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    if i < dim[0]-1:
                        axs[i][j].set_xticklabels([])
                    if i == dim[0]-2 and j == 0:
                        axs[i][j].set_xticklabels([])
                    if j > 0:
                        axs[i][j].set_yticklabels([])
                    k += 1
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('Frequency (Hz)',fontsize=12,y=0.125)
        fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
        fig.text(0.72, 0.075, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.05, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.025, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.2, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
        
        if not path.exists(f'{plot_dir}/comparison/spectra/{data_type}'):
            makedirs(f'{plot_dir}/comparison/spectra/{data_type}')
        
        plt.savefig(f'{plot_dir}/comparison/spectra/{data_type}/{run}_{data_type}_{spec_type}.png', dpi=300)
        plt.close()

        
def plot_wf_comp(plot_dir,syn_times,syn_amps,stn_list,hypdists,data_type,wf_type,home,parameter,project,run,component,start,end):
    """
    Makes a figure comparing observed waveforms to synthetic waveforms with
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
    from obspy import read
    from glob import glob
    from os import path, makedirs
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MultipleLocator, ScalarFormatter
    
    if data_type == 'disp':
        if component != 'avg':
            obs_files = sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/individual/disp/*LX{component}*'))
        else:
            obs_files = sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/average/eucnorm_3comp/disp/*'))
    
    if data_type == 'acc':
        exclusions = ['/CGJI','/CNJI','/LASI','/MLSI','/PPBI','/PSI','/TSI'] # far stations containing surface waves
        if component != 'avg':
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/individual/acc/*HN{component}*')) \
                     if not any(exclude in file for exclude in exclusions)]
        else:
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/average/rotd50/acc/*')) \
                     if not any(exclude in file for exclude in exclusions)]
    
    if data_type == 'vel':
        exclusions = ['/CGJI','/CNJI','/LASI','/MLSI','/PPBI','/PSI','/TSI'] # far stations containing surface waves
        if component != 'avg':
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/individual/vel/*HN{component}*')) \
                     if not any(exclude in file for exclude in exclusions)]
        else:
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/average/rotd50/vel/*')) \
                     if not any(exclude in file for exclude in exclusions)]
    obs_amps = []
    obs_times = []
    for file in obs_files:
        obs_amps.append(read(file)[0].data.tolist())
        obs_times.append(read(file)[0].times('matplotlib').tolist())
    
    # Set figure parameters based on data type
    if data_type == 'disp':
           units = 'm'
           dim = 3,3
    elif data_type == 'acc':
           units = 'm/s/s'
           dim = 3,3
    elif data_type == 'vel':
           units = 'm/s'
           dim = 3,3
    
    # Sort hypdist and get sorted indices
    sort_hypdists = np.sort(hypdists)
    sort_syn_times = [syn_times[i] for i in np.argsort(hypdists)]
    sort_syn_amps = [syn_amps[i] for i in np.argsort(hypdists)]
    sort_obs_times = [obs_times[i] for i in np.argsort(hypdists)]
    sort_obs_amps = [obs_amps[i] for i in np.argsort(hypdists)]
    sort_stn_name = [stn_list[i] for i in np.argsort(hypdists)]
    
    
    # Make figure and subplots
    if data_type == 'disp': 
        if component != 'avg':
            label = f'LY{component}'
        else:
            label = '3-comp eucnorm'
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,8))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                     color='C1',alpha=0.7,lw=0.4,label='synthetic')
                    axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                     'steelblue',alpha=0.7,lw=0.4,label='observed')
                    axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,0.9,label,transform=axs[i][j].transAxes,size=10,horizontalalignment='right')
                    if i < 1:
                        axs[i][j].set_xticklabels([])
                    if i == 1 and j == 0:
                        axs[i][j].set_xticklabels([])
                    
                    # Ticks
                    if np.max(np.abs(sort_obs_amps[k])) > 0.2:
                        axs[i][j].yaxis.set_major_locator(MultipleLocator(0.1))
                    else:
                        axs[i][j].yaxis.set_major_locator(MultipleLocator(0.05))
                    k += 1

        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.72,0.25), framealpha=None, frameon=False)
        fig.delaxes(axs[2][1])
        fig.delaxes(axs[2][2])
        fig.suptitle('Waveform Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('UTC Time(hr:min:sec)',fontsize=12)
        fig.text(0.72, 0.2, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.175, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.15, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.09, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
         
        if not path.exists(f'{plot_dir}/comparison/wf/{data_type}'):
            makedirs(f'{plot_dir}/comparison/wf/{data_type}')
      
        plt.savefig(f'{plot_dir}/comparison/wf/{data_type}/{run}_{data_type}_{component}.png', dpi=300)
        plt.close()
        
        
    else:
        if component != 'avg':
            label = f'HN{component}'
        else:
            label = 'rotd50'
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,9.5))
        k = 0 
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                     color='C1',alpha=0.7,lw=0.4,label='synthetic')
                    axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                    'steelblue',alpha=0.7,lw=0.4,label='observed')
                    axs[i][j].text(0.98,0.9,label,horizontalalignment='right',
                                   transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    # axs[i][j].set_xlim(start, end)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    if i < dim[0]-1:
                        axs[i][j].set_xticklabels([])
                    if i == dim[0]-2 and j == 0:
                        axs[i][j].set_xticklabels([])
                    k += 1
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.suptitle('Waveform Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('UTC Time(hr:min:sec)',fontsize=12,y=0.125)
        fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
        fig.text(0.72, 0.075, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.05, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.025, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.2, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
        
        if not path.exists(f'{plot_dir}/comparison/wf/{data_type}'):
            makedirs(f'{plot_dir}/comparison/wf/{data_type}')
        
        plt.savefig(f'{plot_dir}/comparison/wf/{data_type}/{run}_{data_type}_{component}.png', dpi=300)
        plt.close()
        