#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:27:35 2020

@author: tnye
"""

###############################################################################
# Module with functions used to calculate and plot IM and spectra residuals.
# The functions use IM and spectra values calculated using mentawai.py for
# observed data and synthetic_calc_mpi.py for synthetic data. These functions
# are imported and used in residuals.py.  
###############################################################################


def calc_res(obs_file, home_dir, parameter, project, run, dtype):
    """
    Calculates residuals between synthetic and observed data, and puts residuals
    into a dataframe.
    
    Inputs:
        obs_file(str): Path to observed data flatfile.
        parameter(str): Folder name of parameter being varied.
        project(str): Folder name of specific project within parameter folder.
        run(str): Individual run name within certain project. 
        ln(T/F): If true, calculates the natural log of the residuals.
    Return:
        pgd_res(float): PGD residual. 
        pga_res(float): PGA residual.
        tPGD(float): tPGD residual.
        spectra_res(array): Array of residuals for all the spectra bins.

    """
    
    import numpy as np
    import pandas as pd
    
    # Synthetic values
    syn_df = pd.read_csv(f'{home_dir}/{parameter}/{project}/flatfiles/IMs/{run}_{dtype}.csv')
    
    # Observed values
    obs_df = pd.read_csv(obs_file)
    
    if dtype=='gnss':
        # syn_pgd = np.array(syn_df['pgd'])
        # syn_tPGD_orig = np.array(syn_df['tPGD_origin'])
        # syn_tPGD_parriv = np.array(syn_df['tPGD_parriv'])
        # syn_spectra = np.array(syn_df.iloc[:,24:])
        
        # Remove station KTET from dataframe
            # Don't have access to the data to reprocess like we do with the
            # other stations
        drop_ind = obs_df[obs_df['station'] == 'KTET'].index
        obs_df = obs_df.drop(drop_ind)
        
        syn_pgd = np.delete(np.array(syn_df['pgd']),1) # delete results for ktet
        syn_tPGD = np.delete(np.array(syn_df['tPGD']),1) # delete results for ktet
        syn_spectra = np.delete(np.array(syn_df.iloc[:,23:]),1,axis=0)
        
        # Observed values
        obs_pgd = np.array(obs_df['pgd'])
        obs_tPGD = np.array(obs_df['tPGD_origin'])
        obs_spectra = np.array(obs_df.iloc[:,23:]) 
        
        # Calc residuals
        pgd_res = np.log(obs_pgd) - np.log(syn_pgd)
        tPGD_res_linear = obs_tPGD - syn_tPGD
        tPGD_res_ln = np.log(obs_tPGD) - np.log(syn_tPGD)
        spectra_res = np.log(obs_spectra) - np.log(syn_spectra)
        
        out = [pgd_res,tPGD_res_ln,tPGD_res_linear,spectra_res]
    
    elif dtype=='sm':
        syn_pga = np.array(syn_df['pga'])
        syn_tPGA = np.array(syn_df['tPGA'])
        syn_spectra = np.array(syn_df.iloc[:,23:]) # syn_df only has 1 tPGD column whereas obs_df has 2
    
        # Observed values
        obs_pga = np.array(obs_df['pga'])
        obs_tPGA = np.array(obs_df['tPGA_origin'])
        obs_spectra = np.array(obs_df.iloc[:,23:])

        # Calc residuals
        pga_res = np.log(obs_pga) - np.log(syn_pga)
        tPGA_res_linear = obs_tPGA - syn_tPGA
        tPGA_res_ln = np.log(obs_tPGA) - np.log(syn_tPGA)
        spectra_res = np.log(obs_spectra) - np.log(syn_spectra)
        
        out = [pga_res,tPGA_res_ln,tPGA_res_linear,spectra_res]
    
    return(out)


def plot_spec_res_full(home_dir, parameter, param_vals, data_types, stations, figdir, outliers=True, standard=False):
    """
    
    """
    
    import numpy as np
    import pandas as pd
    from os import path, makedirs
    from glob import glob
    from math import floor
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    import matplotlib
    import matplotlib.ticker as ticker
    
    if parameter == 'risetime':
        val = '5.4' 
        legend_title = 'Mean Risetime (s)'
    elif parameter == 'stress_drop':
        val = '5'
        legend_title = 'Stress Drop (MPa)'
    elif parameter == 'vrupt':
        val = '1.6'
        legend_title = r'Mean $V_{rupt}$ (km/s)'
    else:
        val = 'Standard'
        legend_title = ''
    
    # Residual type to be used in figure name
    label = 'ln Residual'
    
    sns.set_style("whitegrid")
    
    # if 'gnss' in stations:
        
    #     gnss_dfs = sorted(glob(f'{home_dir}/{parameter}/*/flatfiles/residuals/*_gnss.csv'))
    #     gnss_dfs = [file for file in gnss_dfs if not file.endswith('standard_gnss.csv')]
        
    #     # Define bin edges, spectra residual columns, and other data type parameters
    #     bin_edges = [0.004, 0.00509, 0.00648, 0.00825, 0.01051, 0.01337,
    #                  0.01703, 0.02168, 0.02759, 0.03513, 0.04472, 0.05693,
    #                  0.07248, 0.09227, 0.11746, 0.14953, 0.19037, 0.24234,
    #                  0.30852, 0.39276, 0.50000]
        
    #     res_units = 'm*s'
    #     title = f'Displacement Spectra Residuals'
        
    #     # Obtain bin means from bin edges
    #     bin_means = []
    #     for i in range(len(bin_edges)):
    #         if i != 0:
    #             # mean = np.exp((np.log10(bin_edges[i])+np.log10(bin_edges[i-1]))/2)
    #             mean = np.sqrt(bin_edges[i]*bin_edges[i-1])
    #             bin_means.append(mean)
            
    #     # Function to round bin means
    #     def round_sig(x, sig):
    #         return round(x, sig-int(floor(np.log10(abs(x))))-1)
        
    #     # Round bin means so that they look clean for figure
    #     for i, b in enumerate(bin_means):
    #         if b < 0.01:
    #             bin_means[i] = round_sig(b,1)
    #         elif b >= 0.01 and b < 0.1:
    #             bin_means[i] = round_sig(b,1)
    #         elif b >= 0.1 and b < 1:
    #             bin_means[i] = round_sig(b,2)
    #         elif b >= 1 and b < 10:
    #             bin_means[i] = round_sig(b,2)
    #         elif b >= 10:
    #             bin_means[i] = round_sig(b,2)
        
    #     gnss_bin_list = []
    #     gnss_std_bin_list = []
    #     gnss_spec_res_list = np.array([])
    #     gnss_std_spec_res_list = np.array([])
    #     project_list = []
    
    #     for i_project, file in enumerate(gnss_dfs):
            
    #         gnss_res_df = pd.read_csv(file)
            
    #         gnss_spec_res = gnss_res_df.iloc[:,20:].reset_index(drop=True)
            
    #         project = param_vals[i_project]
        
    #         for j_bin in range(len(bin_edges)-1):
                
    #             if j_bin >= 2:
                
    #                 bin_res = np.array(gnss_spec_res.iloc[:,j_bin])
    #                 gnss_bin_list.extend(np.repeat(bin_means[j_bin],len(bin_res)))
    #                 gnss_spec_res_list = np.append(gnss_spec_res_list,bin_res)
    #                 project_list.extend(np.repeat(project,len(bin_res)))
                    
    #                 # Standard parameters
    #                 if standard == True:
    #                     try:
    #                         gnss_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_gnss.csv')
    #                     except:
    #                         gnss_std_df = pd.read_csv(f'{home_dir}/{parameter}/standard/flatfiles/residuals/standard_gnss.csv')
    #                     # gnss_std_df = pd.read_csv(f'{home_dir}/{parameter}/standard/flatfiles/residuals/standard_gnss.csv')
                        
    #                     poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
    #                     ind = np.where(np.in1d(np.array(gnss_std_df['station']), poor_GNSS).reshape(np.array(gnss_std_df['station']).shape)==False)[0]
                
    #                     gnss_std_res = np.array(gnss_std_df.iloc[ind].reset_index(drop=True).iloc[:,20:].iloc[:,j_bin])
    #                     gnss_std_bin_list.extend(np.repeat(bin_means[j_bin],len(gnss_std_res)))
    #                     gnss_std_spec_res_list = np.append(gnss_std_spec_res_list,gnss_std_res)
                    
    #     # Create dataframe for current parameter
    #     gnss_spec_data = {'Project':project_list,'Frequency (Hz)': gnss_bin_list, f'{label} ({res_units})':gnss_spec_res_list}
    #     gnss_spec_df = pd.DataFrame(data=gnss_spec_data)    
        
    #     # Create dataframe for standard parameters
    #     gnss_std_spec_data = {'Frequency (Hz)': gnss_std_bin_list, f'{label} ({res_units})':gnss_std_spec_res_list}
    #     gnss_std_spec_df = pd.DataFrame(data=gnss_std_spec_data)    
        
    #     # Make figure
    #     plt.figure(figsize=(18,6))
    #     ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=gnss_spec_df,
    #                       hue='Project', showfliers=outliers, linewidth=1)
    #     if standard == True:
    #         ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=gnss_std_spec_df,
    #                               showfliers=False,
    #                               boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
    #                               medianprops=dict(lw=1.5,color='black'),
    #                               whiskerprops=dict(lw=2,color='black'),
    #                               capprops=dict(lw=2,color='black'))
    #     ax.set_xticklabels(ax.get_xticklabels(),rotation=15,fontsize=20)
    #     plt.yticks(fontsize=20)
    #     ax.set_title(r'$\bf{Fourier}$ $\bf{Amplitude}$ $\bf{Spectra}$ $\bf{Residuals}$',fontsize=20,pad=15)
    #     ax.axhline(0, ls='--')
    #     handles, labels = ax.get_legend_handles_labels()
    #     handles = handles + [Patch(facecolor='white', edgecolor='black', lw=2, ls='--',label=val)]
    #     labels = labels + ['standard']
    #     ax.set_ylim(-3,3)
    #     ax.set_xlabel('Bin Median Frequency (Hz)', fontsize=20)
    #     ax.set_ylabel(f'{label} ({res_units})', fontsize=20)
    #     leg = ax.legend(handles=handles, loc='lower left', fontsize=18)
    #     leg.set_title(legend_title,prop={'size':18})
    #     plt.subplots_adjust(bottom=0.175, top=0.92, left=0.06, right=0.98, wspace=0.5)
    #     # ax.text(0.98, 0.05, ('Displacement'), size=20, horizontalalignment='right',transform=ax.transAxes)
    #     plt.savefig(f'{figdir}/{parameter}/disp_spectra_res.png', dpi=300)
    #     plt.close()
        
    if 'sm' in stations:
        
        sm_dfs = sorted(glob(f'{home_dir}/{parameter}/*/flatfiles/residuals/*_sm.csv'))
        sm_dfs = [file for file in sm_dfs if not file.endswith('standard_sm.csv')]
        
        # Define bin edges, spectra residual columns, and other data type parameters
        # bin_edges = [4.00000000e-03, 5.91503055e-03, 8.74689659e-03, 1.29345401e-02,
        #        1.91270500e-02, 2.82842712e-02, 4.18255821e-02, 6.18498989e-02,
        #        9.14610104e-02, 1.35248668e-01, 2.00000000e-01, 2.95751527e-01,
        #        4.37344830e-01, 6.46727007e-01, 9.56352500e-01, 1.41421356e+00,
        #        2.09127911e+00, 3.09249495e+00, 4.57305052e+00, 6.76243338e+00,
        #        1.00000000e+01]
        
        bin_edges = np.logspace(np.log10(0.1), np.log10(10), num=11)
        
        # Obtain bin means from bin edges
        bin_means = []
        for i in range(len(bin_edges)):
            if i != 0:
                # mean = np.exp((np.log10(bin_edges[i])+np.log10(bin_edges[i-1]))/2)
                mean = np.sqrt(bin_edges[i]*bin_edges[i-1])
                bin_means.append(mean)
            
        # Function to round bin means
        def round_sig(x, sig):
            return round(x, sig-int(floor(np.log10(abs(x))))-1)
        
        # Round bin means so that they look clean for figure
        for i, b in enumerate(bin_means):
            if b < 0.01:
                bin_means[i] = round_sig(b,1)
            elif b >= 0.01 and b < 0.1:
                bin_means[i] = round_sig(b,1)
            elif b >= 0.1 and b < 1:
                bin_means[i] = round_sig(b,2)
            elif b >= 1 and b < 10:
                bin_means[i] = round_sig(b,2)
            elif b >= 10:
                bin_means[i] = round_sig(b,2)
        
        sm_bin_list = []
        sm_std_bin_list = []
        acc_spec_res_list = np.array([])
        acc_std_spec_res_list = np.array([])
        project_list = []
        
        for i_project, file in enumerate(sm_dfs):
            
            sm_res_df = pd.read_csv(file)
            
            project = param_vals[i_project]
            
            acc_spec_res = sm_res_df.iloc[:,20:].reset_index(drop=True)
        
            for j_bin in range(len(bin_edges)-1):
                
                # Current parameter
                acc_bin_res = np.array(acc_spec_res)[:,j_bin]
                sm_bin_list.extend(np.repeat(bin_means[j_bin],len(acc_bin_res)))
                acc_spec_res_list = np.append(acc_spec_res_list,acc_bin_res)
                project_list.extend(np.repeat(project,len(acc_bin_res)))
                
                if standard == True:
                    # Standard parameters
                    sm_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_sm.csv')
                    # sm_std_df = pd.read_csv(f'{home_dir}/{parameter}/standard/flatfiles/residuals/standard_sm.csv')
                    
                    # Remove strong motion stations farther than 600km to avoid surface waves
                    far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
                    ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
                
                    acc_std_res = np.array(sm_std_df.iloc[ind].reset_index(drop=True).iloc[:,20:].iloc[:,j_bin])
                
                    sm_std_bin_list.extend(np.repeat(bin_means[j_bin],len(acc_std_res)))
                    acc_std_spec_res_list = np.append(acc_std_spec_res_list,acc_std_res)
                        
                
        ###### Acceleration
        res_units = 'm/s'
        title = 'Acceleration Spectra Residuals'
        acc_spec_data = {'Project':project_list,'Frequency (Hz)': sm_bin_list, f'{label} ({res_units})':acc_spec_res_list}
        acc_spec_df = pd.DataFrame(data=acc_spec_data) 
        
        # Create dataframe for standard parameters
        acc_std_spec_data = {'Frequency (Hz)': sm_std_bin_list, f'{label} ({res_units})':acc_std_spec_res_list}
        acc_std_spec_df = pd.DataFrame(data=acc_std_spec_data)    
      
        
        # # Make figure
        # plt.figure(figsize=(18,6))
        # ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=acc_spec_df,
        #                   hue='Project', showfliers=outliers, linewidth=1)
        # if standard == True:
        #     ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=acc_std_spec_df,
        #                       showfliers=False,
        #                       boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
        #                       medianprops=dict(lw=1.5,color='black'),
        #                       whiskerprops=dict(lw=2,color='black'),
        #                       capprops=dict(lw=2,color='black'))
        # ax.set_xticklabels(ax.get_xticklabels(),rotation=15,fontsize=20)
        # plt.yticks(fontsize=20)
        # ax.set_title(r'$\bf{Fourier}$ $\bf{Amplitude}$ $\bf{Spectra}$ $\bf{Residuals}$',fontsize=20,pad=15)
        # ax.axhline(0, ls='--')
        # handles, labels = ax.get_legend_handles_labels()
        # handles = handles + [Patch(facecolor='white',edgecolor='black',lw=2,ls='--',label=val)]
        # labels = labels + ['standard']
        # ax.set_ylim(-4.5,5)
        # ax.set_xlabel('Bin Median Frequency (Hz)', fontsize=20)
        # ax.set_ylabel(f'{label} ({res_units})', fontsize=20)
        # leg = ax.legend(handles=handles, loc='lower left', fontsize=18, ncol=4, framealpha=1)
        # leg.set_title(legend_title,prop={'size':18})
        # plt.subplots_adjust(bottom=0.175, top=0.92, left=0.06, right=0.98, wspace=0.5)
        # # ax.text(0.98, 0.05, ('Acceleration'), size=20, horizontalalignment='right',transform=ax.transAxes)
        # plt.savefig(f'{figdir}/{parameter}/acc_spectra_res.png', dpi=300)
        # # plt.close()
        
        
        # Make figure
        bins = np.unique(acc_std_spec_df['Frequency (Hz)'].values)
        grouped_res_std = [list(acc_std_spec_df[f'{label} ({res_units})'].values[acc_std_spec_df['Frequency (Hz)'].values == fbin]) for fbin in bins]
        projects = np.unique(acc_spec_df.Project.values)
        width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
        w=0.15
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        fig, ax = plt.subplots(1,1,figsize=(18,6))
        handles, labels = ax.get_legend_handles_labels()
        for i, project in enumerate(projects):
            if len(projects) == 3:
                if i == 0:
                    positions = bins - (width(bins,w)/3)
                elif i == 1:
                    positions = bins
                elif i == 2:
                    positions = bins + (width(bins,w)/3)
            if len(projects) == 2:
                if i == 0:
                    positions = bins - (width(bins,w)/4)
                elif i == 1:
                    positions = bins + (width(bins,w)/4)
            project_df = acc_spec_df[acc_spec_df['Project'] == project]
            grouped_res = [list(project_df[f'{label} ({res_units})'].values[project_df['Frequency (Hz)'].values == fbin]) for fbin in bins]
            ax.boxplot(grouped_res,positions=positions,patch_artist=True,
                       boxprops=dict(lw=1,linestyle='--',edgecolor='black',facecolor=colors[i]),
                       medianprops=dict(lw=1,color='black'),
                       whiskerprops=dict(lw=1,color='black'),
                       capprops=dict(lw=1,color='black'),
                       widths=width(bins,w)/len(projects)
                       )
            handles = handles + [Patch(facecolor=colors[i],edgecolor='black',lw=1,ls='-',label=param_vals[i])]
            labels = labels + [param_vals[i]]
        if standard == True:
            ax.boxplot(grouped_res_std,positions=bins,
                       boxprops=dict(lw=2,linestyle='--',color='black'),
                       medianprops=dict(lw=1.5,color='black'),
                       whiskerprops=dict(lw=2,color='black'),
                       capprops=dict(lw=2,color='black'),
                       widths=width(bins,w)
                       )
            handles = handles + [Patch(facecolor='white',edgecolor='black',lw=2,ls='--',label=val)]
            labels = labels + [val]
        ax.set_xscale('log')
        ax.set_xlim(0.1,10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=4))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.tick_params(axis='y', left=True, length=5, which='major', labelsize=20)
        ax.tick_params(axis='x', bottom=True, length=5, which='major', labelsize=20)
        ax.tick_params(axis='x', bottom=True, length=3, which='minor')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_title(r'$\bf{Fourier}$ $\bf{Amplitude}$ $\bf{Spectra}$ $\bf{Residuals}$',fontsize=20,pad=15)
        ax.axhline(0, ls='--')
        # ax.set_ylim(-4.5,5)
        ax.set_ylim(ymin=np.min(acc_std_spec_df[f'{label} ({res_units})'])-1.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=20)
        ax.set_ylabel(f'{label} ({res_units})', fontsize=20)
        leg = ax.legend(handles,labels, loc='lower left', fontsize=18, ncol=4, framealpha=1)
        leg.set_title(legend_title,prop={'size':18})
        plt.subplots_adjust(bottom=0.175, top=0.92, left=0.06, right=0.98, wspace=0.5)
        plt.savefig(f'{figdir}/{parameter}/acc_spectra_res.png', dpi=300)
        # plt.close()


def plot_IM_res_full(home_dir, parameter, projects, param_vals, stations, figdir, outliers=True, standard=False):
    """
    Plots residuals between synthetic and observed IMs for all runs of the
    different projects for one parameter (i.e. stress drop of 0.3, 1.0, and 2.0
    MPa). Plots each IM and spectra bin separately and also plots them all on
    one figure.  
    
    Inputs:
        parameter(str): Folder name of parameter being varied.
        projects(array): Array of folder names of projects within parameter folder.
        param_vals(array): Array of different parameter values (i.e. [0.3, 1.0, 2.0])
        ln(T/F): If true, calculates the natural log of the residuals.
        outliers(T/F): True or False if outliers should be shown in plots. 
        default(T/F): True or false if residuals using default parameters are
                      included. 
        
    Output:
        Just saves the plots to ther respective directories. 
    """
    
    import numpy as np
    import pandas as pd
    from os import path, makedirs
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator, ScalarFormatter
    
    sns.set_style("whitegrid")
    
    # Make deirectory if it does not already exist
    if not path.exists(figdir):
        makedirs(figdir)
    
    # Set up empty lists to store the project names in for the disp IMs, the sm
        # IMs, and the standard parameter IMs
    disp_project_list = []
    sm_project_list = []
    std_disp_project = []
    std_sm_project = []
    
    # Set up empty lists to store the current project residuals in
    pgd_res_list = []
    pga_res_list = []
    tPGD_res_list = []
    std_pgd_res_list = []
    std_pga_res_list = []
    std_tPGD_res_list = []
    
    if parameter == 'risetime':
        val = '5.4' 
    elif parameter == 'stress_drop':
        val = '5'
    elif parameter == 'vrupt':
        val = '1.6'
    else:
        val = '0'
        
    # Add default parameter first if risetime 
    if standard == True and parameter == 'risetime':
        for i, project in enumerate(projects):
            
            if 'gnss' in stations:  
                gnss_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_gnss.csv')
               
                # Remove GNSS stations with low SNR from residual plots
                poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
                ind = np.where(np.in1d(np.array(gnss_std_df['station']), poor_GNSS).reshape(np.array(gnss_std_df['station']).shape)==False)[0]
            
                # Select out residuals 
                std_pgd_res = np.array(gnss_std_df['pgd_res'])[ind]
                std_tPGD_res = np.array(gnss_std_df['tPGD_res_linear'])[ind]
                
                # Get rid of NaNs if there are any
                std_pgd_res = [x for x in std_pgd_res if str(x) != 'nan']
                std_tPGD_res = [x for x in std_tPGD_res if str(x) != 'nan']
                
                # Append residuals from this project to main lists
                # pgd_res_list.append(std_pgd_res)
                # tPGD_res_list.append(std_tPGD_res)
                std_pgd_res_list.extend(std_pgd_res)
                std_tPGD_res_list.extend(std_tPGD_res)
                
                # Get parameter value.  Need different lists for disp and sm runs because 
                    # there are a different number of stations. 
                for j in range(len(std_pgd_res)):
                    std_disp_project.append(val)
                
                # Add empty values for other projects to have same categories
                std_pgd_res_list.extend([None]*len(param_vals))
                std_tPGD_res_list.extend([None]*len(param_vals))
                std_disp_project.extend(param_vals)
            
            if 'sm' in stations:
                sm_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_sm.csv')
            
                # Remove strong motion stations farther than 600km to avoid surface waves
                far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
                ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
            
                # Select out residuals 
                std_pga_res = np.array(sm_std_df['pga_res'])[ind]

                # Get rid of NaNs if there are any
                std_pga_res = [x for x in std_pga_res if str(x) != 'nan']

                # Append residuals from this project to main lists
                # pga_res_list.append(std_pga_res)
                std_pga_res_list.extend(std_pga_res)

                # Get parameter value.  Need different lists for disp and sm runs because 
                    # there are a different number of stations. 
                for j in range(len(std_pga_res)):
                    std_sm_project.append(val)
                
                # Add empty values for other projects to have same categories
                std_pga_res_list.extend([None]*len(param_vals))
                std_sm_project.extend(param_vals)
    
    # Loop through projects and put residuals into lists
    for i, project in enumerate(projects):
        
        # Residual dataframes
        if 'gnss' in stations:
            gnss_res_df = pd.read_csv( f'{home_dir}/{parameter}/{project}/flatfiles/residuals/{project}_gnss.csv')
            
            # # Remove GNSS stations with low SNR from residual plots
            poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
            ind = np.where(np.in1d(np.array(gnss_res_df['station']), poor_GNSS).reshape(np.array(gnss_res_df['station']).shape)==False)[0]
        
            # my_list = ['mentawai.000000','mentawai.000001','mentawai.000002','mentawai.000003']
            # ind = index = np.where(gnss_res_sorted(df['run'].isin(my_list))[0]

            # Select out residuals 
            pgd_res = np.array(gnss_res_df['pgd_res'])[ind]
            tPGD_res = np.array(gnss_res_df['tPGD_res_linear'])[ind]
            
            # Get rid of NaNs if there are any
            pgd_res = [x for x in pgd_res if str(x) != 'nan']
            tPGD_res = [x for x in tPGD_res if str(x) != 'nan']
            
            # Append residuals from this project to main lists
            pgd_res_list.extend(pgd_res)
            tPGD_res_list.extend(tPGD_res)
            
            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(pgd_res)):
                disp_project_list.append(param_vals[i])
            
            # Add empty values for standard to have same categories
            if standard == True:
                pgd_res_list.append(None)
                tPGD_res_list.append(None)
                disp_project_list.append(val)
            
        if 'sm' in stations:
            sm_res_df = pd.read_csv( f'{home_dir}/{parameter}/{project}/flatfiles/residuals/{project}_sm.csv')
            
            # Remove strong motion stations farther than 600km to avoid surface waves
            far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
            ind = np.where(np.in1d(np.array(sm_res_df['station']), far_sm).reshape(np.array(sm_res_df['station']).shape)==False)[0]
        
            # Select out residuals 
            pga_res = np.array(sm_res_df['pga_res'])[ind]
            tPGA_res = np.array(sm_res_df['tPGA_res_linear'])[ind]
            
            # Get rid of NaNs if there are any
            pga_res = [x for x in pga_res if str(x) != 'nan']
            tPGA_res = [x for x in tPGA_res if str(x) != 'nan']
            
            # Append residuals from this project to main lists
            pga_res_list.extend(pga_res)

            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(pga_res)):
                sm_project_list.append(param_vals[i])
            
            # Add empty values for standard to have same categories
            if standard == True:
                pga_res_list.append(None)
                sm_project_list.append(val)

    # Add default parameter last if not risetime
    if standard == True and parameter != 'risetime':
        
        # Residual dataframe for default (aka stdtered) parameters
        # std_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/standard/std/flatfiles/residuals/std_{res}.csv')
        
        if 'gnss' in stations: 
            try:
                gnss_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_gnss.csv')
            except:
                gnss_std_df = pd.read_csv(f'{home_dir}/{parameter}/standard/flatfiles/residuals/standard_gnss.csv')
           
             # Remove GNSS stations with low SNR from residual plots
            poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
            ind = np.where(np.in1d(np.array(gnss_std_df['station']), poor_GNSS).reshape(np.array(gnss_std_df['station']).shape)==False)[0]
        
            # Select out residuals 
            std_pgd_res = np.array(gnss_std_df['pgd_res'])[ind]
            std_tPGD_res = np.array(gnss_std_df['tPGD_res_linear'])[ind]
            
            # Get rid of NaNs if there are any
            std_pgd_res = [x for x in std_pgd_res if str(x) != 'nan']
            std_tPGD_res = [x for x in std_tPGD_res if str(x) != 'nan']
            
            # Append residuals from this project to main lists
            std_pgd_res_list.extend(std_pgd_res)
            std_tPGD_res_list.extend(std_tPGD_res)
            
            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(std_pgd_res)):
                std_disp_project.append(val)
            
            # Add empty values for standard to have same categories
            std_pgd_res_list.extend([None]*len(param_vals))
            std_tPGD_res_list.extend([None]*len(param_vals))
            std_disp_project.extend(param_vals)
        
        if 'sm' in stations:
            sm_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_sm.csv')
            # sm_std_df = pd.read_csv(f'{home_dir}/{parameter}/standard/flatfiles/residuals/standard_sm.csv')
        
            # Remove strong motion stations farther than 600km to avoid surface waves
            far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
            ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
        
            # Select out residuals 
            std_pga_res = np.array(sm_std_df['pga_res'])[ind]


            # Get rid of NaNs if there are any
            std_pga_res = [x for x in std_pga_res if str(x) != 'nan']
        
            # Append residuals from this project to main lists
            std_pga_res_list.extend(std_pga_res)
            
            
            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(std_pga_res)):
                std_sm_project.append(val)

            # Add empty values for standard to have same categories
            std_pga_res_list.extend([None]*len(param_vals))
            std_sm_project.extend(param_vals)
    
    # Set up xlabel and title based on parameter being varied 
    if parameter == 'stress_drop':
        xlabel = 'Stress Drop (Mpa)'
        title = 'Stress Drop IM Residuals'
    elif parameter == 'risetime':
        xlabel = 'Average Risetime (s)'
        title = 'Rise Time IM Residuals'
    elif parameter == 'vrupt':
        xlabel = 'Average Rupture Velocity (km/s)'
        title = r'V$_{rupt}$ IM Residuals'
    else:
        xlabel = ''
        title = 'IM Residuals'
    
    ## Set up data frames
    label = 'ln Residual'
    
    # Change string numbers to floats
    try:
        disp_project_list = [float(string) for string in disp_project_list]
        std_disp_project = [float(string) for string in std_disp_project]
        sm_project_list = [float(string) for string in sm_project_list]
        std_sm_project = [float(string) for string in std_sm_project]
    
    except:
        pass
    
    # PGD
    pgd_data = {'Parameter':disp_project_list, f'{label} (m)':pgd_res_list}
    pgd_df = pd.DataFrame(data=pgd_data).sort_values('Parameter',ascending=True)  
    std_pgd_data = {'Parameter':std_disp_project, f'{label} (m)':std_pgd_res_list}
    std_pgd_df = pd.DataFrame.from_dict(data=std_pgd_data).sort_values('Parameter',ascending=True)  
    
    # tPGD
    tPGD_data = {'Parameter':disp_project_list, f'{label} (s)':tPGD_res_list}
    tPGD_df = pd.DataFrame(data=tPGD_data).sort_values('Parameter',ascending=True)  
    std_tPGD_data = {'Parameter':std_disp_project, f'{label} (m)':std_tPGD_res_list}
    std_tPGD_df = pd.DataFrame.from_dict(data=std_tPGD_data).sort_values('Parameter',ascending=True)  
    
    # PGA
    pga_data = {'Parameter':sm_project_list, f'{label} (m/s/s)':pga_res_list}
    pga_df = pd.DataFrame(data=pga_data).sort_values('Parameter',ascending=True)  
    std_pga_data = {'Parameter':std_sm_project, f'{label} (m)':std_pga_res_list}
    std_pga_df = pd.DataFrame.from_dict(data=std_pga_data).sort_values('Parameter',ascending=True) 
        
    ######################## Plot IMs on one figure ###########################
    
    n_colors = len(projects)  # Set this to your actual number of categories minus 1
    custom_palette = sns.color_palette()[:n_colors]
    
    if 'sm' in stations and 'gnss' in stations:
        
        # Set up Figure
        # fig, axs = plt.subplots(1, 4, figsize=(18,6))
        fig, axs = plt.subplots(1, 3, figsize=(18,6))
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]
        # ax4 = axs[3]
    
        # PGD subplot
        if parameter == 'risetime':
            b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=pgd_df, showfliers=outliers,
                        ax=ax1, palette=custom_palette)
        else:
            b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=pgd_df, showfliers=outliers,
                        ax=ax1)
        if standard == True:
            b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=std_pgd_df, showfliers=outliers,
                        ax=ax1,
                        boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
                        medianprops=dict(lw=1.5,color='black'),
                        whiskerprops=dict(lw=2,color='black'),
                        capprops=dict(lw=2,color='black'))
        b.set(xlabel=None)
        b.set_ylabel('ln (obs/syn) (m)',fontsize=20)
        b.set_xticklabels(ax1.get_xticklabels(),rotation=15)
        yabs_max = abs(max(ax1.get_ylim(), key=abs))
        ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax1.set_ylim(ymin=-2, ymax=2)
        ax1.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax1.yaxis.set_major_locator(MultipleLocator(1))
        ax1.set_title('PGD',fontsize=20)
        ax1.axhline(0, ls='--')
        
        # tPGD subplot
        if parameter == 'risetime':
            b = sns.boxplot(x='Parameter', y=f'{label} (s)', data=tPGD_df, showfliers=outliers,
                        ax=ax2, palette=custom_palette)
        else:
            b = sns.boxplot(x='Parameter', y=f'{label} (s)', data=tPGD_df, showfliers=outliers,
                            ax=ax2)
        if standard == True:
            b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=std_tPGD_df, showfliers=outliers,
                        ax=ax2,
                        boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
                        medianprops=dict(lw=1.5,color='black'),
                        whiskerprops=dict(lw=2,color='black'),
                        capprops=dict(lw=2,color='black'))
        b.set(xlabel=None)
        b.set_ylabel('obs - syn (s)',fontsize=20)
        b.set_xticklabels(ax2.get_xticklabels(),rotation=15)
        yabs_max = abs(max(ax2.get_ylim(), key=abs))
        ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax2.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax2.yaxis.set_major_locator(MultipleLocator(50))
        ax2.set_title('tPGD',fontsize=20)
        ax2.axhline(0, ls='--')
        
        # PGA subplot
        if parameter == 'risetime':
            b = sns.boxplot(x='Parameter', y=f'{label} (m/s/s)', data=pga_df, showfliers=outliers,
                        ax=ax3, palette=custom_palette)
        else:   
            b = sns.boxplot(x='Parameter', y=f'{label} (m/s/s)', data=pga_df, showfliers=outliers,
                        ax=ax3)
        if standard == True:
            b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=std_pga_df, showfliers=outliers,
                        ax=ax3,
                        boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
                        medianprops=dict(lw=1.5,color='black'),
                        whiskerprops=dict(lw=2,color='black'),
                        capprops=dict(lw=2,color='black'))
        b.set(xlabel=None)
        b.set_ylabel(r'ln (obs/syn) (m/s$^2$)',fontsize=20)
        b.set_xticklabels(ax3.get_xticklabels(),rotation=15)
        yabs_max = abs(max(ax3.get_ylim(), key=abs))
        ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax3.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax3.yaxis.set_major_locator(MultipleLocator(1))
        ax3.set_title('PGA',fontsize=20)
        ax3.axhline(0, ls='--')
        
        plt.subplots_adjust(wspace=0.5,bottom=0.2,left=0.06,right=0.98)
        fig.suptitle(r'$\bf{IM}$ $\bf{Residuals}$',fontsize=20)
        fig.supxlabel(xlabel,fontsize=18)
        
        if outliers == True:
            plt.savefig(f'{figdir}/{parameter}/IMs.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'{figdir}/{parameter}/IMs_xout.png', bbox_inches='tight', dpi=300)
        # plt.close()
        
    elif 'gnss' in stations and 'sm' not in stations:
        
        # Set up Figure
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        ax1 = axs[0]
        ax2 = axs[1]

        # Set up xlabel and title based on parameter being varied 
        if parameter == 'stress_drop':
            xlabel = 'Stress Drop (Mpa)'
            title = 'Stress Drop IM Residuals'
        elif parameter == 'risetime':
            xlabel = 'Average Rise Time (s)'
            title = 'Rise Time IM Residuals'
        elif parameter == 'vrupt':
            xlabel = 'Average Rupture Velocity (km/s)'
            title = r'V$_{rupt}$ IM Residuals'
        else:
            xlabel = ''
            title = 'IM Residuals'
        
        # PGD subplot
        pgd_data = {'Parameter':disp_project_list, f'{label} (m)':pgd_res_list}
        pgd_df = pd.DataFrame(data=pgd_data)       
        b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=std_pgd_df, showfliers=outliers,
                    ax=ax1,
                    boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
                    medianprops=dict(lw=1.5,color='black'),
                    whiskerprops=dict(lw=2,color='black'),
                    capprops=dict(lw=2,color='black')) 
        b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=pgd_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax1).set(xlabel=None,ylabel='ln Residual (m)')
        yabs_max = abs(max(ax1.get_ylim(), key=abs))
        ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax1.set_title('PGD')
        ax1.axhline(0, ls='--')
        
        # tPGD subplot
        tPGD_data = {'Parameter':disp_project_list, f'{label} (s)':tPGD_res_list}
        tPGD_df = pd.DataFrame(data=tPGD_data)
        b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=std_tPGD_df, showfliers=outliers,
                    ax=ax2,
                    boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
                    medianprops=dict(lw=1.5,color='black'),
                    whiskerprops=dict(lw=2,color='black'),
                    capprops=dict(lw=2,color='black'))
        b = sns.boxplot(x='Parameter', y=f'{label} (s)', data=tPGD_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax2).set(xlabel=None,ylabel='Residual (s)')
        yabs_max = abs(max(ax2.get_ylim(), key=abs))
        ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax2.set_title('tPGD')
        ax2.axhline(0, ls='--')
        
        if outliers == True:
            plt.savefig(f'{figdir}/IMs_gnss.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'{figdir}/IMs_xout_gnss.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    elif 'sm' in stations and 'gnss' not in stations:
        
        # Set up Figure
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        ax1 = axs[0]
        ax2 = axs[1]
        
        # PGA subplot
        if parameter == 'risetime':
            b = sns.boxplot(x='Parameter', y=f'{label} (m/s/s)', data=pga_df, showfliers=outliers,
                        ax=ax1, palette=custom_palette)
        else:   
            b = sns.boxplot(x='Parameter', y=f'{label} (m/s/s)', data=pga_df, showfliers=outliers,
                        ax=ax1)
        if standard == True:
            b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=std_pga_df, showfliers=outliers,
                        ax=ax1,
                        boxprops=dict(fc=(1,0,0,0),lw=2,ls='--',edgecolor='black'),
                        medianprops=dict(lw=1.5,color='black'),
                        whiskerprops=dict(lw=2,color='black'),
                        capprops=dict(lw=2,color='black'))
        b.set(xlabel=None)
        b.set_ylabel(r'ln Residual (m/s$^2$)',fontsize=18)
        b.set_xticklabels(ax1.get_xticklabels(),rotation=15)
        yabs_max = abs(max(ax1.get_ylim(), key=abs))
        ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax1.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax1.yaxis.set_major_locator(MultipleLocator(1))
        ax1.set_title('PGA',fontsize=18)
        ax1.axhline(0, ls='--')
        
        plt.subplots_adjust(wspace=0.5,bottom=0.2,left=0.06,right=0.98)
        fig.suptitle(r'$\bf{IM}$ $\bf{Residuals}$',fontsize=20)
        fig.supxlabel(xlabel,fontsize=18)
        
        if outliers == True:
            plt.savefig(f'{figdir}/{parameter}/IMs.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'{figdir}/{parameter}/IMs_xout.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    return()




