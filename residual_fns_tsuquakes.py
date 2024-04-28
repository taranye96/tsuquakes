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
        pgv_res(float): PGV residual.
        tPGD_orig_res(float): tPGD from origin residual.
        tPGD_parriv_res(float): tPGD from P-arrival time residual.
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
        
        syn_pgd = np.delete(np.array(syn_df['pgd']),1)
        syn_tPGD_orig = np.delete(np.array(syn_df['tPGD_origin']),1)
        syn_tPGD_parriv = np.delete(np.array(syn_df['tPGD_parriv']),1)
        syn_spectra = np.delete(np.array(syn_df.iloc[:,24:]),1,axis=0)
        
        # Observed values
        obs_pgd = np.array(obs_df['pgd'])
        obs_tPGD_orig = np.array(obs_df['tPGD_origin'])
        obs_tPGD_parriv = np.array(obs_df['tPGD_parriv'])
        obs_spectra = np.array(obs_df.iloc[:,24:])
        
        # Calc residuals
        pgd_res = np.log(obs_pgd) - np.log(syn_pgd)
        tPGD_res_linear = obs_tPGD_orig - syn_tPGD_orig
        tPGD_res_ln = np.log(obs_tPGD_orig) - np.log(syn_tPGD_orig)
        spectra_res = np.log(obs_spectra) - np.log(syn_spectra)
        
        out = [pgd_res,tPGD_res_ln,tPGD_res_linear,spectra_res]
    
    elif dtype=='sm':
        syn_pga = np.array(syn_df['pga'])
        syn_pgv = np.array(syn_df['pgv'])
        syn_tPGA_orig = np.array(syn_df['tPGA_origin'])
        syn_tPGA_parriv = np.array(syn_df['tPGA_parriv'])
        syn_spectra = np.array(syn_df.iloc[:,25:45])
        syn_spectra_v = np.array(syn_df.iloc[:,45:65])
    
        # Observed values
        obs_pga = np.array(obs_df['pga'])
        obs_pgv = np.array(obs_df['pgv'])
        obs_tPGA_orig = np.array(obs_df['tPGA_origin'])
        obs_tPGA_parriv = np.array(obs_df['tPGA_parriv'])
        obs_spectra = np.array(obs_df.iloc[:,25:45])
        obs_spectra_v = np.array(obs_df.iloc[:,45:])

        # Calc residuals
        pga_res = np.log(obs_pga) - np.log(syn_pga)
        pgv_res = np.log(obs_pgv) - np.log(syn_pgv)
        tPGA_res_linear = obs_tPGA_orig - syn_tPGA_orig
        tPGA_res_ln = np.log(obs_tPGA_orig) - np.log(syn_tPGA_orig)
        spectra_res = np.log(obs_spectra) - np.log(syn_spectra)
        spectra_res_v = np.log(obs_spectra_v) - np.log(syn_spectra_v)
        
        out = [pga_res,pgv_res,tPGA_res_ln,tPGA_res_linear,spectra_res,spectra_res_v]
    
    return(out)


def plot_spec_res(home_dir, parameter, project, data_types, stations, outliers=True, standard=False):
    """
    Makes residual boxplot figures of the spectra in the IM flatfile created
    using syn_calc_IMs.py. Each plot is created using one project (i.e. stress
    drop of 2.0 MPa), and each frequency bin has a boxplot for all the runs of
    that project. 
    
        Inputs:
            parameter(str): Name of parameter folder.
            project(str): Name of simulation project.  This will be the main
                          directory where the different runs will be store.
             data_types(array): List of data types to make spectra residual
                               plots for. Options are 'disp', 'acc', 'vel'.
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
    from math import floor
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    
    # Residual type to be used in figure name
    label = 'ln Residual'
   
    # Make sure path to safe plots exists
    if not path.exists(f'{home_dir}/{parameter}/{project}/plots/residuals'):
        makedirs(f'{home_dir}/{parameter}/{project}/plots/residuals')
    
    if 'gnss' in stations:
        
        gnss_res_df = pd.read_csv( f'{home_dir}/{parameter}/{project}/flatfiles/residuals/{project}_gnss.csv')
            
        # Define bin edges, spectra residual columns, and other data type parameters
        bin_edges = [0.004, 0.00509, 0.00648, 0.00825, 0.01051, 0.01337,
                     0.01703, 0.02168, 0.02759, 0.03513, 0.04472, 0.05693,
                     0.07248, 0.09227, 0.11746, 0.14953, 0.19037, 0.24234,
                     0.30852, 0.39276, 0.50000]
        
        # Remove GNSS stations with low SNR from residual plots
        poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
        ind = np.where(np.in1d(np.array(gnss_res_df['station']), poor_GNSS).reshape(np.array(gnss_res_df['station']).shape)==False)[0]
    
        gnss_spec_res = gnss_res_df.iloc[:,20:].loc[ind].reset_index(drop=True)
        
        res_units = 'm*s'
        title = f'{project} Displacement Spectra Residuals'
        
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
        
        gnss_bin_list = []
        gnss_std_bin_list = []
        gnss_spec_res_list = np.array([])
        gnss_std_spec_res_list = np.array([])
        
        for i in range(len(bin_edges)-1):
            
            # Current parameter
            bin_res = np.array(gnss_spec_res.iloc[:,i])
            # bin_res = [x for x in bin_res if str(x) != 'nan']
            gnss_bin_list.extend(np.repeat(bin_means[i],len(bin_res)))
            gnss_spec_res_list = np.append(gnss_spec_res_list,bin_res)
            
            # Standard parameters
            if standard==True:
                gnss_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_gnss.csv')
                
                poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
                ind = np.where(np.in1d(np.array(gnss_std_df['station']), poor_GNSS).reshape(np.array(gnss_std_df['station']).shape)==False)[0]
        
                gnss_std_res = np.array(gnss_std_df.iloc[ind].reset_index(drop=True).iloc[:,20:].iloc[:,i])
                gnss_std_bin_list.extend(np.repeat(bin_means[i],len(gnss_std_res)))
                gnss_std_spec_res_list = np.append(gnss_std_spec_res_list,gnss_std_res)
        
        # Create dataframe for current parameter
        # gnss_spec_res_list = [val for sublist in gnss_spec_res_list for val in sublist]
        gnss_spec_data = {'Frequency (Hz)': gnss_bin_list, f'{label} ({res_units})':gnss_spec_res_list}
        gnss_spec_df = pd.DataFrame(data=gnss_spec_data)    
        
        if standard==True:
            # Create dataframe for standard parameters
            gnss_std_spec_data = {'Frequency (Hz)': gnss_std_bin_list, f'{label} ({res_units})':gnss_std_spec_res_list}
            gnss_std_spec_df = pd.DataFrame(data=gnss_std_spec_data)    
        
        # Make figure
        plt.figure(figsize=(10, 10))
        ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=gnss_spec_df,
                          color='orange', showfliers=outliers, boxprops=dict(alpha=.5))
        if standard==True:
            ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=gnss_std_spec_df,
                              showfliers=False, boxprops=dict(fc=(1,0,0,0), ls='--'))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
        ax.set_title(title)
        ax.axhline(0, ls='--')
        
        if standard==True:
            legend_elements = [Patch(facecolor='orange', alpha=0.5, edgecolor='gray',
                                      label='Varied Parameter'),
                                Patch(facecolor='white', edgecolor='gray', ls='--',
                                      label='Standard Parameters')]
        else:
            legend_elements = [Patch(facecolor='orange', alpha=0.5, edgecolor='gray',
                                      label='Varied Parameter')]
        
        ax.legend(handles=legend_elements, loc='upper left')

        
        if outliers == True:
            plt.savefig(f'{home_dir}/{parameter}/{project}/plots/residuals/disp_spectra_res.png', dpi=300)
        else:
            plt.savefig(f'{home_dir}/{parameter}/{project}/plots/residuals/disp_spectra_xout_res.png', dpi=300)
            
        plt.close()
        
    if 'sm' in stations:
        
        sm_res_df = pd.read_csv( f'{home_dir}/{parameter}/{project}/flatfiles/residuals/{project}_sm.csv')
        
        # Remove strong motion stations farther than 600km to avoid surface waves
        far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
        ind = np.where(np.in1d(np.array(sm_res_df['station']), far_sm).reshape(np.array(sm_res_df['station']).shape)==False)[0]
    
        # Define bin edges, spectra residual columns, and other data type parameters
        bin_edges = [4.00000000e-03, 5.91503055e-03, 8.74689659e-03, 1.29345401e-02,
               1.91270500e-02, 2.82842712e-02, 4.18255821e-02, 6.18498989e-02,
               9.14610104e-02, 1.35248668e-01, 2.00000000e-01, 2.95751527e-01,
               4.37344830e-01, 6.46727007e-01, 9.56352500e-01, 1.41421356e+00,
               2.09127911e+00, 3.09249495e+00, 4.57305052e+00, 6.76243338e+00,
               1.00000000e+01]
        
        acc_spec_res = sm_res_df.iloc[:,21:41].loc[ind].reset_index(drop=True)
        vel_spec_res = sm_res_df.iloc[:,41:61].loc[ind].reset_index(drop=True)
        
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
        vel_spec_res_list = np.array([])
        vel_std_spec_res_list = np.array([])
        
        for i in range(len(bin_edges)-1):
            
            # Current parameter
            acc_bin_res = np.array(acc_spec_res)[:,i]
            # bin_res = [x for x in bin_res if str(x) != 'nan']
            sm_bin_list.extend(np.repeat(bin_means[i],len(acc_bin_res)))
            acc_spec_res_list = np.append(acc_spec_res_list,acc_bin_res)
            
            vel_bin_res = np.array(vel_spec_res)[:,i]
            vel_spec_res_list = np.append(vel_spec_res_list,vel_bin_res)
            
            # Standard parameters
            if standard==True:
                sm_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_sm.csv')
                
                # Remove strong motion stations farther than 600km to avoid surface waves
                far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
                ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
            
                acc_std_res = np.array(sm_std_df.iloc[ind].reset_index(drop=True).iloc[:,21:41].iloc[:,i])
            
                sm_std_bin_list.extend(np.repeat(bin_means[i],len(acc_std_res)))
                acc_std_spec_res_list = np.append(acc_std_spec_res_list,acc_std_res)
                
                vel_std_res = np.array(sm_std_df.iloc[ind].reset_index(drop=True).iloc[:,41:].iloc[:,i])
                vel_std_spec_res_list = np.append(vel_std_spec_res_list,vel_std_res)
        
        ###### Acceleration
        res_units = 'm/s'
        title = 'Acceleration Spectra Residuals'
        acc_spec_data = {'Frequency (Hz)': sm_bin_list, f'{label} ({res_units})':acc_spec_res_list}
        acc_spec_df = pd.DataFrame(data=acc_spec_data) 
        
        if standard==True:
            # Create dataframe for standard parameters
            acc_std_spec_data = {'Frequency (Hz)': sm_std_bin_list, f'{label} ({res_units})':acc_std_spec_res_list}
            acc_std_spec_df = pd.DataFrame(data=acc_std_spec_data)    
          
        
        # Make figure
        plt.figure(figsize=(7,5))
        ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=acc_spec_df,
                          color='orange', showfliers=outliers, boxprops=dict(alpha=.5))
        if standard==True:
            ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=acc_std_spec_df,
                              showfliers=False, boxprops=dict(fc=(1,0,0,0), ls='--'))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30,fontsize=10)
        ax.set_title(title)
        ax.axhline(0, ls='--')
        
        if standard==True:
            legend_elements = [Patch(facecolor='orange', alpha=0.5, edgecolor='gray',
                                      label=f'{project}'),
                                Patch(facecolor='white', edgecolor='gray', ls='--',
                                      label='Standard Parameters')]
        else:
            legend_elements = [Patch(facecolor='orange', alpha=0.5, edgecolor='gray',
                                      label=f'{project}')]
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel(f'{label} ({res_units})', fontsize=12)
        ax.legend(handles=legend_elements, loc='upper right')
        plt.subplots_adjust(bottom=0.15, top=0.94, left=0.1, right=0.98)
        
        if outliers == True:
            plt.savefig(f'{home_dir}/{parameter}/{project}/plots/residuals/acc_spectra_res.png', dpi=300)
        else:
            plt.savefig(f'{home_dir}/{parameter}/{project}/plots/residuals/acc_spectra_xout_res.png', dpi=300)
        
        plt.close()
        
        ###### Velocity
        res_units = 'm'
        title = f'{project} Velocity Spectra Residuals'
        vel_spec_data = {'Frequency (Hz)': sm_bin_list, f'{label} ({res_units})':vel_spec_res_list}
        vel_spec_df = pd.DataFrame(data=vel_spec_data) 
        
        if standard==True:
            # Create dataframe for standard parameters
            vel_std_spec_data = {'Frequency (Hz)': sm_std_bin_list, f'{label} ({res_units})':vel_std_spec_res_list}
            vel_std_spec_df = pd.DataFrame(data=vel_std_spec_data)    
        
        # Make figure
        plt.figure(figsize=(10, 10))
        ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=vel_spec_df,
                          color='orange', showfliers=outliers, boxprops=dict(alpha=.5))
        if standard==True:
            ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=vel_std_spec_df,
                              showfliers=False, boxprops=dict(fc=(1,0,0,0), ls='--'))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
        ax.set_title(title)
        ax.axhline(0, ls='--')
        
        if standard==True:
            legend_elements = [Patch(facecolor='orange', alpha=0.5, edgecolor='gray',
                                      label='Varied Parameter'),
                                Patch(facecolor='white', edgecolor='gray', ls='--',
                                      label='Standard Parameters')]
        else:
          legend_elements = [Patch(facecolor='orange', alpha=0.5, edgecolor='gray',
                                    label='Varied Parameter')]  
        
        ax.legend(handles=legend_elements, loc='upper left')
        
        if outliers == True:
            plt.savefig(f'{home_dir}/{parameter}/{project}/plots/residuals/vel_spectra_res.png', dpi=300)
        else:
            plt.savefig(f'{home_dir}/{parameter}/{project}/plots/residuals/vel_spectra_xout_res.png', dpi=300)
        
        
        plt.close()


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
    
    # Make deirectory if it does not already exist
    if not path.exists(figdir):
        makedirs(figdir)
    
    # Set up empty lists to store the project names in for the disp IMs, the sm
        # IMs, and the standard parameter IMs
    disp_project_list = []
    sm_project_list = []
    
    # Set up empty lists to store the current project residuals in
    pgd_res_list = []
    pga_res_list = []
    pgv_res_list = []
    tPGD_res_list = []
    tPGA_res_list = []
    
    # Add default parameter first if risetime 
    if standard == True and parameter == 'risetime':
        for i, project in enumerate(projects):
            val = '5.4'
            
            if 'gnss' in stations:  
                gnss_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_gnss.csv')
               
                # Remove GNSS stations with low SNR from residual plots
                poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
                ind = np.where(np.in1d(np.array(gnss_std_df['station']), poor_GNSS).reshape(np.array(gnss_std_df['station']).shape)==False)[0]
            
                # Select out residuals 
                std_pgd_res = np.array(gnss_std_df['pgd_res'])[ind]
                std_tPGD_res = np.array(gnss_std_df['tPGD_origin_res'])[ind]
                
                # Get rid of NaNs if there are any
                std_pgd_res = [x for x in std_pgd_res if str(x) != 'nan']
                std_tPGD_res = [x for x in std_tPGD_res if str(x) != 'nan']
                
                # Append residuals from this project to main lists
                pgd_res_list.append(std_pgd_res)
                tPGD_res_list.append(std_tPGD_res)
                
                # Get parameter value.  Need different lists for disp and sm runs because 
                    # there are a different number of stations. 
                for j in range(len(std_pgd_res)):
                    disp_project_list.append(val)
            
            if 'sm' in stations:
                sm_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_sm.csv')
            
                # Remove strong motion stations farther than 600km to avoid surface waves
                far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
                ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
            
                # Select out residuals 
                std_pga_res = np.array(sm_std_df['pga_res'])[ind]
                std_pgv_res = np.array(sm_std_df['pgv_res'])[ind]
                std_tPGA_res = np.array(sm_std_df['tPGA_origin_res'])[ind]

                # Get rid of NaNs if there are any
                std_pga_res = [x for x in std_pga_res if str(x) != 'nan']
                std_pgv_res = [x for x in std_pgv_res if str(x) != 'nan']
                std_tPGA_res = [x for x in std_tPGA_res if str(x) != 'nan']
            
                # Append residuals from this project to main lists
                pga_res_list.append(std_pga_res)
                pgv_res_list.append(std_pgv_res)
                tPGA_res_list.append(std_tPGA_res)
                
                # Get parameter value.  Need different lists for disp and sm runs because 
                    # there are a different number of stations. 
                for j in range(len(std_pga_res)):
                    sm_project_list.append(val)

        
    
    # Loop through projects and put residuals into lists
    for i, project in enumerate(projects):
        
        # Residual dataframes
        if 'gnss' in stations:
            gnss_res_df = pd.read_csv( f'{home_dir}/{parameter}/{project}/flatfiles/residuals/{project}_gnss.csv')
            
            # Remove GNSS stations with low SNR from residual plots
            poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
            ind = np.where(np.in1d(np.array(gnss_res_df['station']), poor_GNSS).reshape(np.array(gnss_res_df['station']).shape)==False)[0]
        
            # Select out residuals 
            pgd_res = np.array(gnss_res_df['pgd_res'])[ind]
            tPGD_res = np.array(gnss_res_df['tPGD_origin_res'])[ind]
            
            # Get rid of NaNs if there are any
            pgd_res = [x for x in pgd_res if str(x) != 'nan']
            tPGD_res = [x for x in tPGD_res if str(x) != 'nan']
            tPGD_res_list.append(tPGD_res)
            
            # Append residuals from this project to main lists
            pgd_res_list.append(pgd_res)
            
            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(pgd_res)):
                disp_project_list.append(param_vals[i])
            
        if 'sm' in stations:
            sm_res_df = pd.read_csv( f'{home_dir}/{parameter}/{project}/flatfiles/residuals/{project}_sm.csv')
            
            # Remove strong motion stations farther than 600km to avoid surface waves
            far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
            ind = np.where(np.in1d(np.array(sm_res_df['station']), far_sm).reshape(np.array(sm_res_df['station']).shape)==False)[0]
        
            # Select out residuals 
            pga_res = np.array(sm_res_df['pga_res'])[ind]
            pgv_res = np.array(sm_res_df['pgv_res'])[ind]
            tPGA_res = np.array(sm_res_df['tPGA_origin_res'])[ind]
            
            # Get rid of NaNs if there are any
            pga_res = [x for x in pga_res if str(x) != 'nan']
            pgv_res = [x for x in pgv_res if str(x) != 'nan']
            tPGA_res = [x for x in tPGA_res if str(x) != 'nan']
            
            # Append residuals from this project to main lists
            pga_res_list.append(pga_res)
            pgv_res_list.append(pgv_res)
            tPGA_res_list.append(tPGA_res)

            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(pga_res)):
                sm_project_list.append(param_vals[i])

    # Add default parameter last if not risetime
    if standard == True and parameter != 'risetime':
        
        if parameter == 'stress_drop':
            val = '5'
        elif parameter == 'vrupt':
            val = '1.7'
        else:
            val = 'Standard'
        
        # Residual dataframe for default (aka stdtered) parameters
        # std_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/standard/std/flatfiles/residuals/std_{res}.csv')
        
        if 'gnss' in stations:  
            gnss_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_gnss.csv')
           
             # Remove GNSS stations with low SNR from residual plots
            poor_GNSS = ['MKMK', 'LNNG', 'LAIS', 'TRTK', 'MNNA', 'BTHL']
            ind = np.where(np.in1d(np.array(gnss_std_df['station']), poor_GNSS).reshape(np.array(gnss_std_df['station']).shape)==False)[0]
        
            # Select out residuals 
            std_pgd_res = np.array(gnss_std_df['pgd_res'])[ind]
            std_tPGD_res = np.array(gnss_std_df['tPGD_origin_res'])[ind]
            
            # Get rid of NaNs if there are any
            std_pgd_res = [x for x in std_pgd_res if str(x) != 'nan']
            std_tPGD_res = [x for x in std_tPGD_res if str(x) != 'nan']
            
            # Append residuals from this project to main lists
            pgd_res_list.append(std_pgd_res)
            tPGD_res_list.append(std_tPGD_res)
            
            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(std_pgd_res)):
                disp_project_list.append(val)
        
        if 'sm' in stations:
            sm_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_sm.csv')
        
            # Remove strong motion stations farther than 600km to avoid surface waves
            far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
            ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
        
            # Select out residuals 
            std_pga_res = np.array(sm_std_df['pga_res'])[ind]
            std_pgv_res = np.array(sm_std_df['pgv_res'])[ind]
            std_tPGA_res = np.array(sm_std_df['tPGA_origin_res'])[ind]

            # Get rid of NaNs if there are any
            std_pga_res = [x for x in std_pga_res if str(x) != 'nan']
            std_pgv_res = [x for x in std_pgv_res if str(x) != 'nan']
            std_tPGA_res = [x for x in std_tPGA_res if str(x) != 'nan']
        
            # Append residuals from this project to main lists
            pga_res_list.append(std_pga_res)
            pgv_res_list.append(std_pgv_res)
            tPGA_res_list.append(std_tPGA_res)
            
            # Get parameter value.  Need different lists for disp and sm runs because 
                # there are a different number of stations. 
            for j in range(len(std_pga_res)):
                sm_project_list.append(val)

        
    ######################## Plot IMs on one figure ###########################

    label = 'ln Residual'
        
    if 'gnss' in stations and 'sm' not in stations:
        
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
        pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
        pgd_data = {'Parameter':disp_project_list, f'{label} (m)':pgd_res_list}
        pgd_df = pd.DataFrame(data=pgd_data)       
                
        sns.boxplot(x='Parameter', y=f'{label} (m)', data=pgd_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax1).set(xlabel=None,ylabel='ln Residual (m)')
        yabs_max = abs(max(ax1.get_ylim(), key=abs))
        ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax1.set_title('PGD')
        ax1.axhline(0, ls='--')
        
        # tPGD subplot
        tPGD_res_list = [val for sublist in tPGD_res_list for val in sublist]
        tPGD_data = {'Parameter':disp_project_list, f'{label} (s)':tPGD_res_list}
        tPGD_df = pd.DataFrame(data=tPGD_data)
                           
        sns.boxplot(x='Parameter', y=f'{label} (s)', data=tPGD_df, showfliers=outliers, boxprops=dict(alpha=.3),
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
        
        # PGA subplot
        pga_res_list = [val for sublist in pga_res_list for val in sublist]
        pga_data = {'Parameter':sm_project_list, f'{label} (m/s/s)':pga_res_list}
        pga_df = pd.DataFrame(data=pga_data)
                           
        sns.boxplot(x='Parameter', y=f'{label} (m/s/s)', data=pga_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax1).set(xlabel=None,ylabel=r'ln Residual (m/s$^2$)')
        yabs_max = abs(max(ax1.get_ylim(), key=abs))
        ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax1.tick_params(left=False,right=False,bottom=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=20)
        ax1.set_title('PGA')
        ax1.axhline(0, ls='--')
        
        # PGV subplot
        pgv_res_list = [val for sublist in pgv_res_list for val in sublist]
        pgv_data = {'Parameter':sm_project_list, f'{label} (m/s)':pgv_res_list}
        pgv_df = pd.DataFrame(data=pgv_data)
                           
        sns.boxplot(x='Parameter', y=f'{label} (m/s)', data=pgv_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax2).set(xlabel=None,ylabel=r'ln Residual (m/s)')
        yabs_max = abs(max(ax2.get_ylim(), key=abs))
        ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax2.set_title('PGV')
        ax2.axhline(0, ls='--')
        
        fig.supxlabel(xlabel,fontsize=20)
        
        if outliers == True:
            plt.savefig(f'{figdir}/IMs_sm.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'{figdir}/IMs_xout_sm.png', bbox_inches='tight', dpi=300)
        # plt.close()
        # plt.show()
    
    elif 'sm' in stations and 'gnss' in stations:
        # Set up Figure
        # fig, axs = plt.subplots(2, 2, figsize=(8, 10))
        # ax1 = axs[0,0]
        # ax2 = axs[0,1]
        # ax3 = axs[1,0]
        # ax4 = axs[1,1]
        
        fig, axs = plt.subplots(1, 4, figsize=(15,6))
        ax1 = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]
        ax4 = axs[3]
    
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
        pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
        pgd_data = {'Parameter':disp_project_list, f'{label} (m)':pgd_res_list}
        pgd_df = pd.DataFrame(data=pgd_data)       
                
        b = sns.boxplot(x='Parameter', y=f'{label} (m)', data=pgd_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax1)
        b.set(xlabel=None)
        b.set_ylabel('ln Residual (m)',fontsize=18)
        yabs_max = abs(max(ax1.get_ylim(), key=abs))
        # ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax1.set_ylim(ymin=-2.5, ymax=2.5)
        ax1.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax1.yaxis.set_major_locator(MultipleLocator(1))
        ax1.set_title('PGD',fontsize=18)
        ax1.axhline(0, ls='--')
        
        # tPGD subplot
        tPGD_res_list = [val for sublist in tPGD_res_list for val in sublist]
        tPGD_data = {'Parameter':disp_project_list, f'{label} (s)':tPGD_res_list}
        tPGD_df = pd.DataFrame(data=tPGD_data)
                           
        b = sns.boxplot(x='Parameter', y=f'{label} (s)', data=tPGD_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax2)
        b.set(xlabel=None)
        b.set_ylabel('Residual (s)',fontsize=18)
        yabs_max = abs(max(ax2.get_ylim(), key=abs))
        ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax2.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax2.set_title('tPGD',fontsize=18)
        ax2.axhline(0, ls='--')
        
        # PGA subplot
        pga_res_list = [val for sublist in pga_res_list for val in sublist]
        pga_data = {'Parameter':sm_project_list, f'{label} (m/s/s)':pga_res_list}
        pga_df = pd.DataFrame(data=pga_data)
                           
        b = sns.boxplot(x='Parameter', y=f'{label} (m/s/s)', data=pga_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax3)
        b.set(xlabel=None)
        b.set_ylabel(r'ln Residual (m/s$^2$)',fontsize=18)
        yabs_max = abs(max(ax3.get_ylim(), key=abs))
        # ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax3.set_ylim(ymin=-2.5, ymax=2.5)
        ax3.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax3.yaxis.set_major_locator(MultipleLocator(1))
        ax3.set_title('PGA',fontsize=18)
        ax3.axhline(0, ls='--')
        
        # PGV subplot
        pgv_res_list = [val for sublist in pgv_res_list for val in sublist]
        pgv_data = {'Parameter':sm_project_list, f'{label} (m/s)':pgv_res_list}
        pgv_df = pd.DataFrame(data=pgv_data)
                           
        b = sns.boxplot(x='Parameter', y=f'{label} (m/s)', data=pgv_df, showfliers=outliers, boxprops=dict(alpha=.3),
                    ax=ax4)
        b.set(xlabel=None)
        b.set_ylabel('ln Residual (m/s)',fontsize=18)
        # yabs_max = abs(max(ax4.get_ylim(), key=abs))
        ax4.set_ylim(ymin=-2.5, ymax=2.5)
        # ax4.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax4.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=18)
        ax4.yaxis.set_major_locator(MultipleLocator(1))
        ax4.set_title('PGV',fontsize=18)
        ax4.axhline(0, ls='--')
        
        plt.subplots_adjust(wspace=0.45,bottom=0.2,left=0.01)
        
        fig.supxlabel(xlabel,fontsize=18)
        
        if outliers == True:
            plt.savefig(f'{figdir}/{parameter}_IMs.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'{figdir}/{parameter}_IMs_xout.png', bbox_inches='tight', dpi=300)
        plt.close()
        # plt.show()
    
    return()


