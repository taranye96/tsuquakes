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


def calc_res(parameter, project, run, ln=True):
    """
    Calculates residuals between synthetic and observed data, and puts residuals
    into a dataframe.
    
    Inputs:
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
    syn_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/IMs/{run}.csv')
    syn_pgd = np.array(syn_df['pgd'])
    syn_pga = np.array(syn_df['pga'])
    syn_pgv = np.array(syn_df['pgv'])
    syn_tPGD_orig = np.array(syn_df['tPGD_origin'])
    syn_tPGD_parriv = np.array(syn_df['tPGD_parriv'])
    syn_tPGA_orig = np.array(syn_df['tPGA_origin'])
    syn_tPGA_parriv = np.array(syn_df['tPGA_parriv'])
    syn_spectra = np.array(syn_df.iloc[:,28:250])
    
    # Observed dataframe contains GNSS and strong motion data. Read in only
        # parts of it if you only have one type of data
    if len(syn_df) == 29:
        obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs2.csv')      # full dataframe
    elif len(syn_df)==16:
        obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs2.csv')[13:] # strong motion stations only
    elif len(syn_df)==13:
        obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs2.csv')[:13] # gnss stations only
    
    # Observed values
    obs_pgd = np.array(obs_df['pgd'])
    obs_pga = np.array(obs_df['pga'])
    obs_pgv = np.array(obs_df['pgv'])
    obs_tPGD_orig = np.array(obs_df['tPGD_origin'])
    obs_tPGD_parriv = np.array(obs_df['tPGD_parriv'])
    obs_tPGA_orig = np.array(obs_df['tPGA_origin'])
    obs_tPGA_parriv = np.array(obs_df['tPGA_parriv'])
    obs_spectra = np.array(obs_df.iloc[:,28:250])
    
    if ln:
        # calc res
        pgd_res = np.log(obs_pgd) - np.log(syn_pgd)
        pga_res = np.log(obs_pga) - np.log(syn_pga)
        pgv_res = np.log(obs_pgv) - np.log(syn_pgv)
        tPGD_orig_res = np.log(obs_tPGD_orig) - np.log(syn_tPGD_orig)
        tPGD_parriv_res = np.log(obs_tPGD_parriv) - np.log(syn_tPGD_parriv)
        tPGA_orig_res = np.log(obs_tPGA_orig) - np.log(syn_tPGA_orig)
        tPGA_parriv_res = np.log(obs_tPGA_parriv) - np.log(syn_tPGA_parriv)
        spectra_res = np.log(obs_spectra) - np.log(syn_spectra)
    
    else:
        # calc res
        pgd_res = obs_pgd - syn_pgd
        pga_res = obs_pga - syn_pga
        pgv_res = obs_pgv - syn_pgv
        tPGD_orig_res = obs_tPGD_orig - syn_tPGD_orig
        tPGD_parriv_res = obs_tPGD_parriv - syn_tPGD_parriv
        tPGA_orig_res = obs_tPGA_orig - syn_tPGA_orig
        tPGA_parriv_res = obs_tPGA_parriv - syn_tPGA_parriv
        spectra_res = obs_spectra - syn_spectra
    
    return (pgd_res,pga_res,pgv_res,tPGD_orig_res,tPGD_parriv_res,tPGA_orig_res,
            tPGA_parriv_res,spectra_res)


def plot_res(parameter, project, rupture_list, ln=True):
    """
    Makes residual boxplot figures of the IMs in the IM flatfile created using
    syn_calc_IMs.py. Each plot is created using one project (i.e. stress drop
    of 2.0 MPa), and each run is a different boxplot on the figure. 
    
        Inputs:
            parameter(str): Name of parameter folder.
            project(str): Name of simulation project.  This will be the main
                          directory where the different runs will be store. 
            rupture_list(array): List of .rupt files created using FakeQuakes.
            ln(T/F): If true, calculates the natural log of the residuals.
        
        Output:
            Just saves the plots to ther respective directories. 
            
    """
    
    import numpy as np
    import pandas as pd
    from os import path, makedirs
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Make paths for plots
    param_dir = '/Users/tnye/FakeQuakes/parameters'
    if not path.exists(f'{param_dir}/{parameter}/{project}/plots/residuals'):
        makedirs(f'{param_dir}/{parameter}/{project}/plots/residuals')
    if not path.exists(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra'):
        makedirs(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra')
    if not path.exists(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/vel'):
        makedirs(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/acc')
    if not path.exists(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/vel'):
        makedirs(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/vel')
    if not path.exists(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/acc'):
        makedirs(f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/disp')
    
    # Set up empty lists to fill 
    pgd_run_list = []
    pga_run_list = []
    pgv_run_list = []
    tPGD_run_list = []
    tPGA_run_list = []
    E_disp_bin8_run_list = []
    E_disp_bin14_run_list = []
    E_disp_bin20_run_list = []
    E_acc_bin8_run_list = []
    E_acc_bin14_run_list = []
    E_acc_bin20_run_list = []
    E_vel_bin8_run_list = []
    E_vel_bin14_run_list = []
    E_vel_bin20_run_list = []
    
    pgd_res_list = []
    pga_res_list = []
    pgv_res_list = []
    tPGD_res_list = []
    tPGA_res_list = []
    E_disp_bin8_list = []
    E_disp_bin14_list = []
    E_disp_bin20_list = []
    E_acc_bin8_list = []
    E_acc_bin14_list = []
    E_acc_bin20_list = []
    E_vel_bin8_list = []
    E_vel_bin14_list = []
    E_vel_bin20_list = []


    for i, rupture in enumerate(rupture_list):
        
        # Remove .rupt to obtain run name
        run = rupture.rsplit('.', 1)[0]
         
        # Residual dataframe
        res_df = pd.read_csv( f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/residuals/runs/{run}_res.csv')
        
        # Pull out residuals 
        pgd_res = np.array(res_df['pgd_res'])
        pga_res = np.array(res_df['pga_res'])
        pgv_res = np.array(res_df['pgv_res'])
        tPGD_res = np.array(res_df['tPGD_origin_res'])
        tPGA_res = np.array(res_df['tPGA_origin_res'])
        E_disp_bin8 = np.array(res_df['E_disp_bin8'])
        E_disp_bin14 = np.array(res_df['E_disp_bin14'])
        E_disp_bin20 =np.array(res_df['E_disp_bin20'])
        E_acc_bin8 = np.array(res_df['E_acc_bin8'])
        E_acc_bin14 = np.array(res_df['E_acc_bin14'])
        E_acc_bin20 =np.array(res_df['E_acc_bin20'])
        E_vel_bin8 = np.array(res_df['E_vel_bin8'])
        E_vel_bin14 = np.array(res_df['E_vel_bin14'])
        E_vel_bin20 =np.array(res_df['E_vel_bin20'])
        
        # Get rid of NaNs
        pgd_res = [x for x in pgd_res if str(x) != 'nan']
        pga_res = [x for x in pga_res if str(x) != 'nan']
        pgv_res = [x for x in pgv_res if str(x) != 'nan']
        tPGD_res = [x for x in tPGD_res if str(x) != 'nan']
        tPGA_res = [x for x in tPGA_res if str(x) != 'nan']
        E_disp_bin8 = [x for x in E_disp_bin8 if str(x) != 'nan']
        E_disp_bin14 = [x for x in E_disp_bin14 if str(x) != 'nan']
        E_disp_bin20 = [x for x in E_disp_bin20 if str(x) != 'nan']
        E_acc_bin8 = [x for x in E_acc_bin8 if str(x) != 'nan']
        E_acc_bin14 = [x for x in E_acc_bin14 if str(x) != 'nan']
        E_acc_bin20 = [x for x in E_acc_bin20 if str(x) != 'nan']
        E_vel_bin8 = [x for x in E_vel_bin8 if str(x) != 'nan']
        E_vel_bin14 = [x for x in E_vel_bin14 if str(x) != 'nan']
        E_vel_bin20 = [x for x in E_vel_bin20 if str(x) != 'nan']
        
        # Append residuals to main list
        pgd_res_list.append(pgd_res)
        pga_res_list.append(pga_res)
        pgv_res_list.append(pgv_res)
        tPGD_res_list.append(tPGD_res)
        tPGA_res_list.append(tPGA_res)
        E_disp_bin8_list.append(E_disp_bin8)
        E_disp_bin14_list.append(E_disp_bin14)
        E_disp_bin20_list.append(E_disp_bin20)
        E_acc_bin8_list.append(E_acc_bin8)
        E_acc_bin14_list.append(E_acc_bin14)
        E_acc_bin20_list.append(E_acc_bin20)
        E_vel_bin8_list.append(E_vel_bin8)
        E_vel_bin14_list.append(E_vel_bin14)
        E_vel_bin20_list.append(E_vel_bin20)

        # get run number
        for j in range(len(pgd_res)):
            pgd_run_list.append(i)
        for j in range(len(pga_res)):
            pga_run_list.append(i)
        for j in range(len(pgv_res)):
            pgv_run_list.append(i)
        for j in range(len(tPGD_res)):
            tPGD_run_list.append(i)
        for j in range(len(tPGA_res)):
            tPGA_run_list.append(i)
        for j in range(len(E_disp_bin8)):
            E_disp_bin8_run_list.append(i)
        for j in range(len(E_disp_bin14)):
            E_disp_bin14_run_list.append(i)
        for j in range(len(E_disp_bin20)):
            E_disp_bin20_run_list.append(i)
        for j in range(len(E_acc_bin8)):
            E_acc_bin8_run_list.append(i)
        for j in range(len(E_acc_bin14)):
            E_acc_bin14_run_list.append(i)
        for j in range(len(E_acc_bin20)):
            E_acc_bin20_run_list.append(i)
        for j in range(len(E_vel_bin8)):
            E_vel_bin8_run_list.append(i)
        for j in range(len(E_vel_bin14)):
            E_vel_bin14_run_list.append(i)
        for j in range(len(E_vel_bin20)):
            E_vel_bin20_run_list.append(i)
    
    # Residual type to be used in figure name
    if ln:
        res='lnres'
    else:
        res='res'

    # PGD boxplot
    pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
    pgd_data = {'Run#':pgd_run_list, 'Residual(m)':pgd_res_list}
    pgd_df = pd.DataFrame(data=pgd_data)       
            
    ax = sns.catplot(x='Run#', y='Residual(m)', data=pgd_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=pgd_df, boxprops=dict(alpha=.3))
    ax.set_title('PGD')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-0.8, 0.8))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/pgd_{res}.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # PGA boxplot
    pga_res_list = [val for sublist in pga_res_list for val in sublist]
    pga_data = {'Run#':pga_run_list, 'Residual(m/s/s)':pga_res_list}
    pga_df = pd.DataFrame(data=pga_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s/s)', data=pga_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s/s)', data=pga_df, boxprops=dict(alpha=.3))
    ax.set_title('PGA')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.8, .8))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/pga_{res}.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # PGV boxplot
    pgv_res_list = [val for sublist in pgv_res_list for val in sublist]
    pgv_data = {'Run#':pgv_run_list, 'Residual(m/s)':pgv_res_list}
    pgv_df = pd.DataFrame(data=pgv_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=pgv_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=pgv_df, boxprops=dict(alpha=.3))
    ax.set_title('PGV')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.8, .8))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/pgv_{res}.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # tPGD boxplot
    tPGD_res_list = [val for sublist in tPGD_res_list for val in sublist]
    tPGD_data = {'Run#':tPGD_run_list, 'Residual(s)':tPGD_res_list}
    tPGD_df = pd.DataFrame(data=tPGD_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(s)', data=tPGD_df)
    ax = sns.boxplot(x='Run#', y='Residual(s)', data=tPGD_df, boxprops=dict(alpha=.3))
    ax.set_title('tPGD')
    ax.axhline(0, ls='--')
    # ax.set(ylim=(-.8, .8))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/tPGD_orig_{res}.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # tPGA boxplot
    tPGA_res_list = [val for sublist in tPGA_res_list for val in sublist]
    tPGA_data = {'Run#':tPGA_run_list, 'Residual(s)':tPGA_res_list}
    tPGA_df = pd.DataFrame(data=tPGA_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(s)', data=tPGA_df)
    ax = sns.boxplot(x='Run#', y='Residual(s)', data=tPGA_df, boxprops=dict(alpha=.3))
    ax.set_title('tPGD')
    ax.axhline(0, ls='--')
    # ax.set(ylim=(-.8, .8))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/tPGA_orig_{res}.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    return()


def plot_spec_res(parameter, project, data_types, ln=True, outliers=True, default=False):
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
    from math import log10, floor
    from itertools import repeat
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    
    # Residual type to be used in figure name
    if ln:
        res='lnres'
        label = 'ln Residual'
    else:
        res='res'
        label = 'Residual'

    # Residual dataframe for specified project
    res_df = pd.read_csv( f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/residuals/{project}_{res}.csv')
   
    if default:
        # Residual dataframe for default (aka standard) parameters
        std_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/standard/std/flatfiles/residuals/std_{res}.csv')
    
    # Make sure path to safe plots exists
    if not path.exists(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/plots/residuals'):
        makedirs(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/plots/residuals')

    # Loop through data_types
    for data in data_types:
        
        # Define bin edges, spectra residual columns, and other data type parameters
        if data == 'disp':
            bin_edges = [0.004, 0.00509, 0.00648, 0.00825, 0.01051, 0.01337,
                         0.01703, 0.02168, 0.02759, 0.03513, 0.04472, 0.05693,
                         0.07248, 0.09227, 0.11746, 0.14953, 0.19037, 0.24234,
                         0.30852, 0.39276, 0.50000]
            spec_res = res_df.iloc[:,24:44]
            if default:
                std_spec_res = std_df.iloc[:,24:44]
            res_units = 'm*s'
            title = f'{project} Displacement Spectra Residuals'
            
        elif data == 'acc':
            bin_edges = [0.004, 0.00592, 0.00875, 0.01293, 0.01913, 0.02828,
                         0.04183, 0.06185, 0.09146, 0.13525, 0.20000, 0.29575,
                         0.43734, 0.64673, 0.95635, 1.41421, 2.09128, 3.09249,
                         4.57305, 6.76243, 10.00000]
            spec_res = res_df.iloc[:,84:104]
            if default:
                std_spec_res = std_df.iloc[:,84:104]
            res_units = 'm/s'
            title = f'{project} Acceleration Spectra Residuals'
        
        elif data == 'vel':
            bin_edges = [0.004, 0.00592, 0.00875, 0.01293, 0.01913, 0.02828,
                         0.04183, 0.06185, 0.09146, 0.13525, 0.20000, 0.29575,
                         0.43734, 0.64673, 0.95635, 1.41421, 2.09128, 3.09249,
                         4.57305, 6.76243, 10.00000]
            spec_res = res_df.iloc[:,144:164]
            if default:
                std_spec_res = std_df.iloc[:,144:164]
            res_units = 'm'
            title = f'{project} Velocity Spectra Residuals'
        
        # Obtain bin means from bin edges
        bin_means = []
        for i in range(len(bin_edges)):
            if i != 0:
                # mean = np.exp((np.log10(bin_edges[i])+np.log10(bin_edges[i-1]))/2)
                mean = np.sqrt(bin_edges[i]*bin_edges[i-1])
                bin_means.append(mean)
            
        # Function to round bin means
        def round_sig(x, sig):
            return round(x, sig-int(floor(log10(abs(x))))-1)
        
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
        
        bin_list = []
        std_bin_list = []
        spec_res_list = []
        std_spec_res_list = []
        
        for i in range(len(bin_edges)-1):
            
            # Current parameter
            bin_res = np.array(spec_res.iloc[:,i])
            bin_res = [x for x in bin_res if str(x) != 'nan']
            bin_list.extend(repeat(bin_means[i],len(bin_res)))
            spec_res_list.append(bin_res)
            
            # Standard parameters
            if default:
                std_res = np.array(std_spec_res.iloc[:,i])
                std_res = [x for x in std_res if str(x) != 'nan']
                std_bin_list.extend(repeat(bin_means[i],len(std_res)))
                std_spec_res_list.append(std_res)
        
        # Create dataframe for current parameter
        spec_res_list = [val for sublist in spec_res_list for val in sublist]
        spec_data = {'Frequency (Hz)': bin_list, f'{label} ({res_units})':spec_res_list}
        spec_df = pd.DataFrame(data=spec_data)    
        
        if default:
            # Create dataframe for standard parameters
            std_spec_res_list = [val for sublist in std_spec_res_list for val in sublist]
            std_spec_data = {'Frequency (Hz)': std_bin_list, f'{label} ({res_units})':std_spec_res_list}
            std_spec_df = pd.DataFrame(data=std_spec_data)    
        
        # Make figure
        plt.figure(figsize=(10, 10))
        ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=spec_df,
                         color='orange', showfliers=outliers, boxprops=dict(alpha=.5))
        if default:
            ax = sns.boxplot(x='Frequency (Hz)', y=f'{label} ({res_units})', data=std_spec_df,
                             showfliers=False, boxprops=dict(fc=(1,0,0,0), ls='--'))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
        ax.set_title(title)
        ax.axhline(0, ls='--')
        
        legend_elements = [Patch(facecolor='orange', alpha=0.5, edgecolor='gray',
                                 label='Varied Parameter'),
                           Patch(facecolor='white', edgecolor='gray', ls='--',
                                 label='Standard Parameters')]
        
        ax.legend(handles=legend_elements, loc='upper left')

        
        if outliers == True:
            plt.savefig(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/plots/residuals/{data}_spec_{res}.png', dpi=300)
        else:
            plt.savefig(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/plots/residuals/{data}_spec_xout_{res}.png', dpi=300)
        plt.close()


def plot_IM_res_full(parameter, projects, param_vals, ln=True, outliers=True, default=False):
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
    
    # Directory for plots using all the runs
    figdir = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}'
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
    
    # Residual type to be used in figure name
    if ln:
        res='lnres'
    else:
        res='res'
    
    # Add default parameter first if risetime 
    if parameter == 'rise_time':
        val = '1'
    
        # Residual dataframe for default (aka stdtered) parameters
        std_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/standard/std/flatfiles/residuals/std_{res}.csv')
        
        # Select out residuals 
        std_pgd_res = np.array(std_df['pgd_res'])
        std_pga_res = np.array(std_df['pga_res'])
        std_pgv_res = np.array(std_df['pgv_res'])
        std_tPGD_res = np.array(std_df['tPGD_origin_res'])
        std_tPGA_res = np.array(std_df['tPGA_origin_res'])
        
        # Get rid of NaNs if there are any
        std_pgd_res = [x for x in std_pgd_res if str(x) != 'nan']
        std_pga_res = [x for x in std_pga_res if str(x) != 'nan']
        std_pgv_res = [x for x in std_pgv_res if str(x) != 'nan']
        std_tPGD_res = [x for x in std_tPGD_res if str(x) != 'nan']
        std_tPGA_res = [x for x in std_tPGA_res if str(x) != 'nan']
        
        # Append residuals from this project to main lists
        pgd_res_list.append(std_pgd_res)
        pga_res_list.append(std_pga_res)
        pgv_res_list.append(std_pgv_res)
        tPGD_res_list.append(std_tPGD_res)
        tPGA_res_list.append(std_tPGA_res)

        # Get parameter value.  Need different lists for disp and sm runs because 
            # there are a different number of stations. 
        for j in range(len(std_pgd_res)):
            disp_project_list.append(val)
        for j in range(len(std_pga_res)):
            sm_project_list.append(val)
    
    # Loop through projects and put residuals into lists
    for i, project in enumerate(projects):
        
        # Residual dataframe
        res_df = pd.read_csv( f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/residuals/{project}_{res}.csv')
        
        # Select out residuals 
        pgd_res = np.array(res_df['pgd_res'])
        pga_res = np.array(res_df['pga_res'])
        pgv_res = np.array(res_df['pgv_res'])
        tPGD_res = np.array(res_df['tPGD_origin_res'])
        tPGA_res = np.array(res_df['tPGA_origin_res'])
        
        # Get rid of NaNs if there are any
        pgd_res = [x for x in pgd_res if str(x) != 'nan']
        pga_res = [x for x in pga_res if str(x) != 'nan']
        pgv_res = [x for x in pgv_res if str(x) != 'nan']
        tPGD_res = [x for x in tPGD_res if str(x) != 'nan']
        tPGA_res = [x for x in tPGA_res if str(x) != 'nan']
        
        # Append residuals from this project to main lists
        pgd_res_list.append(pgd_res)
        pga_res_list.append(pga_res)
        pgv_res_list.append(pgv_res)
        tPGD_res_list.append(tPGD_res)
        tPGA_res_list.append(tPGA_res)

        # Get parameter value.  Need different lists for disp and sm runs because 
            # there are a different number of stations. 
        for j in range(len(pgd_res)):
            disp_project_list.append(param_vals[i])
        for j in range(len(pga_res)):
            sm_project_list.append(param_vals[i])

    # Add default parameter last if stress drop
    if parameter == 'stress_drop' or parameter == 'vrupt':
        if parameter == 'stress_drop':
            val = '5.0'
        elif parameter == 'vrupt':
            val = '1.66'
        
        # Residual dataframe for default (aka stdtered) parameters
        std_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/standard/std/flatfiles/residuals/std_{res}.csv')
        
        # Select out residuals 
        std_pgd_res = np.array(std_df['pgd_res'])
        std_pga_res = np.array(std_df['pga_res'])
        std_pgv_res = np.array(std_df['pgv_res'])
        std_tPGD_res = np.array(std_df['tPGD_origin_res'])
        std_tPGA_res = np.array(std_df['tPGA_origin_res'])
        
        # Get rid of NaNs if there are any
        std_pgd_res = [x for x in std_pgd_res if str(x) != 'nan']
        std_pga_res = [x for x in std_pga_res if str(x) != 'nan']
        std_pgv_res = [x for x in std_pgv_res if str(x) != 'nan']
        std_tPGD_res = [x for x in std_tPGD_res if str(x) != 'nan']
        std_tPGA_res = [x for x in std_tPGA_res if str(x) != 'nan']
        
        # Append residuals from this project to main lists
        pgd_res_list.append(std_pgd_res)
        pga_res_list.append(std_pga_res)
        pgv_res_list.append(std_pgv_res)
        tPGD_res_list.append(std_tPGD_res)
        tPGA_res_list.append(std_tPGA_res)

        # Get parameter value.  Need different lists for disp and sm runs because 
            # there are a different number of stations. 
        for j in range(len(std_pgd_res)):
            disp_project_list.append(val)
        for j in range(len(std_pga_res)):
            sm_project_list.append(val)

    
    ######################## Plot IMs on one figure ###########################
    
    if ln:
        label = 'ln Residual'
    else:
        label = 'Residual'
        
    # Set up Figure
    fig, axs = plt.subplots(2, 3, figsize=(8, 10))
    fig.delaxes(axs[1][2])
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[0,2]
    ax4 = axs[1,0]
    ax5 = axs[1,1]

    # Set up xlabel and title based on parameter being varied 
    if parameter == 'stress_drop':
        xlabel = 'Stress Drop (Mpa)'
        title = 'Stress Drop IM Residuals'
    elif parameter == 'rise_time':
        xlabel = 'Rise Time Factor'
        title = 'Rise Time IM Residuals'
    elif parameter == 'vrupt':
        xlabel = 'Rupture Velocity (km/s)'
        title = 'Vrupt IM Residuals'
    else:
        xlabel = ''
        title = 'IM Residuals'
    
    # PGD subplot
    pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
    pgd_data = {'Parameter':disp_project_list, f'{label} (m)':pgd_res_list}
    pgd_df = pd.DataFrame(data=pgd_data)       
            
    sns.boxplot(x='Parameter', y=f'{label} (m)', data=pgd_df, showfliers=outliers, boxprops=dict(alpha=.3),
                ax=ax1).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax1.set_title('PGD')
    ax1.axhline(0, ls='--')
    
    # PGA subplot
    pga_res_list = [val for sublist in pga_res_list for val in sublist]
    pga_data = {'Parameter':sm_project_list, f'{label} (m/s/s)':pga_res_list}
    pga_df = pd.DataFrame(data=pga_data)
                       
    sns.boxplot(x='Parameter', y=f'{label} (m/s/s)', data=pga_df, showfliers=outliers, boxprops=dict(alpha=.3),
                ax=ax2).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax2.set_title('PGA')
    ax2.axhline(0, ls='--')
    
    # PGV subplot
    pgv_res_list = [val for sublist in pgv_res_list for val in sublist]
    pgv_data = {'Parameter':sm_project_list, f'{label} (m/s)':pgv_res_list}
    pgv_df = pd.DataFrame(data=pgv_data)
                       
    sns.boxplot(x='Parameter', y=f'{label} (m/s)', data=pgv_df, showfliers=outliers, boxprops=dict(alpha=.3),
                ax=ax3).set(xlabel=None)
    yabs_max = abs(max(ax3.get_ylim(), key=abs))
    ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax3.set_title('PGV')
    ax3.axhline(0, ls='--')
    
    # tPGD subplot
    tPGD_res_list = [val for sublist in tPGD_res_list for val in sublist]
    tPGD_data = {'Parameter':disp_project_list, f'{label} (s)':tPGD_res_list}
    tPGD_df = pd.DataFrame(data=tPGD_data)
                       
    sns.boxplot(x='Parameter', y=f'{label} (s)', data=tPGD_df, showfliers=outliers, boxprops=dict(alpha=.3),
                ax=ax4).set(xlabel=None)
    yabs_max = abs(max(ax4.get_ylim(), key=abs))
    ax4.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax4.set_title('tPGD')
    ax4.axhline(0, ls='--')
    
    # tPGA subplot
    tPGA_res_list = [val for sublist in tPGA_res_list for val in sublist]
    tPGA_data = {'Parameter':sm_project_list, f'{label} (s)':tPGA_res_list}
    tPGA_df = pd.DataFrame(data=tPGA_data)
                       
    sns.boxplot(x='Parameter', y=f'{label} (s)', data=tPGA_df, showfliers=outliers, boxprops=dict(alpha=.3),
                ax=ax5).set(xlabel=None)
    yabs_max = abs(max(ax5.get_ylim(), key=abs))
    ax5.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax5.set_title('tPGA')
    ax5.axhline(0, ls='--')
    
    fig.text(0.54, 0.005, xlabel, ha='center')
    fig.suptitle(title, fontsize=12, x=.54)
    fig.tight_layout()
    
    if outliers == True:
        plt.savefig(f'{figdir}/IMs_{res}.png', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(f'{figdir}/IMs_xout_{res}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return()


def plot_spec_res_full(parameter, projects, param_vals, ln=True, outliers=True):
    """
    Plots residuals between synthetic and observed spectra for all runs of the
    different projects for one parameter (i.e. stress drop of 0.3, 1.0, and 2.0
    MPa). Each boxplot design is a different project.  This figure is hard to
    read though because there is too much going on. 
    
    Inputs:
        parameter(str): Folder name of parameter being varied.
        projects(array): Array of folder names of projects within parameter folder.
        param_vals(array): Array of different parameter values (i.e. [0.3, 1.0, 2.0])
        ln(T/F): If true, calculates the natural log of the residuals.
        outliers(T/F): True or False if outliers should be shown in plots. 
        
    Output:
        Just saves the plots to ther respective directories. 
    """
    
    import numpy as np
    import pandas as pd
    from os import path, makedirs
    from math import log10, floor
    from itertools import repeat
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    data_types = ['disp', 'acc', 'vel']
     
    for data in data_types:
    
        plt.figure(figsize=(10, 10))
        
        if parameter == 'stress_drop':
            param = 'Stress Drop'
        elif parameter == 'rise_time':
            param = 'Rise Time'
        elif parameter == 'vrupt':
            param == 'Rupture Velocity'
        
        for i, project in enumerate(projects):
    
            res_df = pd.read_csv( f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/residuals/{project}_res.csv')
        
            if not path.exists(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/plots/residuals'):
                makedirs(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/plots/residuals')
        
            # Define bin edges, spectra residual columns, and other data type parameters
            if data == 'disp':
                bin_edges = [0.004, 0.00509, 0.00648, 0.00825, 0.01051, 0.01337,
                             0.01703, 0.02168, 0.02759, 0.03513, 0.04472, 0.05693,
                             0.07248, 0.09227, 0.11746, 0.14953, 0.19037, 0.24234,
                             0.30852, 0.39276, 0.5]
                spec_res = res_df.iloc[:,24:44]
                res_units = 'm*s'
                title = f'{param} Displacement Spectra Residuals'
                
            elif data == 'acc':
                bin_edges = [0.004, 0.00592, 0.00875, 0.01293, 0.01913, 0.02828,
                             0.04183, 0.06185, 0.09146, 0.13525, 0.20000, 0.29575,
                             0.43734, 0.64673, 0.95635, 1.41421, 2.09128, 3.09249,
                             4.57305, 6.76243, 10.00000]
                spec_res = res_df.iloc[:,84:104]
                res_units = 'm/s'
                title = f'{param} Acceleration Spectra Residuals'
            
            elif data == 'vel':
                bin_edges = [0.004, 0.00592, 0.00875, 0.01293, 0.01913, 0.02828,
                             0.04183, 0.06185, 0.09146, 0.13525, 0.20000, 0.29575,
                             0.43734, 0.64673, 0.95635, 1.41421, 2.09128, 3.09249,
                             4.57305, 6.76243, 10.00000]
                spec_res = res_df.iloc[:,144:164]
                res_units = 'm'
                title = f'{param} Velocity Spectra Residuals'
            
            # Obtain bin means from bin edges
            bin_means = []
            for j in range(len(bin_edges)):
                if j != 0:
                    mean = (bin_edges[i]+bin_edges[j-1])/2
                    bin_means.append(mean)
                
            # Function to round bin means
            def round_sig(x, sig):
                return round(x, sig-int(floor(log10(abs(x))))-1)
            
            # Ruond bin means so that they look clean for figure
            for j, b in enumerate(bin_means):
                if b < 0.01:
                    bin_means[j] = round_sig(b,1)
                elif b >= 0.01 and b < 0.1:
                    bin_means[j] = round_sig(b,1)
                elif b >= 0.1 and b < 1:
                    bin_means[j] = round_sig(b,2)
                elif b >= 1 and b < 10:
                    bin_means[j] = round_sig(b,2)
                elif b >= 10:
                    bin_means[j] = round_sig(b,2)
            
            bin_list = []
            spec_res_list = []
            
            for j in range(25):
                bin_res = np.array(spec_res.iloc[:,j])
                bin_res = [x for x in bin_res if str(x) != 'nan']
                bin_list.extend(repeat(str(bin_means[j]),len(bin_res)))
                spec_res_list.append(bin_res)
        
            spec_res_list = [val for sublist in spec_res_list for val in sublist]
            spec_data = {'Frequency(Hz)': bin_list, f'Residual({res_units})':spec_res_list}
            spec_df = pd.DataFrame(data=spec_data)       
            
            # Make figure
            ax = sns.boxplot(x='Frequency(Hz)', y=f'Residual({res_units})',
                             data=spec_df, color=colors[i], showfliers=outliers,
                             boxprops=dict(alpha=.1))
            ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
            ax.set_title(title)
            ax.legend()
            ax.axhline(0, ls='--')
        
        # Residual type to be used in figure name
        if ln:
            res='lnres'
        else:
            res='res'
        
        plt.savefig(f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/all_runs/{data}_spec_{res}.png', dpi=300)
        plt.close()
