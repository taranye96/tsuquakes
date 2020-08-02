#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:27:35 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from os import path, makedirs
import matplotlib.pyplot as plt
import seaborn as sns

def calc_res(parameter, project, run):
    """
    Calculates residuals between synthetic and observed data, and puts residuals
    into a dataframe.
    
    Inputs:
        parameter(str): Folder name of parameter being varied.
        project(str): Folder name of specific project within parameter folder.
        run(str): Individual run name within certain project. 
        
    Return:
        pgd_res(float): PGD residual. 
        pga_res(float): PGA residual.
        pgv_res(float): PGV residual.
        tPGD_orig_res(float): tPGD from origin residual.
        tPGD_parriv_res(float): tPGD from P-arrival time residual.
        spectra_res(array): Array of residuals for all the spectra bins.

    """
    
    obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')
    
    # Observed values
    obs_pgd = np.array(obs_df['pgd'])
    obs_pga = np.array(obs_df['pga'])
    obs_pgv = np.array(obs_df['pgv'])
    obs_tPGD_orig = np.array(obs_df['tPGD_origin'])
    obs_tPGD_parriv = np.array(obs_df['tPGD_parriv'])
    obs_tPGA_orig = np.array(obs_df['tPGA_origin'])
    obs_tPGA_parriv = np.array(obs_df['tPGA_parriv'])
    obs_spectra = np.array(obs_df.iloc[:,25:250])
    
    # Synthetic values
    syn_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/IMs/{run}.csv')
    # syn_df = pd.read_csv(f'/Users/tnye/tsuquakes/flatfiles/spec_test/{project}/{run}.csv')
    syn_pgd = np.array(syn_df['pgd'])
    syn_pga = np.array(syn_df['pga'])
    syn_pgv = np.array(syn_df['pgv'])
    syn_tPGD_orig = np.array(syn_df['tPGD_origin'])
    syn_tPGD_parriv = np.array(syn_df['tPGD_parriv'])
    syn_tPGA_orig = np.array(syn_df['tPGA_origin'])
    syn_tPGA_parriv = np.array(syn_df['tPGA_parriv'])
    syn_spectra = np.array(syn_df.iloc[:,25:250])
    
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


def plot_res(parameter, project, rupture_list):
    """
    Makes residual boxplot figures of the IMs in the IM flatfile created using
    syn_calc_IMs.py. Each plot is created using one project (i.e. stress drop
    of 2.0 MPa), and each run is a different boxplot on the figure. 
    
        Inputs:
            parameter(str): Name of parameter folder.
            project(str): Name of simulation project.  This will be the main
                          directory where the different runs will be store. 
            rupture_list(array): List of .rupt files created using FakeQuakes.
        
        Output:
            Just saves the plots to ther respective directories. 
            
    """
    
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

    # PGD boxplot
    pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
    pgd_data = {'Run#':pgd_run_list, 'Residual(m)':pgd_res_list}
    pgd_df = pd.DataFrame(data=pgd_data)       
            
    ax = sns.catplot(x='Run#', y='Residual(m)', data=pgd_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=pgd_df, boxprops=dict(alpha=.3))
    ax.set_title('PGD')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-0.8, 0.8))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/pgd_res.png'
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
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/pga_res.png'
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
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/pgv_res.png'
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
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/tPGD_orig_res.png'
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
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/tPGA_orig_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E disp bin 8 spectra boxplot
    E_disp_bin8_list = [val for sublist in E_disp_bin8_list for val in sublist]
    E_disp_bin8_data = {'Run#':E_disp_bin8_run_list, 'Residual(m*s)':E_disp_bin8_list}
    E_disp_bin8_df = pd.DataFrame(data=E_disp_bin8_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin8_df)
    ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin8_df, boxprops=dict(alpha=.3))
    ax.set_title('E Disp Spectra Bin 8')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.5, .5))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/disp/E_disp_bin8_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E disp bin 14 spectra boxplot
    E_disp_bin14_list = [val for sublist in E_disp_bin14_list for val in sublist]
    E_disp_bin14_data = {'Run#':E_disp_bin14_run_list, 'Residual(m*s)':E_disp_bin14_list}
    E_disp_bin14_df = pd.DataFrame(data=E_disp_bin14_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin14_df)
    ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin14_df, boxprops=dict(alpha=.3))
    ax.set_title('E Disp Spectra Bin 14')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.4, .4))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/disp/E_disp_bin14_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E disp bin 20 spectra boxplot
    E_disp_bin20_list = [val for sublist in E_disp_bin20_list for val in sublist]
    E_disp_bin20_data = {'Run#':E_disp_bin20_run_list, 'Residual(m*s)':E_disp_bin20_list}
    E_disp_bin20_df = pd.DataFrame(data=E_disp_bin20_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin20_df)
    ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin20_df, boxprops=dict(alpha=.3))
    ax.set_title('E Disp Spectra Bin 12')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.4, .4))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/disp/E_disp_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E acc bin 8 spectra boxplot
    E_acc_bin8_list = [val for sublist in E_acc_bin8_list for val in sublist]
    E_acc_bin8_data = {'Run#':E_acc_bin8_run_list, 'Residual(m/s)':E_acc_bin8_list}
    E_acc_bin8_df = pd.DataFrame(data=E_acc_bin8_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=E_acc_bin8_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=E_acc_bin8_df, boxprops=dict(alpha=.3))
    ax.set_title('E Acc Spectra Bin 8')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.1, .1))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/acc/E_acc_bin8_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E acc bin 14 spectra boxplot
    E_acc_bin14_list = [val for sublist in E_acc_bin14_list for val in sublist]
    E_acc_bin14_data = {'Run#':E_acc_bin14_run_list, 'Residual(m/s)':E_acc_bin14_list}
    E_acc_bin14_df = pd.DataFrame(data=E_acc_bin14_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=E_acc_bin14_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=E_acc_bin14_df, boxprops=dict(alpha=.3))
    ax.set_title('E Acc Spectra Bin 14')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.1, .1))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/acc/E_acc_bin14_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E acc bin 20 spectra boxplot
    E_acc_bin20_list = [val for sublist in E_acc_bin20_list for val in sublist]
    E_acc_bin20_data = {'Run#':E_acc_bin20_run_list, 'Residual(m/s)':E_acc_bin20_list}
    E_acc_bin20_df = pd.DataFrame(data=E_acc_bin20_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=E_acc_bin20_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=E_acc_bin20_df, boxprops=dict(alpha=.3))
    ax.set_title('E Acc Spectra Bin 20')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.1, .1))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/acc/E_acc_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E vel bin 8 spectra boxplot
    E_vel_bin8_list = [val for sublist in E_vel_bin8_list for val in sublist]
    E_vel_bin8_data = {'Run#':E_vel_bin8_run_list, 'Residual(m)':E_vel_bin8_list}
    E_vel_bin8_df = pd.DataFrame(data=E_vel_bin8_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m)', data=E_vel_bin8_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=E_vel_bin8_df, boxprops=dict(alpha=.3))
    ax.set_title('E Vel Spectra Bin 8')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.2, .2))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/vel/E_vel_bin8_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E vel bin 14 spectra boxplot
    E_vel_bin14_list = [val for sublist in E_vel_bin14_list for val in sublist]
    E_vel_bin14_data = {'Run#':E_vel_bin14_run_list, 'Residual(m)':E_vel_bin14_list}
    E_vel_bin14_df = pd.DataFrame(data=E_vel_bin14_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m)', data=E_vel_bin14_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=E_vel_bin14_df, boxprops=dict(alpha=.3))
    ax.set_title('E Vel Spectra Bin 14')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.2, .2))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/vel/E_vel_bin14_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E vel bin 20 spectra boxplot
    E_vel_bin20_list = [val for sublist in E_vel_bin20_list for val in sublist]
    E_vel_bin20_data = {'Run#':E_vel_bin20_run_list, 'Residual(m)':E_vel_bin20_list}
    E_vel_bin20_df = pd.DataFrame(data=E_vel_bin20_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m)', data=E_vel_bin20_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=E_vel_bin20_df, boxprops=dict(alpha=.3))
    ax.set_title('E Vel Spectra Bin 20')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.2, .2))
    
    figpath = f'{param_dir}/{parameter}/{project}/plots/residuals/spectra/vel/E_vel_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    return()


def plot_res_full(parameter, projects, param_vals):
    """
    Plots residuals between synthetic and observed IMs and spectra for all runs
    of the different projects for one parameter (i.e. stress drop of 0.3, 1.0,
    and 2.0 MPa). Plots each IM and spectra bin separately and also plots them 
    all on one figure.  
    
    Inputs:
        parameter(str): Folder name of parameter being varied.
        projects(array): Array of folder names of projects within parameter folder.
        param_vals(): Array of different parameter values (i.e. [0.3, 1.0, 2.0])
        
    Output:
        Just saves the plots to ther respective directories. 
    """
    
    # Directory for plots using all the runs
    figdir = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/all_runs'
    # Make deirectory if it does not already exist
    if not path.exists(figdir):
        makedirs(figdir)
    
    # Make directories for the spectra plots if they don't already exist
    if not path.exists(f'{figdir}/spectra'):
        makedirs(f'{figdir}/spectra')
    if not path.exists(f'{figdir}/spectra/acc'):
        makedirs(f'{figdir}/spectra/acc')
    if not path.exists(f'{figdir}/spectra/vel'):
        makedirs(f'{figdir}/spectra/vel')
    if not path.exists(f'{figdir}/spectra/disp'):
        makedirs(f'{figdir}/spectra/disp')
    
    # Set up empty lists to store the project names in for the disp IMs and the 
        # sm IMs
    disp_project_list = []
    sm_project_list = []
    
    # Set up empty lists to store the residuals in
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

    # Loop through projects and put residuals into lists
    for i, project in enumerate(projects):
        
        # Residual dataframe
        res_df = pd.read_csv( f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/flatfiles/residuals/{project}_res.csv')
        
        # Select out residuals 
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
        
        # Get rid of NaNs if there are any
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
        
        # Append residuals from this project to main lists
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

        # Get parameter value.  Need different lists for disp and sm runs because 
            # there are a different number of stations. 
        for j in range(len(pgd_res)):
            disp_project_list.append(param_vals[i])
        for j in range(len(pga_res)):
            sm_project_list.append(param_vals[i])


    ################### Plot IMs and Spectra individually #####################
    
    # PGD boxplot
    pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
    pgd_data = {'Parameter': disp_project_list, 'Residual(m)':pgd_res_list}
    pgd_df = pd.DataFrame(data=pgd_data)       
            
    ax = sns.catplot(x='Parameter', y='Residual(m)', data=pgd_df)
    ax = sns.boxplot(x='Parameter', y='Residual(m)', data=pgd_df,boxprops=dict(alpha=.3))
    ax.set_title('PGD')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-0.8, 0.8))
    
    figpath = f'{figdir}/pgd_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # PGA boxplot
    pga_res_list = [val for sublist in pga_res_list for val in sublist]
    pga_data = {'Parameter':sm_project_list, 'Residual(m/s/s)':pga_res_list}
    pga_df = pd.DataFrame(data=pga_data)
                       
    ax = sns.catplot(x='Parameter', y='Residual(m/s/s)', data=pga_df)
    ax = sns.boxplot(x='Parameter', y='Residual(m/s/s)', data=pga_df, boxprops=dict(alpha=.3))
    ax.set_title('PGA')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.8, .8))
    
    figpath = f'{figdir}/pga_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # PGV boxplot
    pgv_res_list = [val for sublist in pgv_res_list for val in sublist]
    pgv_data = {'Parameter':sm_project_list, 'Residual(m/s)':pgv_res_list}
    pgv_df = pd.DataFrame(data=pgv_data)
                       
    ax = sns.catplot(x='Parameter', y='Residual(m/s)', data=pgv_df)
    ax = sns.boxplot(x='Parameter', y='Residual(m/s)', data=pgv_df, boxprops=dict(alpha=.3))
    ax.set_title('PGV')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.8, .8))
    
    figpath = f'{figdir}/pgv_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # tPGD boxplot
    tPGD_res_list = [val for sublist in tPGD_res_list for val in sublist]
    tPGD_data = {'Parameter':disp_project_list, 'Residual(s)':tPGD_res_list}
    tPGD_df = pd.DataFrame(data=tPGD_data)
                       
    ax = sns.catplot(x='Parameter', y='Residual(s)', data=tPGD_df)
    ax = sns.boxplot(x='Parameter', y='Residual(s)', data=tPGD_df, boxprops=dict(alpha=.3))
    ax.set_title('tPGD')
    ax.axhline(0, ls='--')
    # ax.set(ylim=(-.8, .8))
    
    figpath = f'{figdir}/tPGD_orig_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # tPGA boxplot
    tPGA_res_list = [val for sublist in tPGA_res_list for val in sublist]
    tPGA_data = {'Parameter':sm_project_list, 'Residual(s)':tPGA_res_list}
    tPGA_df = pd.DataFrame(data=tPGA_data)
                       
    ax = sns.catplot(x='Parameter', y='Residual(s)', data=tPGA_df)
    ax = sns.boxplot(x='Parameter', y='Residual(s)', data=tPGA_df, boxprops=dict(alpha=.3))
    ax.set_title('tPGA')
    ax.axhline(0, ls='--')
    # ax.set(ylim=(-.8, .8))
    
    figpath = f'{figdir}/tPGA_orig_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E disp bin 8 spectra boxplot
    E_disp_bin8_list = [val for sublist in E_disp_bin8_list for val in sublist]
    E_disp_bin8_data = {'Parameter':disp_project_list, 'Residual(m*s)':E_disp_bin8_list}
    E_disp_bin8_df = pd.DataFrame(data=E_disp_bin8_data)
                       
    ax = sns.catplot(x='Parameter', y='Residual(m*s)', data=E_disp_bin8_df)
    ax = sns.boxplot(x='Parameter', y='Residual(m*s)', data=E_disp_bin8_df, boxprops=dict(alpha=.3))
    ax.set_title('E Disp Spectra Bin 8')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.5, .5))
    
    figpath = f'{figdir}/spectra/disp/E_disp_bin8_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E disp bin 14 spectra boxplot
    E_disp_bin14_list = [val for sublist in E_disp_bin14_list for val in sublist]
    E_disp_bin14_data = {'Run#':disp_project_list, 'Residual(m*s)':E_disp_bin14_list}
    E_disp_bin14_df = pd.DataFrame(data=E_disp_bin14_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin14_df)
    ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin14_df, boxprops=dict(alpha=.3))
    ax.set_title('E Disp Spectra Bin 14')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.4, .4))
    
    figpath = f'{figdir}/spectra/disp/E_disp_bin14_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E disp bin 20 spectra boxplot
    E_disp_bin20_list = [val for sublist in E_disp_bin20_list for val in sublist]
    E_disp_bin20_data = {'Run#':disp_project_list, 'Residual(m*s)':E_disp_bin20_list}
    E_disp_bin20_df = pd.DataFrame(data=E_disp_bin20_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin20_df)
    ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin20_df, boxprops=dict(alpha=.3))
    ax.set_title('E Disp Spectra Bin 12')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.4, .4))
    
    figpath = f'{figdir}/spectra/disp/E_disp_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E acc bin 8 spectra boxplot
    E_acc_bin8_list = [val for sublist in E_acc_bin8_list for val in sublist]
    E_acc_bin8_data = {'Run#':sm_project_list, 'Residual(m/s)':E_acc_bin8_list}
    E_acc_bin8_df = pd.DataFrame(data=E_acc_bin8_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=E_acc_bin8_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=E_acc_bin8_df, boxprops=dict(alpha=.3))
    ax.set_title('E Acc Spectra Bin 8')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.1, .1))
    
    figpath = f'{figdir}/spectra/acc/E_acc_bin8_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E acc bin 14 spectra boxplot
    E_acc_bin14_list = [val for sublist in E_acc_bin14_list for val in sublist]
    E_acc_bin14_data = {'Run#':sm_project_list, 'Residual(m/s)':E_acc_bin14_list}
    E_acc_bin14_df = pd.DataFrame(data=E_acc_bin14_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=E_acc_bin14_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=E_acc_bin14_df, boxprops=dict(alpha=.3))
    ax.set_title('E Acc Spectra Bin 14')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.1, .1))
    
    figpath = f'{figdir}/spectra/acc/E_acc_bin14_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E acc bin 20 spectra boxplot
    E_acc_bin20_list = [val for sublist in E_acc_bin20_list for val in sublist]
    E_acc_bin20_data = {'Run#':sm_project_list, 'Residual(m/s)':E_acc_bin20_list}
    E_acc_bin20_df = pd.DataFrame(data=E_acc_bin20_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=E_acc_bin20_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=E_acc_bin20_df, boxprops=dict(alpha=.3))
    ax.set_title('E Acc Spectra Bin 20')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.1, .1))
    
    figpath = f'{figdir}/spectra/acc/E_acc_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E vel bin 8 spectra boxplot
    E_vel_bin8_list = [val for sublist in E_vel_bin8_list for val in sublist]
    E_vel_bin8_data = {'Run#':sm_project_list, 'Residual(m)':E_vel_bin8_list}
    E_vel_bin8_df = pd.DataFrame(data=E_vel_bin8_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m)', data=E_vel_bin8_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=E_vel_bin8_df, boxprops=dict(alpha=.3))
    ax.set_title('E Vel Spectra Bin 8')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.2, .2))
    
    figpath = f'{figdir}/spectra/vel/E_vel_bin8_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E vel bin 14 spectra boxplot
    E_vel_bin14_list = [val for sublist in E_vel_bin14_list for val in sublist]
    E_vel_bin14_data = {'Run#':sm_project_list, 'Residual(m)':E_vel_bin14_list}
    E_vel_bin14_df = pd.DataFrame(data=E_vel_bin14_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m)', data=E_vel_bin14_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=E_vel_bin14_df, boxprops=dict(alpha=.3))
    ax.set_title('E Vel Spectra Bin 14')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.2, .2))
    
    figpath = f'{figdir}/spectra/vel/E_vel_bin14_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # E vel bin 20 spectra boxplot
    E_vel_bin20_list = [val for sublist in E_vel_bin20_list for val in sublist]
    E_vel_bin20_data = {'Run#':sm_project_list, 'Residual(m)':E_vel_bin20_list}
    E_vel_bin20_df = pd.DataFrame(data=E_vel_bin20_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m)', data=E_vel_bin20_df)
    ax = sns.boxplot(x='Run#', y='Residual(m)', data=E_vel_bin20_df, boxprops=dict(alpha=.3))
    ax.set_title('E Vel Spectra Bin 20')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.2, .2))
    
    figpath = f'{figdir}/spectra/vel/E_vel_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    
    ######################## Plot IMs on one figure ###########################
    
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
        xlabel = 'Rise Time (s)'
        title = 'Rise Time IM Residuals'
    elif parameter == 'vrupt':
        xlabel = 'Rupture Velocity (m/s)'
        title = 'Rupture IM Velocity Residuals'
    
    # PGD subplot
    pgd_data = {'Parameter':disp_project_list, 'Residual (m)':pgd_res_list}
    pgd_df = pd.DataFrame(data=pgd_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m)', data=pgd_df, boxprops=dict(alpha=.3),
                ax=ax1).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax1.set_title('PGD')
    ax1.axhline(0, ls='--')
    
    # PGA subplot
    pga_data = {'Parameter':sm_project_list, 'Residual (m/s/s)':pga_res_list}
    pga_df = pd.DataFrame(data=pga_data)
                       
    sns.boxplot(x='Parameter', y='Residual (m/s/s)', data=pga_df, boxprops=dict(alpha=.3),
                ax=ax2).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax2.set_title('PGA')
    ax2.axhline(0, ls='--')
    
    # PGV subplot
    pgv_data = {'Parameter':sm_project_list, 'Residual (m/s)':pgv_res_list}
    pgv_df = pd.DataFrame(data=pgv_data)
                       
    sns.boxplot(x='Parameter', y='Residual (m/s)', data=pgv_df, boxprops=dict(alpha=.3),
                ax=ax3).set(xlabel=None)
    yabs_max = abs(max(ax3.get_ylim(), key=abs))
    ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax3.set_title('PGV')
    ax3.axhline(0, ls='--')
    
    # tPGD subplot
    tPGD_data = {'Parameter':disp_project_list, 'Residual (s)':tPGD_res_list}
    tPGD_df = pd.DataFrame(data=tPGD_data)
                       
    sns.boxplot(x='Parameter', y='Residual (s)', data=tPGD_df, boxprops=dict(alpha=.3),
                ax=ax4).set(xlabel=None)
    yabs_max = abs(max(ax4.get_ylim(), key=abs))
    ax4.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax4.set_title('tPGD')
    ax4.axhline(0, ls='--')
    
    # tPGA subplot
    tPGA_data = {'Parameter':sm_project_list, 'Residual (s)':tPGA_res_list}
    tPGA_df = pd.DataFrame(data=tPGA_data)
                       
    sns.boxplot(x='Parameter', y='Residual (s)', data=tPGA_df, boxprops=dict(alpha=.3),
                ax=ax5).set(xlabel=None)
    yabs_max = abs(max(ax5.get_ylim(), key=abs))
    ax5.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax5.set_title('tPGA')
    ax5.axhline(0, ls='--')
    
    fig.text(0.54, 0.005, xlabel, ha='center')
    fig.suptitle(title, fontsize=12, x=.54)
    fig.tight_layout()
    
    figpath = f'{figdir}/IMs_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    

    ###################### Plot spectra on one figure #########################
    
    # Set up Figure
    fig, axs = plt.subplots(3, 3, figsize=(10, 12))
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[0,2]
    ax4 = axs[1,0]
    ax5 = axs[1,1]
    ax6 = axs[1,2]
    ax7 = axs[2,0]
    ax8 = axs[2,1]
    ax9 = axs[2,2]
    
    # Set up xlabel and title based on parameter being varied 
    if parameter == 'stress_drop':
        xlabel = 'Stress Drop (Mpa)'
        title = 'Stress Drop Spectra Residuals'
    elif parameter == 'rise_time':
        xlabel = 'Rise Time (s)'
        title = 'Rise Time Spectra Residuals'
    elif parameter == 'vrupt':
        xlabel = 'Rupture Velocity (m/s)'
        title = 'Rupture Velocity Spectra Residuals'
    
    # E disp spectra bin 8 subplot
    E_disp_bin8_data = {'Parameter':disp_project_list, 'Residual (m*s)':E_disp_bin8_list}
    E_disp_bin8_df = pd.DataFrame(data=E_disp_bin8_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m*s)', data=E_disp_bin8_df, boxprops=dict(alpha=.3),
                ax=ax1).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax1.get_ylim(), key=abs))
    ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax1.set_title('E Disp Spectra Bin 8')
    ax1.axhline(0, ls='--')
    
    # E disp spectra bin 14 subplot
    E_disp_bin14_data = {'Parameter':disp_project_list, 'Residual (m*s)':E_disp_bin14_list}
    E_disp_bin14_df = pd.DataFrame(data=E_disp_bin14_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m*s)', data=E_disp_bin14_df, boxprops=dict(alpha=.3),
                ax=ax2).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax2.get_ylim(), key=abs))
    ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax2.set_title('E Disp Spectra Bin 14')
    ax2.axhline(0, ls='--')
    
    # E disp spectra bin 20 subplot
    E_disp_bin20_data = {'Parameter':disp_project_list, 'Residual (m*s)':E_disp_bin20_list}
    E_disp_bin20_df = pd.DataFrame(data=E_disp_bin20_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m*s)', data=E_disp_bin20_df, boxprops=dict(alpha=.3),
                ax=ax3).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax3.get_ylim(), key=abs))
    ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax3.set_title('E Disp Spectra Bin 20')
    ax3.axhline(0, ls='--')
    
    # E acc spectra bin 8 subplot
    E_acc_bin8_data = {'Parameter':sm_project_list, 'Residual (m/s)':E_acc_bin8_list}
    E_acc_bin8_df = pd.DataFrame(data=E_acc_bin8_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m/s)', data=E_acc_bin8_df, boxprops=dict(alpha=.3),
                ax=ax4).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax4.get_ylim(), key=abs))
    ax4.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax4.set_title('E Acc Spectra Bin 8')
    ax4.axhline(0, ls='--')
    
    # E acc spectra bin 14 subplot
    E_acc_bin14_data = {'Parameter':sm_project_list, 'Residual (m/s)':E_acc_bin14_list}
    E_acc_bin14_df = pd.DataFrame(data=E_acc_bin14_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m/s)', data=E_acc_bin14_df, boxprops=dict(alpha=.3),
                ax=ax5).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax5.get_ylim(), key=abs))
    ax5.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax5.set_title('E Acc Spectra Bin 14')
    ax5.axhline(0, ls='--')
    
    # E acc spectra bin 20 subplot
    E_acc_bin20_data = {'Parameter':sm_project_list, 'Residual (m/s)':E_acc_bin20_list}
    E_acc_bin20_df = pd.DataFrame(data=E_acc_bin20_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m/s)', data=E_acc_bin20_df, boxprops=dict(alpha=.3),
                ax=ax6).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax6.get_ylim(), key=abs))
    ax6.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax6.set_title('E Acc Spectra Bin 20')
    ax6.axhline(0, ls='--')
    
    # E vel spectra bin 8 subplot
    E_vel_bin8_data = {'Parameter':sm_project_list, 'Residual (m)':E_vel_bin8_list}
    E_vel_bin8_df = pd.DataFrame(data=E_vel_bin8_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m)', data=E_vel_bin8_df, boxprops=dict(alpha=.3),
                ax=ax7).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax7.get_ylim(), key=abs))
    ax7.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax7.set_title('E Vel Spectra Bin 8')
    ax7.axhline(0, ls='--')
    
    # E vel spectra bin 14 subplot
    E_vel_bin14_data = {'Parameter':sm_project_list, 'Residual (m)':E_vel_bin14_list}
    E_vel_bin14_df = pd.DataFrame(data=E_vel_bin14_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m)', data=E_vel_bin14_df, boxprops=dict(alpha=.3),
                ax=ax8).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax8.get_ylim(), key=abs))
    ax8.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax8.set_title('E Vel Spectra Bin 14')
    ax8.axhline(0, ls='--')
    
    # E vel spectra bin 20 subplot
    E_vel_bin20_data = {'Parameter':sm_project_list, 'Residual (m)':E_vel_bin20_list}
    E_vel_bin20_df = pd.DataFrame(data=E_vel_bin20_data)       
            
    sns.boxplot(x='Parameter', y='Residual (m)', data=E_vel_bin20_df, boxprops=dict(alpha=.3),
                ax=ax9).set(xticklabels=[], xlabel=None)
    yabs_max = abs(max(ax9.get_ylim(), key=abs))
    ax9.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    ax9.set_title('E Vel Spectra Bin 20')
    ax9.axhline(0, ls='--')
    
    fig.text(0.53, 0.005, xlabel, ha='center')
    fig.suptitle(title, fontsize=12, x=.53)
    fig.tight_layout()
    
    figpath = f'{figdir}/spectra_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    return()


