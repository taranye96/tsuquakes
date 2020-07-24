#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:27:35 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd

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
    obs_spectra = np.array(obs_df.iloc[:,25:250])
    
    # Synthetic values
    syn_df = pd.read_csv(f'/Users/tnye/FakeQuakes/{parameter}/{project}/flatfiles/IMs/{run}.csv')
    # syn_df = pd.read_csv(f'/Users/tnye/tsuquakes/flatfiles/spec_test/{project}/{run}.csv')
    syn_pgd = np.array(syn_df['pgd'])
    syn_pga = np.array(syn_df['pga'])
    syn_pgv = np.array(syn_df['pgv'])
    syn_tPGD_orig = np.array(syn_df['tPGD_origin'])
    syn_tPGD_parriv = np.array(syn_df['tPGD_parriv'])
    syn_spectra = np.array(syn_df.iloc[:,25:250])
    
    # calc res
    pgd_res = obs_pgd - syn_pgd
    pga_res = obs_pga - syn_pga
    pgv_res = obs_pgv - syn_pgv
    tPGD_orig_res = obs_tPGD_orig - syn_tPGD_orig
    tPGD_parriv_res = obs_tPGD_parriv - syn_tPGD_parriv
    spectra_res = obs_spectra - syn_spectra
    
    return (pgd_res,pga_res,pgv_res,tPGD_orig_res,tPGD_parriv_res,spectra_res)


def plot_res(parameter, project, runs):
    """
    """
    
    pgd_run_list = []
    pga_run_list = []
    pgv_run_list = []
    tPGD_run_list = []
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
    E_disp_bin8_list = []
    E_disp_bin14_list = []
    E_disp_bin20_list = []
    E_acc_bin8_list = []
    E_acc_bin14_list = []
    E_acc_bin20_list = []
    E_vel_bin8_list = []
    E_vel_bin14_list = []
    E_vel_bin20_list = []


    for i, run in enumerate(runs):
        # Residual dataframe
        res_df = pd.read_csv( f'/Users/tnye/tsuquakes/flatfiles/residuals/{parameter}/{project}/{run}.csv')
        
        # Pull out residuals 
        pgd_res = np.array(res_df['pgd_res'])
        pga_res = np.array(res_df['pga_res'])
        pgv_res = np.array(res_df['pgv_res'])
        tPGD_res = np.array(res_df['tPGD_origin_res'])
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/pgd_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/pga_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/pgv_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/tPGD_orig_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/disp/E_disp_bin8_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/disp/E_disp_bin14_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/disp/E_disp_bin20_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/acc/E_acc_bin8_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/acc/E_acc_bin14_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/acc/E_acc_bin20_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/vel/E_vel_bin8_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/vel/E_vel_bin14_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/vel/E_vel_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    return()


def plot_res_full(param_folder, parameter, projects, param_vals):
    """
    """
    
    # pgd_parameter_list = []
    # pga_parameter_list = []
    # pgv_parameter_list = []
    # tPGD_parameter_list = []
    # E_disp_bin8_parameter_list = []
    # E_disp_bin14_parameter_list = []
    # E_disp_bin20_parameter_list = []
    # E_acc_bin8_parameter_list = []
    # E_acc_bin14_parameter_list = []
    # E_acc_bin20_parameter_list = []
    # E_vel_bin8_parameter_list = []
    # E_vel_bin14_parameter_list = []
    # E_vel_bin20_parameter_list = []
    disp_project_list = []
    sm_project_list = []
    
    pgd_res_list = []
    pga_res_list = []
    pgv_res_list = []
    tPGD_res_list = []
    E_disp_bin8_list = []
    E_disp_bin14_list = []
    E_disp_bin20_list = []
    E_acc_bin8_list = []
    E_acc_bin14_list = []
    E_acc_bin20_list = []
    E_vel_bin8_list = []
    E_vel_bin14_list = []
    E_vel_bin20_list = []


    for i, project in enumerate(projects):
        # Residual dataframe
        res_df = pd.read_csv( f'/Users/tnye/tsuquakes/flatfiles/residuals/{param_folder}/{project}/{project}_res.csv')
        
        # Pull out residuals 
        # pgd_res = np.array(res_df['pgd_res'])
        pga_res = np.array(res_df['pga_res'])
        pgv_res = np.array(res_df['pgv_res'])
        # tPGD_res = np.array(res_df['tPGD_origin_res'])
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
        # pgd_res = [x for x in pgd_res if str(x) != 'nan']
        pga_res = [x for x in pga_res if str(x) != 'nan']
        pgv_res = [x for x in pgv_res if str(x) != 'nan']
        # tPGD_res = [x for x in tPGD_res if str(x) != 'nan']
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
        # pgd_res_list.append(pgd_res)
        pga_res_list.append(pga_res)
        pgv_res_list.append(pgv_res)
        # tPGD_res_list.append(tPGD_res)
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
        # for j in range(len(pgd_res)):
        #     disp_project_list.append(param_vals[i])
        for j in range(len(pga_res)):
            sm_project_list.append(param_vals[i])

    # # PGD boxplot
    # pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
    # pgd_data = {'Parameter':pgd_run_list, 'Residual(m)':pgd_res_list}
    # pgd_df = pd.DataFrame(data=pgd_data)       
            
    # ax = sns.catplot(x='Run#', y='Residual(m)', data=pgd_df)
    # ax = sns.boxplot(x='Run#', y='Residual(m)', data=pgd_df, boxprops=dict(alpha=.3))
    # ax.set_title('PGD')
    # ax.axhline(0, ls='--')
    # ax.set(ylim=(-0.8, 0.8))
    
    # figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/pgd_res.png'
    # plt.savefig(figpath, bbox_inches='tight', dpi=300)
    # plt.close()
    
    # PGA boxplot
    # pga_res_list = [val for sublist in pga_res_list for val in sublist]
    pga_data = {f'{parameter}':sm_project_list, 'Residual(m/s/s)':pga_res_list}
    pga_df = pd.DataFrame(data=pga_data)
                       
    ax = sns.catplot(x=f'{parameter}', y='Residual(m/s/s)', data=pga_df)
    ax = sns.boxplot(x=f'{parameter}', y='Residual(m/s/s)', data=pga_df, boxprops=dict(alpha=.3))
    ax.set_title('PGA')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.8, .8))
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/pga_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/pgv_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    # # tPGD boxplot
    # tPGD_res_list = [val for sublist in tPGD_res_list for val in sublist]
    # tPGD_data = {'Run#':tPGD_run_list, 'Residual(s)':tPGD_res_list}
    # tPGD_df = pd.DataFrame(data=tPGD_data)
                       
    # ax = sns.catplot(x='Run#', y='Residual(s)', data=tPGD_df)
    # ax = sns.boxplot(x='Run#', y='Residual(s)', data=tPGD_df, boxprops=dict(alpha=.3))
    # ax.set_title('tPGD')
    # ax.axhline(0, ls='--')
    # # ax.set(ylim=(-.8, .8))
    
    # figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/tPGD_orig_res.png'
    # plt.savefig(figpath, bbox_inches='tight', dpi=300)
    # plt.close()
    
    # # E disp bin 8 spectra boxplot
    # E_disp_bin8_list = [val for sublist in E_disp_bin8_list for val in sublist]
    # E_disp_bin8_data = {'Run#':E_disp_bin8_run_list, 'Residual(m*s)':E_disp_bin8_list}
    # E_disp_bin8_df = pd.DataFrame(data=E_disp_bin8_data)
                       
    # ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin8_df)
    # ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin8_df, boxprops=dict(alpha=.3))
    # ax.set_title('E Disp Spectra Bin 8')
    # ax.axhline(0, ls='--')
    # ax.set(ylim=(-.5, .5))
    
    # figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/disp/E_disp_bin8_res.png'
    # plt.savefig(figpath, bbox_inches='tight', dpi=300)
    # plt.close()
    
    # # E disp bin 14 spectra boxplot
    # E_disp_bin14_list = [val for sublist in E_disp_bin14_list for val in sublist]
    # E_disp_bin14_data = {'Run#':E_disp_bin14_run_list, 'Residual(m*s)':E_disp_bin14_list}
    # E_disp_bin14_df = pd.DataFrame(data=E_disp_bin14_data)
                       
    # ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin14_df)
    # ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin14_df, boxprops=dict(alpha=.3))
    # ax.set_title('E Disp Spectra Bin 14')
    # ax.axhline(0, ls='--')
    # ax.set(ylim=(-.4, .4))
    
    # figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/disp/E_disp_bin14_res.png'
    # plt.savefig(figpath, bbox_inches='tight', dpi=300)
    # plt.close()
    
    # # E disp bin 20 spectra boxplot
    # E_disp_bin20_list = [val for sublist in E_disp_bin20_list for val in sublist]
    # E_disp_bin20_data = {'Run#':E_disp_bin20_run_list, 'Residual(m*s)':E_disp_bin20_list}
    # E_disp_bin20_df = pd.DataFrame(data=E_disp_bin20_data)
                       
    # ax = sns.catplot(x='Run#', y='Residual(m*s)', data=E_disp_bin20_df)
    # ax = sns.boxplot(x='Run#', y='Residual(m*s)', data=E_disp_bin20_df, boxprops=dict(alpha=.3))
    # ax.set_title('E Disp Spectra Bin 12')
    # ax.axhline(0, ls='--')
    # ax.set(ylim=(-.4, .4))
    
    # figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/disp/E_disp_bin20_res.png'
    # plt.savefig(figpath, bbox_inches='tight', dpi=300)
    # plt.close()
    
    # E acc bin 8 spectra boxplot
    E_acc_bin8_list = [val for sublist in E_acc_bin8_list for val in sublist]
    E_acc_bin8_data = {'Run#':E_acc_bin8_run_list, 'Residual(m/s)':E_acc_bin8_list}
    E_acc_bin8_df = pd.DataFrame(data=E_acc_bin8_data)
                       
    ax = sns.catplot(x='Run#', y='Residual(m/s)', data=E_acc_bin8_df)
    ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=E_acc_bin8_df, boxprops=dict(alpha=.3))
    ax.set_title('E Acc Spectra Bin 8')
    ax.axhline(0, ls='--')
    ax.set(ylim=(-.1, .1))
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/acc/E_acc_bin8_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/acc/E_acc_bin14_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/acc/E_acc_bin20_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/vel/E_vel_bin8_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/vel/E_vel_bin14_res.png'
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
    
    figpath = f'/Users/tnye/tsuquakes/plots/residuals/{parameter}/{project}/spectra/vel/E_vel_bin20_res.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.close()
    
    
    
    # Set up Figure
    fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]
    # fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    # sns.despine(left=True)
    
    # # PGD subplot
    # pgd_data = {f'{parameter}':disp_project_list, 'Residual(m)':pgd_res_list}
    # pgd_df = pd.DataFrame(data=pgd_data)       
            
    # ax1 = sns.catplot(x='Run#', y='Residual(m)', data=pgd_df)
    # ax1 = sns.boxplot(x='Run#', y='Residual(m)', data=pgd_df, boxprops=dict(alpha=.3))
    # ax1.set_title('PGD')
    # ax1.axhline(0, ls='--')
    # ax1.set(ylim=(-0.8, 0.8))
    
    # # tPGD subplot
    # tPGD_data = {f'{parameter}':disp_project_list, 'Residual(s)':tPGD_res_list}
    # tPGD_df = pd.DataFrame(data=tPGD_data)
                       
    # ax2 = sns.catplot(x=f'{parameter}', y='Residual(s)', data=tPGD_df)
    # ax2 = sns.boxplot(x=f'{parameter}', y='Residual(s)', data=tPGD_df, boxprops=dict(alpha=.3))
    # ax2.set_title('tPGD')
    # ax2.axhline(0, ls='--')
    # ax.set(ylim=(-.8, .8))
    
    # PGA subplot
    pga_data = {f'{parameter}':sm_project_list, 'Residual(m/s/s)':pga_res_list}
    pga_df = pd.DataFrame(data=pga_data)
                       
    sns.catplot(x=f'{parameter}', y='Residual(m/s/s)', data=pga_df, ax=ax3)
    sns.boxplot(x=f'{parameter}', y='Residual(m/s/s)', data=pga_df, boxprops=dict(alpha=.3), ax=ax3)
    ax3.set_title('PGA')
    ax3.axhline(0, ls='--')
    ax3.set(ylim=(-.8, .8))
    
    # PGV subplot
    pgv_res_list = [val for sublist in pgv_res_list for val in sublist]
    pgv_data = {f'{parameter}':sm_project_list, 'Residual(m/s)':pgv_res_list}
    pgv_df = pd.DataFrame(data=pgv_data)
                       
    sns.catplot(x=f'{parameter}', y='Residual(m/s)', data=pgv_df, ax=ax4)
    sns.boxplot(x=f'{parameter}', y='Residual(m/s)', data=pgv_df, boxprops=dict(alpha=.3), ax=ax4)
    ax4.set_title('PGV')
    ax4.axhline(0, ls='--')
    ax4.set(ylim=(-.8, .8))
    
    return()


