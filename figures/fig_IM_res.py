#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 19:41:08 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from glob import glob
from math import floor
from os import path, makedirs
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

home_dir = f'/Users/tnye/FakeQuakes/simulations/test_runs_m7.8'

# Set parameters

parameter = 'stress_drop'      # parameter being varied
projects = ['sd0.1','sd1.0','sd2.0']  # array of projects for the parameter
param_vals = ['0.1', '1.0', '2.0']

# parameter = 'risetime'      # parameter being varied
# projects = ['rt2x','rt3x']  # array of projects for the parameter
# param_vals = ['10.8','16.2']        # array of parameter values associated w/ the projects

# parameter = 'vrupt'      # parameter being varied
# projects = ['sf0.3', 'sf0.4']  # array of projects for the parameter
# param_vals = ['1.0', '1.3']        # array of parameter values associated w/ the projects


sns.set_style("whitegrid")

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
    legend_title = 'Mean Rise Time (s)'
elif parameter == 'stress_drop':
    val = '5.0'
    legend_title = 'Stress Parameter (MPa)'
elif parameter == 'vrupt':
    val = '1.6'
    legend_title = r'Mean $V_{rupt}$ (km/s)'
else:
    val = 'Standard'
    legend_title = ''
    
# Add default parameter first if risetime 
if parameter == 'risetime':
    for i, project in enumerate(projects):
        
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
    pgd_res_list.append(None)
    tPGD_res_list.append(None)
    disp_project_list.append(val)
    
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
    pga_res_list.append(None)
    sm_project_list.append(val)

# Add default parameter last if not risetime
if parameter != 'risetime':
    
    # Residual dataframe for default (aka stdtered) parameters
    # std_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/standard/std/flatfiles/residuals/std_{res}.csv')
    
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
    xlabel = 'Stress Parameter (MPa)'
    title = 'Stress Parameter IM Residuals'
elif parameter == 'risetime':
    xlabel = 'Mean Rise Time (s)'
    title = 'Rise Time IM Residuals'
elif parameter == 'vrupt':
    xlabel = r'Mean $V_{rupt}$ (km/s)'
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
tPGD_data = {'Parameter':disp_project_list, 'Residual (s)':tPGD_res_list}
tPGD_df = pd.DataFrame(data=tPGD_data).sort_values('Parameter',ascending=True)  
std_tPGD_data = {'Parameter':std_disp_project, 'Residual (s)':std_tPGD_res_list}
std_tPGD_df = pd.DataFrame.from_dict(data=std_tPGD_data).sort_values('Parameter',ascending=True)  

# PGA
pga_data = {'Parameter':sm_project_list, f'{label} (m/s/s)':pga_res_list}
pga_df = pd.DataFrame(data=pga_data).sort_values('Parameter',ascending=True)  
std_pga_data = {'Parameter':std_sm_project, f'{label} (m/s/s)':std_pga_res_list}
std_pga_df = pd.DataFrame.from_dict(data=std_pga_data).sort_values('Parameter',ascending=True) 


############################ Forier spectra ###############################

sm_dfs = sorted(glob(f'{home_dir}/{parameter}/*/flatfiles/residuals/*_sm.csv'))
sm_dfs = [file for file in sm_dfs if not file.endswith('standard_sm.csv')]

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
        
        # Standard parameters
        sm_std_df = pd.read_csv(f'{home_dir}/standard/flatfiles/residuals/standard_sm.csv')
        # sm_std_df = pd.read_csv(f'{home_dir}/{parameter}/standard/flatfiles/residuals/standard_sm.csv')
        
        # Remove strong motion stations farther than 600km to avoid surface waves
        far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
        ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
    
        acc_std_res = np.array(sm_std_df.iloc[ind].reset_index(drop=True).iloc[:,20:].iloc[:,j_bin])
    
        sm_std_bin_list.extend(np.repeat(bin_means[j_bin],len(acc_std_res)))
        acc_std_spec_res_list = np.append(acc_std_spec_res_list,acc_std_res)


#%%
######################## Plot IMs on one figure ###########################

n_colors = len(projects)  # Set this to your actual number of categories minus 1
custom_palette = sns.color_palette()[:n_colors]

# Set up Figure
layout = [
    ["A", "B", "C"],
    ["D", "D", "D"]]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, axs = plt.subplot_mosaic(layout, figsize=(6.5,5), gridspec_kw={'height_ratios':[1,1.25]})

ax1 = axs['A']
ax2 = axs['B']
ax3 = axs['C']
ax4 = axs['D']

PROPS = {'boxprops':{'edgecolor':'black'},'medianprops':{},'whiskerprops':{},'capprops':{}}

# PGD subplot
ax1.text(-0.25,1.1,r'$\bf{(a)}$',transform=ax1.transAxes,fontsize=10,va='top',ha='right')
if parameter == 'risetime':
    grouped_PGD = [list(list(std_pgd_df['ln Residual (m)'].values[~np.isnan(std_pgd_df['ln Residual (m)'].values)]))]
    grouped_PGD.extend([list(pgd_df['ln Residual (m)'].values[pgd_df['Parameter'].values == float(param)]) for param in param_vals])
    b = ax1.boxplot(grouped_PGD,patch_artist=True,boxprops=dict(lw=1,
                linestyle='--',edgecolor='k'),
                medianprops=dict(lw=1,color='k'),whiskerprops=dict(lw=1,color='k'),
                capprops=dict(lw=1,color='k'),widths=0.8,
                flierprops=dict(marker='d',markersize=5))
    for patch, color in zip(b['boxes'][1:], colors[:len(projects)]):
        patch.set_facecolor(color)
    b['boxes'][0].set_facecolor('none')
    b['boxes'][0].set_edgecolor('dimgray')
    b['boxes'][0].set_linestyle('--')
    b['medians'][0].set_color('dimgray')
    for whisker in b['whiskers'][:2]:  # Adjust indices as needed
        whisker.set(color='dimgray')
    for cap in b['caps'][:2]:  # Adjust indices as needed
        cap.set(color='dimgray')
    b['fliers'][-1].set_color('dimgray')
    labels = []
    labels.append(val)
    labels.extend(param_vals)
    ax1.set_xticklabels(labels)
else:
    grouped_PGD = [list(pgd_df['ln Residual (m)'].values[pgd_df['Parameter'].values == float(param)]) for param in param_vals]
    grouped_PGD.extend([list(list(std_pgd_df['ln Residual (m)'].values[~np.isnan(std_pgd_df['ln Residual (m)'].values)]))])
    b = ax1.boxplot(grouped_PGD,patch_artist=True,boxprops=dict(lw=1,
                linestyle='--',edgecolor='k'),
                medianprops=dict(lw=1,color='k'),whiskerprops=dict(lw=1,color='k'),
                capprops=dict(lw=1,color='k'),widths=0.8,
                flierprops=dict(marker='d',markersize=5))
    for patch, color in zip(b['boxes'][:-1], colors[:len(projects)]):
        patch.set_facecolor(color)
    b['boxes'][-1].set_facecolor('none')
    b['boxes'][-1].set_edgecolor('dimgray')
    b['boxes'][-1].set_linestyle('--')
    b['medians'][-1].set_color('dimgray')
    for whisker in b['whiskers'][-2:]:  # Adjust indices as needed
        whisker.set(color='dimgray')
    for cap in b['caps'][-2:]:  # Adjust indices as needed
        cap.set(color='dimgray')
    b['fliers'][-1].set_color('dimgray')
    labels = param_vals.copy()
    labels.append(val)
    ax1.set_xticklabels(labels)
ax1.xaxis.grid(False)
ax1.set(xlabel=None)
ax1.set_ylabel('ln (obs / sim)',fontsize=11)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=0)
yabs_max = abs(max(ax1.get_ylim(), key=abs))
ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax1.set_ylim(ymin=-2, ymax=2)
ax1.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=10)
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.set_title(r'$\bf{PGD}$',fontsize=11)
ax1.axhline(0, ls=':',c='k',lw=1)

# tPGD subplot
ax2.text(-0.35,1.1,r'$\bf{(b)}$',transform=ax2.transAxes,fontsize=10,va='top',ha='right')
if parameter == 'risetime':
    grouped_tPGD = [list(list(std_tPGD_df['Residual (s)'].values[~np.isnan(std_tPGD_df['Residual (s)'].values)]))]
    grouped_tPGD.extend([list(tPGD_df['Residual (s)'].values[tPGD_df['Parameter'].values == float(param)]) for param in param_vals])
    b = ax2.boxplot(grouped_tPGD,patch_artist=True,boxprops=dict(lw=1,
                linestyle='--',edgecolor='black'),
                medianprops=dict(lw=1,color='black'),whiskerprops=dict(lw=1,color='black'),
                capprops=dict(lw=1,color='black'),widths=0.8,
                flierprops=dict(marker='d',markersize=5))
    for patch, color in zip(b['boxes'][1:], colors[:len(projects)]):
        patch.set_facecolor(color)
    b['boxes'][0].set_facecolor('none')
    b['boxes'][0].set_edgecolor('dimgray')
    b['boxes'][0].set_linestyle('--')
    b['medians'][0].set_color('dimgray')
    for whisker in b['whiskers'][:2]:  # Adjust indices as needed
        whisker.set(color='dimgray')
    for cap in b['caps'][:2]:  # Adjust indices as needed
        cap.set(color='dimgray')
    b['fliers'][-1].set_color('dimgray')
    labels = []
    labels.append(val)
    labels.extend(param_vals)
    ax2.set_xticklabels(labels)
else:
    grouped_tPGD = [list(tPGD_df['Residual (s)'].values[pgd_df['Parameter'].values == float(param)]) for param in param_vals]
    grouped_tPGD.extend([list(list(std_tPGD_df['Residual (s)'].values[~np.isnan(std_tPGD_df['Residual (s)'].values)]))])
    b = ax2.boxplot(grouped_tPGD,patch_artist=True,boxprops=dict(lw=1,
                linestyle='--',edgecolor='black'),
                medianprops=dict(lw=1,color='black'),whiskerprops=dict(lw=1,color='black'),
                capprops=dict(lw=1,color='black'),widths=0.8,
                flierprops=dict(marker='d',markersize=5))
    for patch, color in zip(b['boxes'][:-1], colors[:len(projects)]):
        patch.set_facecolor(color)
    b['boxes'][-1].set_facecolor('none')
    b['boxes'][-1].set_edgecolor('dimgray')
    b['boxes'][-1].set_linestyle('--')
    b['medians'][-1].set_color('dimgray')
    for whisker in b['whiskers'][-2:]:  # Adjust indices as needed
        whisker.set(color='dimgray')
    for cap in b['caps'][-2:]:  # Adjust indices as needed
        cap.set(color='dimgray')
    b['fliers'][-1].set_color('dimgray')
    labels = param_vals.copy()
    labels.append(val)
    ax2.set_xticklabels(labels)
ax2.xaxis.grid(False)
ax2.set_xlabel(xlabel,fontsize=11)
ax2.set_ylabel('obs - sim',fontsize=11)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=0)
yabs_max = abs(max(ax2.get_ylim(), key=abs))
ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax2.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=10)
ax2.set_title(r'$\bf{tPGD}$',fontsize=11)
ax2.axhline(0, ls=':',c='k',lw=1)

# PGA subplot
ax3.text(-0.25,1.1,r'$\bf{(c)}$',transform=ax3.transAxes,fontsize=10,va='top',ha='right')
if parameter == 'risetime':
    grouped_PGA = [list(list(std_pga_df['ln Residual (m/s/s)'].values[~np.isnan(std_pga_df['ln Residual (m/s/s)'].values)]))]
    grouped_PGA.extend([list(pga_df['ln Residual (m/s/s)'].values[pga_df['Parameter'].values == float(param)]) for param in param_vals])
    b = ax3.boxplot(grouped_PGA,patch_artist=True,boxprops=dict(lw=1,
                linestyle='--',edgecolor='black'),
                medianprops=dict(lw=1,color='black'),whiskerprops=dict(lw=1,color='black'),
                capprops=dict(lw=1,color='black'),widths=0.8,
                flierprops=dict(marker='d',markersize=5))
    for patch, color in zip(b['boxes'][1:], colors[:len(projects)]):
        patch.set_facecolor(color)
    b['boxes'][0].set_facecolor('none')
    b['boxes'][0].set_edgecolor('dimgray')
    b['boxes'][0].set_linestyle('--')
    b['medians'][0].set_color('dimgray')
    for whisker in b['whiskers'][:2]:  # Adjust indices as needed
        whisker.set(color='dimgray')
    for cap in b['caps'][:2]:  # Adjust indices as needed
        cap.set(color='dimgray')
    b['fliers'][-1].set_color('dimgray')
    labels = []
    labels.append(val)
    labels.extend(param_vals)
    ax3.set_xticklabels(labels)
else:
    grouped_PGA = [list(pga_df['ln Residual (m/s/s)'].values[pga_df['Parameter'].values == float(param)]) for param in param_vals]
    grouped_PGA.extend([list(list(std_pga_df['ln Residual (m/s/s)'].values[~np.isnan(std_pga_df['ln Residual (m/s/s)'].values)]))])
    b = ax3.boxplot(grouped_PGA,patch_artist=True,boxprops=dict(lw=1,
                linestyle='--',edgecolor='black'),
                medianprops=dict(lw=1,color='black'),whiskerprops=dict(lw=1,color='black'),
                capprops=dict(lw=1,color='black'),widths=0.8,
                flierprops=dict(marker='d',markersize=5))
    for patch, color in zip(b['boxes'][:-1], colors[:len(projects)]):
        patch.set_facecolor(color)
    b['boxes'][-1].set_facecolor('none')
    b['boxes'][-1].set_edgecolor('dimgray')
    b['boxes'][-1].set_linestyle('--')
    b['medians'][-1].set_color('dimgray')
    for whisker in b['whiskers'][-2:]:  # Adjust indices as needed
        whisker.set(color='dimgray')
    for cap in b['caps'][-2:]:  # Adjust indices as needed
        cap.set(color='dimgray')
    b['fliers'][-1].set_color('dimgray')
    labels = param_vals.copy()
    labels.append(val)
    ax3.set_xticklabels(labels)
ax3.xaxis.grid(False)
ax3.set(xlabel=None)
ax3.set_ylabel(r'ln (obs / sim)',fontsize=11)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=0)
yabs_max = abs(max(ax3.get_ylim(), key=abs))
ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax3.tick_params(width=2,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=10)
ax3.yaxis.set_major_locator(MultipleLocator(1))
ax3.set_title(r'$\bf{PGA}$',fontsize=11)
ax3.axhline(0, ls=':',c='k',lw=1)

res_units = 'm/s'
title = 'Acceleration Spectra Residuals'
acc_spec_data = {'Project':project_list,'Frequency (Hz)': sm_bin_list, f'{label} ({res_units})':acc_spec_res_list}
acc_spec_df = pd.DataFrame(data=acc_spec_data) 

# Create dataframe for standard parameters
acc_std_spec_data = {'Frequency (Hz)': sm_std_bin_list, f'{label} ({res_units})':acc_std_spec_res_list}
acc_std_spec_df = pd.DataFrame(data=acc_std_spec_data)    
bins = np.unique(acc_std_spec_df['Frequency (Hz)'].values)
grouped_res_std = [list(acc_std_spec_df[f'{label} ({res_units})'].values[acc_std_spec_df['Frequency (Hz)'].values == fbin]) for fbin in bins]
projects = np.unique(acc_spec_df.Project.values)
width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
w=0.15
handles, labels = ax4.get_legend_handles_labels()
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
    ax4.boxplot(grouped_res,positions=positions,patch_artist=True,
               boxprops=dict(lw=1,linestyle='-',edgecolor='black',facecolor=colors[i]),
               medianprops=dict(lw=1,color='black'),
               whiskerprops=dict(lw=1,color='black'),
               capprops=dict(lw=1,color='black'),
               widths=width(bins,w)/len(projects),
               flierprops=dict(marker='d',markersize=5)
               )
    handles = handles + [Patch(facecolor=colors[i],edgecolor='k',lw=1,ls='-',label=param_vals[i])]
    labels = labels + [param_vals[i]]
    ax4.boxplot(grouped_res_std,positions=bins,
               boxprops=dict(lw=1,linestyle='--',color='dimgray'),
               medianprops=dict(lw=1,color='dimgray',ls='-'),
               whiskerprops=dict(lw=1,color='dimgray'),
               capprops=dict(lw=1,color='dimgray'),
               widths=width(bins,w),
               flierprops=dict(marker='d',linewidth=0.5,markersize=5)
               )
std_handles = [Patch(facecolor='none',edgecolor='dimgray',lw=1,ls='--',label=val)]
std_labels = [val]
if parameter != 'risetime':
    handles = handles + [Patch(facecolor='none',edgecolor='dimgray',lw=1,ls='--',label=val)]
    labels = labels + [val]
else:
    handles = [Patch(facecolor='none',edgecolor='dimgray',lw=1,ls='--',label=val)] + handles
    labels = [val] + labels
ax4.legend(handles, labels, loc='lower left',ncol=4,title=legend_title)
ax4.set_xscale('log')
ax4.set_xlim(0.1,10)
yabs_max = abs(max(ax4.get_ylim(), key=abs))
if parameter =='stress_drop':
    ax4.set_ylim(ymin=-yabs_max-0.75)
elif parameter == 'risetime':
    ax4.set_ylim(ymin=-yabs_max-1)
elif parameter == 'vrupt':
    ax4.set_ylim(ymin=-yabs_max-1.25)
ax4.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
ax4.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=4))
ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax4.tick_params(axis='y', left=True, length=5, which='major', labelsize=11)
ax4.tick_params(axis='x', bottom=True, length=5, which='major', labelsize=11)
ax4.tick_params(axis='x', bottom=True, length=3, which='minor')
ax4.grid(True, which="both", ls="-", alpha=0.5)
ax4.set_title(r'$\bf{Acc}$ $\bf{Fourier}$ $\bf{Amplitude}$ $\bf{Spectra}$',fontsize=11)
ax4.axhline(0, ls=':',c='k',lw=1)
yabs_max = abs(max(ax4.get_ylim(), key=abs))
# ax4.set_ylim(-yabs_max,yabs_max)
ax4.set_xlabel('Frequency (Hz)', fontsize=11)
ax4.set_ylabel('ln (obs / sim)', fontsize=11)
# leg = ax4.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,-0.3),facecolor='white',
#                 frameon=True,fontsize=10,title_fontsize=10,ncol=4,markerscale=2)
# leg.set_title(legend_title,prop={'size':10})
ax4.text(-0.08,1.1,r'$\bf{(d)}$',transform=ax4.transAxes,fontsize=10,va='top',ha='right')

plt.subplots_adjust(wspace=0.5,hspace=0.45,bottom=0.1,left=0.115,right=0.95,top=0.925)
plt.savefig(f'/Users/tnye/tsuquakes/manuscript/figures/{parameter}_residuals.png',dpi=300)
