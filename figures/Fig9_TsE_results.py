#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:06:44 2023

@author: tnye
"""

###############################################################################
# This script makes Figure 9.
###############################################################################


# Imports
import numpy as np
import pandas as pd
from glob import glob
from math import floor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import matplotlib.ticker as ticker
from mudpy import viewFQ

epicenter = [100.14,-3.49]

# Get STF from Han Yue paper
yue_stf_df = pd.read_csv('/Users/tnye/tsuquakes/files/STFs/yue_STF.csv',header=None)
yue_time = yue_stf_df[0]
yue_moment = yue_stf_df[1]

mu1 = 'rt1.234x_sf0.41_sd1.196'
mu2 = 'rt1.954x_sf0.469_sd1.196'
sig1 = 'rt1.4x_sf0.45_sd1.0'
sig2 = 'rt1.4x_sf0.45_sd2.0'
sig3 = 'rt1.75x_sf0.42_sd1.0'
sig4 = 'rt1.75x_sf0.42_sd2.0'

# Get rupture files for TsE scenarios
mu1_rupts = sorted(glob(f'/Users/tnye/tsuquakes/simulations/tse_simulations/{mu1}/output/ruptures/*.rupt'))
mu2_rupts = sorted(glob(f'/Users/tnye/tsuquakes/simulations/tse_simulations/{mu2}/output/ruptures/*.rupt'))
sig1_rupts = sorted(glob(f'/Users/tnye/tsuquakes/simulations/tse_simulations/{sig1}/output/ruptures/*.rupt'))
sig2_rupts = sorted(glob(f'/Users/tnye/tsuquakes/simulations/tse_simulations/{sig2}/output/ruptures/*.rupt'))
sig3_rupts = sorted(glob(f'/Users/tnye/tsuquakes/simulations/tse_simulations/{sig3}/output/ruptures/*.rupt'))
sig4_rupts = sorted(glob(f'/Users/tnye/tsuquakes/simulations/tse_simulations/{sig4}/output/ruptures/*.rupt'))
std_rupts = sorted(glob('/Users/tnye/tsuquakes/simulations/tse_simulations/standard/output/ruptures/*.rupt'))

# Initialize lists for time and moment arrays from the simulation files
mu1_t = []
mu1_M1 = []
mu2_t = []
mu2_M1 = []
sig1_t = []
sig1_M1 = []
sig2_t = []
sig2_M1 = []
sig3_t = []
sig3_M1 = []
sig4_t = []
sig4_M1 = []
std_t = []
std_M1 = []

# Calculate moment rates
exp=18
for rupt in mu1_rupts:
    t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=1,t_total=500,stf_type='dreger',plot=False)
    M1=Mr/(10**exp)
    mu1_t.append(t)
    mu1_M1.append(M1)
for rupt in mu2_rupts:
    t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=1,t_total=500,stf_type='dreger',plot=False)
    M1=Mr/(10**exp)
    mu2_t.append(t)
    mu2_M1.append(M1)
for rupt in sig1_rupts:
    t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=1,t_total=500,stf_type='dreger',plot=False)
    M1=Mr/(10**exp)
    sig1_t.append(t)
    sig1_M1.append(M1)
for rupt in sig2_rupts:
    t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=1,t_total=500,stf_type='dreger',plot=False)
    M1=Mr/(10**exp)
    sig2_t.append(t)
    sig2_M1.append(M1)
for rupt in sig3_rupts:
    t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=1,t_total=500,stf_type='dreger',plot=False)
    M1=Mr/(10**exp)
    sig3_t.append(t)
    sig3_M1.append(M1)
for rupt in sig4_rupts:
    t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=1,t_total=500,stf_type='dreger',plot=False)
    M1=Mr/(10**exp)
    sig4_t.append(t)
    sig4_M1.append(M1)
for rupt in std_rupts:
    t, Mr = viewFQ.source_time_function(rupt,epicenter,dt=1,t_total=500,stf_type='dreger',plot=False)
    M1=Mr/(10**exp)
    std_t.append(t)
    std_M1.append(M1)


#%%
rupt_list = []
for file in mu1_rupts:
    rupt_list.append(file.split('/')[-1].strip('.rupt'))

mu1_data = {'Run':rupt_list, 'Time':mu1_t, 'Moment':mu1_M1}
mu1_df = pd.DataFrame(mu1_data)
mu1_df.to_csv('/Users/tnye/tsuquakes/files/STFs/mua_STF_1s.csv')

mu2_data = {'Run':rupt_list, 'Time':mu2_t, 'Moment':mu2_M1}
mu2_df = pd.DataFrame(mu2_data)
mu2_df.to_csv('/Users/tnye/tsuquakes/files/STFs/mub_STF_1s.csv')

sig1_data = {'Run':rupt_list, 'Time':sig1_t, 'Moment':sig1_M1}
sig1_df = pd.DataFrame(sig1_data)
sig1_df.to_csv('/Users/tnye/tsuquakes/files/STFs/siga_STF_1s.csv')

sig2_data = {'Run':rupt_list, 'Time':sig2_t, 'Moment':sig2_M1}
sig2_df = pd.DataFrame(sig2_data)
sig2_df.to_csv('/Users/tnye/tsuquakes/files/STFs/sigb_STF_1s.csv')

sig3_data = {'Run':rupt_list, 'Time':sig3_t, 'Moment':sig3_M1}
sig3_df = pd.DataFrame(sig3_data)
sig3_df.to_csv('/Users/tnye/tsuquakes/files/STFs/sigc_STF_1s.csv')

sig4_data = {'Run':rupt_list, 'Time':sig4_t, 'Moment':sig4_M1}
sig4_df = pd.DataFrame(sig4_data)
sig4_df.to_csv('/Users/tnye/tsuquakes/files/STFs/sigd_STF_1s.csv')

std_data = {'Run':rupt_list, 'Time':std_t, 'Moment':std_M1}
std_df = pd.DataFrame(std_data)
std_df.to_csv('/Users/tnye/tsuquakes/files/STFs/std_STF_1s.csv')


#%%
# Calculate IM residuals

home_dir = f'/Users/tnye/tsuquakes/simulations'

parameter = 'tse_simulations'      # parameter being varied
projects = [mu1,mu2,sig1,sig2,sig3,sig4]
param_vals = [r'$\mu$-a',r'$\mu$-b',r'$\sigma$-a',r'$\sigma$-b',r'$\sigma$-c',r'$\sigma$-d']        # array of parameter values associated w/ the projects

# sns.set_style("whitegrid")

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

val = 'standard'
legend_title = ''
    
# Add default parameter first if risetime 
for i, project in enumerate(projects):
    
    gnss_std_df = pd.read_csv(f'/Users/tnye/tsuquakes/simulations/tse_simulations/standard/flatfiles/residuals/standard_gnss.csv')
   
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
    
    sm_std_df = pd.read_csv(f'/Users/tnye/tsuquakes/simulations/tse_simulations/standard/flatfiles/residuals/standard_sm.csv')

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

# Set up xlabel and title based on parameter being varied 
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
        sm_std_df = pd.read_csv(f'{home_dir}/{parameter}/standard/flatfiles/residuals/standard_sm.csv')
  
        # Remove strong motion stations farther than 600km to avoid surface waves
        far_sm = ['PPBI', 'PSI', 'CGJI', 'TSI', 'CNJI', 'LASI', 'MLSI']
        ind = np.where(np.in1d(np.array(sm_std_df['station']), far_sm).reshape(np.array(sm_std_df['station']).shape)==False)[0]
    
        acc_std_res = np.array(sm_std_df.iloc[ind].reset_index(drop=True).iloc[:,20:].iloc[:,j_bin])
    
        sm_std_bin_list.extend(np.repeat(bin_means[j_bin],len(acc_std_res)))
        acc_std_spec_res_list = np.append(acc_std_spec_res_list,acc_std_res)


#%%

######################## Plot IMs on one figure ###########################

# n_colors = len(projects)  # Set this to your actual number of categories minus 1
# custom_palette = sns.color_palette()[:n_colors]

# Set up Figure
layout = [
    ["A", "B", "C"],
    ["D", "D", "D"],
    ["E", "E", "null"]]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, axs = plt.subplot_mosaic(layout, figsize=(6.5,6.75), gridspec_kw={'height_ratios':[1,1.25,1]})

ax1 = axs['A']
ax2 = axs['B']
ax3 = axs['C']
ax4 = axs['D']
ax5 = axs['E']
axs['null'].remove()

PROPS = {'boxprops':{'edgecolor':'black'},'medianprops':{},'whiskerprops':{},'capprops':{}}

# PGD subplot
ax1.text(-0.25,1.1,r'$\bf{(a)}$',transform=ax1.transAxes,fontsize=10,va='top',ha='right')
grouped_PGD = [list(list(std_pgd_df['ln Residual (m)'].values[~np.isnan(std_pgd_df['ln Residual (m)'].values)]))]
grouped_PGD.extend([list(pgd_df['ln Residual (m)'].values[pgd_df['Parameter'].values == param]) for param in param_vals])
grouped_PGD = np.array([grouped_PGD[0],grouped_PGD[1]+grouped_PGD[2],
                        grouped_PGD[3]+grouped_PGD[4]+grouped_PGD[3]+grouped_PGD[4]])
b = ax1.boxplot(grouped_PGD,patch_artist=True,boxprops=dict(lw=1,
            linestyle='--',edgecolor='black'),
            medianprops=dict(lw=1,color='black'),widths=0.8,
            flierprops=dict(marker='d',markersize=5))
for i, (box, whisker1, whisker2, cap1, cap2, color) in enumerate(zip(
        b['boxes'][1:], b['whiskers'][2::2], b['whiskers'][3::2], 
        b['caps'][2::2], b['caps'][3::2], colors), start=1):
    box.set(color=color, linewidth=1)
    box.set(facecolor=color)
    whisker1.set(color=color, linewidth=1)
    whisker2.set(color=color, linewidth=1)
    cap1.set(color=color, linewidth=1)
    cap2.set(color=color, linewidth=1)
    
b['boxes'][0].set_facecolor('none')
b['boxes'][0].set_edgecolor('k')
b['boxes'][0].set_linestyle('--')
b['medians'][0].set_color('k')
b['medians'][0].set_linestyle('--')
for whisker in b['whiskers'][:2]:
    whisker.set_linestyle('--')
for cap in b['caps'][:2]:
    cap.set_linestyle('--')
for whisker in b['whiskers'][:2]:  # Adjust indices as needed
    whisker.set(color='k')
for cap in b['caps'][:2]:  # Adjust indices as needed
    cap.set(color='k')
b['fliers'][-1].set_color('k')

labels = []
labels.append(val)
labels.extend([r'TsE-$\mu$',r'TsE-$1\sigma$'])
ax1.set_xticklabels(labels,rotation=22)
ax1.xaxis.grid(False)
ax1.yaxis.grid(True)
ax1.set(xlabel=None)
ax1.set_ylabel('ln (obs / sim)',fontsize=11)
yabs_max = abs(max(ax1.get_ylim(), key=abs))
ax1.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax1.set_ylim(ymin=-2, ymax=2)
ax1.tick_params(width=1,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=10)
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.set_title(r'$\bf{PGD}$',fontsize=11)
ax1.axhline(0, ls=':',c='k',lw=1)

# tPGD subplot
ax2.text(-0.35,1.1,r'$\bf{(b)}$',transform=ax2.transAxes,fontsize=10,va='top',ha='right')
grouped_tPGD = [list(list(std_tPGD_df['Residual (s)'].values[~np.isnan(std_tPGD_df['Residual (s)'].values)]))]
grouped_tPGD.extend([list(tPGD_df['Residual (s)'].values[tPGD_df['Parameter'].values == param]) for param in param_vals])
grouped_tPGD = np.array([grouped_tPGD[0],grouped_tPGD[1]+grouped_tPGD[2],
                        grouped_tPGD[3]+grouped_tPGD[4]+grouped_tPGD[3]+grouped_tPGD[4]])

b = ax2.boxplot(grouped_tPGD,patch_artist=True,boxprops=dict(lw=1,
            linestyle='--',edgecolor='black'),
            medianprops=dict(lw=1,color='black'),widths=0.8,
            flierprops=dict(marker='d',markersize=5))
for i, (box, whisker1, whisker2, cap1, cap2, color) in enumerate(zip(
        b['boxes'][1:], b['whiskers'][2::2], b['whiskers'][3::2], 
        b['caps'][2::2], b['caps'][3::2], colors), start=1):
    box.set(color=color, linewidth=1)
    box.set(facecolor=color)
    whisker1.set(color=color, linewidth=1)
    whisker2.set(color=color, linewidth=1)
    cap1.set(color=color, linewidth=1)
    cap2.set(color=color, linewidth=1)

b['boxes'][0].set_facecolor('none')
b['boxes'][0].set_edgecolor('k')
b['boxes'][0].set_linestyle('--')
b['medians'][0].set_color('k')
b['medians'][0].set_linestyle('--')
for whisker in b['whiskers'][:2]:
    whisker.set_linestyle('--')
for cap in b['caps'][:2]:
    cap.set_linestyle('--')
for whisker in b['whiskers'][:2]:  # Adjust indices as needed
    whisker.set(color='k')
for cap in b['caps'][:2]:  # Adjust indices as needed
    cap.set(color='k')
b['fliers'][-1].set_color('k')

labels = []
labels.append(val)
labels.extend([r'TsE-$\mu$',r'TsE-$1\sigma$'])
ax2.xaxis.grid(False)
ax2.yaxis.grid(True)
# ax2.set_xlabel('Parameter distributions',fontsize=11)
ax2.set_ylabel('obs - sim',fontsize=11)
ax2.set_xticklabels(labels,rotation=22)
yabs_max = abs(max(ax2.get_ylim(), key=abs))
# ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax2.tick_params(width=1,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=10)
ax2.set_title(r'$\bf{tPGD}$',fontsize=11)
ax2.axhline(0, ls=':',c='k',lw=1)

# PGA subplot
ax3.text(-0.25,1.1,r'$\bf{(c)}$',transform=ax3.transAxes,fontsize=10,va='top',ha='right')
grouped_PGA = [list(list(std_pga_df['ln Residual (m/s/s)'].values[~np.isnan(std_pga_df['ln Residual (m/s/s)'].values)]))]
grouped_PGA.extend([list(pga_df['ln Residual (m/s/s)'].values[pga_df['Parameter'].values == param]) for param in param_vals])
grouped_PGA = np.array([grouped_PGA[0],grouped_PGA[1]+grouped_PGA[2],
                        grouped_PGA[3]+grouped_PGA[4]+grouped_PGA[3]+grouped_PGA[4]])

b = ax3.boxplot(grouped_PGA,patch_artist=True,boxprops=dict(lw=1,
            linestyle='--',edgecolor='black'),
            medianprops=dict(lw=1,color='black'),widths=0.8,
            flierprops=dict(marker='d',markersize=5))
for i, (box, whisker1, whisker2, cap1, cap2, color) in enumerate(zip(
        b['boxes'][1:], b['whiskers'][2::2], b['whiskers'][3::2], 
        b['caps'][2::2], b['caps'][3::2], colors), start=1):
    box.set(color=color, linewidth=1)
    box.set(facecolor=color)
    whisker1.set(color=color, linewidth=1)
    whisker2.set(color=color, linewidth=1)
    cap1.set(color=color, linewidth=1)
    cap2.set(color=color, linewidth=1)
    
b['boxes'][0].set_facecolor('none')
b['boxes'][0].set_edgecolor('k')
b['boxes'][0].set_linestyle('--')
b['medians'][0].set_color('k')
b['medians'][0].set_linestyle('--')
for whisker in b['whiskers'][:2]:
    whisker.set_linestyle('--')
for cap in b['caps'][:2]:
    cap.set_linestyle('--')
for whisker in b['whiskers'][:2]:  # Adjust indices as needed
    whisker.set(color='k')
for cap in b['caps'][:2]:  # Adjust indices as needed
    cap.set(color='k')
b['fliers'][-1].set_color('k')

labels = []
labels.append(val)
labels.extend([r'TsE-$\mu$',r'TsE-$1\sigma$'])
ax3.set(xlabel=None)
ax3.set_ylabel(r'ln (obs / sim)',fontsize=11)
ax3.set_xticklabels(labels,rotation=22)
yabs_max = abs(max(ax3.get_ylim(), key=abs))
ax3.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax3.tick_params(width=1,length=5,right=False,top=False,labelleft=True,labelright=False,labelbottom=True,labeltop=False,labelsize=10)
ax3.yaxis.set_major_locator(MultipleLocator(1))
ax3.set_title(r'$\bf{PGA}$',fontsize=11)
ax3.axhline(0, ls=':',c='k',lw=1)
ax3.yaxis.grid(True)

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
grouped_projects = ['mu','sigma']
for i, project in enumerate(grouped_projects):
    if i == 0:
        positions = bins - (width(bins,w)/4)
    elif i == 1:
        positions = bins + (width(bins,w)/4)
        
    project_df = acc_spec_df[acc_spec_df['Project'].str.contains(project, case=False)]
    grouped_res = [list(project_df[f'{label} ({res_units})'].values[project_df['Frequency (Hz)'].values == fbin]) for fbin in bins]
    ax4.boxplot(grouped_res,positions=positions,patch_artist=True,
                boxprops=dict(lw=1,linestyle='--',edgecolor=colors[i],facecolor=colors[i]),
                medianprops=dict(lw=1,color='black'),
                whiskerprops=dict(lw=1,color=colors[i]),
                capprops=dict(lw=1,color=colors[i]),
                widths=width(bins,w)/2,
                flierprops=dict(marker='d',markersize=5)
                )
    handles = handles + [Patch(facecolor=colors[i],edgecolor=colors[i],lw=1,ls='-',label=param_vals[i])]
    labels = labels + [param_vals[i]]
    ax4.boxplot(grouped_res_std,positions=bins,
                boxprops=dict(lw=1,linestyle='--',color='black'),
                medianprops=dict(lw=1,color='black',ls='--'),
                whiskerprops=dict(lw=1,color='black',ls='--'),
                capprops=dict(lw=1,color='black',ls='--'),
                widths=width(bins,w),
                flierprops=dict(marker='d',linewidth=0.5,markersize=5)
                )
handles = [Patch(facecolor='white',edgecolor='black',lw=1,ls='--',label=val)] + handles 
labels = ['standard',r'TsE-$\mu$',r'TsE-$1\sigma$']
ax4.legend(handles, labels, loc='lower left',ncol=4,title=legend_title,fontsize=10)
ax4.set_xscale('log')
ax4.set_xlim(0.1,10)
ax4.set_ylim(ymin=-yabs_max-1.5)
yabs_max = abs(max(ax4.get_ylim(), key=abs))
ax4.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
ax4.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=4))
ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax4.tick_params(axis='y', left=True, length=5, which='major', labelsize=11)
ax4.tick_params(axis='x', bottom=True, length=5, which='major', labelsize=11)
ax4.tick_params(axis='x', bottom=True, length=3, which='minor')
ax4.grid(True, which="both", ls="-", alpha=0.5)
ax4.set_title(r'$\bf{Acc}$ $\bf{Fourier}$ $\bf{Amplitude}$ $\bf{Spectra}$',fontsize=11)
ax4.axhline(0, ls=':',c='k',lw=1)
ax4.set_xlabel('Frequency (Hz)', fontsize=11)
ax4.set_ylabel('ln (obs / sim)', fontsize=11)
ax4.text(-0.08,1.1,r'$\bf{(d)}$',transform=ax4.transAxes,fontsize=10,va='top',ha='right')
ax4.grid(True, which="both", ls="-", alpha=0.5)

ax5.plot(yue_time,yue_moment,c='k',lw=1.5,label='Yue14 STF')
ax5.plot(std_df['Time'].loc[i],np.mean(std_df['Moment']),c='k',lw=1.5,ls='--',label='standard')
ax5.plot(mu1_df['Time'].loc[0],np.mean(np.concatenate([mu1_df['Moment'].values,mu2_df['Moment'].values])),c=colors[0],lw=1.5,label=r'TsE-$\mu$')  
ax5.plot(mu1_df['Time'].loc[0],np.mean(np.concatenate([sig1_df['Moment'].values,
         sig2_df['Moment'].values,sig3_df['Moment'].values,sig4_df['Moment'].values])),c=colors[1],lw=1.5,label=r'TsE-$1\sigma$')  
ax5.tick_params(axis='x', length=5, which='major', labelsize=11)
ax5.tick_params(axis='y', length=5, which='major', labelsize=11)
ax5.grid(True, which="both", ls="-", alpha=0.5)
ax5.legend(loc='upper right',fontsize=10)
ax5.text(-0.18,1.1,r'$\bf{(e)}$',transform=ax5.transAxes,fontsize=10,va='top',ha='right')
ax5.set_xlim(0,120)
ax5.set_ylim(0,17)    
ax5.set_xlabel('Time (s)',fontsize=11)
ax5.set_ylabel(r'Moment rate, $10^{18}$ Nm/s',fontsize=11)
ax5.set_title(r'$\bf{STFs}$',fontsize=11)
plt.subplots_adjust(wspace=0.55,hspace=0.5,bottom=0.075,left=0.13,right=0.95,top=0.95)
plt.savefig(f'/Users/tnye/tsuquakes/manuscript/figures/Fig9_TsE_residuals.png',dpi=300)
