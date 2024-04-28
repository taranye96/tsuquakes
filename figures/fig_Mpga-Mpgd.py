#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:15:38 2023

@author: tnye
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

sahakian_df1 = pd.read_csv('/Users/tnye/tsuquakes/data/sahakian2019/mag_differences_KStests_finalmag.csv')
sahakian_df2 = pd.read_csv('/Users/tnye/tsuquakes/data/sahakian2019/mag_differences_KStests_pgdmag.csv')
obs_M = sahakian_df1.mw.values
obs_Mdiff = sahakian_df2.magdiff_KSpvalue.values

Mdiff_df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpga-Mpgd_results_m7.8.csv')


#%% 

plt.rcParams.update({'font.size': 10})
# markers=['X','^','s','v','.','d','D','<','*','H','8','o','P','>','p','h']
markers=['o','o','o','o','o','o','o','o','*','o','o','o','o','o','o','o']
labels=['Ecuador2016','Ibaraki2011','Illapel2015','Iquique2014','Iquique_aftershock2014',
        'Iwate2011','Maule2010','Melinka2016','Mentawai2010','Miyagi2011A','Miyagi2011B',
        'Nepal2015','Nepal_aftershock2015','Nicoya2012','Tohoku2011','Tokachi2003']

layout = [["A", "B"],["C", "C"]]

# Plot Magnitude residuals
# fig, ax = plt.subplots(1,1,figsize=(6,5))
fig, axs = plt.subplot_mosaic(layout, figsize=(6.5,5.75), gridspec_kw={'height_ratios':[0.5,2]})

axs['A'].hist(Mdiff_df['Mpgd'],color='gray',bins=5)
axs['A'].axvline(7.6,c='r')
axs['A'].set_xlabel(r'$M_{PGD}$')
axs['A'].set_ylabel('Counts')
axs['A'].text(-0.2, 1, r'$\bf{(a)}$', ha='right', va='top', transform=axs['A'].transAxes)
axs['B'].hist(Mdiff_df['Mpga'],color='gray')
axs['B'].axvline(6.35,c='r')
axs['B'].set_xlabel(r'$M_{PGA}$')
axs['B'].text(-0.15, 1, r'$\bf{(b)}$', ha='right', va='top', transform=axs['B'].transAxes)

for i in range(len(obs_M)):
    # axs['C'].scatter(obs_M[i],obs_Mdiff[i],s=100,marker=markers[i],c='mediumaquamarine',label=labels[i])
    if i == 0:
        axs['C'].scatter(obs_M[i],obs_Mdiff[i],s=100,marker=markers[i],c='mediumaquamarine',label='Typical Eqs')
    elif markers[i] == '*':
        axs['C'].scatter(obs_M[i],obs_Mdiff[i],s=140,marker=markers[i],c='mediumorchid',label='Tsunami Eq')
    else:
        axs['C'].scatter(obs_M[i],obs_Mdiff[i],s=100,marker=markers[i],c='mediumaquamarine')
    

std = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='standard')[0]],positions=[7.8],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='mediumaquamarine',lw=1)),whiskerprops=dict(color='mediumaquamarine'),capprops=dict(color='mediumaquamarine'),medianprops=dict(color='mediumaquamarine',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'mediumaquamarine','markeredgecolor':'none'})
mu = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']!='standard')[0]],positions=[7.8],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='mediumorchid',lw=1)),whiskerprops=dict(color='mediumorchid'),capprops=dict(color='mediumorchid'),medianprops=dict(color='mediumorchid',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'mediumorchid','markeredgecolor':'none'})
# sigb = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd2.0')[0]],positions=[7.78],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='mediumorchid',lw=1)),whiskerprops=dict(color='mediumorchid'),capprops=dict(color='mediumorchid'),medianprops=dict(color='mediumorchid',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'mediumorchid','markeredgecolor':'none'})
# sigd = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd2.0')[0]],positions=[7.82],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='hotpink',lw=1)),whiskerprops=dict(color='hotpink'),capprops=dict(color='hotpink'),medianprops=dict(color='hotpink',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'hotpink','markeredgecolor':'none'})
# siga = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd1.0')[0]],positions=[7.78],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='mediumturquoise',lw=1)),whiskerprops=dict(color='mediumturquoise'),capprops=dict(color='mediumturquoise'),medianprops=dict(color='mediumturquoise',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'mediumturquoise','markeredgecolor':'none'})
# sigc = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd1.0')[0]],positions=[7.82],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='salmon',lw=1)),whiskerprops=dict(color='salmon'),capprops=dict(color='salmon'),medianprops=dict(color='salmon',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'salmon','markeredgecolor':'none'})
# mub = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.742_sf0.422_sd1.428')[0]],positions=[7.78],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='orangered',lw=1)),whiskerprops=dict(color='orangered'),capprops=dict(color='orangered'),medianprops=dict(color='orangered',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'orangered','markeredgecolor':'none'})
# mua = axs['C'].boxplot(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.422_sf0.395_sd1.428')[0]],positions=[7.82],patch_artist=True,boxprops=(dict(facecolor='none',edgecolor='yellowgreen',lw=1)),whiskerprops=dict(color='yellowgreen'),capprops=dict(color='yellowgreen'),medianprops=dict(color='yellowgreen',lw=1),flierprops={'marker': 'o','markersize':4,'markerfacecolor':'yellowgreen','markeredgecolor':'none'})

axs['C'].set_xticks([7.0,7.5,8.0,8.5,9.0])
axs['C'].set_xticklabels([7.0,7.5,8.0,8.5,9.0])
axs['C'].xaxis.set_minor_locator(MultipleLocator(0.1))
axs['C'].axhline(0,c='gray',ls='--',lw=0.8)
axs['C'].set_ylim(ymax=1)
axs['C'].set_xlim(7,9.1)
axs['C'].grid(which='both',alpha=0.25)
axs['C'].set_ylabel(r'$M_{PGA}-M_{PGD}$')
axs['C'].set_xlabel(r'$M_{w}$')
handles, labels = axs['C'].get_legend_handles_labels()
# handles.extend([std['boxes'][0],mua['boxes'][0],mub['boxes'][0],siga['boxes'][0],sigb['boxes'][0],sigc['boxes'][0],sigd['boxes'][0]])
# labels.extend(['Standard',r'$\mu$-a',r'$\mu$-b',r'1$\sigma$-a',r'1$\sigma$-b',r'1$\sigma$-c',r'1$\sigma$-d'])
handles.extend([std['boxes'][0],mu['boxes'][0]])
labels.extend(['Typical Eq Simulations',r'Tsunami Eq Simulations'])
axs['C'].legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,-0.14),facecolor='white',
                frameon=True,fontsize=8,markerscale=0.75,ncol=2)
axs['C'].text(-0.08, 1, r'$\bf{(c)}$', ha='right', va='top', transform=axs['C'].transAxes)
plt.subplots_adjust(bottom=0.175,top=0.98,left=0.135,right=0.9,hspace=0.3,wspace=0.3)
plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/Mpga-Mpgd_revised.png',dpi=300)
# plt.close()


#%%

data = {
        r'$\mu$-a':Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.422_sf0.395_sd1.428')[0]],
        r'$\mu$-b':Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.742_sf0.422_sd1.428')[0]],
        r'$\sigma$-a':Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd1.0')[0]],
        r'$\sigma$-b':Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd2.0')[0]],
        r'$\sigma$-c':Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd1.0')[0]],
        r'$\sigma$-d':Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd2.0')[0]]
        }

categories = list(data.keys())
values = list(data.values())

fig, ax = plt.subplots(1,1,figsize=(3,1.5))
ax.boxplot(values, labels=categories, patch_artist=True,
           boxprops=dict(facecolor='none', edgecolor='mediumorchid', linestyle='--', lw=1),
           whiskerprops=dict(color='mediumorchid', linestyle='--'),
           capprops=dict(color='mediumorchid', linestyle='--'),
           medianprops=dict(color='mediumorchid', linestyle='--', lw=1),
           flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'mediumorchid', 'markeredgecolor': 'none'})
# ax.set_title('Tsunami Earthquake Parameter Set',size=9,weight='bold')
# ax.set_ylabel(r'$M_{PGA}-M_{PGD}$',size=8)
ax.tick_params(axis='both',which='major',labelsize=8)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.set_facecolor('none')
ax.yaxis.tick_right()
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.patch.set_alpha(0)

plt.subplots_adjust(bottom=0.15,right=0.85,left=0.05)
plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/Mpga-Mpgd_tse.png',dpi=300)


#%%
fig, axs = plt.subplots(1,2,figsize=(6,3))
axs[0].hist(Mdiff_df['Mpgd'])
axs[1].hist(Mdiff_df['Mpga'])

plt.subplots_adjust(left=0.195,right=0.9)




#%%

fig, axs = plt.subplots(3,3,figsize=(6.5,5))
axs[0,0].hist(obs_Mdiff,bins=20)
axs[0,1].hist(Mdiff_standard_df['Mpga-Mpgd'].values,bins=10)
axs[0,2].remove()
axs[1,0].hist(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.422_sf0.395_sd1.581')[0]],bins=10)
axs[1,1].hist(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.742_sf0.422_sd1.581')[0]],bins=10)
axs[1,2].hist(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd1.0')[0]],bins=10)
axs[2,0].hist(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd2.0')[0]],bins=10)
axs[2,1].hist(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd1.0')[0]],bins=10)
axs[2,2].hist(Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd2.0')[0]],bins=10)

axs[0,0].set_xlim(-2,1)
axs[0,1].set_xlim(-2,1)
axs[1,0].set_xlim(-2,1)
axs[1,1].set_xlim(-2,1)
axs[1,2].set_xlim(-2,1)
axs[2,0].set_xlim(-2,1)
axs[2,1].set_xlim(-2,1)
axs[2,2].set_xlim(-2,1)

plt.subplots_adjust(bottom=0.15,right=0.95,left=0.1,top=0.95,hspace=0.3,wspace=0.3)





