#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:55:53 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from scipy.optimize import root_scalar
from itertools import combinations
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator
from math import ceil
import gmm_call as gmm

## Kernel tests
# kernel = C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
# kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#     noise_level=1, noise_level_bounds=(1e-5, 1e1)
# )
# kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#     noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))


#%%

###############################################################################
# 1D test with stress drop
###############################################################################

sm_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/HF_mean_residuals_corrected.csv')

stress = sm_df['stress drop'].values
HF_res = sm_df['HF res'].values

# Define input
oneD_data = np.array(stress).reshape(-1, 1)

# Define prediction grid
oneD_x = np.linspace(0.1,4,1000)
oneD_X = oneD_x.reshape(-1, 1)

# Run Gaussian process
# oneD_gp = GaussianProcessRegressor(kernel=None,n_restarts_optimizer=9)
# oneD_gp = GaussianProcessRegressor(kernel=C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"),n_restarts_optimizer=9)
# oneD_gp = GaussianProcessRegressor(kernel=RBF(1.0, length_scale_bounds="fixed"),n_restarts_optimizer=9)

# kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#     noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)
# )
# kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#     noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)
# )

# Kernel for 1D regression
# oneD_kernel = RBF(1.0, length_scale_bounds=(0.4,2.9))
oneD_kernel = RBF(2.9, length_scale_bounds=("fixed"))

oneD_gp = GaussianProcessRegressor(kernel=oneD_kernel,n_restarts_optimizer=9)

# Get grid predictions
oneD_gp.fit(oneD_data, HF_res)
oneD_pred, oneD_std = oneD_gp.predict(oneD_X, return_std=True)
hf_res_std = np.std(oneD_pred)

# Define the mean function based on the GPR predictions
def oneD_objective_function(x):
    y_mean, _ = oneD_gp.predict(np.array([x]).reshape(-1, 1), return_std=True)
    return abs(y_mean[0])

initial_guess = np.array([1.5])
x_cross_zero = minimize(oneD_objective_function, initial_guess, method='Nelder-Mead', tol=1e-7).x[0]

print(x_cross_zero)

#%%

###############################################################################
# twoD test with risetime and vrupt
###############################################################################

gnss_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_parameter_residuals_exNGNG.csv')
# gnss_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_event_residuals_exNGNG.csv')

kfactor = gnss_df['k-factor'].values
ssf = gnss_df['ssf'].values
tPGD_res = gnss_df['tPGD res'].values

# Define input
twoD_data = np.column_stack((kfactor, ssf))

# Define prediction grid
twoD_x = np.linspace(1,2.3,100)
twoD_y = np.linspace(0.37,0.49,100)
twoD_X = np.column_stack((twoD_x, twoD_y))
x_mesh, y_mesh = np.meshgrid(twoD_x, twoD_y)
twoD_x_plot = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

# Kernel for 2D regressoin

# Calculate pairwise distances
pairwise_distances = []
for pair in combinations(twoD_data, 2):
    distance = np.linalg.norm(pair[0] - pair[1])
    pairwise_distances.append(distance)

# Find minimum and maximum distances
min_spacing = np.unique(pairwise_distances)[1]
max_spacing = np.max(pairwise_distances)

# twoD_kernel = RBF(0.7, length_scale_bounds=(0.7,max_spacing))
twoD_kernel = RBF(max_spacing, length_scale_bounds=("fixed"))

# Run Gaussian process
twoD_gp = GaussianProcessRegressor(kernel=twoD_kernel,n_restarts_optimizer=10)
twoD_gp.fit(twoD_data, tPGD_res)
twoD_pred, twoD_std = twoD_gp.predict(twoD_x_plot, return_std=True)
tpgd_res_std = np.std(twoD_pred)

# Get grid predictions
z_pred_mesh = twoD_pred.reshape(x_mesh.shape)
z_std_mesh = twoD_std.reshape(x_mesh.shape)

# Define residual ranges for plotting
res_lowerlim = -ceil(np.max(np.abs(z_pred_mesh)))
res_upperlim = ceil(np.max(np.abs(z_pred_mesh)))
# std_lowerlim = 0
# std_upperlim = 0.005

# Define the objective function
def twoD_objective_function(xy):
    x, y = xy
    # print(f'x = {x}, y = {y}')
    mean_value, _ = twoD_gp.predict(np.array([[x, y]]), return_std=True)
    return np.abs(mean_value)

# Initial guess for the minimum (can be any twoD point)
initial_guess_1 = np.array([1.65,0.415])
result_1 = minimize(twoD_objective_function, initial_guess_1, method='Nelder-Mead',tol=1e-7)
min_coords_1 = result_1.x

initial_guess_2 = np.array([2,0.445])
result_2 = minimize(twoD_objective_function, initial_guess_2, method='Nelder-Mead',tol=1e-7)
min_coords_2 = result_2.x

initial_guess_3 = np.array([1.2,0.405])
result_3 = minimize(twoD_objective_function, initial_guess_3, method='Nelder-Mead',tol=1e-7)
min_coords_3 = result_3.x


print(f'Ideal Parameters')
print(min_coords_1)
print(min_coords_2)
print(min_coords_3)


#%%

##### Plot Results #####

line_x = np.array([1,2.3])
line_y = line_x*0.088 + 0.270
line_x2 = np.array([5.4,12.42])
line_y2 = line_y*3.24

# Set up 2D prediction colormap
pred_cmap = plt.get_cmap('seismic') 
pred_cNorm  = colors.Normalize(vmin=res_lowerlim, vmax=res_upperlim)
pred_scalarMap = cmx.ScalarMappable(norm=pred_cNorm, cmap=pred_cmap)

# Set up 2D standard deviation colormap
std_lowerlim = 7e-7
std_upperlim = 1.75e-6
std_cmap = plt.get_cmap('viridis') 
std_cNorm  = colors.Normalize(vmin=std_lowerlim, vmax=std_upperlim)
std_scalarMap = cmx.ScalarMappable(norm=std_cNorm, cmap=std_cmap)

## Initialize figure
layout = [["A", "B"],["null","null"],["C", "C"]]
fig, axs = plt.subplot_mosaic(layout, figsize=(6.5,6.5), gridspec_kw={'height_ratios':[2,0,1.25]})
axs['null'].remove()
plt.subplots_adjust(left=0.09,right=0.91,bottom=0.175, top=0.925, wspace=0.5, hspace=0.1)

# 2D prediction
pred_ax = axs['A']
pred_ax.grid(False)
pred_ax.set_xlabel(r'$\bf{Rise}$ $\bf{time}$ $\bf{(s)}$',size=10)
pred_ax.set_ylabel(r'$\bf{V_{rupt}}$ $\bf{(km/s)}$',size=10)
pred_ax.set_xlim(5.4,12.4)
pred_ax.set_ylim(1.2,1.6)
pred_ax.set_yticks([1.2,1.3, 1.4, 1.5, 1.6])
rise_ticks = np.array([1,1.3,1.6,1.9,2.2])
pred_ax.set_xticks(np.around(rise_ticks*5.4,1))
pred_ax.tick_params(axis='both',rotation=0,labelsize=10)

pred_ax2 = pred_ax.twinx().twiny()
temp1 = pred_ax.twinx()
pred_ax2.grid(False)
temp1.grid(False)
pred_ax2.set_xlabel(r'$\bf{k-factor}$',size=10)
pred_ax2.set_xlim(1,2.3)
pred_ax2.set_ylim(0.37,0.49)
pred_ax2.set_yticks([0.37,0.4,0.43,0.46,0.49])
pred_ax2.set_xticks(rise_ticks)
pred_ax2.tick_params(axis='both',labelsize=10)
pred_ax2.contourf(x_mesh, y_mesh, z_pred_mesh, levels=750, cmap='seismic', vmin=res_lowerlim, vmax=res_upperlim)
pred_ax2.contour(x_mesh, y_mesh, z_pred_mesh, levels=[0], linewidths=0.8, linestyles='-', colors='k')
pred_ax2.contour(x_mesh, y_mesh, z_pred_mesh, levels=[0-tpgd_res_std,0+tpgd_res_std], linewidths=0.8, linestyles='--', colors='k')
pred_ax2.plot([0],[0],c='k',lw=0.8,label='0-residual line')
pred_ax2.plot([0],[0],c='k',lw=0.8,ls='--',label=r'GPR residuals std dev')
pred_cbar = fig.colorbar(pred_scalarMap,ticks=[-30,-15,0,15,30],ax=pred_ax2,pad=0.2,orientation='horizontal')
pred_cbar.set_label(r'GPR $\delta_{tPGD}$ prediction (s)', size=10)
temp1.tick_params(left=False,right=False,top=False,bottom=False,labelleft=False,
                  labelright=False,labeltop=False,labelbottom=False)

# for i in range(len(kfactor)):
for i in range(70):
    # if i == 0:
    #     pred_ax2.scatter(kfactor[i],ssf[i],facecolors='none',edgecolors='k',lw=0.4,s=40,label='Regression parameters sampled')
    # else:
    #     pred_ax2.scatter(kfactor[i],ssf[i],facecolors='none',edgecolors='k',lw=0.4,s=40)
    if i == 0:
        pred_ax2.scatter(kfactor[i*30],ssf[i*30],edgecolors='k',facecolors=pred_scalarMap.to_rgba(np.mean(tPGD_res[:30])),lw=0.4,s=40,label='Regression parameters sampled')
    else:
        pred_ax2.scatter(kfactor[i*30],ssf[i*30],edgecolors='k',facecolors=pred_scalarMap.to_rgba(np.mean(tPGD_res[i*30:i*30+30])),lw=0.4,s=40)
# pred_ax2.scatter([1.422,1.742,1.2,2.0],[0.395,0.422,0.41,0.42],c='k',lw=1,s=40,alpha=1,label='TsE parameters evalauted')
pred_ax2.scatter([1.954,1.234,1.4,1.75],[0.469,0.410,0.45,0.42],c='k',lw=1,s=40,alpha=1,label='TsE parameters evalauted')
# pred_ax2.plot(line_x,line_y,c='yellow',lw=2)

temp1.plot([0,0],[0,0],c='yellow',lw=2,label='Linear fit')
temp1.legend(loc='upper left')
pred_ax2.set_ylabel('SSF',size=10) 

# 2D Std dev
std_ax = axs['B']
std_xax2 = std_ax.twiny()
std_yax2 = std_ax.twinx()
std_ax.grid(False)
std_xax2.grid(False)
std_yax2.grid(False)
std_ax.set_xlabel(r'$\bf{Rise}$ $\bf{time}$ $\bf{(s)}$',size=10)
std_xax2.set_xlabel(r'$\bf{k-factor}$',size=10)
std_yax2.set_ylabel(r'$\bf{V_{rupt}}$ $\bf{(km/s)}$',rotation=270,labelpad=15,size=10)
std_xax2.set_xlim(1,2.3)
std_ax.set_ylim(0.37,0.49)
std_ax.set_xlim(5.4,12.4)
std_yax2.set_ylim(1.2,1.6)
std_ax.set_yticks([0.37,0.4,0.43,0.46,0.49])
std_yax2.set_yticks([1.2,1.3, 1.4, 1.5, 1.6])
rise_ticks = np.array([1,1.3,1.6,1.9,2.2])
std_xax2.set_xticks(rise_ticks)
std_ax.set_xticks(np.around(rise_ticks*5.4,1))
std_xax2.contourf(x_mesh, y_mesh, z_std_mesh, levels=750, cmap='viridis', vmin=std_lowerlim, vmax=std_upperlim)
std_xax2.plot([0],[0],c='k',lw=0.8,label='0-residual line')
std_xax2.plot([0],[0],c='k',lw=0.8,ls='--',label=r'$1\sigma$ residuals')
std_cbar = plt.colorbar(std_scalarMap,ax=std_xax2, orientation='horizontal',pad=0.2)
std_cbar.set_label(r'GPR std dev', size=10)
# std_ax.tick_params(labelleft=False)
std_ax.tick_params(axis='both',labelsize=10)
std_xax2.tick_params(axis='x',labelsize=10)
std_yax2.tick_params(axis='y',labelsize=10)
std_ax.set_ylabel(r'$\bf{SSF}$',size=10)

# 1D stress drop
axs['C'].grid(alpha=0.25)
axs['C'].scatter(stress,HF_res,facecolors='none',edgecolors='k',lw=0.2,s=40,alpha=0.7)
axs['C'].scatter([1.0,1.419,2.0],[0,0,0],c='k',lw=1,s=40,alpha=1)
axs['C'].plot(oneD_x, oneD_pred, label=r"GPR $\delta_{HF}$ prediction")
axs['C'].fill_between(
    oneD_X.ravel(),
    oneD_pred - 2*oneD_std,
    oneD_pred + 2*oneD_std,
    alpha=0.5,
    label='95% confidence',
)
axs['C'].set_xlim(0,3.5)
axs['C'].axhline(hf_res_std,lw=1,ls='--',c='k')
axs['C'].axhline(-hf_res_std,lw=1,ls='--',c='k')
axs['C'].axvline(x_cross_zero,ls='-',c='k',lw=1)
axs['C'].set_ylim(-3,3)
axs['C'].set_xlabel(r'$\bf{Stress}$ $\bf{Parameter}$ $\bf{(MPa)}$',fontsize=10)
axs['C'].set_ylabel(r'$\bf{\delta_{HF}}$',fontsize=10)
axs['C'].tick_params(axis='both', which='major', labelsize=10)
# axs['C'].text(1.035, 0.6, r'$\bf{SSF}$', ha='left', transform=axs['A'].transAxes)

# tpgd_handles, tpgd_labels = pred_ax2.get_legend_handles_labels()
# sd_handles, sd_labels = axs['C'].get_legend_handles_labels()
# handles = tpgd_handles+sd_handles
# labels = tpgd_labels+sd_labels
axs['C'].legend(loc='upper right',fontsize=10)
handles, labels = pred_ax2.get_legend_handles_labels()
handles[2].set_facecolor('none')
axs['C'].legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5,-0.275),facecolor='white',
                frameon=True,fontsize=10,ncol=2)
# pred_ax2.legend(handles, labels, loc='upper center',bbox_to_anchor=(1.3,-1.95),facecolor='white',
#                 frameon=True,fontsize=10,ncol=2)
plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/unannotated/GPR_figure_revised.png',dpi=300)
# plt.close()

