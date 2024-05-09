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
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from scipy.optimize import root_scalar
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from math import ceil
import gmm_call as gmm

# Read in csv files with mean residuals
# gnss_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_mean_residuals.csv')


#%%

###############################################################################
# 1D test with stress drop
###############################################################################

sm_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/HF_mean_residuals_corrected.csv')

# GPR parameters
length_scale = 1.0

# ind = np.where(sm_df['stress drop']!=5.0)[0]
ind = np.where(sm_df['stress drop']<=3.0)[0]
stress = sm_df['stress drop'].values[ind]
HF_res = sm_df['HF res'].values[ind]


# Get GMM std dev
rrup = np.arange(10,600,10)
vs30 = vs30 = np.full_like(rrup,760)
M = vs30 = np.full_like(rrup,7.84)
ln_pga_ngasub, sd_nga = gmm.parker2020(M,rrup,vs30,'PGA')

# Define input
data = np.array(stress).reshape(-1, 1)

# Define prediction grid
x = np.linspace(0.1,4,1000)
X = x.reshape(-1, 1)

# Run Gaussian process
# gp = GaussianProcessRegressor(kernel=None,n_restarts_optimizer=9)
gp = GaussianProcessRegressor(kernel=RBF(length_scale=5.0), n_restarts_optimizer=9)

# Get grid predictions
gp.fit(data, HF_res)
mean_prediction, std_prediction = gp.predict(X, return_std=True)
res_std = np.std(mean_prediction)

# Define the mean function based on the GPR predictions
def mean_function(x):
    y_mean, _ = gp.predict(np.array([x]).reshape(-1, 1), return_std=True)
    return abs(y_mean[0])

# Define the mean function based on the GPR predictions
def objective_function(x):
    y_mean, _ = gp.predict(np.array([x]).reshape(-1, 1), return_std=True)
    return abs(y_mean[0])

# # Use numerical optimization to find the root (where mean function crosses zero)
# # result = root_scalar(mean_function, bracket=[-5, 5])
# result = root_scalar(mean_function, bracket=[0.1, 2])
# x_cross_zero = result.root

initial_guess = np.array([1.5])
x_cross_zero = minimize(objective_function, initial_guess, method='Nelder-Mead', tol=1e-7).x[0]


##### Plot Results #####

fig, ax = plt.subplots(1,1,figsize=(4,3.75))
ax.grid(alpha=0.25)
ax.scatter(stress,HF_res,facecolors='none',edgecolors='k',lw=0.2,s=40,alpha=0.7,label='Locations of regression runs')
ax.scatter([1.0,1.581,2.0],[0,0,0],c='k',lw=1,s=40,alpha=1,label='Locations of ideal runs')
ax.plot(x, mean_prediction, label="Mean prediction")
ax.fill_between(
    X.ravel(),
    mean_prediction - 2*std_prediction,
    mean_prediction + 2*std_prediction,
    alpha=0.5,
    label='95% confidence',
)
ax.set_xlim(0,3.5)
ax.set_ylim(-2.5,5)
ax.axhline(res_std,lw=1,ls='--',c='k',label=r'1$\sigma$ residuals')
ax.axhline(-res_std,lw=1,ls='--',c='k')
ax.axhline(0,ls='-',c='k',lw=1,label='0-residual line')
ax.axvline(x_cross_zero,ls='-',c='r',lw=1,label='Ideal stress drop')
ax.legend(bbox_to_anchor=(0.45,-0.25),loc='upper center',fontsize=9,ncol=2)
ax.set_xlabel('Stress Drop (MPa)',fontsize=10)
ax.set_ylabel('Mean IM residuals (ln)',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.subplots_adjust(left=0.14,right=0.95,bottom=0.4, top=0.95)
# plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/HF_regression_1D_m7.8.png',dpi=300)
# plt.close()


#%%

###############################################################################
# 2D test with risetime and vrupt
###############################################################################

exNGNG = True

if exNGNG == True:
    gnss_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_parameter_residuals_exNGNG.csv')
else:
    gnss_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_parameter_residuals.csv')

kfactor = gnss_df['k-factor'].values
ssf = gnss_df['ssf'].values
tPGD_res = gnss_df['tPGD res'].values

ind = np.where(gnss_df['ssf']>=0.37)[0]
# my_list = ['mentawai.000000','mentawai.000001','mentawai.000002','mentawai.000003']
# ind = index = np.where((df['ssf']>=0.4) & (df['stress drop']>1.5) & (df['run'].isin(my_list)))[0]
kfactor = gnss_df['k-factor'].values[ind]
ssf = gnss_df['ssf'].values[ind]
tPGD_res = gnss_df['tPGD res'].values[ind]

# Define input
x = kfactor
y = ssf
data = np.column_stack((x, y))

# Define prediction grid
x = np.linspace(1,2.3,100)
y = np.linspace(0.37,0.49,100)
X = np.column_stack((x, y))
x_mesh, y_mesh = np.meshgrid(x, y)
x_plot = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

# Run Gaussian process
gp = GaussianProcessRegressor(kernel=None,n_restarts_optimizer=9)
gp.fit(data, tPGD_res)
mean_prediction, std_prediction = gp.predict(x_plot, return_std=True)
res_std = np.std(mean_prediction)

# Get grid predictions
z_pred_mesh = mean_prediction.reshape(x_mesh.shape)
z_std_mesh = std_prediction.reshape(x_mesh.shape)

# Define residual ranges for plotting
res_lowerlim = -ceil(np.max(np.abs(z_pred_mesh)))
res_upperlim = ceil(np.max(np.abs(z_pred_mesh)))
std_lowerlim = 0
std_upperlim = 0.005

# Define the objective function
def objective_function(xy):
    x, y = xy
    # print(f'x = {x}, y = {y}')
    mean_value, _ = gp.predict(np.array([[x, y]]), return_std=True)
    return np.abs(mean_value)

# Initial guess for the minimum (can be any 2D point)
initial_guess = np.array([1.4,0.39])
result_1 = minimize(objective_function, initial_guess, method='Nelder-Mead',tol=1e-7)
min_coords_1 = result_1.x

initial_guess = np.array([1.7,0.42])
result_2 = minimize(objective_function, initial_guess, method='Nelder-Mead',tol=1e-7)
min_coords_2 = result_2.x

print(f'Ideal Parameters')
print(min_coords_1)
print(min_coords_2)


##### Plot Results #####

# Set up colormap
pred_cmap = plt.get_cmap('seismic') 
pred_cNorm  = colors.Normalize(vmin=res_lowerlim, vmax=res_upperlim)
pred_scalarMap = cmx.ScalarMappable(norm=pred_cNorm, cmap=pred_cmap)

std_cmap = plt.get_cmap('viridis') 
std_cNorm  = colors.Normalize(vmin=std_lowerlim, vmax=std_upperlim)
std_scalarMap = cmx.ScalarMappable(norm=std_cNorm, cmap=std_cmap)

# Plot 2D contour plot 
fig, axs = plt.subplots(1,2,figsize=(6.5,4.5))
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.25,top=0.9,wspace=0.3)

pred_ax = axs[0]
pred_ax2 = pred_ax.twinx().twiny()
std_ax = axs[1]
std_xax2 = std_ax.twiny()
std_yax2 = std_ax.twinx()
temp1 = pred_ax.twinx()

pred_ax.set_xlabel('Risetime (s)')
pred_ax.set_ylabel(r'$V_{rupt}$ (km/s)')
pred_ax2.set_xlabel('k-factor')
temp1.set_ylabel('SSF',labelpad=32)
pred_ax2.set_xlim(1,2.3)
pred_ax2.set_ylim(0.37,0.49)
pred_ax.set_xlim(5.4,12.4)
pred_ax.set_ylim(1.2,1.6)
pred_ax2.set_yticks([0.37,0.4,0.43,0.46,0.49])
pred_ax.set_yticks([1.2,1.3, 1.4, 1.5, 1.6])
rise_ticks = np.array([1,1.3,1.6,1.9,2.2])
pred_ax2.set_xticks(rise_ticks)
pred_ax.set_xticks(np.around(rise_ticks*5.4,1))
pred_ax.tick_params(axis='x',rotation=0,labelsize=10)
pred_ax2.contourf(x_mesh, y_mesh, z_pred_mesh, levels=750, cmap='seismic', vmin=res_lowerlim, vmax=res_upperlim)
pred_ax2.contour(x_mesh, y_mesh, z_pred_mesh, levels=[0], linewidths=0.8, linestyles='-', colors='k')
pred_ax2.contour(x_mesh, y_mesh, z_pred_mesh, levels=[0-res_std,0+res_std], linewidths=0.8, linestyles='--', colors='k')
pred_ax2.plot([0],[0],c='k',lw=0.8,label='0-residual line')
pred_ax2.plot([0],[0],c='k',lw=0.8,ls='--',label=r'$1\sigma$ residuals')
pred_cbar = fig.colorbar(pred_scalarMap,ticks=[-30,-15,0,15,30],ax=pred_ax2,pad=0.2,orientation='horizontal',label=r'$\delta_{tPGD}$ (s)')
temp1.tick_params(left=False,right=False,top=False,bottom=False,labelleft=False,
                  labelright=False,labeltop=False,labelbottom=False)

for i in range(len(kfactor)):
    if i == 0:
        pred_ax2.scatter(kfactor[i],ssf[i],facecolors='none',edgecolors='k',lw=0.4,s=40,label='Locations of regression runs')
    else:
        pred_ax2.scatter(kfactor[i],ssf[i],facecolors='none',edgecolors='k',lw=0.4,s=40)
pred_ax2.scatter([1.422,1.742,1.2,2.0],[0.395,0.422,0.41,0.42],c='k',lw=1,s=40,alpha=1,label='Locations of ideal runs')

std_ax.set_xlabel('Risetime (s)')
std_xax2.set_xlabel('k-factor')
std_yax2.set_ylabel(r'$V_{rupt}$ (km/s)')
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
std_xax2.contour(x_mesh, y_mesh, z_std_mesh, levels=[0], linewidths=0.8, linestyles='-', colors='k')
std_xax2.contour(x_mesh, y_mesh, z_std_mesh, levels=[0-res_std,0+res_std], linewidths=0.8, linestyles='--', colors='k')
std_xax2.plot([0],[0],c='k',lw=0.8,label='0-residual line')
std_xax2.plot([0],[0],c='k',lw=0.8,ls='--',label=r'$1\sigma$ residuals')
std_cbar = plt.colorbar(std_scalarMap,ticks=[0,0.001,0.002,0.003,0.004,0.005], ax=std_xax2, orientation='horizontal',pad=0.2,label=r'$\delta_{tPGD}$ std dev')
std_cbar.formatter.set_powerlimits((0, 0))
std_ax.tick_params(labelleft=False)
std_ax.tick_params(axis='x')

pred_ax2.legend(bbox_to_anchor=(0.5,-0.6), loc='upper center')

if exNGNG == True:
    plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/tPGD_regression_2D_exNGNG_m7.8.png',dpi=300)
else:
    plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/tPGD_regression_2D_m7.8.png',dpi=300)
# plt.close()

