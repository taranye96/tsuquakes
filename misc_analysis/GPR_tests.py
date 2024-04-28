#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:19:23 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from scipy.optimize import root_scalar
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from math import ceil
import gmm_call as gmm

## Default kernel
# kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")

## Example kernel
# kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#     noise_level=1, noise_level_bounds=(1e-5, 1e1))

## Example kernel 2
# kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
#     noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))

kernels = [
            1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 3.4)) + WhiteKernel(noise_level=0.5),
            1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 3.4)) + WhiteKernel(noise_level=1),
            1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 3.4)) + WhiteKernel(noise_level=1e-2),
            1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 3.4)) + WhiteKernel(noise_level=1e-3)
          ]
                                                                                    
labels = [0.5, 1, 0.01, 0.001]

#%%

sm_df = pd.read_csv('/Users/tnye/tsuquakes/gaussian_process/HF_mean_residuals_corrected.csv')

stress = sm_df['stress drop'].values
HF_res = sm_df['HF res'].values

# Define input
oneD_data = np.array(stress).reshape(-1, 1)

# Define prediction grid
oneD_x = np.linspace(0.1,4,100)
oneD_X = oneD_x.reshape(-1, 1)

#%%

fig, ax = plt.subplots(1,1)

mean_list = []

for i, kernel in enumerate(kernels):

    oneD_gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
    
    # Get grid predictions
    oneD_gp.fit(oneD_data, HF_res)
    oneD_pred, oneD_std = oneD_gp.predict(oneD_X, return_std=True)
    hf_res_std = np.std(oneD_pred)
    
    mean_list.append(oneD_pred)
    
    # Plot
    ax.grid(alpha=0.25)
    ax.scatter(stress,HF_res,facecolors='none',edgecolors='k',lw=0.2,s=40,alpha=0.7)
    ax.errorbar(oneD_x, oneD_pred, oneD_std, alpha=0.6, label = f'Noise level = {labels[i]}')
    ax.set_xlim(0,3.5)
    ax.set_ylim(-3,3)
    ax.set_xlabel(r'$\bf{Stress}$ $\bf{Parameter}$ $\bf{(MPa)}$',fontsize=10)
    ax.set_ylabel(r'$\bf{\delta_{HF}}$',fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Define the mean function based on the GPR predictions
    def oneD_objective_function(x):
        y_mean, _ = oneD_gp.predict(np.array([x]).reshape(-1, 1), return_std=True)
        return abs(y_mean[0])

    initial_guess = np.array([1.5])
    x_cross_zero = minimize(oneD_objective_function, initial_guess, method='Nelder-Mead', tol=1e-7).x[0]
    print(x_cross_zero)

ax.legend(loc='upper right')

plt.savefig('/Users/tnye/tsuquakes/plots/misc/GPR_test/noise.png',dpi=300)


#%%

fig, ax = plt.subplots(1,1)
ax.grid(alpha=0.25)
ax.scatter(stress,HF_res,facecolors='none',edgecolors='k',lw=0.2,s=40,alpha=0.7)



oneD_gp = GaussianProcessRegressor(kernel=None,n_restarts_optimizer=9)

# Get grid predictions
oneD_gp.fit(oneD_data, HF_res)
oneD_pred, oneD_std = oneD_gp.predict(oneD_X, return_std=True)
hf_res_std = np.std(oneD_pred)


ax.errorbar(oneD_x, oneD_pred, oneD_std)
ax.set_xlim(0,3.5)
ax.set_ylim(-3,3)
ax.set_xlabel(r'$\bf{Stress}$ $\bf{Parameter}$ $\bf{(MPa)}$',fontsize=10)
ax.set_ylabel(r'$\bf{\delta_{HF}}$',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_title("CK(1.0,'fixed') * RBF(1.0,'fixed')")

# Define the mean function based on the GPR predictions
def oneD_objective_function(x):
    y_mean, _ = oneD_gp.predict(np.array([x]).reshape(-1, 1), return_std=True)
    return abs(y_mean[0])

initial_guess = np.array([1.5])
x_cross_zero = minimize(oneD_objective_function, initial_guess, method='Nelder-Mead', tol=1e-7).x[0]
print(x_cross_zero)

plt.savefig('/Users/tnye/tsuquakes/plots/misc/GPR_test/default.png',dpi=300)


#%%

kernel = ConstantKernel(3.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")

fig, ax = plt.subplots(1,1)
ax.grid(alpha=0.25)
ax.scatter(stress,HF_res,facecolors='none',edgecolors='k',lw=0.2,s=40,alpha=0.7)



oneD_gp = GaussianProcessRegressor(kernel=None,n_restarts_optimizer=9)

# Get grid predictions
oneD_gp.fit(oneD_data, HF_res)
oneD_pred, oneD_std = oneD_gp.predict(oneD_X, return_std=True)
hf_res_std = np.std(oneD_pred)


ax.errorbar(oneD_x, oneD_pred, oneD_std)
ax.set_xlim(0,3.5)
ax.set_ylim(-3,3)
ax.set_xlabel(r'$\bf{Stress}$ $\bf{Parameter}$ $\bf{(MPa)}$',fontsize=10)
ax.set_ylabel(r'$\bf{\delta_{HF}}$',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_title("CK(3.0,'fixed') * RBF(1.0,'fixed')")

# Define the mean function based on the GPR predictions
def oneD_objective_function(x):
    y_mean, _ = oneD_gp.predict(np.array([x]).reshape(-1, 1), return_std=True)
    return abs(y_mean[0])

initial_guess = np.array([1.5])
x_cross_zero = minimize(oneD_objective_function, initial_guess, method='Nelder-Mead', tol=1e-7).x[0]
print(x_cross_zero)

plt.savefig('/Users/tnye/tsuquakes/plots/misc/GPR_test/CK_3.png',dpi=300)


#%%

kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * 1.0 * RBF(length_scale=1, length_scale_bounds='fixed')
# kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds=(1e-3, 3.4))

fig, ax = plt.subplots(1,1)
ax.grid(alpha=0.25)
ax.scatter(stress,HF_res,facecolors='none',edgecolors='k',lw=0.2,s=40,alpha=0.7)

oneD_gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)

# Get grid predictions
oneD_gp.fit(oneD_data, HF_res)
oneD_pred, oneD_std = oneD_gp.predict(oneD_X, return_std=True)
hf_res_std = np.std(oneD_pred)


ax.errorbar(oneD_x, oneD_pred, oneD_std, label="Kernel = CK(1.0, 'fixed') * RBF(1.0, 'fixed')")

ax.set_xlim(0,3.5)
ax.set_ylim(-3,3)
ax.set_xlabel(r'$\bf{Stress}$ $\bf{Parameter}$ $\bf{(MPa)}$',fontsize=10)
ax.set_ylabel(r'$\bf{\delta_{HF}}$',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_title(str("CK(1.0,'fixed') * 1.0 * RBF(1,'fixed')"))

oneD_gp = GaussianProcessRegressor(kernel=kernels[3],n_restarts_optimizer=9)

# Get grid predictions
oneD_gp.fit(oneD_data, HF_res)
oneD_pred, oneD_std = oneD_gp.predict(oneD_X, return_std=True)
hf_res_std = np.std(oneD_pred)

plt.savefig('/Users/tnye/tsuquakes/plots/misc/GPR_test/test_c.png',dpi=300)


#%%
# from plotly import plotly
import GPy
GPy.plotting.change_plotting_library('plotly')
Y = np.array(HF_res).reshape(-1, 1)
X = np.array(stress).reshape(-1, 1)

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = GPy.models.GPRegression(X,Y,kernel)

from IPython.display import display
display(m)

fig = m.plot()
GPy.plotting.show(fig, filename='basic_gp_regression_notebook')





try:
        #===========================================================================
        # Load in your plotting library here and
        # save it under the name plotting_library!
        # This is hooking the library in
        # for the usage in GPy:
        if lib not in supported_libraries:
            raise ValueError("Warning: Plotting library {} not recognized, currently supported libraries are: \n {}".format(lib, ", ".join(supported_libraries)))
        if lib == 'matplotlib':
            import matplotlib
            from .matplot_dep.plot_definitions import MatplotlibPlots
            from .matplot_dep import visualize, mapping_plots, priors_plots, ssgplvm, svig_plots, variational_plots, img_plots
            current_lib[0] = MatplotlibPlots()
        if lib in ['plotly', 'plotly_online']:
            import plotly
            from .plotly_dep.plot_definitions import PlotlyPlotsOnline
            current_lib[0] = PlotlyPlotsOnline(**kwargs)
        if lib == 'plotly_offline':
            import plotly
            from .plotly_dep.plot_definitions import PlotlyPlotsOffline
            current_lib[0] = PlotlyPlotsOffline(**kwargs)
        if lib == 'none':
            current_lib[0] = None
        inject_plotting()
        #===========================================================================
    except (ImportError, NameError):
        config.set('plotting', 'library', 'none')
        raise
        import warnings
        warnings.warn(ImportWarning("You spevified {} in your configuration, but is not available. Install newest version of {} for plotting".format(lib, lib)))


