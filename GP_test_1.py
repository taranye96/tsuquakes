#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:43:08 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from glob import glob
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from pyDOE import lhs
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import root_scalar
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as ticker

flatfiles = glob('/Users/tnye/FakeQuakes/simulations/test_suite/stress_drop/*/flatfiles/residuals/*_sm.csv')

# risetime = np.array([])
# vrupt = np.array([])
# stress = np.array([])
# residuals = np.array([])

# for file in flatfiles:
#     df = pd.read_csv(file)
#     try:
#         val = float(file.split('/')[-1].split('_')[0].strip('sd'))
#     except:
#         val = 5.0
#     stress = np.append(stress, [val]*len(df))
#     risetime = np.append(risetime, [11.2]*len(df))
#     vrupt = np.append(vrupt, [1.0]*len(df))
#     residuals = np.append(residuals, df.pga_res.values)

risetime = []
vrupt = []
stress = []
pga_residuals = []
pgv_residuals = []

for file in flatfiles:
    df = pd.read_csv(file)
    try:
        val = float(file.split('/')[-1].split('_')[0].strip('sd'))
    except:
        val = 5.0
    stress.extend([val]*len(df))
    risetime.extend([11.2]*len(df))
    vrupt.extend([1.0]*len(df))
    pga_residuals.extend(df.pga_res.values)
    pgv_residuals.extend(df.pgv_res.values)

IM_mean_res = (np.array(pga_residuals) + np.array(pgv_residuals)) / 2


#%%

###############################################################################
# 1D test with stress drop
###############################################################################

# Define input
data = np.array(stress).reshape(-1, 1)

# Define prediction grid
x = np.linspace(0.1,5,1000)
X = x.reshape(-1, 1)

# Run Gaussian process
gp = GaussianProcessRegressor(kernel=None,n_restarts_optimizer=9)

# Get grid predictions
gp.fit(data, IM_mean_res)
mean_prediction, std_prediction = gp.predict(X, return_std=True)

# Define the mean function based on the GPR predictions
def mean_function(x):
    y_mean, _ = gp.predict(np.array([x]).reshape(-1, 1), return_std=True)
    return y_mean[0]

# Use numerical optimization to find the root (where mean function crosses zero)
# result = root_scalar(mean_function, bracket=[-5, 5])
result = root_scalar(mean_function, bracket=[0.1, 5])
x_cross_zero = result.root


##### Plot Results #####

fig, ax = plt.subplots(1,1)
plt.scatter(stress, IM_mean_res, label='Simulations IM mean')
plt.semilogx(x, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    # mean_prediction - 1.96 * std_prediction,
    # mean_prediction + 1.96 * std_prediction,
    mean_prediction - 3*std_prediction,
    mean_prediction + 3*std_prediction,
    alpha=0.5,
    # label=r"95% confidence interval",
    label=r"3$\sigma$",
)
plt.axhline(0,ls='--')
plt.axvline(x_cross_zero,ls='--')
plt.legend()
plt.xlabel('Stress Drop (MPa)')
plt.ylabel('Mean IM residuals (ln)')



#%%

###############################################################################
# 2D test with stress drop
###############################################################################

# Define input
x_stress = stress
y_stress = stress
data = np.column_stack((x_stress, y_stress))

# Define prediction grid
x = np.linspace(0.1,5,1000)
y = np.linspace(0.1,5,1000)
X = np.column_stack((x, y))
x_mesh, y_mesh = np.meshgrid(x, y)
x_plot = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

# Run Gaussian process
gp = GaussianProcessRegressor(kernel=None,n_restarts_optimizer=9)
gp.fit(data, IM_mean_res)
mean_prediction, std_prediction = gp.predict(x_plot, return_std=True)

# Get grid predictions
z_pred_mesh = mean_prediction.reshape(x_mesh.shape)

# # Define the mean function based on the GPR predictions
# def gp_mean_covariance(x, y):
#     mean = gp.predict(np.array([[x, y]]))
#     return mean[0]

# # Define the objective function to minimize (i.e., the mean prediction - 0)
# def objective_function(coords):
#     x, y = coords
#     mean = gp_mean_covariance(x, y)
#     return abs(mean)

# # Define the objective function to minimize (i.e., the mean prediction - 0)
# def objective_function(params):
#     mean = abs(gp.predict(np.array([params])))
#     return mean

# Define the objective function
def objective_function(xy):
    x, y = xy
    # print(f'x = {x}, y = {y}')
    mean_value, _ = gp.predict(np.array([[x, y]]), return_std=True)
    return np.abs(mean_value)

# Initial guess for the minimum (can be any 2D point)
initial_guess = np.array([1.1,1.1])
minimize(objective_function, initial_guess, method='Nelder-Mead', tol=1e-1)

# Perform the minimization
result = minimize(objective_function, initial_guess, method='Nelder-Mead')
# methods: BFGS, Nelder-Mead
# result = root_scalar(objective_function, args=(0,), method='brentq', bracket=[1,1])

# Extract the minimum and its coordinates
minimum_value = result.fun
min_coords = result.x


##### Plot Results #####

fig, ax = plt.subplots(1,1)
plt.scatter(stress, IM_mean_res, label='Simulations IM mean')
plt.scatter(x_mesh.ravel(), mean_prediction, label="Mean prediction")
# plt.fill_between(
#     X.ravel(),
#     # mean_prediction - 1.96 * std_prediction,
#     # mean_prediction + 1.96 * std_prediction,
#     mean_prediction - std_prediction,
#     mean_prediction + std_prediction,
#     alpha=0.5,
#     # label=r"95% confidence interval",
#     label=r"$\sigma$",
# )
# plt.axhline(0,ls='--')
# plt.axvline(x_cross_zero,ls='--')
plt.legend()
plt.xlabel('Stress Drop (MPa)')
plt.ylabel('Mean IM residuals (ln)')

# Set up colormap
cmap = plt.get_cmap('seismic') 
cNorm  = colors.Normalize(vmin=-2, vmax=2)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

# Plot 2D contour plot 
fig, ax = plt.subplots(1,1)
ax.contourf(x_mesh, y_mesh, z_pred_mesh, levels=500, cmap='seismic', vmin=-2, vmax=2)
cbar = fig.colorbar(scalarMap,ticks=[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
ax.set_xlabel('Stress Drop (MPa)')
ax.set_ylabel('Stress Drop (MPa)')
ax.set_title('2D Gaussian Process Contour Plot')
plt.show()

# Plot 2D contour plot loglog
fig, ax = plt.subplots(1,1)
ax.contourf(x_mesh, y_mesh, z_pred_mesh, levels=500, cmap='seismic', vmin=-2, vmax=2)
cbar = fig.colorbar(scalarMap)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Stress Drop (MPa)')
ax.set_ylabel('Stress Drop (MPa)')
ax.set_title('2D Gaussian Process Contour Plot')
plt.show()

# Plot 3D contour plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh, y_mesh, z_pred_mesh, cmap='seismic', vmin=-2, vmax=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Gaussian Process Surface Plot')
plt.show()


#%%

# data = stress

# # Initialize the kernel for the Gaussian Process
# kernel = 1.0 * RBF(length_scale=1.0)

# # Initialize the Gaussian Process Regressor
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# gp.fit(data, residuals)

# def find_parameter_for_output_zero(gp_regressor, X):
#     # Generate points in the input space for prediction
#     x_pred = np.linspace(np.min(X), np.max(X), 1000)[:, np.newaxis]

#     # Make predictions for the input points
#     y_pred, _ = gp_regressor.predict(x_pred, return_std=True)

#     # Find the parameter value closest to 0
#     closest_to_zero_idx = np.argmin(np.abs(y_pred))

#     return x_pred[closest_to_zero_idx][0]


# parameter_for_zero_output = find_parameter_for_output_zero(gp, data.reshape(-1, 1))

# print("Parameter value for output closest to 0:", parameter_for_zero_output)

data = np.column_stack((risetime, vrupt, stress))

# Initialize the kernel for the Gaussian Process
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0, 1.0], (1e-2, 1e2))

# Initialize the Gaussian Process Regressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

gp.fit(data, residuals)

def residual_prediction(params):
    # Predict the residual at the given parameters
    residual_pred, _ = gp.predict(np.array(params).reshape(1, -1), return_std=True)
    return abs(residual_pred[0])


# Initial guess for the parameters
initial_guess = [1.2, 0.42, 0.8]

# Minimize the residual_prediction function to find the optimal parameters
result = minimize(residual_prediction, initial_guess, method='L-BFGS-B')

# Extract the optimized parameters
optimal_params = result.x
print(optimal_params)







# Create a grid of parameter combinations for plotting
grid_size = 100
X_plot, Y_plot, Z_plot = np.meshgrid(np.linspace(min(X), max(X), grid_size),
                                     np.linspace(min(Y), max(Y), grid_size),
                                     np.linspace(min(Z), max(Z), grid_size))

# Flatten the grid to create input data for prediction
data_plot = np.column_stack((X_plot.ravel(), Y_plot.ravel(), Z_plot.ravel()))

# Predict residuals for the grid points
residuals_pred, _ = gp.predict(data_plot, return_std=True)

# Reshape the predictions back to the grid shape
residuals_pred = residuals_pred.reshape(X_plot.shape)

# Plot the contour plot
plt.contourf(X_plot[:,:,0], Y_plot[:,:,0], residuals_pred[:,:,0], cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Residuals')
plt.title('Gaussian Process Regression')
plt.scatter(optimal_params[0], optimal_params[1], color='red', label='Optimal Params')
plt.legend()
plt.show()



class GaussianProcess:
    """A Gaussian Process class for creating and exploiting  
    a Gaussian Process model"""
    
    def __init__(self, n_restarts, optimizer):
        """Initialize a Gaussian Process model
        
        Input
        ------
        n_restarts: number of restarts of the local optimizer
        optimizer: algorithm of local optimization"""
        
        self.n_restarts = n_restarts
        self.optimizer = optimizer

X_train = np.column_stack((risetime, vrupt))
y_train = residuals
pipe = Pipeline([('scaler', MinMaxScaler()), 
         ('GP', GaussianProcess(n_restarts=10, optimizer='L-BFGS-B'))])
pipe.fit(X_train, y_train)


