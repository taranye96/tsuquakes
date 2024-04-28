#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:56:26 2023

@author: tnye
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

# Initialize arrays
risetime = []
vrupt = []
stress = []

# List of parameters
rt_vals = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
sf_vals = [0.37, 0.40, 0.43, 0.46, 0.49]
sd_vals = [0.1, 0.5, 1.0, 1.5, 2.0]
residuals = []

# Loop over parameter combinations
for rt_val in rt_vals:
    for sf_val in sf_vals:
        for sd_val in sd_vals:
            risetime.append(rt_val)
            vrupt.append(sf_val)
            stress.append(sd_val)
            
            if rt_val == 1.5 and sf_val == 0.43 and sd_val == 1.0:
                residuals.append(0)
                print(0)
            else:
                residuals.append(np.random.randint(-200, 100) / 200.0)

# # Combine the X, Y, and Z coordinates into a single 2D array
# # with shape (n_samples, 3)
# X = [0.1, 1, 2, 5]
# Y = [5.6, 11.2, 16.8, 16.8]
# Z = [1.6, 1.3, 1.0, 1.0]
# residuals = [-1, 0.1, 1, 1.3]

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






