#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:02:53 2023

@author: tnye
"""

import numpy as np

# Three data points (x, y)
# data_points = [(1.422, 0.395), (1.742, 0.422), (2.01, 0.45)]
data_points = [(5.4*1.422, 3.25*0.395), (5.4*1.742, 3.25*0.422), (5.4*2.01, 3.25*0.45)]

# Extract x and y values
x_data, y_data = zip(*data_points)

# Use numpy's polyfit to approximate the line
m, b = np.polyfit(x_data, y_data, 1)

# The equation of the line: y = mx + b
equation = f'y = {m:.2f}x + {b:.2f}'

print("Approximated Line Equation:")
print(equation)



x = np.array([5.4*1.422, 5.4*1.742, 5.4*2.01])
y = np.array([3.25*0.395, 3.25*0.422, 3.25*0.45])

#find line of best fit
a, b = np.polyfit(x, y, 1)


plt.figure()
plt.scatter(x,y)
plt.plot(x,a*x+b)