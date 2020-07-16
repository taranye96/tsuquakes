#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:59:05 2020

@author: tnye
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/pgd_compare.csv')
diego_pgd = np.array(df['pgd_diego'])
me_pgd = np.array(df['pgd_me'])

x = [0, 0.4]
y = x

plt.scatter(diego_pgd,me_pgd)
plt.plot(x, y, 'r-')
plt.xlim(0,0.4)
plt.ylim(0,0.4)
plt.xlabel('Diego PGD (m)')
plt.ylabel('My PGD (m)')
plt.title('PGD Comparison')

plt.savefig('/Users/tnye/tsuquakes/plots/pgd_comp.png', dpi=300)
plt.close()
