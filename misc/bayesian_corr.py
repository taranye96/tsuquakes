#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:00:28 2021

@author: tnye
"""

from obspy.core import read
import numpy as np
import pandas as pd
from numpy import genfromtxt,where,ones,arange,polyfit,tile,zeros
import pymc3 as pm  
from theano.compile.ops import as_op
import theano.tensor as tt
from theano import shared
import matplotlib.pyplot as plt


#get the data
#pwaves = genfromtxt('/Users/sydneydybing/Downloads/DT2019_Cascadia_Amplitudes.txt')
# What exactly are the colunns here? Amplitude of waves at a specific time?
# My equivalent = peak strain at a specific time

#build the target function, misfit to this is what is being minimized
@as_op(itypes=[tt.dvector, tt.dscalar, tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dvector])
def two_straight_lines(x,m1,m2,xinter,x0,y0):
    '''
    input x coordiantes are in x
    slopes are m1 and m2
    intercept of left hand line is b1 
    intersection of two lines is at xinter
    
    Note that y intercept of second straight line is dependent on b1 and xinter
      and defined entirely by them (so that the lines touch).
    '''
    
    #output vector
    yout = ones(len(x))
    
    #before building the first straight line, calculate the intercept
    b1 = y0 - m1*x0
    
    #build first straight line segment
    yout = m1*x + b1
    
    #find points that are after the intersection and make the second segment
    i=where(x>xinter)[0]
    
    #define second y intercept
    b2 = m1*xinter + b1 - m2*xinter
    
    #make second straight line
    yout[i] = m2*x[i] + b2

    return yout

#build the target function, misfit tot his is what is being minimized
def non_theano_two_straight_lines(x,m1,m2,xinter,x0,y0):
    '''
    input x coordiantes are in x
    slopes are m1 and m2
    intercept of left hand line is b1 
    intersection of two lines is at xinter
    
    Note that y intercept of second straight line is dependent on b1 and xinter
      and defined entirely by them (so that the lines touch).
    '''
    
    #output vector
    yout = ones(len(x))
    
    #before building the first straight line, calculate the intercept
    b1 = y0 - m1*x0
    
    #build first straight line segment
    yout = m1*x + b1
    
    #find points that are after the intersection and make the second segment
    i=where(x>xinter)[0]
    
    #define second y intercept
    b2 = m1*xinter + b1 - m2*xinter
    
    #make second straight line
    yout[i] = m2*x[i] + b2

    return yout

##### Barbour Data Info #####

# Reaad in dataframes
rt_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/melgar_hayes2017.csv')
sd_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/ye2016.csv')

# Obtain origin times form the dfs
rt_datetimes = np.array(rt_df['origin time'])
rt_USGSID = np.array(rt_df['#USGS ID'])
rt_types = np.array(rt_df['type'])
rt_mag = np.array(rt_df['Mw'])
rt_depths = np.array(rt_df['depth(km)'])
sd_dates = np.array(sd_df['Date'])
sd_times = np.array(sd_df['Time'])

# Obtain rise time and stress drops
all_vrupt = np.array(rt_df['rupture vel(km/s)'])
all_apparent_stress = np.array(sd_df['σa(MPa)'])

# Initialize origin lists
rt_origins = np.array([])
sd_origins = np.array([])

# Loop through rise time df
for origin in rt_datetimes:
    short_orig = origin.split('.')[0]
    new_orig = short_orig.split(':')[0] + ':' + short_orig.split(':')[1]
    rt_origins = np.append(rt_origins, new_orig)

# Loop through stress drop df
for i, date in enumerate(sd_dates):
    yyyy = date.split('-')[0]
    mth = date.split('-')[1]
    dd = date.split('-')[2]
    hr = sd_times[i].split(':')[0]
    mm = sd_times[i].split(':')[1]
    
    origin = yyyy + '-' + mth + '-' + dd + 'T' + hr + ':' + mm
    sd_origins = np.append(sd_origins, origin)

# Find common events between both datasets
vrupt = []
apparent_stress = [] 
common_events = []
common_depths = []
common_mag = []
common_IDs = []
color = []

# tsq_ID = ['p000jqvm', 'p000fn2b', 'p000a45f', 'p000ah4q', 'p000hnj4', 'c000f1s0', 'p000fjta']
tsq_ID = ['p000ensm', 'p000hnj4', 'p0007dmb', 'p0006djk']
for i, element in enumerate(rt_origins):
    
    # Only select megathrust events
    if rt_types[i] == "i":
        
        if element in sd_origins:
            
            common_events.append(element.split('T')[0])
            common_depths.append(rt_depths[i])
            common_mag.append(rt_mag[i])
            common_IDs.append(rt_USGSID[i])
            if rt_USGSID[i] in tsq_ID:
                color.append(1)
            else:
                color.append(0)
            
            # Find indexes of rise times and stress drops for common events
            vrupt_ind = i
            sd_ind = np.where(sd_origins == element)[0][0]
            
            # Find rise times and stress drops for common events
            vrupt.append(all_vrupt[vrupt_ind])
            apparent_stress.append(all_apparent_stress[sd_ind])

# Convert form apparent stress to stress drop
stress_drop = []
for stress in apparent_stress:
    sd = 4.3*stress
    stress_drop.append(sd)

log10_stress = np.log10(stress_drop)
log10_vrupt = np.log10(vrupt)

#split into x and y vectors
# xobserved = times[205:1281]
# print(xobserved.shape)

# yobserved = log10_data[205:1281]
# print(yobserved.shape)

# x0 = xobserved[0]
# y0 = yobserved[0]
x0 = log10_stress[0]
y0 = log10_vrupt[0]

# xobserved = times[205:400]
# yobserved = log10_data[205:400]

# in order to pass the x variable into the target function it needs to be 
# converted to a Theano "shared" variable
# theano_xobserved = shared(xobserved)
theano_xobserved = shared(log10_stress)
theano_x0 = shared(x0)
theano_y0 = shared(y0)

# MCMC run parameters, these are good numbers for a "production" run. If you are
# fooling arund these can be lower to iterate faster
Nburn = 1000 # burn in samples that get discarded
Nmcmc = 2000 # bump to at least 5-10k
Nchains = 1

#bounds for the prior distributions
m1_low = 0 ; m1_high = 5 # lowest slope 0, highest 5
m2_low = 0 ; m2_high = 10
b1_low = -1 ; b1_high = 1 # lowest y-intercept -20, highest 0
# xinter_low = 11 ; xinter_high = 13 # location of the line slope change
xinter_low = 1 ; xinter_high = 2 # location of the line slope change

#define the Bayesian model
with pm.Model()as model:
    
    #Use normal distributions as priors
    m1 = pm.Normal('m1', mu=0.5, sigma=1)
    m2 = pm.Normal('m2', mu=-0.1, sigma=5)
    # b1 = pm.Normal('b1', mu=-5, sigma=5)
    xinter = pm.Uniform('xinter', lower=xinter_low, upper=xinter_high)
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)

    #this is the model
    likelihood = pm.Normal('y', mu=two_straight_lines(theano_xobserved,m1,m2,xinter,theano_x0,theano_y0),
                            observed=yobserved,sigma=sigma)
#    likelihood = pm.Normal('y', mu=one_straight_line(xobserved,m1,b1),observed=yobserved,
#                           sigma=sigma)
    
    # NUTS sampler (default) is gradient based and won't work, use metropolis
    step = pm.Metropolis()
    
    #This runs the mcmc sampler
    mcmc = pm.sample(Nmcmc, tune = Nburn,cores = Nchains,step=step)


# done, now is post-processing to get the data out of the sampler

##Unwrap coeficients
m1 = mcmc['m1'].mean() #Youc an also get the uncertainties by getting the std. dev.
# b1 = mcmc['b1'].mean()
m2 = mcmc['m2'].mean()
xinter = mcmc['xinter'].mean()
b1 = y0 - m1*x0
b2 = m1*xinter + b1 - m2*xinter

#make plot to check stuff
xpredicted = arange(xobserved.min(),xobserved.max()+0.1,0.1)
ypredicted=ones(len(xpredicted))
ypredicted = m1*xpredicted +b1
i=where(xpredicted>xinter)[0]
ypredicted[i]=m2*xpredicted[i]+b2

#get one-sigma region (need to obtain a ton of forward models and get stats)
N=len(mcmc.get_values('m1'))

m1_array=mcmc.get_values('m1')
np.savez('/Users/sydneydybing/StrainProject/M6_500km_sel/StrainData_sel/Trimmed/PeakStrains/MCMC_npz/' + quake + '/' + sta + '_m1.npz')
m2_array=mcmc.get_values('m2')
np.savez('/Users/sydneydybing/StrainProject/M6_500km_sel/StrainData_sel/Trimmed/PeakStrains/MCMC_npz/' + quake + '/' + sta + '_m2.npz')
# b1_array=mcmc.get_values('b1')
xinter_array=mcmc.get_values('xinter')

# plt.hist(m1_array)
# plt.show()

yfit = zeros((len(xpredicted),N))
for k in range(N):
    yfit[:,k] = non_theano_two_straight_lines(xpredicted,m1_array[k],m2_array[k],xinter_array[k],x0,y0)

mu = yfit.mean(1)
sig = yfit.std(1)*1.95 #for 95% confidence
mu_plus=mu+sig
mu_minus=mu-sig



#least squares
mls,bls = polyfit(xobserved,yobserved,1)

plt.figure()
plt.plot(xobserved,yobserved,label='observed')
#plt.plot(xpredicted,ypredicted,c='r',label='predicted')
plt.plot(xpredicted,mu,c='r',label='predicted')
plt.plot(xpredicted,xpredicted*mls+bls,c='k',label='lstsq')
plt.legend()
plt.fill_between(xpredicted,mu_plus,mu_minus,color='r',alpha=0.2) #95% confidence interval
plt.xlabel('Time (s) - p-wave at 10s')
plt.ylabel('log(PST)')
# plt.savefig('/Users/sydneydybing/StrainProject/M6_500km_sel/StrainData_sel/Trimmed/PeakStrains/MCMC_figs/' + quake + '/' + sta + '.jpg', format="JPEG", dpi=400)
plt.close()




