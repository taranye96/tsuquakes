#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:56:29 2023

@author: tnye
"""

###############################################################################
# Script that computes a stochastic version of the parameter scaling
# relationships.
###############################################################################


def get_stochastic_vrupt(stress):
    
    import numpy as np
    from numpy.random import normal
    
    coefficients = [0.119, 0.357]
    std = 0.092
        
    log_vrupt_mean = coefficients[0]*np.log10(stress) + coefficients[1]
    vrupt = 10**normal(log_vrupt_mean,std)
    
    return(vrupt)

def get_stochastic_risetime(vrupt):
    
    import numpy as np
    from numpy.random import normal
    
    coefficients = [-0.920, 1.270]
    std = 0.222
        
    log_rise_mean = coefficients[0]*np.log10(vrupt) + coefficients[1]
    risetime = 10**normal(log_rise_mean,std)
    
    return(risetime)


# stress = 1
# vrupt = get_stochastic_vrupt(stress)

# vrupt = 1
# risetime = get_stochastic_risetime(vrupt)
