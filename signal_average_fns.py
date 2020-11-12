#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:24:50 2020

@author: tnye
"""

###############################################################################
# Module with functions used to calculate various averages of record components.
# These functions are imported and used in synthetic_calc_mpi.py and 
# synthetic_calc.py.   
###############################################################################


def get_geom_avg_3comp(E_record, N_record, Z_record):
    """
    Get the geometric average of the three components of a record.

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
        Z_record(array): Vertical component trace data.
    
    Return:
        geom_avg(array): Record of geometric average.
    """

    import numpy as np

    geom_avg = np.cbrt(E_record * N_record * Z_record)

    return (geom_avg)


def get_geom_avg_2comp(E_record, N_record):
    """
    Get the geometric average of two components of a record (most likely the
    horizontal components).

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
    
    Return:
        geom_avg(array): Record of geometric average.
    """

    import numpy as np

    geom_avg = np.sqrt(E_record * N_record)

    return (geom_avg)


def get_eucl_norm_3comp(E_record, N_record, Z_record):
    """
    Get the euclidean norm of the three components of a record.  This is
    equivalent to calculating the magnitude of a vector. 

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
        Z_record(array): Vertical component trace data.
    
    Return:
        eucl_norm(array): Record of euclidian norm.
    """

    import numpy as np

    eucl_norm = np.sqrt(E_record**2 + N_record**2 + Z_record**2)

    return (eucl_norm)


def get_eucl_norm_2comp(E_record, N_record):
    """
    Get the euclidean norm of two components of a record (most likely the
    horizontal components).  This is equivalent to calculating the magnitude of
    a vector. 

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
    
    Return:
        eucl_norm(array): Full record of euclidian norm.
    """

    import numpy as np

    eucl_norm = np.sqrt(E_record**2 + N_record**2)

    return (eucl_norm)