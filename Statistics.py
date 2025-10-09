import numpy as np
import pandas as pd

def kge(meas,mod):
    """
    Returns:
    kge_score (float): The KGE score.
    """
    # Calculate mean and standard deviation of observed and simulated data
    meas_mean = np.mean(meas)
    mod_mean = np.mean(mod)
    meas_std = np.std(meas)
    mod_std = np.std(mod)

    # Calculate correlation coefficient (r)
    r = np.corrcoef(meas, mod)[0, 1]

    # Calculate bias (beta)
    beta = mod_mean / meas_mean

    # Calculate variability ratio (gamma)
    gamma = mod_std / meas_std

    # Calculate KGE
    kge_score = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    return kge_score

def R2(meas, mod):
    meas = np.array(meas).ravel()
    mod  = np.array(mod).ravel()
    correlation_matrix = np.corrcoef(meas, mod)
    correlation_xy = correlation_matrix[0, 1]
    return correlation_xy**2
