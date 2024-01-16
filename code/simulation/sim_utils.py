import numpy as np


def safe_mean(arr):
    # Masking invalid values (out of bounds) with False
    valid_values = ~np.isnan(arr)
    # Calculating the mean of valid values
    mean = np.mean(arr[valid_values])
    return mean

def safe_max(arr):
    # Masking invalid values (out of bounds) with False
    valid_values = ~np.isnan(arr)
    # Calculating the mean of valid values
    mx = np.max(arr[valid_values])
    return mx