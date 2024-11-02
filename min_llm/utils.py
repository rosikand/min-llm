"""
file: utils.py
----------------- 
This file holds various utility functions for min_llm.  
"""

import numpy as np 


def add(a, b):
    """
    Function: add
    Description: adds a and b (mainly here as a test function)
    Args:
        a: int
        b: int
    Returns:
        int
    """
    return a + b


def add_arrays(arr1, arr2):
    """
    Function: add_arrays
    Description: Add two NumPy arrays element-wise with error handling.

    Args:
    arr1 (numpy.ndarray): First input array
    arr2 (numpy.ndarray): Second input array
    
    Returns:
    numpy.ndarray: Element-wise sum of input arrays
    
    Raises:
    ValueError: If arrays have incompatible shapes
    TypeError: If inputs are not NumPy arrays
    """

    # Check if inputs are NumPy arrays
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        raise TypeError("Both inputs must be NumPy arrays")
    
    # Check if arrays have compatible shapes
    if arr1.shape != arr2.shape:
        raise ValueError(f"Arrays have incompatible shapes: {arr1.shape} and {arr2.shape}")
    
    # Perform addition
    return arr1 + arr2
