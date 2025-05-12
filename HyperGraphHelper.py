"""
HyperGraphHelper

Contains auxiliary code for working with the 
HyperGraph class, the HyperGraphSI class, and the 
topological inequality measures.


2024/06/05 --- ML, SD
"""

import copy
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy import special
from scipy import optimize
from typing import List, Dict, Set, Union, Callable, Tuple

def hpdi(f: ArrayLike, p: float = 0.9) -> ArrayLike:
    '''Calculate the highest posterior density interval along the
       columns of the input array.

       Input
       f - n x m array n observations of m quantities.
       p - returned are intervals that contain probability mass p

       Output
       low_high - 2 x m array containing highest posterior density intervals
                  that contain probability mass p. The first row holds the
                  lower bound, the second the upper bound of the interval.
    '''

    n = f.shape[0]  # number of rows (observations)
    m = f.shape[1]  # number of columns (observed quantities)

    low_high = np.zeros((2, m))

    for i in range(m):
        pn = int(n * p)  # number of samples corresponding to probability mass p

        # samples in increasing order
        f_sorted = np.sort(f[:, i])

        # the boundaries s.t. if the last entry of left_boundaries is the rightmost
        # value at which it is at all possible to contain probability mass p and
        # vice versa for right
        left = f_sorted[:(n - pn)]
        right = f_sorted[pn:]

        # all the intervals left[i] to right[i] contain probability mass p. The
        # one of highest density is the one of shortest length.
        length = right - left
        idx_start = np.argmin(length, axis=0)
        idx_end = idx_start + pn

        low_high[0, i] = f_sorted[idx_start]
        low_high[1, i] = f_sorted[idx_end]

    return low_high

def logbinning(x:ArrayLike, nbins:int=10, base:int=10, normalize:bool=True)->Tuple[ArrayLike]:
    """Bins the values x in nbins bins of exponentially increasing size.

    Input
    x - the input values to be binned
    nbins - the number of bins
    base - the base of the logarithm
    normalize - return counts or probability density
    
    Output
    counts - the binned x values
    centers - the bin centers
    edges - the bin edges
    """

    x = np.asarray(x)

    # to avoid zeros in log
    x = x[x != 0]

    xmax = np.max(x)
    xmin = np.min(x)

    bins = np.logspace(np.log(xmin)/np.log(base), np.log(xmax)/np.log(base), num=nbins+1, base=base)

    counts, edges = np.histogram(x, bins=bins, density=normalize)
    centers = 0.5*(bins[1:] + bins[:-1])

    return counts, centers, edges

def linbinning(x:ArrayLike, nbins:int=10, normalize:bool=True)->Tuple[ArrayLike]:
    """Bins the values x in nbins bins of linearly increasing size.
    
    Input
    x - the values to be binned
    nbins - the number of bins
    normalize - return counts or  probability density

    Output
    counts - the binned x values
    centers - the bin centers
    edges - the bin edges
    """

    x = np.asarray(x)

    xmax = np.max(x)
    xmin = np.min(x)

    bins = np.linspace(xmin, xmax, num=nbins+1)

    counts, edges = np.histogram(x, bins=bins, density=normalize)
    centers = 0.5*(bins[1:] + bins[:-1])

    return counts, centers, edges