"""
Fairness Measures

Contains the fairness measures from the paper
Zappalà et al. (2024).

ML, SD - 2024/05/11
"""

import copy
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as stats
import scipy.special as special
from typing import List, Dict, Set, Union


def acquisition_fairness(group_sequences:List[ArrayLike], n:List[int], f:float=1.)->float:
    """Calculate the acquisition fairness [Zappalà et al. (2024)] i.e. the average of the 
       ratio of fraction of minority nodes reached to the fraction of minority nodes in the network at
       the time when a fraction f of all nodes has been reached by the information spread. The function
       assumes that nodes of group 1 are the minority nodes.

        Input
        group_sequences - list of arrays of group indices in order of infection assuming.
        n - number of nodes of each group in the network in the form [n0, n1]
        f - fraction of nodes reached by the spread

        Output
        af - acquisition fairness
    """

    # exctract the number of nodes in groups 0 and 1, with 1 coding for the minority group
    n0, n1 = n

    # calculate the total number of nodes
    N = n0 + n1

    # calculate the fraction of minority nodes among all nodes
    total_minority = n1/N

    # calculate the fraction of minority nodes reached when a fraction of all nodes is reached.
    # in simply summing up the group_sequence we use that minority membership is coded as 1.
    informed_minority = np.array([np.sum(group_seq[:np.min([group_seq.shape[0] - 1, int(f*N)])])/np.min([group_seq.shape[0] - 1, int(f*N)]) for group_seq in group_sequences])

    # calculate the acquisition fairness
    af = np.mean(informed_minority/total_minority)

    return af

def diffusion_fairness(time_sequences_majority:List[ArrayLike], time_sequences_minority:List[ArrayLike], N:int, f:float=1.)->float:
    """Diffusion fairness [Zappalà et al. (2024)] is defined as the quotient of the average
       duration of spread when seeds are in the minority group and when seeds are in the majority group.
       The function assumes that nodes of group 1 are the minority nodes.
       
       Input
       time_sequences_majority - list of arrays of infection times when starting from majority nodes.
       time_sequences_minority - list of arrays of infection times when starting from minority nodes.
       N - number of nodes in the network
       f - fraction of nodes reached by the spread

       Output
       df - diffusion fairness
    """
    
    # calculate the time at which a fraction f of nodes is reached when seeding at minority or majority respectively.
    delta_t_majority = np.array([time_seq[np.min([time_seq.shape[0] - 1, int(f * N)])] for time_seq in time_sequences_majority])
    delta_t_minority = np.array([time_seq[np.min([time_seq.shape[0] - 1, int(f * N)])] for time_seq in time_sequences_minority])

    # calculate the diffusion fairness
    df = np.mean(delta_t_majority)/np.mean(delta_t_minority)
    
    return df