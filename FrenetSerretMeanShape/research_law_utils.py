import numpy as np
from scipy import signal
from signal_utils import *

def cut_min_maxPersist(t, f_sdot, f_kappa, f_tau, threshold, resample_factor=1):
    sdot_signal = f_sdot(t)/np.max(f_sdot(t))
    list_ind_minimum = minimum_fromMaxPersistence(sdot_signal, threshold)
    n_parts = len(list_ind_minimum)-1
    parts_t = np.empty((n_parts), dtype=object)
    parts_sdot = np.empty((n_parts), dtype=object)
    parts_kappa = np.empty((n_parts), dtype=object)
    parts_tau = np.empty((n_parts), dtype=object)
    # t = np.linspace(0,1,resample_factor)
    for i in range(n_parts):
        t_i = t[list_ind_minimum[i]+1:list_ind_minimum[i+1]-1]
        # parts_t[i] = np.linspace(t_i[0], t_i[-1], resample_factor*len(t_i))
        parts_t[i] = np.linspace(t_i[0], t_i[-1], resample_factor)
        parts_sdot[i] = f_sdot(parts_t[i])
        parts_kappa[i] = f_kappa(parts_t[i])
        parts_tau[i] = f_tau(parts_t[i])
    return parts_t, parts_sdot, parts_kappa, parts_tau


def compute_determinant(sdot, kappa, tau, abs=False, normalized=False):
    C = np.power(sdot,6)*np.power(kappa,2)*tau
    if abs==True:
        C = np.abs(C)
    if normalized==True:
        # C = C/np.linalg.norm(C)
        C = C/np.max(C)

    return C
