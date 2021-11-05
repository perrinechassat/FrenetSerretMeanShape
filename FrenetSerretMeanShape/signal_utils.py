import numpy as np
from scipy import signal

def take_subset(data, t_grid, bornes):
    bornes_ind = (int(bornes[0]/t_grid[-1]*data.shape[0]), int(bornes[1]/t_grid[-1]*data.shape[0]))
    return data[bornes_ind[0]:bornes_ind[1],:], t_grid[bornes_ind[0]:bornes_ind[1]]

def ind_bornes(grid, bornes):
    bornes_ind = (int(bornes[0]/grid[-1]*len(grid)), int(bornes[1]/grid[-1]*len(grid)))
    return bornes_ind

def bornes_peaks(grid, curve, nb_peaks):
    n = len(curve)
    peaks, prop = signal.find_peaks(curve, height=0, distance=n*0.0025)
    ind = np.argsort(prop["peak_heights"])[-nb_peaks:]
    bornes = np.sort(grid[peaks[ind]])
    return bornes
