import numpy as np
from scipy import signal
import sys
sys.path.insert(1, '../../Persistence1D-master/python/')
from persistence1d import *

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

def minimum_fromMaxPersistence(signal, threshold):
    ExtremaAndPersistence = RunPersistence(signal)
    #~ Keep only those extrema with a persistence larger than threshold.
    Filtered = [t for t in ExtremaAndPersistence if t[1] > threshold]
    # #~ Sort the list of extrema by persistence.
    Sorted = sorted(Filtered, key=lambda ExtremumAndPersistence: ExtremumAndPersistence[0])
    list_ind_minimum = []
    for i, E in enumerate(Sorted):
        if (i % 2 == 0):
            list_ind_minimum.append(E[0])
    return list_ind_minimum


def minimum_fromMaxPersistence_nb(signal, number):
    ExtremaAndPersistence = RunPersistence(signal)
    #~ Sort the list of extrema by persistence.
    Sorted = sorted(ExtremaAndPersistence, key=lambda ExtremumAndPersistence: ExtremumAndPersistence[0])
    list_ind_minimum = []
    for k, E in enumerate(Sorted):
        if (k % 2 == 0):
            list_ind_minimum.append(E)
    Sorted_bis = sorted(list_ind_minimum, key=lambda list_ind_minimum: -list_ind_minimum[1])
    list_ind = []
    for k, E in enumerate(Sorted_bis):
        list_ind.append(E[0])
    min_ind = list_ind[:int(np.min([number, len(list_ind)]))]
    return min_ind
