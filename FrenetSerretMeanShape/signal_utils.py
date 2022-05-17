import numpy as np
from scipy import signal
import sys
sys.path.insert(1, '../../Persistence1D-master/python/')
from persistence1d import *
from sklearn.metrics import mean_squared_error
from scipy import optimize
from scipy import interpolate
from scipy import integrate
from visu_utils import *

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

def maximum_fromMaxPersistence(signal, threshold):
    ExtremaAndPersistence = RunPersistence(signal)
    #~ Keep only those extrema with a persistence larger than threshold.
    Filtered = [t for t in ExtremaAndPersistence if t[1] > threshold]
    # #~ Sort the list of extrema by persistence.
    Sorted = sorted(Filtered, key=lambda ExtremumAndPersistence: ExtremumAndPersistence[0])
    list_ind_maximum = []
    for i, E in enumerate(Sorted):
        if (i % 2 != 0):
            list_ind_maximum.append(E[0])
    return list_ind_maximum

def find_maxs(signal, threshold):
    ExtremaAndPersistence = RunPersistence(signal)
    Sorted = sorted(ExtremaAndPersistence, key=lambda ExtremumAndPersistence: ExtremumAndPersistence[0])
    list_ind_maximum = []
    for i in range(len(Sorted)):
        if (i % 2 != 0):
            dist = signal[Sorted[i][0]] - signal[Sorted[i-1][0]] + signal[Sorted[i][0]] - signal[Sorted[i+1][0]]
            if dist > threshold:
                list_ind_maximum.append(Sorted[i][0])
    return list_ind_maximum


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



""" Functions for Sigma Lognormal approximation of signal """

def compute_mu(t_a, t_b, a_a, a_b):
    return np.log((t_a-t_b)/(np.exp(-a_a)-np.exp(-a_b)))

def compute_t0(t_a, mu, a_a):
    return t_a - np.exp(mu-a_a)

def compute_D(v_a, sigma, mu, a_a):
    return v_a*sigma*np.sqrt(2*np.pi)*np.exp(mu + a_a**2/(2*sigma**2)) - a_a

def compute_a(sigma2, i):
    if i=='inf':
        return 3/2 * sigma2 + np.sqrt(sigma2*(sigma2/4 + 1))
    elif i=='max':
        return sigma2
    elif i=='sup':
        return 3/2 * sigma2 - np.sqrt(sigma2*(sigma2/4 + 1))
    else:
        print('i is not correct.')

def pdf_sigma_lognormal(t0, D, mu, sigma):
    def f(x):
        return D*(np.exp(-(np.log(x-t0) - mu)**2 / (2 * sigma**2)) / ((x-t0) * sigma * np.sqrt(2 * np.pi)))
    return f

def extractLogNormalParameters(tm, t1, t2, v, t):
    param = []

    # alpha = 1, beta = m
    if not np.isnan([tm, t1]).any():
        logr_1m = np.log(v(t1)/v(tm))
        sigma2_1m = -2 - 2*logr_1m - 1/(2*logr_1m)
        a_1 = compute_a(sigma2_1m, 'inf')
        a_m = compute_a(sigma2_1m, 'max')
        mu_1m = compute_mu(t1, tm, a_1, a_m)
        t0_1m = compute_t0(t1, mu_1m, a_1)
        D_1m = compute_D(v(t1), np.sqrt(sigma2_1m), mu_1m, a_1)
        param.append([t0_1m, D_1m, mu_1m, sigma2_1m])

    # alpha = 1, beta = 2
    if not np.isnan([t2, t1]).any():
        logr_12 = np.log(v(t1)/v(t2))
        sigma2_12 = -2 + 2*np.sqrt(1 + logr_12**2)
        a_1 = compute_a(sigma2_12, 'inf')
        a_2 = compute_a(sigma2_12, 'sup')
        mu_12 = compute_mu(t1, t2, a_1, a_2)
        t0_12 = compute_t0(t1, mu_12, a_1)
        D_12 = compute_D(v(t1), np.sqrt(sigma2_12), mu_12, a_1)
        param.append([t0_12, D_12, mu_12, sigma2_12])

    # alpha = m, beta = 2
    if not np.isnan([t2, tm]).any():
        logr_m2 = np.log(v(t2)/v(tm))
        sigma2_m2 = -2 - 2*logr_m2 - 1/(2*logr_m2)
        a_m = compute_a(sigma2_m2, 'max')
        a_2 = compute_a(sigma2_m2, 'sup')
        mu_m2 = compute_mu(tm, t2, a_m, a_2)
        t0_m2 = compute_t0(tm, mu_m2, a_m)
        D_m2 = compute_D(v(tm), np.sqrt(sigma2_m2), mu_m2, a_m)
        param.append([t0_m2, D_m2, mu_m2, sigma2_m2])

    n = len(param)
    # print(param)
    # print([t-param[i][0] for i in range(n)])
    mse = np.ones(n)*np.inf
    v_true = v(t)
    for i in range(n):
        if not np.isnan(param[i]).any():
            v_pred = pdf_sigma_lognormal(param[i][0], param[i][1], param[i][2], np.sqrt(param[i][3]))(t)
            try:
                mse[i] = mean_squared_error(v_true, v_pred)
            except:
                continue

    # print(mse)
    if np.isinf(mse).all():
        return param[0]
    else:
        best_param = np.array(param)[np.argmin(mse)]
        return best_param


def optimizeLogNormalParam(param, v, t):

    def y(theta, t):
        return pdf_sigma_lognormal(theta[0], theta[1], theta[2], np.sqrt(theta[3]))(t)

    def fun(theta):
        return y(theta, t) - v(t)

    theta0 = param
    res = optimize.least_squares(fun, theta0)
    if res.success==False:
        print('optimization of log-normal parameters do not converged.')

    return res.x


def find_relevant_points(v, t): #il faudra peut être ajouter les points min, max au cas ou plusieurs points d'inflections : à voir
    # function to find the time points corresponding to maximum and inflections points
    pts = np.ones(3)*np.nan

    vdot = interpolate.interp1d(t, np.gradient(v(t)))
    vdotdot = interpolate.interp1d(t, np.gradient(vdot(t)))
    t1, t2 = t[np.argmax(vdot(t))], t[np.argmin(vdot(t))]
    res_max = optimize.root_scalar(vdot, bracket=[t1, t2])
    pts[0] = res_max.root

    if np.sign(vdotdot(pts[0]))!=np.sign(vdotdot(t[0])):
        res_inf1 = optimize.root_scalar(vdotdot, bracket=[t[0], pts[0]])
        pts[1] = res_inf1.root

    if np.sign(vdotdot(pts[0]))!=np.sign(vdotdot(t[-1])):
        res_inf2 = optimize.root_scalar(vdotdot, bracket=[pts[0], t[-1]])
        pts[2] = res_inf2.root

    return pts


def approx_stroke_by_lognormal(v, t):
    pts = find_relevant_points(v, t)
    print('relevant points : ', pts)

    param0 = extractLogNormalParameters(pts[0], pts[1], pts[2], v, t)
    print(param0)
    t_ = np.linspace(t[0], pts[2], 100)
    param_opt = optimizeLogNormalParam(param0, v, t_)
    print(param_opt)
    return param_opt, pdf_sigma_lognormal(param_opt[0], param_opt[1], param_opt[2], np.sqrt(param_opt[3]))


def substrac_strokes(v, F):
    def v_r(t):
        return v(t) - sum(f(t) for f in F)
    return v_r


def reconstruct_signal(F):
    n = len(F)
    if n>0:
        def g(t):
            return sum(f(t) for f in F)
    else:
        def g(t):
            return 0*t
    return g


def SNR(v_o, v_r, t):
    f1 = lambda x : v_o(x)**2
    f2 = lambda x : (v_o(x)-v_r(x))**2
    snr = 10*np.log10(integrate.quad(f1, t[0], t[-1])[0]/integrate.quad(f2, t[0], t[-1])[0])
    return snr


def cut_signal_minimum(v, t):
    vdot = interpolate.interp1d(t, np.gradient(v(t)))
    ind_max = find_maxs(v(t), np.max(v(t))/10)
    ind_max = [0] + ind_max + [len(t)-1]
    n = len(ind_max)
    m = int(np.min((n-1,2)))
    print(n, ind_max)
    t_min = []
    for i in range(m):
        t0 = t[ind_max[i]]
        t1 = t[ind_max[i+1]]

        t_min.append(optimize.fminbound(v, t0, t1))

    if len(t_min) >= 2:
        return [t_min[0], t_min[1]]
    else:
        return t_min[0]


def approx_signal_sum_lognormal(v_o, t, threshold):
    P = []
    F = []
    t_bounds = cut_signal_minimum(v_o,t)
    snr = 0
    v_s = substrac_strokes(v_o, F)
    v_r = reconstruct_signal(F)
    while snr<threshold and len(t_bounds)>1:
        ti = np.linspace(t_bounds[0], t_bounds[1], 100)
        print(ti[0], ti[-1])
        pi, vi = approx_stroke_by_lognormal(v_s, ti)
        P.append(pi)
        F.append(vi)
        v_r = reconstruct_signal(F)
        snr = SNR(v_o, v_r, t)
        v_s = substrac_strokes(v_o, F)
        t_bounds = cut_signal_minimum(v_s,t)
        plot_array_2D(t, [v_o(t), v_s(t), v_r(t)], ' ')
    return P, F
