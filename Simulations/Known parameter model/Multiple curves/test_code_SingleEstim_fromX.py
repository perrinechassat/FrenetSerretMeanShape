import sys
import os.path
sys.path.insert(1, '../../../FrenetSerretMeanShape')
from frenet_path import *
from trajectory import *
from model_curvatures import *
from estimation_algo_utils import *
from maths_utils import *
from simu_utils import *
from optimization_utils import opti_loc_poly_traj
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import cumtrapz
from skopt import gp_minimize
from skopt.plots import plot_convergence
from pickle import *
import dill as pickle
from timeit import default_timer as timer
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#
# """ --------------------------------------------------------------------ADD VAR From X EXACT -------------------------------------------------------------------------------------- """
# print('------------------ ADD VAR From X EXACT --------------------')
#
# """ DEFINE DATA """
#
# """ Parameters of the simulation """
# n_curves = 25
# nb_S = 100
# concentration = 10
# K = concentration*np.eye(3)
# n_MC = 90
#
# """ Definition of reference TNB and X """
# L0 = 5
# init0 = np.eye(3)
# s0 = np.linspace(0,L0,nb_S)
# domain_range = (0.0,5.0)
# true_curv0 = lambda s: np.exp(np.sin(s))
# true_tors0 = lambda s: 0.2*s - 0.5
# Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
# Q0.frenet_serret_solve()
# X0 = Q0.data_trajectory
#
# """ Definition of population TNB """
#
# param_kappa = [1, 2.5]
# param_tau = [1, 2.5]
# sigma_kappa = 0.2
# sigma_tau = 0.08
# param_noise = {"param_kappa" : param_kappa, "param_tau" : param_tau, "sigma_kappa" : sigma_kappa, "sigma_tau" : sigma_tau}
# sigma_e = 0.05
# param_loc_poly_deriv = { "h_min" : 0.1, "h_max" : 0.2, "nb_h" : 50}
# param_loc_poly_TNB = {"h" : 0.2, "p" : 3, "iflag": [1,1], "ibound" : 1}
# n_resamples = nb_S
# t = np.linspace(0, 1, nb_S)
#
# array_TruePopFP, array_TruePopFP_Noisy = np.empty((n_MC), dtype=object), np.empty((n_MC), dtype=object)
#
# res = Parallel(n_jobs=-1)(delayed(simul_populationTNB_additiveVar)(n_curves, L0, s0, K, param_kappa, param_tau, sigma_kappa, sigma_tau, true_curv0, true_tors0, Noisy_flag) for i in range(n_MC))
#
# for k in range(n_MC):
#     array_TruePopFP[k] = res[k][0]
#     array_TruePopFP_Noisy[k] = res[k][1]
#
# res = Parallel(n_jobs=-1)(delayed(add_noise_X_and_preprocess_MultipleCurves)(array_TruePopFP[i], sigma_e, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=False) for i in range(n_MC))
#
# array_PopFP_LP = np.empty((n_MC), dtype=object)
# array_PopFP_GS = np.empty((n_MC), dtype=object)
#
# for k in range(n_MC):
#     array_PopFP_LP[k] = res[k][1]
#     array_PopFP_GS[k] = res[k][2]
#
# d_GS = np.zeros(n_MC)
# d_LP = np.zeros(n_MC)
#
# for k in range(n_MC):
#
#     """ delta GS """
#     out = Parallel(n_jobs=-1)(delayed(geodesic_dist)(array_PopFP_GS[k].frenet_paths[i].data, array_TruePopFP[k].frenet_paths[i].data) for i in range(n_curves))
#     d_GS[k] = np.mean(out)
#
#     """ delta LP """
#     out = Parallel(n_jobs=-1)(delayed(geodesic_dist)(array_PopFP_LP[k].frenet_paths[i].data, array_TruePopFP[k].frenet_paths[i].data) for i in range(n_curves))
#     d_LP[k] = np.mean(out)
#
# print('Results of simulation on multiple curves from X with sigma='+str(sigma_e)+' and '+str(nb_S)+' points, repeted '+str(n_MC)+' times:')
# print('Delta_GS : ', d_GS.mean(), d_GS.std())
# print('Delta_LP : ', d_LP.mean(), d_LP.std())
#
#


""" --------------------------------------------------------------------WARP VAR From X NOISY -------------------------------------------------------------------------------------- """
print('------------------ WARP VAR From X NOISY --------------------')


""" Definition of reference TNB and X"""
L0 = 1
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
domain_range = (0.0,1.0)
true_curv0 = lambda s: 10*np.abs(np.sin(3*s))+10
true_tors0 = lambda s: -10*np.sin(2*np.pi*s)
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory


""" Definition of population TNB """

start1 = timer()

sigma_e = 0.01
param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 0.2, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S
t = np.linspace(0, 1, nb_S)
s0_fun = lambda s: s
l = 1
param_shape_warp = np.linspace(-l, l, n_curves)

array_TruePopFP, array_TruePopFP_Noisy = np.empty((n_MC), dtype=object), np.empty((n_MC), dtype=object)

res = Parallel(n_jobs=-1)(delayed(simul_populationTNB_ShapeWarping)(n_curves, t, s0_fun, K, param_shape_warp, true_curv0, true_tors0, Noisy_flag) for i in range(n_MC))

for k in range(n_MC):
    array_TruePopFP[k] = res[k][0]
    array_TruePopFP_Noisy[k] = res[k][1]


res = Parallel(n_jobs=-1)(delayed(add_noise_X_and_preprocess_MultipleCurves)(array_TruePopFP[i], sigma_e, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=False) for i in range(n_MC))

array_PopFP_LP = np.empty((n_MC), dtype=object)
array_PopFP_GS = np.empty((n_MC), dtype=object)

for k in range(n_MC):
    array_PopFP_LP[k] = res[k][1]
    array_PopFP_GS[k] = res[k][2]

d_GS = np.zeros(n_MC)
d_LP = np.zeros(n_MC)

for k in range(n_MC):

    """ delta GS """
    out = Parallel(n_jobs=-1)(delayed(geodesic_dist)(array_PopFP_GS[k].frenet_paths[i].data, array_TruePopFP[k].frenet_paths[i].data) for i in range(n_curves))
    d_GS[k] = np.mean(out)

    """ delta LP """
    out = Parallel(n_jobs=-1)(delayed(geodesic_dist)(array_PopFP_LP[k].frenet_paths[i].data, array_TruePopFP[k].frenet_paths[i].data) for i in range(n_curves))
    d_LP[k] = np.mean(out)

print('Results of simulation on multiple curves from X with sigma='+str(sigma_e)+' and '+str(nb_S)+' points, repeted '+str(n_MC)+' times:')
print('Delta_GS : ', d_GS.mean(), d_GS.std())
print('Delta_LP : ', d_LP.mean(), d_LP.std())
