""" Code to estimate the three different means from curves in the manifold S² """

import sys
import os.path
sys.path.insert(1, '../../FrenetSerretMeanShape')
from frenet_path import *
from trajectory import *
from model_curvatures import *
from estimation_algo_utils import *
from maths_utils import *
from simu_utils import *
from optimization_utils import opti_loc_poly_traj
from generative_model_spherical_curves import *
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

# '----------------------------------------------------------------------------------------- Without Noise ------------------------------------------------------------------------'
#
# """ Parameters of the simulation """
# n_curves = 25
# N = n_curves
# nb_S = 100
# n_MC = 90
# hyperparam = [0.015, 0.01, 0.01]
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.02, 0.1), "bounds_lcurv" : (1e-4, 1), "bounds_ltors" : (1e-9, 1e-3)}
# nb_knots = 30
# param_model = {"nb_basis" : 30, "domain_range" : (0,1)}
# smoothing = {"flag": False, "method": "karcher_mean"}
# sigma_e = 0
#
# """--------------------------------------------------------- Frenet-Serret Mean -----------------------------------------------------------------"""
#
# """ Preprocessing """
# n_resamples = 100
# param_loc_poly_deriv = { "h_min" : 0.1, "h_max" : 0.2, "nb_h" : 20}
# param_loc_poly_TNB = {"h" : 16, "p" : 3, "iflag": [1,1], "ibound" : 0}
# domain_range = (0,1)
# t = np.linspace(0,1,nb_S)
#
# res = Parallel(n_jobs=-1)(delayed(simul_Frame_sphere)(n_curves, nb_S, domain_range, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, {"ind":True,"val":1}, True, sigma_e)
#                             for i in range(n_MC))
#
# array_PopNewFrame_LP = np.empty((n_MC), dtype=object)
# array_PopNewFrame = np.empty((n_MC), dtype=object)
# array_PopTraj = np.empty((n_MC), dtype=object)
# array_kGeod_Extrins = np.empty((n_MC), dtype=object)
# array_meanL = np.empty((n_MC), dtype=object)
# list_echec = []
#
# for k in range(n_MC):
#     array_PopNewFrame_LP[k] = res[k][1]
#     array_PopNewFrame[k] = res[k][2]
#     array_PopTraj[k] = res[k][0]
#     array_kGeod_Extrins[k] = res[k][3]
#     array_meanL[k] = res[k][4]
#     if res[k][5]==False:
#         list_echec.append(k)
#
# print("True mean...")
#
# Mu, X_tab, V_tab = generative_model_spherical_curves(n_curves, 20, nb_S, domain_range)
# X, NewFrame_LP, NewFrame, k_geod_extrins, uccessLocPoly = pre_process_data_sphere(Mu, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True)
# Alpha_new, Alpha, Alpha_Q_LP, Alpha_Q_GS, Alpha_theta_extrins, Alpha_successLocPoly = pre_process_data(NewFrame_LP.data[:,0,:].transpose()/X.L, NewFrame_LP.grid_obs, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True)
# k_geod_theo = [np.dot(np.cross(NewFrame_LP.data[:,0,:].transpose()[i,:],NewFrame_LP.data[:,1,:].transpose()[i,:]), Alpha.derivatives[:,6:9][i,:]) for i in range(n_resamples)]
#
#
# param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.02, 0.1), "bounds_lcurv" : (1e-4, 1), "bounds_ltors" : (1e-9, 1e-3)}
#
# print("Individual estimations...")
#
# domain_range=(0.0,1.0)
#
# array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
#
# for n in range(n_curves):
#
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopNewFrame_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#                                 for i in range(n_MC))
#     for i in range(n_MC):
#         array_SmoothFPIndiv[i,n] = out[i][0]
#         array_resOptIndiv[i,n] = out[i][1]
#         # if array_resOptIndiv[i,n][1]==True:
#         #     array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
#         #     array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
# # print("Mean estimations Frenet Serret with alignment...")
# #
# # array_SmoothPopFP0 = np.empty((n_MC), dtype=object)
# # array_resOpt0 = np.empty((n_MC), dtype=object)
# #
# # out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_PopFP_NewFrame[i], param_model, smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=True, lam=100) for i in range(n_MC))
#
# # for k in range(n_MC):
# #     array_SmoothPopFP0[k] = out[k][0]
# #     array_resOpt0[k] = out[k][1]
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.02, 0.1), "bounds_lcurv" : (1e-4, 1), "bounds_ltors" : (1e-9, 1e-3)}
#
# print("Mean estimations Frenet Serret without alignment...")
#
# array_SmoothPopFP1 = np.empty((n_MC), dtype=object)
# array_resOpt1 = np.empty((n_MC), dtype=object)
#
# # out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_PopFP_NewFrame[i], param_model, smoothing=smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt) for i in range(n_MC))
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopNewFrame_LP[i], domain_range, nb_knots, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True) for i in range(n_MC))
#
# for k in range(n_MC):
#     array_SmoothPopFP1[k] = out[k][0]
#     array_resOpt1[k] = out[k][1]
#
# print("Mean estimations SRVF...")
#
# array_SRVF_mean = np.empty((n_MC), dtype=object)
# array_SRVF_gam = np.empty((n_MC), dtype=object)
#
# array_SRVF = Parallel(n_jobs=-1)(delayed(compute_mean_SRVF)(array_PopTraj[i], t) for i in range(n_MC))
#
# for i in range(n_MC):
#     array_SRVF_mean[i] = array_SRVF[i][0]
#     array_SRVF_gam[i] = array_SRVF[i][1]
#
# print("Mean estimations Arithmetic...")
#
# array_Arithmetic_mean = Parallel(n_jobs=-1)(delayed(compute_mean_Arithmetic)(array_PopTraj[i]) for i in range(n_MC))
#
#
# """ SAVING DATA """
#
# print('Saving the data...')
#
# filename = "SphereCurves_estimation_K_geod"+'_nCalls_'+str(param_bayopt["n_calls"])+'_sigma_e_'+str(sigma_e)+'_n_MC_'+str(n_MC)
# dic = {"N_curves": n_curves, "param_bayopt" : param_bayopt, "param_model" : param_model, "n_MC" : n_MC,
# # "resOpt0" : array_resOpt0, "SmoothPopFP0" : array_SmoothPopFP0,
# "resOpt1" : array_resOpt1, "SmoothPopFP1" : array_SmoothPopFP1, "kGeod_Extrins" : array_kGeod_Extrins,
# "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB,
# "PopNewFrame_LP" : array_PopNewFrame_LP, "PopNewFrame" : array_PopNewFrame, "PopTraj" : array_PopTraj, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam, "Arithmetic_mean" : array_Arithmetic_mean, "mean_L" : array_meanL, "k_geod_theo" : k_geod_theo, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "list_echec" : list_echec}
#
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')
#


'----------------------------------------------------------------------------------------- With Noise ------------------------------------------------------------------------'

""" Parameters of the simulation """
n_curves = 25
N = n_curves
nb_S = 100
n_MC = 90
hyperparam = [0.015, 0.01, 0.01]
param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.02, 0.07), "bounds_lcurv" : (1e-4, 1), "bounds_ltors" : (1e-9, 1e-3)}
nb_knots = 30
param_model = {"nb_basis" : 30, "domain_range" : (0,1)}
smoothing = {"flag": False, "method": "karcher_mean"}
sigma_e = 0.03

"""--------------------------------------------------------- Frenet-Serret Mean -----------------------------------------------------------------"""

""" Preprocessing """
n_resamples = 100
param_loc_poly_deriv = { "h_min" : 0.1, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 8, "p" : 3, "iflag": [1,1], "ibound" : 1}
domain_range = (0,1)
t = np.linspace(0,1,nb_S)

res = Parallel(n_jobs=-1)(delayed(simul_Frame_sphere)(n_curves, nb_S, domain_range, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, {"ind":True,"val":1}, True, sigma_e)
                            for i in range(n_MC))

array_PopNewFrame_LP = np.empty((n_MC), dtype=object)
array_PopNewFrame = np.empty((n_MC), dtype=object)
array_PopTraj = np.empty((n_MC), dtype=object)
array_kGeod_Extrins = np.empty((n_MC), dtype=object)
array_meanL = np.empty((n_MC), dtype=object)
list_echec = []

for k in range(n_MC):
    array_PopNewFrame_LP[k] = res[k][1]
    array_PopNewFrame[k] = res[k][2]
    array_PopTraj[k] = res[k][0]
    array_kGeod_Extrins[k] = res[k][3]
    array_meanL[k] = res[k][4]
    if res[k][5]==False:
        list_echec.append(k)

print("True mean...")

Mu, X_tab, V_tab = generative_model_spherical_curves(n_curves, 20, nb_S, domain_range)
X, NewFrame_LP, NewFrame, k_geod_extrins, successLocPoly = pre_process_data_sphere(Mu, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True)
Alpha_new, Alpha, Alpha_Q_LP, Alpha_Q_GS, Alpha_theta_extrins, Alpha_successLocPoly = pre_process_data(NewFrame_LP.data[:,0,:].transpose()/X.L, NewFrame_LP.grid_obs, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True)
k_geod_theo = [np.dot(np.cross(NewFrame_LP.data[:,0,:].transpose()[i,:],NewFrame_LP.data[:,1,:].transpose()[i,:]), Alpha.derivatives[:,6:9][i,:]) for i in range(n_resamples)]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.02, 0.07), "bounds_lcurv" : (1e-4, 1), "bounds_ltors" : (1e-9, 1e-3)}

print("Individual estimations...")

domain_range=(0.0,1.0)

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopNewFrame_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]

# print("Mean estimations Frenet Serret with alignment...")
#
# array_SmoothPopFP0 = np.empty((n_MC), dtype=object)
# array_resOpt0 = np.empty((n_MC), dtype=object)
#
# out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_PopFP_NewFrame[i], param_model, smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=True, lam=100) for i in range(n_MC))

# for k in range(n_MC):
#     array_SmoothPopFP0[k] = out[k][0]
#     array_resOpt0[k] = out[k][1]
param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.02, 0.07), "bounds_lcurv" : (1e-4, 1), "bounds_ltors" : (1e-9, 1e-3)}

print("Mean estimations Frenet Serret without alignment...")

array_SmoothPopFP1 = np.empty((n_MC), dtype=object)
array_resOpt1 = np.empty((n_MC), dtype=object)

# out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_PopFP_NewFrame[i], param_model, smoothing=smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt) for i in range(n_MC))
out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopNewFrame_LP[i], domain_range, nb_knots, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True) for i in range(n_MC))

for k in range(n_MC):
    array_SmoothPopFP1[k] = out[k][0]
    array_resOpt1[k] = out[k][1]

print("Mean estimations SRVF...")

array_SRVF_mean = np.empty((n_MC), dtype=object)
array_SRVF_gam = np.empty((n_MC), dtype=object)

array_SRVF = Parallel(n_jobs=-1)(delayed(compute_mean_SRVF)(array_PopTraj[i], t) for i in range(n_MC))

for i in range(n_MC):
    array_SRVF_mean[i] = array_SRVF[i][0]
    array_SRVF_gam[i] = array_SRVF[i][1]

print("Mean estimations Arithmetic...")

array_Arithmetic_mean = Parallel(n_jobs=-1)(delayed(compute_mean_Arithmetic)(array_PopTraj[i]) for i in range(n_MC))


""" SAVING DATA """

print('Saving the data...')

filename = "SphereCurves_estimation_K_geod"+'_nCalls_'+str(param_bayopt["n_calls"])+'_sigma_e_'+str(sigma_e)+'_n_MC_'+str(n_MC)
dic = {"N_curves": n_curves, "param_bayopt" : param_bayopt, "param_model" : param_model, "n_MC" : n_MC,
# "resOpt0" : array_resOpt0, "SmoothPopFP0" : array_SmoothPopFP0,
"resOpt1" : array_resOpt1, "SmoothPopFP1" : array_SmoothPopFP1, "kGeod_Extrins" : array_kGeod_Extrins,
"param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB,
"PopNewFrame_LP" : array_PopNewFrame_LP, "PopNewFrame" : array_PopNewFrame, "PopTraj" : array_PopTraj, "SRVF_mean" :array_SRVF_mean,
"SRVF_gam" :array_SRVF_gam, "Arithmetic_mean" : array_Arithmetic_mean, "mean_L" : array_meanL, "k_geod_theo" : k_geod_theo, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "list_echec" : list_echec}


if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')
