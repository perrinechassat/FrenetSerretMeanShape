import sys
import os.path
sys.path.insert(1, '../../../FrenetSerretMeanShape')
sys.path.insert(1, '../../Sphere')
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
# """ ---------------------------------------------------------------------------ADD VAR From TNB NOISY -------------------------------------------------------------------------------------- """
# print('------------------ ADD VAR From TNB NOISY --------------------')
#
# """ DEFINE DATA """
#
# """ Parameters of the simulation """
# n_curves = 25
# nb_S = 100
# concentration = 10
# K = concentration*np.eye(3)
# n_MC = 90
# hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
# nb_knots = 15
# Noisy_flag = True
# str_Noise = '_Noisy_' #'_Exact_'
#
# """ Definition of reference TNB """
#
# L0 = 5
# init0 = np.eye(3)
# s0 = np.linspace(0,L0,nb_S)
# domain_range = (0.0,L0)
# true_curv0 = lambda s: np.exp(np.sin(s))
# true_tors0 = lambda s: 0.2*s - 0.5
# Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
# Q0.frenet_serret_solve()
# X0 = Q0.data_trajectory
#
#
# """ Definition of population TNB """
#
# start1 = timer()
#
# param_kappa = [1, 2.5]
# param_tau = [1, 2.5]
# sigma_kappa = 0.3
# sigma_tau = 0.3
# param_noise = {"param_kappa" : param_kappa, "param_tau" : param_tau, "sigma_kappa" : sigma_kappa, "sigma_tau" : sigma_tau}
#
# array_TruePopFP, array_TruePopFP_Noisy = np.empty((n_MC), dtype=object), np.empty((n_MC), dtype=object)
#
# res = Parallel(n_jobs=-1)(delayed(simul_populationTNB_additiveVar)(n_curves, L0, s0, K, param_kappa, param_tau, sigma_kappa, sigma_tau, true_curv0, true_tors0, Noisy_flag) for i in range(n_MC))
#
# for k in range(n_MC):
#     array_TruePopFP[k] = res[k][0]
#     array_TruePopFP_Noisy[k] = res[k][1]
#
# """ ESTIMATION """
#
#
# """ Monte carlo """
#
# print("Individual estimations...")
#
# param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-3, 10)}
#
# array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_k_indiv = []
#
# for n in range(n_curves):
#
#     # out = Parallel(n_jobs=-1)(delayed(adaptative_estimation)(array_TruePopFP_Noisy[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#     #                             for i in range(n_MC))
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_TruePopFP_Noisy[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#                                 for i in range(n_MC))
#
#     for i in range(n_MC):
#         array_SmoothFPIndiv[i,n] = out[i][0]
#         # array_k_indiv.append(array_SmoothFPIndiv[i,n].k)
#         array_resOptIndiv[i,n] = out[i][1]
#         if array_resOptIndiv[i,n][1]==True:
#             array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
#             array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
#
# print("Mean estimations...")
#
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-2, 100)}
#
# array_SmoothPopFP = np.empty((n_MC), dtype=object)
# array_SmoothThetaFP = np.empty((n_MC), dtype=object)
# array_resOpt = np.empty((n_MC), dtype=object)
# array_k = []
#
# # out = Parallel(n_jobs=-1)(delayed(adaptative_estimation)(array_TruePopFP_Noisy[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
# #                             for i in range(n_MC))
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_TruePopFP_Noisy[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_MC))
#
# for k in range(n_MC):
#     array_SmoothPopFP[k] = out[k][0]
#     # array_k.append(array_SmoothPopFP[k].k)
#     array_resOpt[k] = out[k][1]
#     if array_resOpt[k][1]==True:
#         array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
#         array_SmoothThetaFP[k].frenet_serret_solve()
#
# duration = timer() - start1
# print('total time', duration)
#
#
# """ SAVING DATA """
#
# print('Saving the data...')
#
# filename = "MultipleEstimationAddVarFromTNB_SingleEstim_SmoothInit_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
# dic = {"N_curves": n_curves, "X0" : X0, "Q0" : Q0, "true_curv0" : true_curv0, "true_tors0" : true_tors0, "L" : L0, "param_bayopt" : param_bayopt, "K" : concentration, "nb_S" : nb_S, "nb_knots" : nb_knots, "n_MC" : n_MC,
# "resOpt" : array_resOpt, "TruePopFP" : array_TruePopFP, "TruePopFP_Noisy" : array_TruePopFP_Noisy, "SmoothPopFP" : array_SmoothPopFP, "SmoothThetaFP" : array_SmoothThetaFP, "param_noise" : param_noise,
# "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv, "array_k_indiv" : array_k_indiv, "array_k" : array_k}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')

# """ ---------------------------------------------------------------------------ADD VAR From TNB EXACT -------------------------------------------------------------------------------------- """
# print('------------------ ADD VAR From TNB EXACT --------------------')
#
# """From TNB"""
#
# """ DEFINE DATA """
#
# """ Parameters of the simulation """
# n_curves = 25
# nb_S = 100
# concentration = 10
# K = concentration*np.eye(3)
# n_MC = 90
# hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
# nb_knots = 15
# Noisy_flag = False
# str_Noise = '_Exact_'
#
# """ Definition of reference TNB """
#
# L0 = 5
# init0 = np.eye(3)
# s0 = np.linspace(0,L0,nb_S)
# domain_range = (0.0,L0)
# true_curv0 = lambda s: np.exp(np.sin(s))
# true_tors0 = lambda s: 0.2*s - 0.5
# Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
# Q0.frenet_serret_solve()
# X0 = Q0.data_trajectory
#
#
# """ Definition of population TNB """
#
# start1 = timer()
#
# param_kappa = [1, 2.5]
# param_tau = [1, 2.5]
# sigma_kappa = 0.3
# sigma_tau = 0.3
# param_noise = {"param_kappa" : param_kappa, "param_tau" : param_tau, "sigma_kappa" : sigma_kappa, "sigma_tau" : sigma_tau}
#
# array_TruePopFP, array_TruePopFP_Noisy = np.empty((n_MC), dtype=object), np.empty((n_MC), dtype=object)
#
# res = Parallel(n_jobs=-1)(delayed(simul_populationTNB_additiveVar)(n_curves, L0, s0, K, param_kappa, param_tau, sigma_kappa, sigma_tau, true_curv0, true_tors0, Noisy_flag) for i in range(n_MC))
#
# for k in range(n_MC):
#     array_TruePopFP[k] = res[k][0]
#     array_TruePopFP_Noisy[k] = res[k][1]
#     # print(array_TruePopFP[k].data.shape, array_TruePopFP_Noisy[k].data.shape)
#
# """ ESTIMATION """
#
# """ Monte carlo """
#
# print("Individual estimations...")
#
# param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-3, 10)}
#
# array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_k_indiv = []
#
# for n in range(n_curves):
#
#     # out = Parallel(n_jobs=-1)(delayed(adaptative_estimation)(array_TruePopFP_Noisy[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#     #                             for i in range(n_MC))
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_TruePopFP_Noisy[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#                                 for i in range(n_MC))
#
#     for i in range(n_MC):
#         array_SmoothFPIndiv[i,n] = out[i][0]
#         # array_k_indiv.append(array_SmoothFPIndiv[i,n].k)
#         array_resOptIndiv[i,n] = out[i][1]
#         if array_resOptIndiv[i,n][1]==True:
#             array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
#             array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
#
# print("Mean estimations...")
#
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-2, 100)}
#
# array_SmoothPopFP = np.empty((n_MC), dtype=object)
# array_SmoothThetaFP = np.empty((n_MC), dtype=object)
# array_resOpt = np.empty((n_MC), dtype=object)
# array_k = []
#
# # out = Parallel(n_jobs=-1)(delayed(adaptative_estimation)(array_TruePopFP_Noisy[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
# #                             for i in range(n_MC))
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_TruePopFP_Noisy[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_MC))
#
# for k in range(n_MC):
#     array_SmoothPopFP[k] = out[k][0]
#     # array_k.append(array_SmoothPopFP[k].k)
#     array_resOpt[k] = out[k][1]
#     if array_resOpt[k][1]==True:
#         array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
#         array_SmoothThetaFP[k].frenet_serret_solve()
#
# duration = timer() - start1
# print('total time', duration)
#
#
# """ SAVING DATA """
#
# print('Saving the data...')
#
# filename = "MultipleEstimationAddVarFromTNB_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
# dic = {"N_curves": n_curves, "X0" : X0, "Q0" : Q0, "true_curv0" : true_curv0, "true_tors0" : true_tors0, "L" : L0, "param_bayopt" : param_bayopt, "K" : concentration, "nb_S" : nb_S, "nb_knots" : nb_knots, "n_MC" : n_MC,
# "resOpt" : array_resOpt, "TruePopFP" : array_TruePopFP, "TruePopFP_Noisy" : array_TruePopFP_Noisy, "SmoothPopFP" : array_SmoothPopFP, "SmoothThetaFP" : array_SmoothThetaFP, "param_noise" : param_noise,
# "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv, "array_k_indiv" : array_k_indiv, "array_k" : array_k}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')


""" ---------------------------------------------------------------------------ADD VAR From X EXACT -------------------------------------------------------------------------------------- """
print('------------------ ADD VAR From X EXACT --------------------')

""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 40
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.14), "bounds_lcurv" : (1e-10, 0.01), "bounds_ltors" : (1e-5, 10)}
Noisy_flag = False

""" Definition of reference TNB and X"""
L0 = 5
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
domain_range = (0.0,5.0)
true_curv0 = lambda s: np.exp(np.sin(s))
true_tors0 = lambda s: 0.2*s - 0.5
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory


""" Definition of population TNB """

start1 = timer()

sigma_e = 0
param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 0.05, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S
t = np.linspace(0, 1, nb_S)

"""Load Data"""
filename = "MultipleEstimationAddVarFromX_SingleEstim__nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_3'

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["PopFP_LP"]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.03, 0.14), "bounds_lcurv" : (1e-10, 0.01), "bounds_ltors" : (1e-5, 10)}

print("Individual estimations...")
#
domain_range=(0.0,1.0)

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationAddVarFromX_SingleEstim__nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')


""" ---------------------------------------------------------------------------ADD VAR From X NOISY -------------------------------------------------------------------------------------- """
print('------------------ ADD VAR From X NOISY --------------------')

""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 40
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.04, 0.14), "bounds_lcurv" : (1e-10, 0.01), "bounds_ltors" : (1e-5, 10)}
Noisy_flag = False

""" Definition of reference TNB and X"""
L0 = 5
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
domain_range = (0.0,5.0)
true_curv0 = lambda s: np.exp(np.sin(s))
true_tors0 = lambda s: 0.2*s - 0.5
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory


""" Definition of population TNB """

start1 = timer()

param_kappa = [1, 2.5]
param_tau = [1, 2.5]
sigma_kappa = 0.2
sigma_tau = 0.08
param_noise = {"param_kappa" : param_kappa, "param_tau" : param_tau, "sigma_kappa" : sigma_kappa, "sigma_tau" : sigma_tau}
sigma_e = 0.05
param_loc_poly_deriv = { "h_min" : 0.1, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 0.2, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S
t = np.linspace(0, 1, nb_S)

"""Load Data"""
filename = "MultipleEstimationAddVarFromX_SingleEstim__nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_3'

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["PopFP_LP"]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.04, 0.14), "bounds_lcurv" : (1e-10, 0.01), "bounds_ltors" : (1e-5, 10)}

print("Individual estimations...")
#
domain_range=(0.0,1.0)

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()

#
duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationAddVarFromX_SingleEstim__nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')



""" --------------------------------------------------------------------------- WARP VAR From TNB EXACT -------------------------------------------------------------------------------------- """
print('------------------ WARP VAR From TNB EXACT --------------------')



"""From TNB"""

""" DEFINE DATA Exact and l=1"""

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 15
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.09), "bounds_lcurv" : (1e-6, 0.1), "bounds_ltors" : (1e-6, 0.1)}
Noisy_flag = False
str_Noise = '_Exact_'

""" Definition of reference TNB """

L0 = 1
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
domain_range = (0.0,L0)
true_curv0 = lambda s: 10*np.abs(np.sin(3*s))+10
true_tors0 = lambda s: -10*np.sin(2*np.pi*s)
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory

""" Definition of population TNB """

start1 = timer()

t = np.linspace(0,1,nb_S)
s0_fun = lambda s: s
l=1
param_shape_warp = np.linspace(-l, l, n_curves)

"""Load Data"""
filename = "MultipleEstimationShapeWarpFromTNB_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_2'

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["TruePopFP_Noisy"]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.03, 0.09), "bounds_lcurv" : (1e-6, 0.1), "bounds_ltors" : (1e-6, 0.1)}

print("Individual estimations...")

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()


duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationShapeWarpFromTNB_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')



""" --------------------------------------------------------------------------- WARP VAR From TNB NOISY -------------------------------------------------------------------------------------- """
print('------------------ WARP VAR From TNB NOISY --------------------')


""" DEFINE DATA Noisy and l=1"""

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12]
nb_knots = 15
param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (1e-6, 0.1), "bounds_ltors" : (1e-6, 0.1)}
Noisy_flag = True
str_Noise = '_Noisy_'

""" Definition of reference TNB """

L0 = 1
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
domain_range = (0.0,L0)
true_curv0 = lambda s: 10*np.abs(np.sin(3*s))+10
true_tors0 = lambda s: -10*np.sin(2*np.pi*s)
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory

""" Definition of population TNB """

start1 = timer()

t = np.linspace(0,1,nb_S)
s0_fun = lambda s: s
l=1
param_shape_warp = np.linspace(-l, l, n_curves)

"""Load Data"""
filename = "MultipleEstimationShapeWarpFromTNB_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_2'

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["TruePopFP_Noisy"]

print("Individual estimations...")

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()


duration = timer() - start1
print('total time', duration)

""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationShapeWarpFromTNB_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')


""" --------------------------------------------------------------------------- WARP VAR From X EXACT -------------------------------------------------------------------------------------- """
print('------------------ WARP VAR From X EXACT --------------------')


""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12]
nb_knots = 15
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}
Noisy_flag = False

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

sigma_e = 0
param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 0.05, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S
t = np.linspace(0, 1, nb_S)
s0_fun = lambda s: s
l = 1
param_shape_warp = np.linspace(-l, l, n_curves)

"""Load Data"""
filename = "MultipleEstimationShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["PopFP_LP"]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}

print("Individual estimations...")
#
domain_range=(0.0,1.0)

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()

#
duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}


if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')


""" --------------------------------------------------------------------------- WARP VAR From X NOISY -------------------------------------------------------------------------------------- """
print('------------------ WARP VAR From X NOISY --------------------')



""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12]
nb_knots = 30
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}
Noisy_flag = False

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
param_loc_poly_deriv = { "h_min" : 0.15, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 0.2, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S
t = np.linspace(0, 1, nb_S)
s0_fun = lambda s: s
l = 1
param_shape_warp = np.linspace(-l, l, n_curves)

"""Load Data"""
filename = "MultipleEstimationShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["PopFP_LP"]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}

print("Individual estimations...")
#
domain_range=(0.0,1.0)

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()

#
duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')




# """Unknown model From X"""
#
# """______________________________________________________________________sig_p=0.02 and sig_e=0__________________________________________________________________________________"""
#
#
# """ DEFINE DATA """
#
# """ Parameters of the simulation """
# n_curves = 25
# nb_S = 100
# sigma_p = 0.02
# sigma_e = 0
# n_MC = 90
# hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
# nb_knots = 15
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
# Noisy_flag = False
#
# """ Definition of reference TNB and X"""
#
# L0 = 5
# phi_ref = (1,0.9,0.8)
# f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
# df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
# ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
# dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])
#
# t = np.linspace(0,L0,nb_S)
#
# def true_curv(t,phi):
#     return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
# def true_torsion(t,phi):
#     return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)
#
#
# """ Definition of population TNB """
#
# start1 = timer()
#
# param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
# param_loc_poly_TNB = {"h" : 0.07, "p" : 3, "iflag": [1,1], "ibound" : 0}
# n_resamples = nb_S
#
# res = Parallel(n_jobs=-1)(delayed(simul_populationCurves_UnknownModel)(f, t, phi_ref, sigma_e, sigma_p, n_curves, n_resamples, param_loc_poly_deriv,
#     param_loc_poly_TNB, {"ind":True,"val":L0}, False) for i in range(n_MC))
#
# array_PopFP_LP = np.empty((n_MC), dtype=object)
# array_PopFP_GS = np.empty((n_MC), dtype=object)
# array_PopTraj = np.empty((n_MC), dtype=object)
# array_ThetaExtrins = np.empty((n_MC), dtype=object)
# array_Phi = np.empty((n_MC), dtype=object)
# array_Xmean = np.empty((n_MC), dtype=object)
# array_Qmean = np.empty((n_MC), dtype=object)
# array_ThetaMeanExtrins = np.empty((n_MC), dtype=object)
# array_ThetaMeanTrue = np.empty((n_MC), dtype=object)
#
# for k in range(n_MC):
#     array_PopFP_LP[k] = res[k][1]
#     array_PopFP_GS[k] = res[k][2]
#     array_PopTraj[k] = res[k][0]
#     array_ThetaExtrins[k] = res[k][3]
#     array_Phi[k] = res[k][4]
#     array_Xmean[k] = res[k][5]
#     array_Qmean[k] = res[k][6]
#     array_ThetaMeanExtrins[k] = res[k][7]
#     array_ThetaMeanTrue[k] = np.array([[true_curv(t_,array_Phi[k].mean(axis=0)) for t_ in t],[true_torsion(t_,array_Phi[k].mean(axis=0)) for t_ in t]])
#
# """ ESTIMATION """
#
# """ Monte carlo """
#
# print("Individual estimations...")
#
# domain_range=(0.0,1.0)
# param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
#
# for n in range(n_curves):
#
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#                                 for i in range(n_MC))
#     for i in range(n_MC):
#         array_SmoothFPIndiv[i,n] = out[i][0]
#         array_resOptIndiv[i,n] = out[i][1]
#         if array_resOptIndiv[i,n][1]==True:
#             array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
#             array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
#
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# print("Mean estimations Frenet Serret...")
#
# array_SmoothPopFP = np.empty((n_MC), dtype=object)
# array_SmoothThetaFP = np.empty((n_MC), dtype=object)
# array_resOpt = np.empty((n_MC), dtype=object)
#
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_MC))
#
# s0 = np.linspace(0,1,nb_S)
# for k in range(n_MC):
#     array_SmoothPopFP[k] = out[k][0]
#     array_resOpt[k] = out[k][1]
#     if array_resOpt[k][1]==True:
#         array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
#         array_SmoothThetaFP[k].frenet_serret_solve()
#
# duration = timer() - start1
# print('total time', duration)
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
# filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')
#
#
# """_____________________________________________________________________________sig_p=0.05 and sig_e=0__________________________________________________________________________________"""
#
#
# """ DEFINE DATA """
#
# """ Parameters of the simulation """
# n_curves = 25
# nb_S = 100
# sigma_p = 0.05
# sigma_e = 0
# n_MC = 90
# hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
# nb_knots = 15
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
# Noisy_flag = False
#
# """ Definition of reference TNB and X"""
#
# L0 = 5
# phi_ref = (1,0.9,0.8)
# f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
# df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
# ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
# dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])
#
# t = np.linspace(0,L0,nb_S)
#
# def true_curv(t,phi):
#     return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
# def true_torsion(t,phi):
#     return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)
#
#
# """ Definition of population TNB """
#
# start1 = timer()
#
# param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
# param_loc_poly_TNB = {"h" : 0.07, "p" : 3, "iflag": [1,1], "ibound" : 0}
# n_resamples = nb_S
#
# res = Parallel(n_jobs=-1)(delayed(simul_populationCurves_UnknownModel)(f, t, phi_ref, sigma_e, sigma_p, n_curves, n_resamples, param_loc_poly_deriv,
#     param_loc_poly_TNB, {"ind":True,"val":L0}, False) for i in range(n_MC))
#
# array_PopFP_LP = np.empty((n_MC), dtype=object)
# array_PopFP_GS = np.empty((n_MC), dtype=object)
# array_PopTraj = np.empty((n_MC), dtype=object)
# array_ThetaExtrins = np.empty((n_MC), dtype=object)
# array_Phi = np.empty((n_MC), dtype=object)
# array_Xmean = np.empty((n_MC), dtype=object)
# array_Qmean = np.empty((n_MC), dtype=object)
# array_ThetaMeanExtrins = np.empty((n_MC), dtype=object)
# array_ThetaMeanTrue = np.empty((n_MC), dtype=object)
#
# for k in range(n_MC):
#     array_PopFP_LP[k] = res[k][1]
#     array_PopFP_GS[k] = res[k][2]
#     array_PopTraj[k] = res[k][0]
#     array_ThetaExtrins[k] = res[k][3]
#     array_Phi[k] = res[k][4]
#     array_Xmean[k] = res[k][5]
#     array_Qmean[k] = res[k][6]
#     array_ThetaMeanExtrins[k] = res[k][7]
#     array_ThetaMeanTrue[k] = np.array([[true_curv(t_,array_Phi[k].mean(axis=0)) for t_ in t],[true_torsion(t_,array_Phi[k].mean(axis=0)) for t_ in t]])
#
# """ ESTIMATION """
#
# """ Monte carlo """
#
# print("Individual estimations...")
#
# domain_range=(0.0,1.0)
# param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
#
# for n in range(n_curves):
#
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#                                 for i in range(n_MC))
#     for i in range(n_MC):
#         array_SmoothFPIndiv[i,n] = out[i][0]
#         array_resOptIndiv[i,n] = out[i][1]
#         if array_resOptIndiv[i,n][1]==True:
#             array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
#             array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
# print("Mean estimations Frenet Serret...")
#
#
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# array_SmoothPopFP = np.empty((n_MC), dtype=object)
# array_SmoothThetaFP = np.empty((n_MC), dtype=object)
# array_resOpt = np.empty((n_MC), dtype=object)
#
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_MC))
#
# s0 = np.linspace(0,1,nb_S)
# for k in range(n_MC):
#     array_SmoothPopFP[k] = out[k][0]
#     array_resOpt[k] = out[k][1]
#     if array_resOpt[k][1]==True:
#         array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
#         array_SmoothThetaFP[k].frenet_serret_solve()
#
# duration = timer() - start1
# print('total time', duration)
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
# filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')
#
#
#
# """_____________________________________________________________________________sig_p=0.02 and sig_e=0.03_________________________________________________________________________________"""
#
# """ DEFINE DATA """
#
# """ Parameters of the simulation """
# n_curves = 25
# nb_S = 100
# sigma_p = 0.02
# sigma_e = 0.03
# n_MC = 90
# hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
# nb_knots = 15
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
# Noisy_flag = False
#
# """ Definition of reference TNB and X"""
#
# L0 = 5
# phi_ref = (1,0.9,0.8)
# f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
# df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
# ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
# dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])
#
# t = np.linspace(0,L0,nb_S)
#
# def true_curv(t,phi):
#     return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
# def true_torsion(t,phi):
#     return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)
#
#
# """ Definition of population TNB """
#
# start1 = timer()
#
# param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
# param_loc_poly_TNB = {"h" : 0.2, "p" : 3, "iflag": [1,1], "ibound" : 0}
# n_resamples = nb_S
#
# res = Parallel(n_jobs=-1)(delayed(simul_populationCurves_UnknownModel)(f, t, phi_ref, sigma_e, sigma_p, n_curves, n_resamples, param_loc_poly_deriv,
#     param_loc_poly_TNB, {"ind":True,"val":L0}, False) for i in range(n_MC))
#
# array_PopFP_LP = np.empty((n_MC), dtype=object)
# array_PopFP_GS = np.empty((n_MC), dtype=object)
# array_PopTraj = np.empty((n_MC), dtype=object)
# array_ThetaExtrins = np.empty((n_MC), dtype=object)
# array_Phi = np.empty((n_MC), dtype=object)
# array_Xmean = np.empty((n_MC), dtype=object)
# array_Qmean = np.empty((n_MC), dtype=object)
# array_ThetaMeanExtrins = np.empty((n_MC), dtype=object)
# array_ThetaMeanTrue = np.empty((n_MC), dtype=object)
#
# for k in range(n_MC):
#     array_PopFP_LP[k] = res[k][1]
#     array_PopFP_GS[k] = res[k][2]
#     array_PopTraj[k] = res[k][0]
#     array_ThetaExtrins[k] = res[k][3]
#     array_Phi[k] = res[k][4]
#     array_Xmean[k] = res[k][5]
#     array_Qmean[k] = res[k][6]
#     array_ThetaMeanExtrins[k] = res[k][7]
#     array_ThetaMeanTrue[k] = np.array([[true_curv(t_,array_Phi[k].mean(axis=0)) for t_ in t],[true_torsion(t_,array_Phi[k].mean(axis=0)) for t_ in t]])
#
# """ ESTIMATION """
#
# """ Monte carlo """
#
# print("Individual estimations...")
#
# domain_range=(0.0,1.0)
# param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
#
# for n in range(n_curves):
#
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#                                 for i in range(n_MC))
#     for i in range(n_MC):
#         array_SmoothFPIndiv[i,n] = out[i][0]
#         array_resOptIndiv[i,n] = out[i][1]
#         if array_resOptIndiv[i,n][1]==True:
#             array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
#             array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
# print("Mean estimations Frenet Serret...")
#
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# array_SmoothPopFP = np.empty((n_MC), dtype=object)
# array_SmoothThetaFP = np.empty((n_MC), dtype=object)
# array_resOpt = np.empty((n_MC), dtype=object)
#
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_MC))
#
# s0 = np.linspace(0,1,nb_S)
# for k in range(n_MC):
#     array_SmoothPopFP[k] = out[k][0]
#     array_resOpt[k] = out[k][1]
#     if array_resOpt[k][1]==True:
#         array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
#         array_SmoothThetaFP[k].frenet_serret_solve()
#
# duration = timer() - start1
# print('total time', duration)
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
# filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')
#
#
#
# """_____________________________________________________________________________sig_p=0.05 and sig_e=0.03_________________________________________________________________________________"""
#
#
# """ DEFINE DATA """
#
# """ Parameters of the simulation """
# n_curves = 25
# nb_S = 100
# sigma_p = 0.05
# sigma_e = 0.03
# n_MC = 90
# hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
# nb_knots = 15
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
# Noisy_flag = False
#
# """ Definition of reference TNB and X"""
#
# L0 = 5
# phi_ref = (1,0.9,0.8)
# f = lambda t,phi: np.array([np.cos(phi[0]*t), np.sin(phi[1]*t), phi[2]*t]).transpose()
# df = lambda t,phi: np.array([-phi[0]*np.sin(phi[0]*t), phi[1]*np.cos(phi[1]*t), phi[2]])
# ddf = lambda t,phi: np.array([-(phi[0]**2)*np.cos(phi[0]*t), -(phi[1]**2)*np.sin(phi[1]*t), 0])
# dddf = lambda t,phi: np.array([np.power(phi[0],3)*np.sin(phi[0]*t), -np.power(phi[1],3)*np.cos(phi[1]*t), 0])
#
# t = np.linspace(0,L0,nb_S)
#
# def true_curv(t,phi):
#     return np.linalg.norm(np.cross(df(t,phi),ddf(t,phi)))/np.power(np.linalg.norm(df(t,phi)),3)
# def true_torsion(t,phi):
#     return np.dot(np.cross(df(t,phi),ddf(t,phi)), dddf(t,phi))/np.power(np.linalg.norm(np.cross(df(t,phi),ddf(t,phi))),2)
#
#
# """ Definition of population TNB """
#
# start1 = timer()
#
# param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
# param_loc_poly_TNB = {"h" : 0.2, "p" : 3, "iflag": [1,1], "ibound" : 0}
# n_resamples = nb_S
#
# res = Parallel(n_jobs=-1)(delayed(simul_populationCurves_UnknownModel)(f, t, phi_ref, sigma_e, sigma_p, n_curves, n_resamples, param_loc_poly_deriv,
#     param_loc_poly_TNB, {"ind":True,"val":L0}, False) for i in range(n_MC))
#
# array_PopFP_LP = np.empty((n_MC), dtype=object)
# array_PopFP_GS = np.empty((n_MC), dtype=object)
# array_PopTraj = np.empty((n_MC), dtype=object)
# array_ThetaExtrins = np.empty((n_MC), dtype=object)
# array_Phi = np.empty((n_MC), dtype=object)
# array_Xmean = np.empty((n_MC), dtype=object)
# array_Qmean = np.empty((n_MC), dtype=object)
# array_ThetaMeanExtrins = np.empty((n_MC), dtype=object)
# array_ThetaMeanTrue = np.empty((n_MC), dtype=object)
#
# for k in range(n_MC):
#     array_PopFP_LP[k] = res[k][1]
#     array_PopFP_GS[k] = res[k][2]
#     array_PopTraj[k] = res[k][0]
#     array_ThetaExtrins[k] = res[k][3]
#     array_Phi[k] = res[k][4]
#     array_Xmean[k] = res[k][5]
#     array_Qmean[k] = res[k][6]
#     array_ThetaMeanExtrins[k] = res[k][7]
#     array_ThetaMeanTrue[k] = np.array([[true_curv(t_,array_Phi[k].mean(axis=0)) for t_ in t],[true_torsion(t_,array_Phi[k].mean(axis=0)) for t_ in t]])
#
# """ ESTIMATION """
#
# """ Monte carlo """
#
# print("Individual estimations...")
#
# domain_range=(0.0,1.0)
# param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
# array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
#
# for n in range(n_curves):
#
#     out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
#                                 for i in range(n_MC))
#     for i in range(n_MC):
#         array_SmoothFPIndiv[i,n] = out[i][0]
#         array_resOptIndiv[i,n] = out[i][1]
#         if array_resOptIndiv[i,n][1]==True:
#             array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
#             array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()
#
# print("Mean estimations Frenet Serret...")
#
# param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.1), "bounds_lcurv" : (1e-9, 1e-3), "bounds_ltors" : (1e-9, 1e-3)}
#
# array_SmoothPopFP = np.empty((n_MC), dtype=object)
# array_SmoothThetaFP = np.empty((n_MC), dtype=object)
# array_resOpt = np.empty((n_MC), dtype=object)
#
# out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=False)
#                             for i in range(n_MC))
#
# s0 = np.linspace(0,1,nb_S)
# for k in range(n_MC):
#     array_SmoothPopFP[k] = out[k][0]
#     array_resOpt[k] = out[k][1]
#     if array_resOpt[k][1]==True:
#         array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
#         array_SmoothThetaFP[k].frenet_serret_solve()
#
# duration = timer() - start1
# print('total time', duration)
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
# filename = "MultipleEstimationUnknownModel_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_e_"+str(sigma_e)+"_sigma_p_"+str(sigma_p)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
# dic = {"N_curves": n_curves, "L" : L0, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots, "n_MC" : n_MC, "resOpt" : array_resOpt, "SmoothPopFP" : array_SmoothPopFP,
# "SmoothThetaFP" : array_SmoothThetaFP, "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "sigma_p" : sigma_p,
# "PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" :array_SRVF_mean,
# "SRVF_gam" :array_SRVF_gam,"Arithmetic_mean" : array_Arithmetic_mean, "array_Phi" : array_Phi, "array_Xmean" : array_Xmean, "array_Qmean" : array_Qmean, "ThetaMeanExtrins" : array_ThetaMeanExtrins,
# "ThetaMeanTrue" : array_ThetaMeanTrue, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}
#
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()
#
# print('END !')




""" --------------------------------------------------------------------------- WARP VAR From X+h EXACT -------------------------------------------------------------------------------------- """
print('------------------ WARP VAR From X+h EXACT --------------------')

""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 15
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}
Noisy_flag = False

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

sigma_e = 0
param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 0.08, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S
t = np.linspace(0, 1, nb_S)
s0_fun = lambda s: s
param_time_warp = np.linspace(-0.1, 0.1, n_curves)
l = 1
param_shape_warp = np.linspace(-l, l, n_curves)

"""Load Data"""
filename = "MultipleEstimationTimeAndShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["PopFP_LP"]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}

print("Individual estimations...")
#
domain_range=(0.0,1.0)

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()

#
duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationTimeAndShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')



""" --------------------------------------------------------------------------- WARP VAR From X+h NOISY -------------------------------------------------------------------------------------- """
print('------------------ WARP VAR From X+h NOISY --------------------')


""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 15
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}
Noisy_flag = False

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
param_loc_poly_deriv = { "h_min" : 0.1, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 20, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S
t = np.linspace(0, 1, nb_S)
s0_fun = lambda s: s
param_time_warp = np.linspace(-0.1, 0.1, n_curves)
l = 1
param_shape_warp = np.linspace(-l, l, n_curves)

"""Load Data"""
filename = "MultipleEstimationTimeAndShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])

fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()

array_PopFP_LP = dic["PopFP_LP"]

param_bayopt = {"n_splits":  10, "n_calls" : 50, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (0.00001, 0.1), "bounds_ltors" : (0.00001, 0.1)}

print("Individual estimations...")
#
domain_range=(0.0,1.0)

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(single_estim_optimizatinon)(array_PopFP_LP[i].frenet_paths[n], domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=False, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()

#
duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationTimeAndShapeWarpFromX_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_l_'+str(l)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])+'_indiv_estim'
dic = {"PopFP_LP" : array_PopFP_LP, "SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv}


if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')