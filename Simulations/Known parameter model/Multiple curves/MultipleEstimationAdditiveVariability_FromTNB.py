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



""" ---------------------------------------------------------------------------ADD VAR From TNB NOISY -------------------------------------------------------------------------------------- """
print('------------------ ADD VAR From TNB NOISY --------------------')

""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 15
Noisy_flag = True
str_Noise = '_Noisy_' #'_Exact_'

""" Definition of reference TNB """

L0 = 5
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
domain_range = (0.0,L0)
true_curv0 = lambda s: np.exp(np.sin(s))
true_tors0 = lambda s: 0.2*s - 0.5
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory


""" Definition of population TNB """

start1 = timer()

param_kappa = [1, 2.5]
param_tau = [1, 2.5]
sigma_kappa = 0.3
sigma_tau = 0.3
param_noise = {"param_kappa" : param_kappa, "param_tau" : param_tau, "sigma_kappa" : sigma_kappa, "sigma_tau" : sigma_tau}

array_TruePopFP, array_TruePopFP_Noisy = np.empty((n_MC), dtype=object), np.empty((n_MC), dtype=object)

res = Parallel(n_jobs=-1)(delayed(simul_populationTNB_additiveVar)(n_curves, L0, s0, K, param_kappa, param_tau, sigma_kappa, sigma_tau, true_curv0, true_tors0, Noisy_flag) for i in range(n_MC))

for k in range(n_MC):
    array_TruePopFP[k] = res[k][0]
    array_TruePopFP_Noisy[k] = res[k][1]

""" ESTIMATION """


""" Monte carlo """

print("Individual estimations...")

param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-3, 10)}

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_k_indiv = []

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_TruePopFP_Noisy[i].frenet_paths[n], param_model, smoothing, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=False)
                                for i in range(n_MC))

    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()


print("Mean estimations...")

param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-2, 100)}

array_SmoothPopFP = np.empty((n_MC), dtype=object)
array_SmoothThetaFP = np.empty((n_MC), dtype=object)
array_resOpt = np.empty((n_MC), dtype=object)
array_k = []

out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_TruePopFP_Noisy[i], param_model, smoothing, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=False)
                            for i in range(n_MC))

for k in range(n_MC):
    array_SmoothPopFP[k] = out[k][0]
    array_resOpt[k] = out[k][1]
    if array_resOpt[k][1]==True:
        array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
        array_SmoothThetaFP[k].frenet_serret_solve()

duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationAddVarFromTNB_SingleEstim_SmoothInit_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
dic = {"N_curves": n_curves, "X0" : X0, "Q0" : Q0, "true_curv0" : true_curv0, "true_tors0" : true_tors0, "L" : L0, "param_bayopt" : param_bayopt, "K" : concentration, "nb_S" : nb_S, "nb_knots" : nb_knots, "n_MC" : n_MC,
"resOpt" : array_resOpt, "TruePopFP" : array_TruePopFP, "TruePopFP_Noisy" : array_TruePopFP_Noisy, "SmoothPopFP" : array_SmoothPopFP, "SmoothThetaFP" : array_SmoothThetaFP, "param_noise" : param_noise,
"SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv, "array_k_indiv" : array_k_indiv, "array_k" : array_k}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')

""" ---------------------------------------------------------------------------ADD VAR From TNB EXACT -------------------------------------------------------------------------------------- """
print('------------------ ADD VAR From TNB EXACT --------------------')

"""From TNB"""

""" DEFINE DATA """

""" Parameters of the simulation """
n_curves = 25
nb_S = 100
concentration = 10
K = concentration*np.eye(3)
n_MC = 90
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 15
Noisy_flag = False
str_Noise = '_Exact_'

""" Definition of reference TNB """

L0 = 5
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
domain_range = (0.0,L0)
true_curv0 = lambda s: np.exp(np.sin(s))
true_tors0 = lambda s: 0.2*s - 0.5
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory


""" Definition of population TNB """

start1 = timer()

param_kappa = [1, 2.5]
param_tau = [1, 2.5]
sigma_kappa = 0.3
sigma_tau = 0.3
param_noise = {"param_kappa" : param_kappa, "param_tau" : param_tau, "sigma_kappa" : sigma_kappa, "sigma_tau" : sigma_tau}

array_TruePopFP, array_TruePopFP_Noisy = np.empty((n_MC), dtype=object), np.empty((n_MC), dtype=object)

res = Parallel(n_jobs=-1)(delayed(simul_populationTNB_additiveVar)(n_curves, L0, s0, K, param_kappa, param_tau, sigma_kappa, sigma_tau, true_curv0, true_tors0, Noisy_flag) for i in range(n_MC))

for k in range(n_MC):
    array_TruePopFP[k] = res[k][0]
    array_TruePopFP_Noisy[k] = res[k][1]
    # print(array_TruePopFP[k].data.shape, array_TruePopFP_Noisy[k].data.shape)

""" ESTIMATION """

""" Monte carlo """

print("Individual estimations...")

param_bayopt = {"n_splits":  10, "n_calls" : 30, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-3, 10)}
param_model = {"nb_basis" : nb_knots, "domain_range": (0.0,L0)}
smoothing = {"flag":False, "method":"karcher_mean"}

array_SmoothFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv = np.empty((n_MC, n_curves), dtype=object)
array_k_indiv = []

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_TruePopFP_Noisy[i].frenet_paths[n], param_model, smoothing, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=False)
                                for i in range(n_MC))

    for i in range(n_MC):
        array_SmoothFPIndiv[i,n] = out[i][0]
        array_resOptIndiv[i,n] = out[i][1]
        if array_resOptIndiv[i,n][1]==True:
            array_SmoothThetaFPIndiv[i,n] = FrenetPath(array_SmoothFPIndiv[i,n].grid_obs, array_SmoothFPIndiv[i,n].grid_obs, init=array_SmoothFPIndiv[i,n].data[:,:,0], curv=array_SmoothFPIndiv[i,n].curv, tors=array_SmoothFPIndiv[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv[i,n].frenet_serret_solve()


print("Mean estimations...")

param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.2, 0.6), "bounds_lcurv" : (10e-3, 10), "bounds_ltors" : (10e-2, 100)}
param_model = {"nb_basis" : nb_knots, "domain_range": (0.0,L0)}
smoothing = {"flag": False, "method": "karcher_mean"}

array_SmoothPopFP = np.empty((n_MC), dtype=object)
array_SmoothThetaFP = np.empty((n_MC), dtype=object)
array_resOpt = np.empty((n_MC), dtype=object)
array_k = []

out = Parallel(n_jobs=-1)(delayed(global_estimation)(array_TruePopFP_Noisy[i], param_model, smoothing, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=False)
                            for i in range(n_MC))

for k in range(n_MC):
    array_SmoothPopFP[k] = out[k][0]
    array_resOpt[k] = out[k][1]
    if array_resOpt[k][1]==True:
        array_SmoothThetaFP[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP[k]), curv=array_SmoothPopFP[k].mean_curv, tors=array_SmoothPopFP[k].mean_tors, dim=3)
        array_SmoothThetaFP[k].frenet_serret_solve()

duration = timer() - start1
print('total time', duration)


""" SAVING DATA """

print('Saving the data...')

filename = "MultipleEstimationAddVarFromTNB_SingleEstim_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+str_Noise+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])
dic = {"N_curves": n_curves, "X0" : X0, "Q0" : Q0, "true_curv0" : true_curv0, "true_tors0" : true_tors0, "L" : L0, "param_bayopt" : param_bayopt, "K" : concentration, "nb_S" : nb_S, "nb_knots" : nb_knots, "n_MC" : n_MC,
"resOpt" : array_resOpt, "TruePopFP" : array_TruePopFP, "TruePopFP_Noisy" : array_TruePopFP_Noisy, "SmoothPopFP" : array_SmoothPopFP, "SmoothThetaFP" : array_SmoothThetaFP, "param_noise" : param_noise,
"SmoothFPIndiv" : array_SmoothFPIndiv, "resOptIndiv" : array_resOptIndiv, "SmoothThetaFPIndiv" : array_SmoothThetaFPIndiv, "array_k_indiv" : array_k_indiv, "array_k" : array_k}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')
