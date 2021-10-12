import sys
import os.path
sys.path.insert(1, './FrenetSerretMeanShape')
from frenet_path import *
from trajectory import *
from model_curvatures import *
from estimation_algo_utils_2 import *
from maths_utils import *
from simu_utils import *
from visu_utils import *
from optimization_utils import opti_loc_poly_traj
import numpy as np
from pickle import *
import dill as pickle
from timeit import default_timer as timer
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


""" Define parameter of the simulation """

# number of curves
n_curves = 10
# number of discretization points
nb_S = 100
# parameter to add noise to the initial Frenet paths
concentration = 10
K = concentration*np.eye(3)
# number of iteration of MonteCarlo
n_MC = 5
# hyperparameters fixed to run the algorithm without optimization.
hyperparam = [0.03, 0.1, 0.1]
# parameters of the bayesian optimisation: number of splits for cross validation, number of calls in bayesian optimization, bounds for search of optimal parameters for h, lambda1 and lambda2.
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.04, 0.08), "bounds_lcurv" : (1e-6, 0.1), "bounds_ltors" : (1e-6, 0.1)}
# False if we want to do the simulation with exact values of the Frenet paths, True if we want noisy values
Noisy_flag = False
# definition of the reference parameters curv0, tors0, frenet path Q0 and mean curve X0
L0 = 1
init0 = np.eye(3)
s0 = np.linspace(0,L0,nb_S)
true_curv0 = lambda s: 10*np.abs(np.sin(3*s))+10
true_tors0 = lambda s: -10*np.sin(2*np.pi*s)
Q0 = FrenetPath(s0, s0, init=init0, curv=true_curv0, tors=true_tors0, dim=3)
Q0.frenet_serret_solve()
X0 = Q0.data_trajectory
# parameters of the model for smoothing spline
param_model = {"nb_basis" : 15, "domain_range": (0.0,L0)}
# parameters for smoothing the Frenet paths (keep to False for now)
smoothing = {"flag": False, "method": "karcher_mean"}
# parameters for shape warping functions
param_shape_warp = np.linspace(-1, 1, n_curves)
t = np.linspace(0, 1, nb_S)
# Parameters for preprocessing and estimation of the X values
sigma_e = 0
param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 0.05, "p" : 3, "iflag": [1,1], "ibound" : 0}
n_resamples = nb_S


""" Simulation of a corresponding population of Frenet paths """
print("Generation of the population...")

array_TruePopFP, array_TruePopFP_Noisy = np.empty((n_MC), dtype=object), np.empty((n_MC), dtype=object)

res = Parallel(n_jobs=-1)(delayed(simul_populationTNB_ShapeWarping_TNB)(n_curves, s0, K, param_shape_warp, true_curv0, true_tors0, Noisy_flag) for i in range(n_MC))

for k in range(n_MC):
    array_TruePopFP[k] = res[k][0]
    array_TruePopFP_Noisy[k] = res[k][1]


""" Individual estimations of the parameters from TNB data """
print("Individual estimations from TNB...")

array_SmoothFPIndiv_TNB = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv_TNB = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv_TNB = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(global_estimation_2)(array_TruePopFP_Noisy[i].frenet_paths[n], param_model, smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv_TNB[i,n] = out[i][0]
        array_resOptIndiv_TNB[i,n] = out[i][1]
        if array_resOptIndiv_TNB[i,n][1]==True:
            array_SmoothThetaFPIndiv_TNB[i,n] = FrenetPath(array_SmoothFPIndiv_TNB[i,n].grid_obs, array_SmoothFPIndiv_TNB[i,n].grid_obs, init=array_SmoothFPIndiv_TNB[i,n].data[:,:,0], curv=array_SmoothFPIndiv_TNB[i,n].curv, tors=array_SmoothFPIndiv_TNB[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv_TNB[i,n].frenet_serret_solve()


""" Estimations of the mean parameters from TNB data """
print("Estimations of mean parameters from TNB...")

array_SmoothPopFP_TNB = np.empty((n_MC), dtype=object)
array_SmoothThetaFP_TNB = np.empty((n_MC), dtype=object)
array_resOpt_TNB = np.empty((n_MC), dtype=object)

out = Parallel(n_jobs=-1)(delayed(global_estimation_2)(array_TruePopFP_Noisy[i], param_model, smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=True, lam=100)
                            for i in range(n_MC))

for k in range(n_MC):
    array_SmoothPopFP_TNB[k] = out[k][0]
    array_resOpt_TNB[k] = out[k][1]
    if array_resOpt_TNB[k][1]==True:
        array_SmoothThetaFP_TNB[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP_TNB[k]), curv=array_SmoothPopFP_TNB[k].mean_curv, tors=array_SmoothPopFP_TNB[k].mean_tors, dim=3)
        array_SmoothThetaFP_TNB[k].frenet_serret_solve()


""" Preprocessing and estimation of the Frenet paths from X data """
print("Preprocessing and estimation of TNB from X data...")

res = Parallel(n_jobs=-1)(delayed(add_noise_X_and_preprocess_MultipleCurves)(array_TruePopFP[i], sigma_e, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":L0}, locpolyTNB_local=False) for i in range(n_MC))

array_PopFP_LP = np.empty((n_MC), dtype=object)
array_PopFP_GS = np.empty((n_MC), dtype=object)
array_PopTraj = np.empty((n_MC), dtype=object)
array_ThetaExtrins = np.empty((n_MC), dtype=object)

for k in range(n_MC):
    array_PopFP_LP[k] = res[k][1]
    array_PopFP_GS[k] = res[k][2]
    array_PopTraj[k] = res[k][0]
    array_ThetaExtrins[k] = res[k][3]


""" Individual estimations of the parameters from X data """
print("Individual estimations from X...")

array_SmoothFPIndiv_X = np.empty((n_MC, n_curves), dtype=object)
array_resOptIndiv_X = np.empty((n_MC, n_curves), dtype=object)
array_SmoothThetaFPIndiv_X = np.empty((n_MC, n_curves), dtype=object)

for n in range(n_curves):

    out = Parallel(n_jobs=-1)(delayed(global_estimation_2)(array_PopFP_LP[i].frenet_paths[n], param_model, smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=False)
                                for i in range(n_MC))
    for i in range(n_MC):
        array_SmoothFPIndiv_X[i,n] = out[i][0]
        array_resOptIndiv_X[i,n] = out[i][1]
        if array_resOptIndiv_X[i,n][1]==True:
            array_SmoothThetaFPIndiv_X[i,n] = FrenetPath(array_SmoothFPIndiv_X[i,n].grid_obs, array_SmoothFPIndiv_X[i,n].grid_obs, init=array_SmoothFPIndiv_X[i,n].data[:,:,0], curv=array_SmoothFPIndiv_X[i,n].curv, tors=array_SmoothFPIndiv_X[i,n].tors, dim=3)
            array_SmoothThetaFPIndiv_X[i,n].frenet_serret_solve()


""" Estimations of the mean parameters from X data """
print("Estimations of mean parameters from X...")

array_SmoothPopFP_X = np.empty((n_MC), dtype=object)
array_SmoothThetaFP_X = np.empty((n_MC), dtype=object)
array_resOpt_X = np.empty((n_MC), dtype=object)

out = Parallel(n_jobs=-1)(delayed(global_estimation_2)(array_PopFP_LP[i],  param_model, smoothing, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, alignment=True, lam=100)
                            for i in range(n_MC))

for k in range(n_MC):
    array_SmoothPopFP_X[k] = out[k][0]
    array_resOpt_X[k] = out[k][1]
    if array_resOpt_X[k][1]==True:
        array_SmoothThetaFP_X[k] = FrenetPath(s0, s0, init=mean_Q0(array_SmoothPopFP_X[k]), curv=array_SmoothPopFP_X[k].mean_curv, tors=array_SmoothPopFP_X[k].mean_tors, dim=3)
        array_SmoothThetaFP_X[k].frenet_serret_solve()


""" Estimations of the mean parameters from X data with SRVF method """
print("Estimations of mean shapes with SRVF method...")
array_SRVF_mean = np.empty((n_MC), dtype=object)
array_SRVF_gam = np.empty((n_MC), dtype=object)

array_SRVF = Parallel(n_jobs=-1)(delayed(compute_mean_SRVF)(array_PopTraj[i], t) for i in range(n_MC))

for i in range(n_MC):
    array_SRVF_mean[i] = array_SRVF[i][0]
    array_SRVF_gam[i] = array_SRVF[i][1]


""" Estimations of the mean parameters from X data with arithmetic method """
print("Estimations of mean shapes with arithmetic method...")

array_Arithmetic_mean = Parallel(n_jobs=-1)(delayed(compute_mean_Arithmetic)(array_PopTraj[i]) for i in range(n_MC))


""" Save results """
print('Saving the results...')

filename = "Demo_Simulation_Results_nbS_"+str(nb_S)+"N_curves"+str(n_curves)+"_sigma_"+str(sigma_e)+"_K_"+str(concentration)+'_nMC_'+str(n_MC)+'_nCalls_'+str(param_bayopt["n_calls"])

dic = {"N_curves": n_curves, "X0" : X0, "Q0" : Q0, "true_curv0" : true_curv0, "true_tors0" : true_tors0, "L" : L0, "param_bayopt" : param_bayopt, "K" : concentration, "nb_S" : nb_S,
"n_MC" : n_MC, "param_shape_warp" : param_shape_warp, "param_model" : param_model, "smoothing" : smoothing, "param_loc_poly_deriv" : param_loc_poly_deriv,
"param_loc_poly_TNB" : param_loc_poly_TNB, "sigma" : sigma_e, "TruePopFP" : array_TruePopFP, "TruePopFP_Noisy" : array_TruePopFP_Noisy,
"PopFP_LP" : array_PopFP_LP, "PopFP_GS" : array_PopFP_GS, "PopTraj" : array_PopTraj, "ThetaExtrins" : array_ThetaExtrins,
"SmoothFPIndiv_TNB" : array_SmoothFPIndiv_TNB, "resOptIndiv_TNB" : array_resOptIndiv_TNB, "SmoothThetaFPIndiv_TNB" : array_SmoothThetaFPIndiv_TNB,
"resOpt_TNB" : array_resOpt_TNB, "SmoothPopFP_TNB" : array_SmoothPopFP_TNB, "SmoothThetaFP_TNB" : array_SmoothThetaFP_TNB,
"SmoothFPIndiv_X" : array_SmoothFPIndiv_X, "resOptIndiv_X" : array_resOptIndiv_X, "SmoothThetaFPIndiv_X" : array_SmoothThetaFPIndiv_X,
"resOpt_X" : array_resOpt_X, "SmoothPopFP_X" : array_SmoothPopFP_X, "SmoothThetaFP_X" : array_SmoothThetaFP_X,
"SRVF_mean" :array_SRVF_mean, "SRVF_gam" : array_SRVF_gam, "Arithmetic_mean" : array_Arithmetic_mean}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

""" Plot results """
print('Plot the results...')

true_kappa = true_curv0(s0)
true_tau = true_tors0(s0)

# From TNB
list_kappa = []
list_tau = []
for k in range(n_MC):
    if array_resOpt_TNB[k][1]==True:
        list_kappa.append(array_SmoothThetaFP_TNB[k].curv(s0))
        list_tau.append(array_SmoothThetaFP_TNB[k].tors(s0))
plot_curvatures_grey_bis(s0, list_kappa, list_tau, true_kappa[None,:], true_tau[None,:], ["True Mean"], "FS Mean", title="Results of estimation from TNB")

# From X
list_kappa = []
list_tau = []
for k in range(n_MC):
    if array_resOpt_X[k][1]==True:
        list_kappa.append(array_SmoothThetaFP_X[k].curv(s0))
        list_tau.append(array_SmoothThetaFP_X[k].tors(s0))
plot_curvatures_grey_bis(s0, list_kappa, list_tau, true_kappa[None,:], true_tau[None,:], ["True Mean"], "FS Mean", title="Results of estimation from X")

# Mean Shapes
features_FS = []
features_SRVF = []
features_Arithm = []
for k in range(n_MC):
    features_FS.append(centering_and_rotate(X0/L0, array_SmoothThetaFP_X[k].data_trajectory))
    features_SRVF.append(centering_and_rotate(X0/L0, array_SRVF_mean[k]))
    features_Arithm.append(centering_and_rotate(X0/L0, array_Arithmetic_mean[k]))

X_mean = centering_and_rotate(X0/L0, X0/L0)
plot_3D_res_simu(features_FS, features_SRVF, features_Arithm, X_mean)


print('END !')
