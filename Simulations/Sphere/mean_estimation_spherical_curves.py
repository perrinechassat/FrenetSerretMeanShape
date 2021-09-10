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


""" Parameters of the simulation """
n_curves = 25
N = n_curves
nb_S = 100
# n_MC = 100
hyperparam = [0.006, 1e-12, 1e-12, 1e-12]
nb_knots = 40
param_bayopt = {"n_splits":  10, "n_calls" : 80, "bounds_h" : (0.03, 0.14), "bounds_lcurv" : (1e-6, 1), "bounds_ltors" : (1e-6, 1)}


""" Generate data """
Mu, X_tab, V_tab = generative_model_spherical_curves(n_curves, 20, nb_S, (0,1))
t = np.linspace(0,1,nb_S)


"""--------------------------------------------------------- Frenet-Serret Mean -----------------------------------------------------------------"""

""" Preprocessing """
n_resamples = 100
param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 50}
param_loc_poly_TNB = {"h" : 15, "p" : 3, "iflag": [1,1], "ibound" : 0}
domain_range = (0,1)

array_X = []
array_Xnew = []
array_Xnew_scale = []
array_Q_LP = []
array_Q_GS = []
array_ThetaExtrins = []
list_L = []

for i in range(n_curves):
    X_newi, Xi, Q_LPi, Q_GSi, theta_extrinsi, successLocPoly = pre_process_data(X_tab[:,:,i], t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True)
    print(successLocPoly)
    array_X.append(Xi)
    list_L.append(Xi.L)
    array_Xnew_scale.append(X_newi/Xi.L)
    array_Q_LP.append(Q_LPi)
    array_Q_GS.append(Q_GSi)
    array_Xnew.append(X_newi)
    array_ThetaExtrins.append(theta_extrinsi)

mean_L = np.mean(list_L)

""" Mean Estimation with alignement """
TruePopFrenetPath = PopulationFrenetPath(array_Q_LP)

print('Begin FS mean estimation V1...')
SmoothPopFrenetPath1, resOpt_mean1 = single_estim_optimizatinon(TruePopFrenetPath, domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=True, lam=100)
SmoothThetaPopFP1 = FrenetPath(SmoothPopFrenetPath1.grids_obs[0], SmoothPopFrenetPath1.grids_obs[0], init=mean_Q0(SmoothPopFrenetPath1), curv=SmoothPopFrenetPath1.mean_curv, tors=SmoothPopFrenetPath1.mean_tors, dim=3)
SmoothThetaPopFP1.frenet_serret_solve()
print('End FS mean estimation.')


print('Begin FS mean estimation V2...')
SmoothPopFrenetPath2, resOpt_mean2 = single_estim_optimizatinon_sphere(TruePopFrenetPath, domain_range, nb_knots, mean_L, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=True, lam=100)
SmoothThetaPopFP2 = FrenetPath(SmoothPopFrenetPath2.grids_obs[0], SmoothPopFrenetPath2.grids_obs[0], init=mean_Q0(SmoothPopFrenetPath2), curv=SmoothPopFrenetPath2.mean_curv, tors=SmoothPopFrenetPath2.mean_tors, dim=3)
SmoothThetaPopFP2.frenet_serret_solve()
print('End FS mean estimation.')

print('Begin FS mean estimation V3...')
SmoothPopFrenetPath3, resOpt_mean3 = single_estim_optimizatinon_sphere_double(TruePopFrenetPath, domain_range, nb_knots, mean_L, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=True, lam=100)
SmoothThetaPopFP3 = FrenetPath(SmoothPopFrenetPath3.grids_obs[0], SmoothPopFrenetPath3.grids_obs[0], init=mean_Q0(SmoothPopFrenetPath3), curv=SmoothPopFrenetPath3.mean_curv, tors=SmoothPopFrenetPath3.mean_tors, dim=3)
SmoothThetaPopFP3.frenet_serret_solve()
print('End FS mean estimation.')


"""--------------------------------------------------------- SRVF Mean -----------------------------------------------------------------"""

print("SRVF Mean estimation...")
t = np.linspace(0,1,n_resamples)
SRVF_mean = compute_mean_SRVF(array_Xnew, t)

"""--------------------------------------------------------- Arithmetic Mean -----------------------------------------------------------------"""

print("Arithmetic Mean estimation...")
Arithmetic_mean = compute_mean_Arithmetic(array_Xnew_scale)


""" SAVING DATA """

print('Saving the data...')

filename = "SphereCurves_FS_estimation_Mu3"+'_nCalls_'+str(param_bayopt["n_calls"])
dic = {"data": X_tab, "array_X" : array_X, "array_Xnew" : array_Xnew, "t" : t, "array_Q_LP" : array_Q_LP, "array_Q_GS" : array_Q_GS, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots,
 "SmoothPopFrenetPath1" : SmoothPopFrenetPath1, "resOpt_mean1": resOpt_mean1, "SmoothThetaPopFP1" : SmoothThetaPopFP1,
 "SmoothPopFrenetPath2" : SmoothPopFrenetPath2, "resOpt_mean2": resOpt_mean2, "SmoothThetaPopFP2" : SmoothThetaPopFP2,
 "SmoothPopFrenetPath3" : SmoothPopFrenetPath3, "resOpt_mean3": resOpt_mean3, "SmoothThetaPopFP3" : SmoothThetaPopFP3,
 "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" : SRVF_mean, "Arithmetic_mean" : Arithmetic_mean}


if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')
