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
param_bayopt = {"n_splits":  10, "n_calls" : 10, "bounds_h" : (0.03, 0.14), "bounds_lcurv" : (1e-6, 0.1), "bounds_ltors" : (1e-6, 0.1)}


""" Generate data """
Mu, X_tab, V_tab = generative_model_spherical_curves(n_curves, 20, nb_S, (0,1))
t = np.linspace(0,1,nb_S)


"""--------------------------------------------------------- Frenet-Serret Mean -----------------------------------------------------------------"""

""" Preprocessing """
n_resamples = 100
param_loc_poly_deriv = { "h_min" : 0.01, "h_max" : 0.2, "nb_h" : 20}
param_loc_poly_TNB = {"h" : 15, "p" : 3, "iflag": [1,1], "ibound" : 0}
domain_range = (0,1)

array_X = []
array_Xnew = []
array_Xnew_scale = []
array_Q_LP = []
array_Q_GS = []
array_ThetaExtrins = []

for i in range(n_curves):
    X_newi, Xi, Q_LPi, Q_GSi, theta_extrinsi, successLocPoly = pre_process_data(X_tab[:,:,i], t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True)
    print(successLocPoly)
    array_X.append(Xi)
    array_Xnew_scale.append(X_newi/Xi.L)
    array_Q_LP.append(Q_LPi)
    array_Q_GS.append(Q_GSi)
    array_Xnew.append(X_newi)
    array_ThetaExtrins.append(theta_extrinsi)

""" Mean Estimation with alignement """
TruePopFrenetPath = PopulationFrenetPath(array_Q_LP)

print('Begin FS mean estimation...')
SmoothPopFrenetPath, resOpt_mean = single_estim_optimizatinon(TruePopFrenetPath, domain_range, nb_knots, tracking=False, hyperparam=hyperparam, opt=True, param_bayopt=param_bayopt, multicurves=True, alignment=True, lam=100)
SmoothThetaPopFP = FrenetPath(SmoothPopFrenetPath.grids_obs[0], SmoothPopFrenetPath.grids_obs[0], init=mean_Q0(SmoothPopFrenetPath), curv=SmoothPopFrenetPath.mean_curv, tors=SmoothPopFrenetPath.mean_tors, dim=3)
SmoothThetaPopFP.frenet_serret_solve()
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

filename = "SphereCurves_FS_estimation_only_mean_adaptative"+'_nCalls_'+str(param_bayopt["n_calls"])
dic = {"data": X_tab, "array_X" : array_X, "array_Xnew" : array_Xnew, "t" : t, "array_Q_LP" : array_Q_LP, "array_Q_GS" : array_Q_GS, "param_bayopt" : param_bayopt, "nb_knots" : nb_knots,
 "SmoothPopFrenetPath" : SmoothPopFrenetPath, "resOpt_mean": resOpt_mean, "SmoothThetaPopFP" : SmoothThetaPopFP,
 "param_loc_poly_deriv" : param_loc_poly_deriv, "param_loc_poly_TNB" : param_loc_poly_TNB, "ThetaExtrins" : array_ThetaExtrins, "SRVF_mean" : SRVF_mean, "Arithmetic_mean" : Arithmetic_mean}


if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('END !')
