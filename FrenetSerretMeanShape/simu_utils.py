import sys
import os.path
sys.path.insert(1, '../Simulations/Sphere')
from frenet_path import *
from trajectory import *
from model_curvatures import *
from maths_utils import *
from generative_model_spherical_curves import *
# from estimation_algo_utils import *
from optimization_utils import opti_loc_poly_traj

import numpy as np
from scipy.linalg import expm, polar, logm
from scipy.integrate import cumtrapz
from scipy.interpolate import splrep, splder, sproot, splev, interp1d
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.matrices import Matrices
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.riemannian_metric import RiemannianMetric
import fdasrsf as fs

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.registration import ElasticRegistration, ShiftRegistration, landmark_registration_warping
from skfda.preprocessing.registration.elastic import elastic_mean
from skfda.misc import metrics
from joblib import Parallel, delayed
from timeit import default_timer as timer
import torch

from numba.experimental import jitclass
from numba import int32, float64, cuda, float32, objmode, njit, prange
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


""" Set of functions useful for the simulation codes """


def add_noise_FrenetPath(K, Q):
    Q_noisy = FrenetPath(Q.grid_obs, Q.grid_obs, dim=3)
    for j in range(len(Q.grid_obs)):
        Q_noisy.data[:,:,j] = np.dot(Q.data[:,:,j], matrix_rnd(K))
    Q_noisy.data_trajectory = np.transpose(cumtrapz(Q_noisy.data[:,0,:], Q_noisy.grid_obs, initial=0))
    return Q_noisy



def simul_populationTNB_additiveVar(N_samples, L0, s0, K, param_kappa, param_tau, sigma_kappa, sigma_tau, true_curv0, true_tors0, Noisy_flag):
    range_intvl = [0,L0]
    n_interp = 1000
    GP_noise_kappa = simulate_populationGP(N_samples, range_intvl, n_interp, param_kappa)
    GP_noise_tau = simulate_populationGP(N_samples, range_intvl, n_interp, param_tau)

    def curv_fct(k):
        return lambda s: np.abs(true_curv0(s) + sigma_kappa*GP_noise_kappa[k](s))
    def tors_fct(k):
        return lambda s: true_tors0(s) + sigma_tau*GP_noise_tau[k](s)

    data_pop_TNB = []
    for k in range(N_samples):
        data_pop_TNB.append(FrenetPath(s0, s0, init=matrix_rnd(K), curv=curv_fct(k), tors=tors_fct(k), dim=3))
        data_pop_TNB[k].frenet_serret_solve()
    data_pop_TNB_noisy = []
    if Noisy_flag==True:
        for k in range(N_samples):
            data_pop_TNB_noisy.append(add_noise_FrenetPath(K, data_pop_TNB[k]))
    else:
        data_pop_TNB_noisy = data_pop_TNB

    PopTNB = PopulationFrenetPath(data_pop_TNB)
    PopTNB_noisy = PopulationFrenetPath(data_pop_TNB_noisy)

    return PopTNB, PopTNB_noisy


def simul_populationTNB_TimeAndShapeWarping(N_samples, t, s0_fun, K, param_time_warp, param_shape_warp, true_curv0, true_tors0, Noisy_flag):

    list_S = []
    list_L = []
    for i in range(N_samples):
        list_S.append(gamma(s0_fun(h(t,param_time_warp[i])),param_shape_warp[i]))
        list_L.append(list_S[i][-1])

    def curv_fct(k):
        return lambda s: omega_prime(s,param_shape_warp[k])*true_curv0(omega(s,param_shape_warp[k]))*list_L[k]
    def tors_fct(k):
        return lambda s: omega_prime(s,param_shape_warp[k])*true_tors0(omega(s,param_shape_warp[k]))*list_L[k]

    data_pop_TNB = []
    for k in range(N_samples):
        data_pop_TNB.append(FrenetPath(list_S[k], list_S[k], init=matrix_rnd(K), curv=curv_fct(k), tors=tors_fct(k), dim=3))
        data_pop_TNB[k].frenet_serret_solve()
    data_pop_TNB_noisy = []
    if Noisy_flag==True:
        for k in range(N_samples):
            data_pop_TNB_noisy.append(add_noise_FrenetPath(K, data_pop_TNB[k]))
    else:
        data_pop_TNB_noisy = data_pop_TNB

    PopTNB = PopulationFrenetPath(data_pop_TNB)
    PopTNB_noisy = PopulationFrenetPath(data_pop_TNB_noisy)

    return PopTNB, PopTNB_noisy


def simul_populationTNB_TimeWarping(N_samples, t, s0_fun, K, param_time_warp, param_shape_warp, true_curv0, true_tors0, Noisy_flag):

    list_S = []
    list_L = []
    for i in range(N_samples):
        list_S.append(s0_fun(h(t,param_time_warp[i])))
        list_L.append(list_S[i][-1])

    data_pop_TNB = []
    for k in range(N_samples):
        data_pop_TNB.append(FrenetPath(list_S[k], list_S[k], init=matrix_rnd(K), curv=true_curv0, tors=true_tors0, dim=3))
        data_pop_TNB[k].frenet_serret_solve()
    data_pop_TNB_noisy = []
    if Noisy_flag==True:
        for k in range(N_samples):
            data_pop_TNB_noisy.append(add_noise_FrenetPath(K, data_pop_TNB[k]))
    else:
        data_pop_TNB_noisy = data_pop_TNB

    PopTNB = PopulationFrenetPath(data_pop_TNB)
    PopTNB_noisy = PopulationFrenetPath(data_pop_TNB_noisy)

    return PopTNB, PopTNB_noisy


def simul_populationTNB_ShapeWarping(N_samples, t, s0_fun, K, param_shape_warp, true_curv0, true_tors0, Noisy_flag):

    list_S = []
    list_L = []
    for i in range(N_samples):
        list_S.append(gamma(s0_fun(t),param_shape_warp[i]))
        list_L.append(list_S[i][-1])

    def curv_fct(k):
        return lambda s: omega_prime(s,param_shape_warp[k])*true_curv0(omega(s,param_shape_warp[k]))*list_L[k]
    def tors_fct(k):
        return lambda s: omega_prime(s,param_shape_warp[k])*true_tors0(omega(s,param_shape_warp[k]))*list_L[k]

    data_pop_TNB = []
    for k in range(N_samples):
        data_pop_TNB.append(FrenetPath(list_S[k], list_S[k], init=matrix_rnd(K), curv=curv_fct(k), tors=tors_fct(k), dim=3))
        data_pop_TNB[k].frenet_serret_solve()
    data_pop_TNB_noisy = []
    if Noisy_flag==True:
        for k in range(N_samples):
            data_pop_TNB_noisy.append(add_noise_FrenetPath(K, data_pop_TNB[k]))
    else:
        data_pop_TNB_noisy = data_pop_TNB

    PopTNB = PopulationFrenetPath(data_pop_TNB)
    PopTNB_noisy = PopulationFrenetPath(data_pop_TNB_noisy)

    return PopTNB, PopTNB_noisy


def simul_populationTNB_ShapeWarping_TNB(N_samples, s0, K, param_shape_warp, true_curv0, true_tors0, Noisy_flag):

    list_L = []
    for i in range(N_samples):
        list_L.append(s0[-1])

    def curv_fct(k):
        return lambda s: omega_prime(s,param_shape_warp[k])*true_curv0(omega(s,param_shape_warp[k]))*list_L[k]
    def tors_fct(k):
        return lambda s: omega_prime(s,param_shape_warp[k])*true_tors0(omega(s,param_shape_warp[k]))*list_L[k]

    data_pop_TNB = []
    for k in range(N_samples):
        data_pop_TNB.append(FrenetPath(s0, s0, init=matrix_rnd(K), curv=curv_fct(k), tors=tors_fct(k), dim=3))
        data_pop_TNB[k].frenet_serret_solve()
    data_pop_TNB_noisy = []
    if Noisy_flag==True:
        for k in range(N_samples):
            data_pop_TNB_noisy.append(add_noise_FrenetPath(K, data_pop_TNB[k]))
    else:
        data_pop_TNB_noisy = data_pop_TNB

    PopTNB = PopulationFrenetPath(data_pop_TNB)
    PopTNB_noisy = PopulationFrenetPath(data_pop_TNB_noisy)

    return PopTNB, PopTNB_noisy




def simul_populationCurves_UnknownModel(f, time, phi_ref, sigma_e, sigma_p, N, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind, locpolyTNB_local):
    phi_array = np.random.normal(loc=phi_ref, scale=sigma_p, size=(N,3))
    array_Traj =  []
    PopQ_LP = []
    PopQ_GS = []
    array_ThetaExtrins = []

    for i in range(N):
        X, Q_LP, Q_GS, theta_extrins = add_noise_X_and_preprocess(f(time, phi_array[i]), sigma_e, time, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind, locpolyTNB_local)
        array_Traj.append(X)
        PopQ_GS.append(Q_GS)
        PopQ_LP.append(Q_LP)
        array_ThetaExtrins.append(theta_extrins)


    phi_mean = np.mean(phi_array, axis=0)
    X_mean, Q_LP_mean, Q_GS_mean, theta_extrins_mean = add_noise_X_and_preprocess(f(time, phi_mean), 0.0, time, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind, locpolyTNB_local)

    return array_Traj, PopulationFrenetPath(PopQ_LP), PopulationFrenetPath(PopQ_GS), array_ThetaExtrins, phi_array, X_mean, Q_LP_mean, theta_extrins_mean



def pre_process_data_fast(data, t_init, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True):
    X = Trajectory(data, t_init)
    # Estimation des dérivées et de s(t)
    h_opt = opti_loc_poly_traj(X.data, X.t, param_loc_poly_deriv['h_min'], param_loc_poly_deriv['h_max'], param_loc_poly_deriv['nb_h'])
    # print(h_opt)
    X.loc_poly_estimation(X.t, 5, h_opt)
    X.compute_S(scale=scale_ind["ind"])

    if scale_ind["ind"]==True:
        new_grid_S = np.linspace(0,1,n_resamples)
    else:
        new_grid_S = np.linspace(0,X.S(X.t)[-1],n_resamples)
    Q_LP, vkappa, Param, Param0, vparam, successLocPoly = X.TNB_locPolyReg(grid_in=X.S(X.t), grid_out=new_grid_S, h=param_loc_poly_TNB['h'], p=param_loc_poly_TNB['p'], iflag=param_loc_poly_TNB['iflag'],
     ibound=param_loc_poly_TNB['ibound'], local=locpolyTNB_local)

    if scale_ind==False:
        Q_LP.grid_obs = np.linspace(0,scale_ind["val"],n_resamples)
        Q_LP.grid_eval = np.linspace(0,scale_ind["val"],n_resamples)

    Q_GS = X.TNB_GramSchmidt(X.t)
    curv_extrins, tors_extrins = X.theta_extrinsic_formula(X.t)
    theta_extrins = [curv_extrins, tors_extrins]

    return X, Q_LP, Q_GS, theta_extrins, successLocPoly



def add_noise_X_and_preprocess(X0, sigma, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=False):
    success = False
    k = 0
    param_TNB = param_loc_poly_TNB.copy()
    while success==False and k<10:
        # print(param_loc_poly_TNB["h"])
        if sigma!=0:
            noise = np.random.randn(X0.shape[0], X0.shape[1])
            data_X_noisy = np.add(X0, sigma*noise)
        else:
            data_X_noisy = X0
        X, Q_LP, Q_GS, theta_extrins, success = pre_process_data_fast(data_X_noisy, t, n_resamples, param_loc_poly_deriv, param_TNB, scale_ind, locpolyTNB_local)
        if locpolyTNB_local==False:
            param_TNB["h"]+=0.01
        else:
            param_TNB["h"]+=2
        k+=1
    if k==10:
        print("ECHEC")
    return X, Q_LP, Q_GS, theta_extrins


def add_noise_X_and_preprocess_MultipleCurves(PopFrenetPath, sigma, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=False):
    N_samples = PopFrenetPath.nb_samples
    # array_Traj = np.empty((N_samples), dtype=object)
    array_Traj = []
    PopQ_LP = []
    PopQ_GS = []
    # array_ThetaExtrins = np.empty((N_samples), dtype=object)
    array_ThetaExtrins = []
    for i in range(N_samples):
        # print(i)
        X, Q_LP, Q_GS, theta_extrins = add_noise_X_and_preprocess(PopFrenetPath.frenet_paths[i].data_trajectory, sigma, t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind, locpolyTNB_local)
        # array_Traj[i], array_ThetaExtrins[i] = X, theta_extrins
        array_Traj.append(X)
        PopQ_GS.append(Q_GS)
        PopQ_LP.append(Q_LP)
        array_ThetaExtrins.append(theta_extrins)
    return array_Traj, PopulationFrenetPath(PopQ_LP), PopulationFrenetPath(PopQ_GS), array_ThetaExtrins



def compute_mean_SRVF(array_Traj, t):
    K = len(array_Traj)
    if isinstance(array_Traj[0], Trajectory):
        M,n = array_Traj[0].data.shape
        beta_X = np.zeros((n,M,K))
        for i in range(K):
            beta_X[:,:,i] = np.transpose(array_Traj[i].data)
    else:
        M,n = array_Traj[0].shape
        beta_X = np.zeros((n,M,K))
        for i in range(K):
            beta_X[:,:,i] = np.transpose(array_Traj[i])

    obj_X = fs.fdacurve(beta=beta_X,N=M,scale=False)
    obj_X.srvf_align()
    # obj_X.karcher_mean(parallel=False, cores=-1)

    mean_srvf = np.transpose(obj_X.beta_mean)
    srvf_traj = mean_srvf - mean_srvf[0,:]

    gam_srvf = obj_X.gams
    gamI = fs.utility_functions.SqrtMeanInverse(gam_srvf)
    time0 = (t[-1] - t[0]) * gamI + t[0]
    for k in range(K):
        gam_srvf[:, k] = np.interp(time0, t, gam_srvf[:, k])

    return srvf_traj, gam_srvf


def compute_mean_Arithmetic(array_Traj):
    K = len(array_Traj)
    if isinstance(array_Traj[0], Trajectory):
        M,n = array_Traj[0].data.shape
        data_X = np.zeros((K,M,n))
        for i in range(K):
            data_X[i,:,:] = array_Traj[i].data
    else:
        M,n = array_Traj[0].shape
        data_X = np.zeros((K,M,n))
        for i in range(K):
            data_X[i,:,:] = array_Traj[i]

    mean = np.mean(data_X, axis=0)
    mean = mean - mean[0,:]
    return mean


def pre_process_data(data, t_init, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True):

    def x(t): return interpolate.griddata(t_init, data, t, method='cubic')
    t_new = np.linspace(0,t_init[-1],n_resamples)
    X_new = x(t_new)

    X = Trajectory(data, t_init)
    # Estimation des dérivées et de s(t)
    h_opt = opti_loc_poly_traj(X.data, X.t, param_loc_poly_deriv['h_min'], param_loc_poly_deriv['h_max'], param_loc_poly_deriv['nb_h'])
    # print(h_opt)
    X.loc_poly_estimation(X.t, 5, h_opt)
    X.compute_S(scale=scale_ind["ind"])

    if scale_ind["ind"]==True:
        new_grid_S = np.linspace(0,1,n_resamples)
    else:
        new_grid_S = np.linspace(0,X.S(X.t)[-1],n_resamples)
    Q_LP, vkappa, Param, Param0, vparam, successLocPoly = X.TNB_locPolyReg(grid_in=X.S(X.t), grid_out=new_grid_S, h=param_loc_poly_TNB['h'], p=param_loc_poly_TNB['p'], iflag=param_loc_poly_TNB['iflag'],
     ibound=param_loc_poly_TNB['ibound'], local=locpolyTNB_local)

    Q_GS = X.TNB_GramSchmidt(t_new)
    curv_extrins, tors_extrins = X.theta_extrinsic_formula(t_new)
    theta_extrins = [curv_extrins, tors_extrins]

    return X_new, X, Q_LP, Q_GS, theta_extrins, successLocPoly



def pre_process_data_sphere(data, t_init, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind={"ind":True,"val":1}, locpolyTNB_local=True):

    X = Trajectory(data, t_init)
    h_opt = opti_loc_poly_traj(X.data, X.t, param_loc_poly_deriv['h_min'], param_loc_poly_deriv['h_max'], param_loc_poly_deriv['nb_h'])
    X.loc_poly_estimation(X.t, 5, h_opt)
    X.compute_S(scale=scale_ind["ind"])

    if scale_ind["ind"]==True:
        new_grid_S = np.linspace(0,1,n_resamples)
    else:
        new_grid_S = np.linspace(0,X.S(X.t)[-1],n_resamples)

    """ estimation TNB local poly """
    success = False
    k = 0
    param_TNB = param_loc_poly_TNB.copy()
    while success==False and k<15:
        # print(param_TNB['h'])
        Q_LP, vkappa, Param, Param0, vparam, success = X.TNB_locPolyReg(grid_in=X.S(X.t), grid_out=new_grid_S, h=param_TNB['h'], p=param_TNB['p'], iflag=param_TNB['iflag'],
         ibound=param_loc_poly_TNB['ibound'], local=locpolyTNB_local)
        if locpolyTNB_local==False:
            param_TNB["h"]+=0.01
        else:
            param_TNB["h"]+=2
        k+=1
    if k==15:
        print("ECHEC")

    if scale_ind["ind"]==True:
        alpha = Q_LP.data_trajectory*X.L
    else:
        alpha = Q_LP.data_trajectory
    beta = Q_LP.data[:,0,:]
    gamma = np.transpose(np.cross(alpha, np.transpose(beta)))
    Q = np.zeros((3, 3, n_resamples))
    Q[:,0,:] = np.transpose(alpha)
    Q[:,1,:] = beta
    Q[:,2,:] = gamma
    NewFrame = FrenetPath(new_grid_S, new_grid_S, data=Q)

    return X, Q_LP, NewFrame, success


def simul_Frame_sphere(N, nb_S, domain_range, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind, locpolyTNB_local):
    """ Generate data """
    Mu, X_tab, V_tab = generative_model_spherical_curves(N, 20, nb_S, domain_range)
    t = np.linspace(domain_range[0],domain_range[1],nb_S)

    array_Traj =  []
    PopQ_LP = []
    Pop_NewFrame = []
    list_L = []

    for i in range(N):
        X, Q_LP, NewFrame, successLocPoly = pre_process_data_sphere(X_tab[:,:,i], t, n_resamples, param_loc_poly_deriv, param_loc_poly_TNB, scale_ind, locpolyTNB_local)
        # print(successLocPoly)
        # if successLocPoly==False:
        #     return [], [], [], []
        array_Traj.append(X)
        PopQ_LP.append(Q_LP)
        Pop_NewFrame.append(NewFrame)
        list_L.append(X.L)

    mean_L = np.mean(list_L)

    return array_Traj, PopulationFrenetPath(PopQ_LP), PopulationFrenetPath(Pop_NewFrame), mean_L




"""Warpings functions"""

def omega(s,a):
    if np.abs(a)<1e-15:
        return s
    else:
        return np.around(np.log(s*(np.exp(a)-1)+1)/a, decimals=6)

def omega_prime(s,a):
    if np.abs(a)<1e-15:
        return 1 + s*0
    else:
        return (1/a)*(np.exp(a)-1)*(1/(s*(np.exp(a)-1)+1))

def gamma(s,a):
    if np.abs(a)<1e-15:
        return s
    else:
        return (np.exp(a*s) - 1)/(np.exp(a) - 1)

def gamma_prime(s,a):
    if np.abs(a)<1e-15:
        return 1 + s*0
    else:
        return a/(np.exp(a) - 1) * np.exp(a*s)

def h(s,a):
    if np.abs(a)<1e-15:
        return s
    else:
        return (np.sin(2*np.pi*s) + s/a)*a

def h_prime(s,a):
    if np.abs(a)<1e-15:
        return 1 + s*0
    else:
        return (2*np.pi*np.cos(2*np.pi*s) + 1/a)*a
#
# def gamma_5(s,a):
#     if np.abs(a)<1e-15:
#         return s
#     else:
#         return 5*(np.exp(a*s/5) - 1)/(np.exp(a) - 1)
