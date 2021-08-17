import sys
from frenet_path import *
from trajectory import *
from model_curvatures import *
from maths_utils import *
from optimization_utils import *
from alignment_utils import *

import numpy as np
from scipy.linalg import expm, polar, logm
from scipy.integrate import cumtrapz
from scipy.interpolate import splrep, splder, sproot, splev, interp1d
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.matrices import Matrices
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.riemannian_metric import RiemannianMetric

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.registration import ElasticRegistration, ShiftRegistration, landmark_registration_warping
from skfda.preprocessing.registration.elastic import elastic_mean
from skfda.misc import metrics
import fdasrsf as fs
from joblib import Parallel, delayed
from timeit import default_timer as timer
import torch

from numba.experimental import jitclass
from numba import int32, float64, cuda, float32, objmode, njit, prange
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


""" Set of functions for the estimation of the mean Frenet Path and mean curvatures """


def compute_A(theta):
    """
    Compute the matrix 3x3 A_theta
    ...
    """
    return np.array([[0, -theta[0], 0], [theta[0], 0, -theta[1]], [0, theta[1], 0]])


def lie_smoother_q(q, SingleFrenetPath, Model, p):
    """
    Step of function Lie smoother.
    ...
    """

    SO3 = SpecialOrthogonal(3)
    Obs_q = SingleFrenetPath.data[:,:,SingleFrenetPath.neighbor_obs[q]]
    Obs_q = np.rollaxis(Obs_q, 2)

    kappa_q = np.apply_along_axis(Model.curv.function, 0, SingleFrenetPath.grid_double[q])
    tau_q = np.apply_along_axis(Model.tors.function, 0, SingleFrenetPath.grid_double[q])
    theta_q = np.stack((SingleFrenetPath.delta[q]*kappa_q, SingleFrenetPath.delta[q]*tau_q), axis=1)

    A_q = np.apply_along_axis(compute_A, 1, theta_q)
    A_q = torch.from_numpy(A_q)
    exp_A_q = torch.matrix_exp(A_q)
    Obs_q_torch = torch.from_numpy(Obs_q)
    mat_product = Obs_q_torch
    ind_norm = torch.where(torch.linalg.matrix_norm(A_q) > 0.)[0]
    mat_product[ind_norm,:,:] = torch.matmul(Obs_q_torch[ind_norm,:,:].double(), exp_A_q[ind_norm,:,:].double())

    mean = FrechetMean(metric=SO3.metric, max_iter=100, point_type='matrix', verbose=True)
    mean.fit(mat_product.cpu().detach().numpy(), SingleFrenetPath.weight[q])
    mu_q = mean.estimate_

    return mu_q
    # mu_q_proj = SO3.projection(mu_q)
    # return mu_q_proj

def lie_smoother_i(SingleFrenetPath, Model, p):
    """
    Step of function Lie smoother.
    ...
    """
    M = np.zeros((p,p,SingleFrenetPath.nb_grid_eval))

    for q in range(SingleFrenetPath.nb_grid_eval):
        M[:,:,q] = lie_smoother_q(q, SingleFrenetPath, Model, p)

    SmoothFrenetPath = FrenetPath(grid_obs=SingleFrenetPath.grid_eval, grid_eval=SingleFrenetPath.grid_eval, data=M)
    return SmoothFrenetPath


def lie_smoother(PopFrenetPath, Model):
    """
    LieSmoother
    Smoothing by local averaging based on the geodesic distance in SO(p),
    i.e. we compute local Karcher mean. It can use the information on the curvature and torsion defined in Model for a better estimation.
    ...
    """

    if isinstance(PopFrenetPath, FrenetPath):
        N_samples = 1
    else:
        N_samples = PopFrenetPath.nb_samples

    p = PopFrenetPath.dim

    if N_samples==1:
        return lie_smoother_i(PopFrenetPath, Model, p)
    else:
        data_smoothfrenetpath = []
        for i in range(N_samples):
            data_smoothfrenetpath.append(lie_smoother_i(PopFrenetPath.frenet_paths[i], Model, p))

        return PopulationFrenetPath(data_smoothfrenetpath)


@njit
def compute_sort_unique_val(S, Omega, Kappa, Tau):
    """
    Step of function Compute Raw Curvature, compute the re-ordering of the data.
    ...
    """
    uniqueS = np.unique(S)
    nb_unique_val = len(uniqueS)
    mOmega = np.zeros(nb_unique_val)
    mKappa = np.zeros(nb_unique_val)
    mTau   = np.zeros(nb_unique_val)
    for ijq in range(nb_unique_val):
        id_ijq      = np.where(S==uniqueS[ijq])[0]
        Omega_ijq   = Omega[id_ijq]
        Kappa_ijq   = Kappa[id_ijq]
        Tau_ijq     = Tau[id_ijq]
        mOmega[ijq] = np.sum(Omega_ijq)
        if mOmega[ijq]>0:
            mKappa[ijq] = (np.ascontiguousarray(Omega_ijq[np.where(Omega_ijq>0)]) @ np.ascontiguousarray(np.transpose(Kappa_ijq[np.where(Omega_ijq>0)])))/mOmega[ijq]
            mTau[ijq]   = (np.ascontiguousarray(Omega_ijq[np.where(Omega_ijq>0)]) @ np.ascontiguousarray(np.transpose(Tau_ijq[np.where(Omega_ijq>0)])))/mOmega[ijq]
        else:
            mKappa[ijq] = 0
            mTau[ijq]   = 0

    return uniqueS, mOmega, mKappa, mTau


@njit
def compute_Rq_boucle(dim, N_q, Obs_q, data, u_q, q, nb_grid):
    """
    Step of function Compute Raw Curvature
    ...
    """
    R_q = np.zeros((dim,dim,N_q))

    for j in range(N_q):
        if (q!=0 or j!=0) and (q!=nb_grid-1 or j!=N_q-1):
            R_q[:,:,j] = -my_log_M3(np.transpose(np.ascontiguousarray(data))@np.ascontiguousarray(Obs_q[:,:,j]))/u_q[j]

    return R_q


def compute_Rq(q, FrenetPath, SmoothFrenetPath):
    """
    Step of function Compute Raw Curvature
    ...
    """
    N_q = len(FrenetPath.neighbor_obs[q])
    Obs_q = FrenetPath.data[:,:,FrenetPath.neighbor_obs[q]]
    w_q = FrenetPath.weight[q]
    u_q = np.copy(FrenetPath.delta[q])
    omega_q = np.multiply(w_q,np.power(u_q,2))

    if q!=0 and q!=FrenetPath.nb_grid_eval-1:
        v_q = np.where(u_q==0)[0]
        u_q[u_q==0] = 1

    R_q = compute_Rq_boucle(FrenetPath.dim, N_q, Obs_q, SmoothFrenetPath.data[:,:,q], u_q, q, FrenetPath.nb_grid_eval)

    if q!=0 and q!=FrenetPath.nb_grid_eval-1:
        R_q[:,:,v_q] = np.abs(0*R_q[:,:,v_q])

    kappa = np.squeeze(R_q[1,0,:])
    tau = np.squeeze(R_q[2,1,:])

    return omega_q.tolist(), kappa.tolist(), tau.tolist()


def compute_raw_curvatures_without_alignement(PopulationFrenetPath, h, PopulationSmoothFrenetPath):
    """
    Compute the weighted instantaneous rate of change of the Frenet frames without alignment between samples.
    They are noisy and often needs to be smoothed by splines
    ...
    """

    N_samples = PopulationFrenetPath.nb_samples
    PopulationFrenetPath.compute_neighbors(h)
    if N_samples==1:
        Omega, S, Kappa, Tau = [], [], [], []
        for q in range(PopulationFrenetPath.nb_grid_eval):
            if q==0:
                s = np.zeros(len(PopulationFrenetPath.neighbor_obs[q]))
            elif q==PopulationFrenetPath.nb_grid_eval-1:
                s = PopulationFrenetPath.length*np.ones(len(PopulationFrenetPath.neighbor_obs[q]))
            else:
                s = PopulationFrenetPath.grid_double[q]
            S += list(s)
            omega_q, kappa, tau = compute_Rq(q, PopulationFrenetPath, PopulationSmoothFrenetPath)
            Omega = np.append(Omega, omega_q)
            Kappa = np.append(Kappa, kappa)
            Tau = np.append(Tau, tau)
    else:
        Omega, S, Kappa, Tau = [], [], [], []
        for i in range(N_samples):
            for q in range(PopulationFrenetPath.frenet_paths[i].nb_grid_eval):
                if q==0:
                    s = np.zeros(len(PopulationFrenetPath.frenet_paths[i].neighbor_obs[q]))
                elif q==PopulationFrenetPath.frenet_paths[i].nb_grid_eval-1:
                    s = PopulationFrenetPath.frenet_paths[i].length*np.ones(len(PopulationFrenetPath.frenet_paths[i].neighbor_obs[q]))
                else:
                    s = PopulationFrenetPath.frenet_paths[i].grid_double[q]
                S += list(s)
                omega_q, kappa, tau = compute_Rq(q, PopulationFrenetPath.frenet_paths[i], PopulationSmoothFrenetPath.frenet_paths[i])
                Omega = np.append(Omega, omega_q)
                Kappa = np.append(Kappa, kappa)
                Tau = np.append(Tau, tau) # Kappa and Tau contains the empirical estimation of the curvature and torsion.

    Ms, Momega, Mkappa, Mtau = compute_sort_unique_val(np.around(S, 8), Omega, Kappa, Tau)

    return Mkappa, Mtau, Ms, Momega


def compute_raw_curvatures_i(SingleFrenetPath, SmoothFrenetPath):
    omega_i = []
    kappa_i = []
    s_i = []
    tau_i = []
    for q in range(SingleFrenetPath.nb_grid_eval):
        if q==0:
            s = np.zeros(len(SingleFrenetPath.neighbor_obs[q]))
        elif q==SingleFrenetPath.nb_grid_eval-1:
            s = SingleFrenetPath.length*np.ones(len(SingleFrenetPath.neighbor_obs[q]))
        else:
            s = SingleFrenetPath.grid_double[q]
        s_i = np.append(s_i, s.tolist())

        omega_q, kappa, tau = compute_Rq(q, SingleFrenetPath, SmoothFrenetPath)
        omega_i = np.append(omega_i, omega_q)
        kappa_i = np.append(kappa_i, kappa)
        tau_i = np.append(tau_i, tau)

    Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_sort_unique_val(np.around(s_i, 5), omega_i, kappa_i, tau_i)
    return Ms_i, Momega_i, Mkappa_i, Mtau_i



def compute_raw_curvatures_alignement_init(PopulationFrenetPath, h, PopulationSmoothFrenetPath, lam):
    """
    Compute the weighted instantaneous rate of change of the Frenet frames with alignment between samples.
    They are noisy and often needs to be smoothed by splines
    In this function we do the research of the warping functions.
    ...
    """
    N_samples = PopulationFrenetPath.nb_samples
    PopulationFrenetPath.compute_neighbors(h)

    if N_samples==1:
        Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_raw_curvatures_i(PopulationFrenetPath, PopulationSmoothFrenetPath)
        return Mkappa_i, Mtau_i, Ms_i, Momega_i

    else:
        Omega, S, Kappa, Tau = [], [], [], []
        """for alignement all FrenetPath needs to have the same grid"""

        for i in range(N_samples):
            Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_raw_curvatures_i(PopulationFrenetPath.frenet_paths[i], PopulationSmoothFrenetPath.frenet_paths[i])
            # if len(Ms_i)!=len(S[i-1]):
            S.append(Ms_i)
            Omega.append(Momega_i)
            Kappa.append(Mkappa_i)
            Tau.append(Mtau_i)

        # Alignment of curves
        Omega = np.asarray(Omega)
        sum_omega = np.sum(Omega, axis=0)
        ind_nozero = np.where(sum_omega!=0.)
        sum_omega = sum_omega[ind_nozero]
        Omega = np.squeeze(Omega[:,ind_nozero])
        Kappa = np.squeeze(np.asarray(Kappa)[:,ind_nozero])
        Tau = np.squeeze(np.asarray(Tau)[:,ind_nozero])
        S = S[0][ind_nozero]


        theta = np.stack((np.transpose(Kappa), np.abs(np.transpose(Tau))))
        theta[np.isnan(theta)] = 0.0
        theta[np.isinf(theta)] = 0.0

        res = align_vect_curvatures_fPCA(theta, np.squeeze(S), np.transpose(Omega), num_comp=3, cores=-1, smoothdata=False, MaxItr=20, lam=lam)
        theta_align, gam_theta, weighted_mean_theta = res.fn, res.gamf, res.mfn

        gam_fct = np.empty((gam_theta.shape[1]), dtype=object)
        for i in range(gam_theta.shape[1]):
            gam_fct[i] = interp1d(np.squeeze(S), gam_theta[:,i])

        weighted_mean_kappa = np.transpose(weighted_mean_theta[0])
        tau_align, weighted_mean_tau = warp_curvatures(np.transpose(Tau), gam_fct, np.squeeze(S), np.transpose(Omega))

        return weighted_mean_kappa, weighted_mean_tau, S, sum_omega, gam_fct, res


def compute_raw_curvatures_alignement_boucle(PopulationFrenetPath, h, PopulationSmoothFrenetPath, prev_gam):
    """
    Compute the weighted instantaneous rate of change of the Frenet frames with alignment between samples.
    They are noisy and often needs to be smoothed by splines
    In this function we only apply the warping functions found previously.
    ...
    """
    N_samples = PopulationFrenetPath.nb_samples
    PopulationFrenetPath.compute_neighbors(h)

    if N_samples==1:
        Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_raw_curvatures_i(PopulationFrenetPath, PopulationSmoothFrenetPath)
        return Mkappa_i, Mtau_i, Ms_i, Momega_i

    else:
        Omega, S, Kappa, Tau = [], [], [], []
        """for alignement all FrenetPath needs to have the same grid"""

        for i in range(N_samples):
            Ms_i, Momega_i, Mkappa_i, Mtau_i = compute_raw_curvatures_i(PopulationFrenetPath.frenet_paths[i], PopulationSmoothFrenetPath.frenet_paths[i])
            # if len(Ms_i)!=len(S[i-1]):
            S.append(Ms_i)
            Omega.append(Momega_i)
            Kappa.append(Mkappa_i)
            Tau.append(Mtau_i)

        ## VERSION WITH FDA
        Omega = np.asarray(Omega)
        sum_omega = np.sum(Omega, axis=0)
        ind_nozero = np.where(sum_omega!=0.)
        sum_omega = sum_omega[ind_nozero]
        Omega = np.squeeze(Omega[:,ind_nozero])
        Kappa = np.squeeze(np.asarray(Kappa)[:,ind_nozero])
        Tau = np.squeeze(np.asarray(Tau)[:,ind_nozero])
        S = S[0][ind_nozero]

        # WARP KAPPA BY THE PREVIOUS GAM Functions
        kappa_align, weighted_mean_kappa = warp_curvatures(np.transpose(Kappa), prev_gam, np.squeeze(S), np.transpose(Omega))
        tau_align, weighted_mean_tau = warp_curvatures(np.transpose(Tau), prev_gam, np.squeeze(S), np.transpose(Omega))
        weighted_mean_kappa = np.transpose(weighted_mean_kappa)
        weighted_mean_tau = np.transpose(weighted_mean_tau)

        return weighted_mean_kappa, weighted_mean_tau, S, sum_omega, prev_gam, kappa_align, tau_align



def global_estimation_init(PopFrenetPath, SmoothPopulationFrenetPath_init, Model, x, opt_alignment=False, lam=0.0, gam={"flag" : False, "value" : None}):

    if opt_alignment==False:
        mKappa, mTau, mS, mOmega = compute_raw_curvatures_without_alignement(PopFrenetPath, x[0], SmoothPopulationFrenetPath_init)
        align_results = collections.namedtuple('align_fPCA', ['convergence'])
        res = align_results(True)
    elif opt_alignment==True and gam["flag"]==False:
        mKappa, mTau, mS, mOmega, gam, res = compute_raw_curvatures_alignement_init(PopFrenetPath, x[0], SmoothPopulationFrenetPath_init, lam)
    else:
        mKappa, mTau, mS, mOmega, gam, kappa_align, tau_align = compute_raw_curvatures_alignement_boucle(PopFrenetPath, x[0], SmoothPopulationFrenetPath_init, gam["value"])
        align_results = collections.namedtuple('align_fPCA', ['convergence'])
        res = align_results(True)

    theta_curv = Model.curv.smoothing(mS, mKappa, mOmega, x[1])
    theta_torsion = Model.tors.smoothing(mS, mTau, mOmega, x[2])
    theta0 = np.concatenate((theta_curv,theta_torsion))

    return theta0, gam, res


def global_estimation(PopFrenetPath, SmoothPopulationFrenetPath_init, Model, x, opt_tracking=False, opt_alignment=False, lam=0.0, gam={"flag" : False, "value" : None}):

    epsilon = 10e-3
    N_samples = PopFrenetPath.nb_samples

    theta0, gam, res = global_estimation_init(PopFrenetPath, SmoothPopulationFrenetPath_init, Model, x, opt_alignment=opt_alignment, lam=lam, gam=gam)
    if res.convergence==False:
        return SmoothPopulationFrenetPath_init, False
    theta  = theta0
    Dtheta = theta
    k=0

    # Resmooth data with model information
    while np.linalg.norm(Dtheta)>=epsilon*np.linalg.norm(theta) and k<31:

        SmoothPopulationFrenet = lie_smoother(PopFrenetPath,Model)

        if opt_alignment==True:
            mKappa, mTau, mS, mOmega, gam, kappa_align, tau_align = compute_raw_curvatures_alignement_boucle(PopFrenetPath, x[0], SmoothPopulationFrenet, gam)
        else:
            mKappa, mTau, mS, mOmega = compute_raw_curvatures_without_alignement(PopFrenetPath, x[0], SmoothPopulationFrenet)

        theta_curv = Model.curv.smoothing(mS, mKappa, mOmega, x[1])
        theta_torsion = Model.tors.smoothing(mS, mTau, mOmega, x[2])

        thetakm1 = theta
        theta = np.concatenate((theta_curv, theta_torsion))
        Dtheta = theta - thetakm1
        SmoothPopulationFrenet_final = SmoothPopulationFrenet
        SmoothPopulationFrenet_final.set_estimate_theta(Model.curv.function, Model.tors.function)
        if N_samples!=1 and opt_alignment==True:
            SmoothPopulationFrenet_final.set_gam_functions(gam)
        k += 1

    if k==31:
        return SmoothPopulationFrenet_final, False
    elif k==0:
        return SmoothPopulationFrenetPath_init, True
    else:
        return SmoothPopulationFrenet_final, True



def adaptative_estimation(TrueFrenetPath, domain_range, nb_basis, tracking=False, hyperparam=None, opt=False, param_bayopt=None, multicurves=False, alignment=False, lam=0.0):

    curv_smoother = BasisSmoother(domain_range=domain_range, nb_basis=nb_basis)
    tors_smoother = BasisSmoother(domain_range=domain_range, nb_basis=nb_basis)

    if opt==True:
        if multicurves==True:
            Opt_fun = lambda x: objective_multiple_curve(param_bayopt["n_splits"], TrueFrenetPath, curv_smoother, tors_smoother, x, alignment, lam)
            x = bayesian_optimisation(Opt_fun, param_bayopt["n_calls"], [param_bayopt["bounds_h"], param_bayopt["bounds_lcurv"], param_bayopt["bounds_ltors"]])
        else:
            start = timer()
            Opt_fun = lambda x: objective_single_curve(param_bayopt["n_splits"], TrueFrenetPath, curv_smoother, tors_smoother, x)
            x = bayesian_optimisation(Opt_fun, param_bayopt["n_calls"], [param_bayopt["bounds_h"], param_bayopt["bounds_lcurv"], param_bayopt["bounds_ltors"]])
            duration = timer() - start
            print('Time for bayesian optimisation: ', duration)

        curv_smoother.reinitialize()
        tors_smoother.reinitialize()
        res_opt = x
    else:
        x = hyperparam
        res_opt = x

    Model_theta = Model(curv_smoother, tors_smoother)
    TrueFrenetPath.compute_neighbors(x[0])
    SmoothFrenetPath0 = lie_smoother(TrueFrenetPath,Model_theta)
    SmoothFrenetPath_fin, ind_conv = global_estimation(TrueFrenetPath, SmoothFrenetPath0, Model_theta, x, opt_tracking=tracking, opt_alignment=alignment, lam=lam)

    return SmoothFrenetPath_fin, [x,ind_conv]




""" Functions for the optimization of hyperparameters """


def step_cross_val(curv_smoother, tors_smoother, test_index, train_index, SingleFrenetPath, X_true_temp, hyperparam):
    """
    Step of cross validation in the case of estimation of curvature and torsion on a single curve. The error is computed here on X.
    ...
    """

    t_train = SingleFrenetPath.grid_obs[train_index]
    data_train = SingleFrenetPath.data[:,:,train_index]
    train_FrenetPath = FrenetPath(t_train, SingleFrenetPath.grid_obs, data=data_train)

    curv_smoother.reinitialize()
    tors_smoother.reinitialize()

    Model_test = Model(curv_smoother, tors_smoother)
    train_FrenetPath.compute_neighbors(hyperparam[0])
    pred_FrenetPath0 = lie_smoother(train_FrenetPath,Model_test)
    pred_FrenetPath = global_estimation(train_FrenetPath,pred_FrenetPath0, Model_test, hyperparam, opt_tracking=False)


    temp_FrenetPath_Q0 = FrenetPath(SingleFrenetPath.grid_obs, SingleFrenetPath.grid_eval, init=pred_FrenetPath.data[:,:,0], curv=pred_FrenetPath.curv, tors=pred_FrenetPath.tors)
    temp_FrenetPath_Q0.frenet_serret_solve()
    X_temp_Q0 = temp_FrenetPath_Q0.data_trajectory

    #alignement des centres
    X_temp_Q0 = X_temp_Q0 - fs.curve_functions.calculatecentroid(np.transpose(X_temp_Q0))
    #rotations
    X_temp_Q0_new = fs.curve_functions.find_best_rotation(np.transpose(X_true_temp), np.transpose(X_temp_Q0))[0]

    fd_FS = FDataGrid(X_temp_Q0_new[:,test_index], SingleFrenetPath.grid_obs[test_index])
    fd_true = FDataGrid(np.transpose(X_true_temp)[:,test_index], SingleFrenetPath.grid_obs[test_index])
    lp_dist = metrics.lp_distance(fd_true, fd_FS, p=2).mean()

    return lp_dist


def step_cross_val_on_Q(curv_smoother, tors_smoother, test_index, train_index, SingleFrenetPath, hyperparam):
    """
    Step of cross validation in the case of estimation of curvature and torsion on a single curve. The error is computed here on Q.
    ...
    """
    t_train = SingleFrenetPath.grid_obs[train_index]
    t_test = SingleFrenetPath.grid_obs[test_index]
    data_train = SingleFrenetPath.data[:,:,train_index]
    train_FrenetPath = FrenetPath(t_train, SingleFrenetPath.grid_obs, data=data_train)

    curv_smoother.reinitialize()
    tors_smoother.reinitialize()

    Model_test = Model(curv_smoother, tors_smoother)
    train_FrenetPath.compute_neighbors(hyperparam[0])
    pred_FrenetPath0 = lie_smoother(train_FrenetPath,Model_test)
    pred_FrenetPath, ind_conv = global_estimation(train_FrenetPath, pred_FrenetPath0, Model_test, hyperparam, opt_tracking=False)

    if ind_conv==True:
        temp_FrenetPath_Q0 = FrenetPath(SingleFrenetPath.grid_obs, SingleFrenetPath.grid_eval, init=pred_FrenetPath.data[:,:,0], curv=pred_FrenetPath.curv, tors=pred_FrenetPath.tors)
        temp_FrenetPath_Q0.frenet_serret_solve()

        dist = geodesic_dist(np.rollaxis(SingleFrenetPath.data[:,:,test_index], 2), np.rollaxis(temp_FrenetPath_Q0.data[:,:,test_index], 2))
        return dist
    else:
        return 100



def objective_single_curve(n_splits, SingleFrenetPath, curv_smoother, tors_smoother, hyperparam):
    """
    Objective function in case of estimation for a single curve that do the cross validation.
    ...
    """
    print(hyperparam)
    err = []
    # err2 =[]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    k = 0
    curv_smoother.reinitialize()
    tors_smoother.reinitialize()

    for train_index, test_index in kf.split(SingleFrenetPath.grid_obs):

        dist = step_cross_val_on_Q(curv_smoother, tors_smoother, test_index, train_index, SingleFrenetPath,  hyperparam)

        if np.isnan(dist):
            print('nan value', k)
        else:
            err.append(dist)

        k += 1

    # err = Parallel(n_jobs=10)(delayed(step_cross_val_on_Q)(curv_smoother, tors_smoother, test_index, train_index, SingleFrenetPath,  hyperparam)
    #     for train_index, test_index in kf.split(SingleFrenetPath.grid_obs))

    return np.mean(np.array(err))




def step_cross_val_on_Q_multiple_curves(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath,  hyperparam, alignment, lam, gam={"flag" : False, "value" : None}):
    """
    Step of cross validation in the case of estimation on multiple curves. The error is computed here on Q.
    ...
    """
    n_curves = PopFrenetPath.nb_samples

    train_PopFP_data = []
    for i in range(n_curves):
        train_PopFP_data.append(FrenetPath(PopFrenetPath.grids_obs[i][train_index], PopFrenetPath.grids_obs[i], data=np.copy(PopFrenetPath.data[i][:,:,train_index])))

    train_PopFP = PopulationFrenetPath(train_PopFP_data)

    curv_smoother.reinitialize()
    tors_smoother.reinitialize()

    Model_test = Model(curv_smoother, tors_smoother)
    train_PopFP.compute_neighbors(hyperparam[0])
    pred_PopFP0 = lie_smoother(train_PopFP,Model_test)
    pred_PopFP, ind_conv = global_estimation(train_PopFP, pred_PopFP0, Model_test, hyperparam, opt_tracking=False, opt_alignment=alignment, lam=lam, gam=gam)

    if ind_conv==True:
        Q0 = mean_Q0(pred_PopFP)

        temp_FrenetPath_Q0 = FrenetPath(PopFrenetPath.grids_obs[0], PopFrenetPath.grids_obs[0], init=Q0, curv=pred_PopFP.mean_curv, tors=pred_PopFP.mean_tors)
        temp_FrenetPath_Q0.frenet_serret_solve()

        dist = np.zeros(n_curves)
        for i in range(n_curves):
            dist[i] = geodesic_dist(np.rollaxis(PopFrenetPath.data[i][:,:,test_index], 2), np.rollaxis(temp_FrenetPath_Q0.data[:,:,test_index], 2))

        return dist.mean()
    else:
        return 100


def objective_multiple_curve(n_splits, PopFrenetPath, curv_smoother, tors_smoother, hyperparam, alignment, lam):
    """
    Objective function in case of estimation for multiple curves that do the cross validation.
    ...
    """
    print(hyperparam)
    err = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    k = 0
    curv_smoother.reinitialize()
    tors_smoother.reinitialize()
    print('begin parallel cross val')
    start = timer()

    if alignment==True:
        Model_theta = Model(curv_smoother, tors_smoother)
        PopFrenetPath.compute_neighbors(hyperparam[0])
        SmoothPopFrenetPath0 = lie_smoother(PopFrenetPath,Model_theta)
        mKappa, mTau, mS, mOmega, gam_val, res0 = compute_raw_curvatures_alignement_init(PopFrenetPath, hyperparam[0], SmoothPopFrenetPath0, lam)
        gam = {"flag" : True, "value" : gam_val}
        if res0.nb_itr==20:
            return 100

    for train_index, test_index in kf.split(PopFrenetPath.frenet_paths[0].grid_obs):
        # print('------- step ', k, ' cross validation --------')
        if alignment==True:
            dist = step_cross_val_on_Q_multiple_curves(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath,  hyperparam, alignment, lam, gam)
        else:
            dist = step_cross_val_on_Q_multiple_curves(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath,  hyperparam, alignment, lam)

        if np.isnan(dist):
            print('nan value', k)
        else:
            err.append(dist)
        k += 1

    # if alignment==True:
    #     err = Parallel(n_jobs=10)(delayed(step_cross_val_on_Q_multiple_curves)(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath,  hyperparam, alignment, lam, gam)
    #         for train_index, test_index in kf.split(PopFrenetPath.frenet_paths[0].grid_obs))
    # else:
    #     err = Parallel(n_jobs=10)(delayed(step_cross_val_on_Q_multiple_curves)(curv_smoother, tors_smoother, test_index, train_index, PopFrenetPath,  hyperparam, alignment, lam)
    #         for train_index, test_index in kf.split(PopFrenetPath.frenet_paths[0].grid_obs))

    duration = timer() - start
    print('cross val', duration)
    return np.mean(np.array(err))