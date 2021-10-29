import numpy as np
from scipy.linalg import logm, svd, expm
import scipy.linalg
from sklearn.gaussian_process.kernels import Matern
import fdasrsf as fs
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d, UnivariateSpline
from numpy.linalg import norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
import collections
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.matrices import Matrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.backend as gs
from numba.experimental import jitclass
from numba import int32, float64, cuda, float32, objmode, njit
import torch
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.registration import ElasticRegistration, ShiftRegistration, landmark_registration_warping
from skfda.preprocessing.registration.elastic import elastic_mean
from skfda.misc import metrics
import time
import multiprocessing
import functools

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                return
        return inner
    return decorator

""" Set of mathematical functions useful """

def exp_matrix(A):
    alpha = np.linalg.norm(A,'fro')/np.sqrt(2)
    if alpha>0:
        return expm(A)
    else:
        return np.eye(len(A))

@njit
def my_log_M3(R):
    """
    Compute the matrix logarithm in R^3, with Rodrigues Formula
    ...
    """
    N = np.linalg.norm(R-np.eye(len(R)))
    if np.isnan(N) or np.isinf(N) or N<10e-6:
        return np.zeros((len(R),len(R)))
    else:
        vecA = np.zeros(3)
        c = 0.5*(np.trace(R)-1)
        if c>0:
            trR = min(c,1)
        else:
            trR = max(c,-1)
        theta = np.arccos(trR)
        if np.abs(theta)>10e-6:
            beta = theta/(2*np.sin(theta))
        else:
            beta = 0.5 * (1 + np.square(theta)/6)

        vecA[0]= -R[0,1]+R[1,0]
        vecA[1]= R[0,2]-R[2,0]
        vecA[2]= -R[1,2]+R[2,1]
        vecA = beta*vecA
        return np.array([[0, -vecA[0], vecA[1]], [vecA[0], 0, -vecA[2]], [-vecA[1], vecA[2], 0]])


# @njit
def matrix_rnd(F):
    '''
    Simulate one observation from Matrix Fisher Distribution in SO(3) parametrized
    with matrix F = K* Omega, where K is the concentration matrix and Omega is the mean direction.
    Algorithm proposed by Michael Habeck,  "Generation of three dimensional random rotations in fitting
    and matching problems", Computational Statistics, 2009.
    ...
    '''
    U,L,V = np.linalg.svd(F, hermitian=True)
    alpha,beta,gamma,_,_,_ = euler_gibbs_sampling(L,600)
    S = np.zeros((3,3))
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    cosb = np.cos(beta)
    sinb = np.sin(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)

    S[0,0] = cosa*cosb*cosg-sina*sing
    S[0,1] = sina*cosb*cosg+cosa*sing
    S[0,2] = -sinb*cosg
    S[1,0] = -cosa*cosb*sing-sina*cosg
    S[1,1] = -sina*cosb*sing+cosa*cosg
    S[1,2] = sinb*sing
    S[2,0] = cosa*sinb
    S[2,1] = sina*sinb
    S[2,2] = cosb

    R = U @ S @ V
    return R


# @njit
def euler_gibbs_sampling(Lambda,n):
    '''
    Gibbs sampling by Habeck 2009 for the simulation of random matrices
    ...
    '''
    beta = 0
    Alpha = np.zeros(n)
    Beta  = np.zeros(n)
    Gamma = np.zeros(n)
    for i in range(n):
        kappa_phi = (Lambda[0]+Lambda[1])*(np.cos(0.5*beta))**2
        kappa_psi = (Lambda[0]-Lambda[1])*(np.sin(0.5*beta))**2
        if kappa_phi < 1e-6:
            phi = 2*np.pi*np.random.random()
        else:
            phi = np.random.vonmises(0,kappa_phi)
        psi = 2*np.pi*np.random.random()
        u  = int(np.random.random()<0.5)
        alpha = (phi+psi)/2 + np.pi*u
        gamma = (phi-psi)/2 + np.pi*u
        Alpha[i] = alpha
        Gamma[i] = gamma
        kappa_beta = (Lambda[0]+Lambda[1])*np.cos(phi)+(Lambda[0]-Lambda[1])*np.cos(psi)+2*Lambda[2]
        r = np.random.random()
        x = 1+2*np.log(r+(1-r)*np.exp(-kappa_beta))/kappa_beta
        beta = np.arccos(x)
        Beta[i] = beta

    return alpha,beta,gamma,Alpha,Beta,Gamma

# @njit
def simulate_populationGP(Npop, range_intvl, n_interp, param_kern):
    """
    Simulate a population of gaussian processes
    ...
    """
    X = np.expand_dims(np.linspace(range_intvl[0], range_intvl[1], n_interp), 1)
    kernel = 1.0 * Matern(length_scale=param_kern[0], nu=param_kern[1])
    C = kernel.__call__(X, X)  # Kernel of data points
    F = []
    for j in range(Npop):
        ys = np.random.multivariate_normal(mean=np.zeros(n_interp), cov=C)
        f = interp1d(np.squeeze(X), ys)
        F.append(f)
    return np.squeeze(F)

@njit
def weighted_mean(f, weights):
    """
    Compute the weighted mean of set of functions
    ...
    """
    sum_weights = np.sum(weights, axis=1)
    N = f.shape[1]
    M = f.shape[0]
    mfw = np.zeros(M)
    for j in range(M):
        if sum_weights[j]>0:
            mfw[j] = (np.ascontiguousarray(f[j,:]) @ np.ascontiguousarray(weights[j,:]))/sum_weights[j]
        else:
            mfw[j] = 0
    return mfw

@njit
def weighted_mean_vect(f, weights):
    """
    Compute the weighted mean of a vector of functions
    ...
    """
    sum_weights = np.sum(weights, axis=1)
    n = f.shape[0]
    N = f.shape[2]
    M = f.shape[1]
    mfw = np.zeros((n,M))
    for i in range(n):
        for j in range(M):
            if sum_weights[j]>0:
                mfw[i,j] = (np.ascontiguousarray(f[i,j,:]) @ np.ascontiguousarray(weights[j,:]))/sum_weights[j]
            else:
                mfw[i,j] = 0
    return mfw

# @with_timeout(20)
def geodesic_dist(data1,data2):
    """
    Compute the geodesic distance between two rotation matrices
    ...
    """
    if data1.shape[0]==3:
        data1 = np.rollaxis(data1, 2)
    if data2.shape[0]==3:
        data2 = np.rollaxis(data2, 2)

    SO3 = SpecialOrthogonal(3)
    R = estimate_optimal_rotation(data1, data2)
    gdist = SO3.metric.dist(np.matmul(data1,R),data2)
    return np.mean(gdist)
    

def mean_geodesic_dist(PopFP1, PopFP2):
    N1 = PopFP1.nb_samples
    N2 = PopFP2.nb_samples
    if N1==N2 and N1!=1:
        out = Parallel(n_jobs=-1)(delayed(geodesic_dist)(PopFP1.frenet_paths[i].data, PopFP2.frenet_paths[i].data) for i in range(N1))
        return np.mean(out)
    elif N1==N2 and N1==1:
        return geodesic_dist(PopFP1.data,PopFP2.data)
    else:
        raise Exception('PopFP1 and PopFP2 don\'t have the same number of curves.')


def estimate_optimal_rotation(data1,data2):
    """
    Estimate the optimal rotation between two rotation matrices
    ...
    """
    SO3 = SpecialOrthogonal(3)
    R = np.matmul(np.rollaxis(np.transpose(data1), 2),data2)
    mean = FrechetMean(metric=SO3.metric, point_type='matrix')
    mean.fit(R)
    return SO3.projection(mean.estimate_)


def L2_dist(X1, X2, grid):
    """
    Compute the L² distance between two curves in R³
    ...
    """
    #alignement des centres
    X2 = X2 - fs.curve_functions.calculatecentroid(np.transpose(X2))
    #rotations
    X2_new = fs.curve_functions.find_best_rotation(np.transpose(X1), np.transpose(X2))[0]
    fd_X2 = FDataGrid(X2_new, grid)
    fd_X1 = FDataGrid(np.transpose(X1), grid)
    l2_dist = metrics.lp_distance(fd_X1, fd_X2, p=2).mean()
    return l2_dist


def FisherRao_dist(X1, X2):
    """
    Compute the Fisher-Rao distance between two curves in R³
    ...
    """
    # #alignement des centres
    # X2 = X2 - fs.curve_functions.calculatecentroid(np.transpose(X2))
    # #rotations
    # X2_new = fs.curve_functions.find_best_rotation(np.transpose(X1), np.transpose(X2))[0]
    # fd_X2 = FDataGrid(X2_new, grid)
    # fd_X1 = FDataGrid(np.transpose(X1), grid)
    # FR_dist = metrics.fisher_rao_distance(fd_X1, fd_X2).mean()
    FR_dist = fs.curve_functions.elastic_distance_curve(X1, X2, scale=True)
    return FR_dist


def centering_and_rotate(X1, X2):
    """
    Function that center X2 to its centroid and find the optimal roatation to align X2 to X1 (Proscrutes analysis)
    ...
    """
    X2 = X2 - fs.curve_functions.calculatecentroid(np.transpose(X2))
    X2_new = fs.curve_functions.find_best_rotation(np.transpose(X1), np.transpose(X2))[0]
    return X2_new.transpose()


def mean_Q0(PopFrenetPath):
    """
    Compute the mean initial condition of a population of Frenet paths
    ...
    """
    n_curves = PopFrenetPath.nb_samples
    array_Q0 = np.zeros((n_curves, 3, 3))
    SO3 = SpecialOrthogonal(3)
    for i in range(n_curves):
        array_Q0[i,:,:] = PopFrenetPath.data[i][:,:,0]
    mean = FrechetMean(metric=SO3.metric, point_type='matrix')
    mean.fit(array_Q0)
    return SO3.projection(mean.estimate_)
