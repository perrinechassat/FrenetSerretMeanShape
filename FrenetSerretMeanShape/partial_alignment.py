import numpy as np
from scipy.integrate import trapz, cumtrapz, quad
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.misc import derivative
from joblib import Parallel, delayed
import time
import collections
import optimum_reparamN2_C as orN2_C
import sys
from maths_utils import *
from frenet_path import *
from visu_utils import *
import fdasrsf as fs
from alignment_utils import *

'''___________________________________________ Partial Alignement _______________________________________________'''


def align_subparts_gam(a,b,c,d):
    f = lambda s: ((c-d)/(a-b))*(s-a) + c
    return f


def compute_n(a, D):
    # f goes from [0,1] to [a,b]
    # f_inv from [a,b] to [0,1]
    f = lambda s: (D*s + a)*(a<=(D*s + a)<=a+D) + (D+a)*((D*s + a)>a+D) + a*((D*s + a)<a)
    f_inv = lambda s: ((s-a)/D)*(0<=((s-a)/D)<=1) + 1*(((s-a)/D)>1) + 0*(((s-a)/D)<0)
    # f = lambda s: (D*s + a)
    # f_inv = lambda s: ((s-a)/D)
    return f, f_inv

def compute_f(alpha, beta):
    # f = lambda s: np.round(alpha*s + beta, decimals=8)
    # f_inv = lambda s: np.round((s-beta)/alpha, decimals=8)
    # f = lambda s: (alpha*s + beta)*(beta<=(alpha*s + beta)<=alpha+beta) + (alpha+beta)*((alpha*s + beta)>alpha+beta) + alpha*((alpha*s + beta)<alpha)
    # f_inv = lambda s: ((s-beta)/alpha)*(0<=((s-beta)/alpha)<=1) + 1*(((s-beta)/alpha)>1) + 0*(((s-beta)/alpha)<0)
    f = lambda s: (alpha*s + beta)
    f_inv = lambda s: ((s-beta)/alpha)
    return f, f_inv



def partial_align_1d_fromInt(X1, X2, init_ind2, sub_int1, sub_int2, lbda=0.0):
    a, b, c, d = sub_int1[0], sub_int1[1], sub_int2[0], sub_int2[1]
    alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
    out = partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta, lbda)
    return out


def partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta, lbda=0):
    # step 1:
    b = a+D
    f, f_inv = compute_f(alpha, beta)
    X2tilde = lambda s: alpha*X2(f(s))

    t = np.linspace(0,1,200)
    n, n_inv = compute_n(a, D)

    x1 = np.array([D*X1(n(t_)) for t_ in t])
    x2 = np.array([D*X2tilde(n(t_)) for t_ in t])
    gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    gam_inv = fs.utility_functions.invertGamma(gam)
    gamma = interp1d(t, gam, fill_value=([gam[0]], [gam[-1]]), bounds_error=False)
    gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([n(gamma(n_inv(j))) for j in np.linspace(a,b,200)]), k=5, s=0.00001)
    gamma_tilde_prime = gamma_tilde.derivative()
    gamma_tilde_inv = interp1d(np.linspace(a,b,200), np.array([n(gamma_inv(n_inv(t_))) for t_ in np.linspace(a,b,200)]), fill_value=([a], [b]), bounds_error=False)

    # step 3:
    # q1 = np.sqrt(x1)
    # q2 = np.sqrt(x2)
    # gam = fs.utility_functions.optimum_reparam(q1, t, q2, lam=lbda)
    # gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
    # gam = (gam - gam[0]) / (gam[-1] - gam[0])
    # gam_inv = fs.utility_functions.invertGamma(gam)
    # gamma = interp1d(t, gam, fill_value=([gam[0]], [gam[-1]]), bounds_error=False)
    # gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    # gam_dev = np.gradient(gam, 1/np.double(200 - 1))
    # gamma_prime = interp1d(t, gam_dev, fill_value=([gam_dev[0]], [gam_dev[-1]]), bounds_error=False)
    # step 4:
    # gamma_tilde = lambda s : n(gamma(n_inv(s)))
    # gamma_tilde_inv = lambda s : n(gamma_inv(n_inv(s)))
    # gamma_tilde_interp = UnivariateSpline(np.linspace(0,1,200), np.array([gamma(j) for j in np.linspace(0,1,200)]), k=5)
    # gamma_tilde_prime = gamma_tilde_interp.derivative()
    # gamma_tilde = lambda s : n(gamma_tilde_interp(n_inv(s)))
    # gamma_tilde_inv = lambda s : n(gamma_inv(n_inv(s)))
    # gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([n(gamma(n_inv(j))) for j in np.linspace(a,b,200)]), k=5)
    # gamma_tilde_prime = gamma_tilde_interp.derivative()
    # gamma_tilde_inv = interp1d(gamma_tilde(np.linspace(a,b,200)), np.linspace(a,b,200), fill_value=([a], [b]), bounds_error=False)

    # step 5:

    def g(s):
        if a<=s<=b:
            return gamma_tilde(s)
        elif s<a:
            return gamma_tilde_prime(a)*(s-a) + gamma_tilde(a)
        elif s>b:
            return gamma_tilde_prime(b)*(s-b) + gamma_tilde(b)

    def g_prime(s):
        if a<=s<=b:
            return gamma_tilde_prime(s)
        elif s<a:
            return gamma_tilde_prime(a)
        elif s>b:
            return gamma_tilde_prime(b)

    def g_inv(s):
        if a<=s<=b:
            return gamma_tilde_inv(s)
        elif s<a:
            return (1/gamma_tilde_prime(a))*(s-gamma_tilde(a)) + a
        elif s>b:
            return (1/gamma_tilde_prime(b))*(s-gamma_tilde(b)) + b

    A, B = g_inv(f_inv(init_ind2[0])), g_inv(f_inv(init_ind2[1]))
    g_tilde = UnivariateSpline(np.linspace(A,B,400), np.array([g(j) for j in np.linspace(A,B,400)]), k=4, s=0.001)
    g_tilde_prime = g_tilde.derivative()

    def X2new(s):
        return X2tilde(g_tilde(s))*g_tilde_prime(s)

    partial_align_results = collections.namedtuple('partial_align', ['X2new', 'A', 'B', 'g', 'g_prime', 'g_inv', 'gamma', 'gamma_inv', 'gamma_prime', 'gam'])
    out = partial_align_results(X2new, A, B, g_tilde, g_tilde_prime, g_inv, gamma_tilde, gamma_tilde_inv, gamma_tilde_prime, gam)
    return out


def extend_X(X, int):
    A,B = int[0], int[1]
    def X_tilde(s):
        if A<=s<=B:
            return X(s)
        elif s<A:
            return X(A)
        else:
            return X(B)
    return X_tilde

def H(a, D, X1, X2new, int1, int2new, ratio):
    min_int = np.min([int1[0],int2new[0]])
    max_int = np.max([int1[1],int2new[1]])
    x1_tilde = extend_X(X1, int1)
    x2_tilde = extend_X(X2new, int2new)

    # grid1 = np.linspace(min_int, a, int(abs(min_int-a)/ratio)+1) #, decimals=6)
    # grid2 = np.linspace(a, a+D, int(abs(D)/ratio)+1)
    # grid3 = np.linspace(a+D, max_int, int(abs(max_int-a-D)/ratio)+1) #, decimals=6)
    # dist = np.array([x1_tilde(grid2[i])-x2_tilde(grid2[i]) for i in range(len(grid2))])
    # dist_shape = np.trapz(dist**2, grid2)
    # # dist_shape = np.trapz(abs(dist), grid2)
    # p1 = np.array([x1_tilde(t)-x2_tilde(t) for t in grid1])
    # p2 = np.array([x1_tilde(t)-x2_tilde(t) for t in grid3])
    # pen = np.trapz(p1**2,grid1)+ np.trapz(p2**2,grid3)
    # # pen = np.trapz(abs(p1),grid1)+ np.trapz(abs(p2),grid3)
    # print('dist_shape 2', dist_shape, 'dist_shape 1', np.trapz(abs(dist), grid2), 'pen 2', pen, 'pen 1', np.trapz(abs(p1),grid1)+ np.trapz(abs(p2),grid3) )
    # # print('L1', dist_shape+pen, 'L2', np.trapz(dist**2, grid2)+np.trapz(p1**2,grid1)+ np.trapz(p2**2,grid3))
    #
    grid = np.linspace(min_int, max_int, int(abs(max_int-min_int)/ratio)+1)
    tot_dist = np.trapz(np.array([x1_tilde(grid[i])-x2_tilde(grid[i]) for i in range(len(grid))])**2, grid)
    # print('tot_dist', tot_dist)
    return tot_dist


def cost_gridsearch(X1, X2, init_ind1, init_ind2, lbda):
    def func(x):
        a, b, c, d = x[0], x[1], x[2], x[3]
        alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
        out = partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta, lbda)
        cost = H(a, D, X1, out.X2new, init_ind1, [out.A, out.B], 1/1000)
        return cost
    return func


def partial_align_1d_bis(X1, X2, init_ind2, a, D, alpha, beta, lbda=0):
    # step 1:
    b = a+D
    f, f_inv = compute_f(alpha, beta)
    X2tilde = lambda s: alpha*X2(f(s))
    # X2tilde = lambda s: X2(f(s))

    t = np.linspace(0,1,200)
    n, n_inv = compute_n(a, D)

    x1 = np.array([D*X1(n(t_)) for t_ in t])
    x2 = np.array([D*X2tilde(n(t_)) for t_ in t])
    # q1 = np.sqrt(x1)
    # q2 = np.sqrt(x2)
    # gam = fs.utility_functions.optimum_reparam(q1, t, q2, lam=lbda)
    gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])

    M = gam.size
    gam_dev = np.gradient(gam, 1 / np.double(M - 1))
    tmp = np.interp((t[-1] - t[0]) * gam + t[0], t, x2)
    x2_temp = tmp * gam_dev

    print('norm 2', np.linalg.norm(x1 - x2_temp, 2))
    print('norm 1', np.linalg.norm(x1 - x2_temp, 1))

    # gam_inv = fs.utility_functions.invertGamma(gam)
    n_gam = np.array([n(g_) for g_ in gam])
    gamma = interp1d(np.array([n(t_) for t_ in t]), n_gam, fill_value=([n_gam[0]], [n_gam[-1]]), bounds_error=False)
    # gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([gamma(j) for j in np.linspace(a,b,200)]), k=5)
    gamma_tilde_prime = gamma_tilde.derivative()
    #
    def X2new(s):
        return X2tilde(gamma(s))*gamma_tilde_prime(s)

    partial_align_results = collections.namedtuple('partial_align', ['X2new', 'gamma', 'gamma_smooth', 'gamma_prime', 'gam', 'a', 'b'])
    out = partial_align_results(X2new, gamma, gamma_tilde, gamma_tilde_prime, gam, a, b)
    # partial_align_results = collections.namedtuple('partial_align', ['gamma', 'gam'])
    # out = partial_align_results(gamma, gam)
    return out


def data_fitting_criterion(X1, X2, init_ind1, init_ind2, lbda):
    def func(x):
        a, b, c, d = x[0], x[1], x[2], x[3]
        alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
        f, f_inv = compute_f(alpha, beta)
        X2tilde = lambda s: alpha*X2(f(s))
        t = np.linspace(0,1,200)
        n, n_inv = compute_n(a, D)
        x1 = np.array([D*X1(n(t_)) for t_ in t])
        x2 = np.array([D*X2tilde(n(t_)) for t_ in t])
        gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
        gam = (gam - gam[0]) / (gam[-1] - gam[0])
        M = gam.size
        gam_dev = np.gradient(gam, 1 / np.double(M - 1))
        tmp = np.interp((t[-1] - t[0]) * gam + t[0], t, x2)
        x2_temp = tmp * gam_dev

        # print('shape dist:', np.trapz(abs(x1 - x2_temp), t), np.linalg.norm(x1 - x2_temp, 1))
        # print('weighted shape dist 2:', np.power(init_ind1[1]/(b-a), 2)*np.power(init_ind2[1]/(d-c), 2)*np.trapz(abs(x1 - x2_temp), t))
        # print('weighted shape dist 1:', np.power(init_ind1[1]/(b-a), 1)*np.power(init_ind2[1]/(d-c), 1)*np.trapz(abs(x1 - x2_temp), t))
        # print('pen:', lbda*np.trapz(abs(np.ones(200,)-gam_dev), t))
        # print('phase dist:', np.trapz(abs(np.ones(200,)-gam_dev), t))
        # print(init_ind1[1], b-a, init_ind2[1], d-c)
        # print('weights:', np.power(init_ind1[1]/(b-a), 3), np.power(init_ind2[1]/(d-c), 3))
        cost = np.power(init_ind1[1]/(b-a), 2)*np.power(init_ind2[1]/(d-c), 2)*np.trapz(abs(x1 - x2_temp), t) #+ lbda*np.trapz(abs(np.ones(200,)-gam_dev), t)
        # dy = np.trapz(abs(x1 - x2_temp), t)
        # dx = np.trapz(abs(np.ones(200,)-gam_dev), t)
        return cost
    return func

def partial_align_1d_v3(X1, X2, init_ind2, a, D, alpha, beta, lbda=0):
    # step 1:
    b = a+D
    f, f_inv = compute_f(alpha, beta)
    X2tilde = lambda s: alpha*X2(f(s))
    # X2tilde = lambda s: X2(f(s))

    t = np.linspace(0,1,200)
    n, n_inv = compute_n(a, D)

    x1 = np.array([D*X1(n(t_)) for t_ in t])
    x2 = np.array([D*X2tilde(n(t_)) for t_ in t])
    gam = optimum_reparam_1d(x2, t, x1, lam=lbda)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    # gam_inv = fs.utility_functions.invertGamma(gam)
    n_gam = np.array([n(g_) for g_ in gam])
    gamma = interp1d(np.array([n(t_) for t_ in t]), n_gam, fill_value=([n_gam[0]], [n_gam[-1]]), bounds_error=False)
    # gamma_inv = interp1d(t, gam_inv, fill_value=([gam_inv[0]], [gam_inv[-1]]), bounds_error=False)
    gamma_tilde = UnivariateSpline(np.linspace(a,b,200), np.array([gamma(j) for j in np.linspace(a,b,200)]), k=5)
    gamma_tilde_prime = gamma_tilde.derivative()
    #
    def X1new(s):
        return X1(gamma_tilde(s))*gamma_tilde_prime(s)

    partial_align_results = collections.namedtuple('partial_align', ['X1new', 'X2new', 'gamma', 'gamma_smooth', 'gamma_prime', 'gam', 'a', 'b'])
    out = partial_align_results(X1new, X2tilde, gamma, gamma_tilde, gamma_tilde_prime, gam, a, b)
    # partial_align_results = collections.namedtuple('partial_align', ['gamma', 'gam'])
    # out = partial_align_results(gamma, gam)

    return out


def cost_bis(X1, X2, init_ind1, init_ind2, lbda):
    def func(x):
        a, b, c, d = x[0], x[1], x[2], x[3]
        alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
        print(alpha, beta)
        out = partial_align_1d_bis(X1, X2, init_ind2, a, D, alpha, beta, lbda)
        grid = np.linspace(a,b,200)
        cost = np.power(init_ind1[1]/(b-a), 3)*np.power(init_ind2[1]/(d-c), 3)*np.trapz((X1(grid)-out.X2new(grid))**2, grid) + lbda*np.trapz((np.ones(200,)-out.gamma_prime(grid))**2, grid)
        # cost = np.power(init_ind1[1]/(b-a), 1)*np.power(init_ind2[1]/(d-c), 1)*np.trapz((X1(grid)-out.X2new(grid))**2, grid) + lbda*np.trapz((np.ones(200,)-out.gamma_prime(grid))**2, grid)
        # cost = (1 - abs((b-a)/init_ind1[1]))*(1 - abs((d-c)/init_ind2[1]))*np.trapz((X1(grid)-out.X2new(grid))**2, grid) + lbda*np.trapz((np.ones(200,)-out.gamma_prime(grid))**2, grid)
        print('shape dist:', np.trapz((X1(grid)-out.X2new(grid))**2, grid))
        print('weighted shape dist:', np.power(init_ind1[1]/(b-a), 3)*np.power(init_ind2[1]/(d-c), 3)*np.trapz((X1(grid)-out.X2new(grid))**2, grid))
        print('pen:', lbda*np.trapz((np.ones(200,)-out.gamma_prime(grid))**2, grid))
        print(init_ind1[1], b-a, init_ind2[1], d-c)
        print('weights:', np.power(init_ind1[1]/(b-a), 3), np.power(init_ind2[1]/(d-c), 3))
        return cost
    return func

def make_grid(list1, list2, dist=0):
    c1 = np.concatenate(np.stack(np.meshgrid(list1,list1), axis=-1))
    L1 =  []
    for x in c1:
        if x[0]<x[1]:
            if x[1]-x[0]>=dist:
                L1.append(x)
    L1 = np.unique(L1, axis=0)
    c2 = np.concatenate(np.stack(np.meshgrid(list2,list2), axis=-1))
    L2 =  []
    for x in c2:
        if x[0]<x[1]:
            if x[1]-x[0]>=dist:
                L2.append(x)
    L2 = np.unique(L2, axis=0)
    L_final = np.empty((len(L1), len(L2)), dtype=object)
    for i in range(len(L1)):
        for j in range(len(L2)):
            L_final[i,j] = np.concatenate([L1[i], L2[j]])
    return np.concatenate(L_final)



def estim_gam_from_mu(mu, X1, X2, lbda=0, n_disc_pts=200):

    a, b, c, d = mu[0], mu[1], mu[2], mu[3]
    alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
    f, f_inv = compute_f(alpha, beta)
    X2tilde = lambda s: alpha*X2(f(s))

    n, n_inv = compute_n(a, D)
    t = np.linspace(0,1,n_disc_pts)

    x1 = np.array([D*X1(n(t_)) for t_ in t])
    x2 = np.array([D*X2tilde(n(t_)) for t_ in t])

    gam = optimum_reparam_1d(x1, t, x2, lam=lbda)
    gam = (gam - gam[0]) / (gam[-1] - gam[0])
    gam_inv = fs.utility_functions.invertGamma(gam)

    n_gam = np.array([n(g_) for g_ in gam])
    gamma = interp1d(np.array([n(t_) for t_ in t]), n_gam, fill_value=([n_gam[0]], [n_gam[-1]]), bounds_error=False)
    n_gam_inv = np.array([n(g_) for g_ in gam_inv])
    gamma_inv = interp1d(np.array([n(t_) for t_ in t]), n_gam_inv, fill_value=([n_gam_inv[0]], [n_gam_inv[-1]]), bounds_error=False)

    gam_from_mu = collections.namedtuple('gam_from_mu', ['gam', 'gam_inv', 'gamma', 'gamma_inv', 'f', 'f_inv', 'n', 'n_inv', 'a', 'b', 'alpha', 'beta', 'c', 'd'])
    out = gam_from_mu(gam, gam_inv, gamma, gamma_inv, f, f_inv, n, n_inv, a, b, alpha, beta, c, d)
    return out


def extrapolation_gam(gam, gam_inv, mu, init_ind2, n_pts=400):
    a, b, c, d = mu[0], mu[1], mu[2], mu[3]
    alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
    n, n_inv = compute_n(a, b-a)
    f, f_inv = compute_f(alpha, beta)
    n_pts_gam = len(gam)
    gamma_tilde = UnivariateSpline(np.linspace(a,b,n_pts_gam), np.array([n(j) for j in gam]), k=5, s=0.00001)
    gamma_tilde_prime = gamma_tilde.derivative()
    gamma_tilde_inv = interp1d(np.linspace(a,b,n_pts_gam), np.array([n(j) for j in gam_inv]), fill_value=([a], [b]), bounds_error=False)

    def g(s):
        if a<=s<=b:
            return gamma_tilde(s)
        elif s<a:
            return gamma_tilde_prime(a)*(s-a) + gamma_tilde(a)
        elif s>b:
            return gamma_tilde_prime(b)*(s-b) + gamma_tilde(b)

    def g_prime(s):
        if a<=s<=b:
            return gamma_tilde_prime(s)
        elif s<a:
            return gamma_tilde_prime(a)
        elif s>b:
            return gamma_tilde_prime(b)

    def g_inv(s):
        if a<=s<=b:
            return gamma_tilde_inv(s)
        elif s<a:
            return (1/gamma_tilde_prime(a))*(s-gamma_tilde(a)) + a
        elif s>b:
            return (1/gamma_tilde_prime(b))*(s-gamma_tilde(b)) + b

    A, B = g_inv(f_inv(init_ind2[0])), g_inv(f_inv(init_ind2[1]))
    g_tilde = UnivariateSpline(np.linspace(A,B,n_pts), np.array([g(j) for j in np.linspace(A,B,n_pts)]), k=4, s=0.001)
    g_tilde_prime = g_tilde.derivative()
    g_tilde_inv = UnivariateSpline(np.linspace(init_ind2[0], init_ind2[1],n_pts), np.array([g_inv(f_inv(t_)) for t_ in np.linspace(init_ind2[0], init_ind2[1],n_pts)]), k=4, s=0.001)

    return g_tilde, g_tilde_prime, g_tilde_inv


def estim_h_extrapolation(X, init_ind, g_inv_tab, n_pts=400):

    grid = np.linspace(init_ind[0], init_ind[1], n_pts)
    h_inv = UnivariateSpline(grid, np.mean(g_inv_tab, axis=0), k=4, s=0.001)
    h = UnivariateSpline(h_inv(grid), grid, k=4, s=0.001)
    h_prime = h.derivative()
    h_grid = np.linspace(h_inv(grid[0]), h_inv(grid[-1]), n_pts)
    h_bounds = np.array([h_inv(grid[0]), h_inv(grid[-1])])
    new_X_arr = X(h(h_grid))*h_prime(h_grid)
    new_X = interp1d(h_grid, new_X_arr)

    opt_warp_extrapolation = collections.namedtuple('opt_warp_extrapolation', ['new_X', 'new_X_arr', 'h', 'h_inv', 'h_prime', 'h_grid', 'h_bounds'])
    out = opt_warp_extrapolation(new_X, new_X_arr, h, h_inv, h_prime, h_grid, h_bounds)
    return out


def estim_h_partial_mean(X, init_ind, g_inv_tab, alphabeta_tab, ab_tab, n_pts=400):

    grid = np.linspace(init_ind[0], init_ind[1], n_pts)
    h_arr = mean_partial_samples(g_inv_tab, ab_tab, grid)
    alpha_mean = np.mean(alphabeta_tab[:,0])
    beta_mean = np.mean(alphabeta_tab[:,1])
    f_mean, f_inv_mean = compute_f(alpha_mean, beta_mean)

    h = UnivariateSpline(grid, h_arr, k=4, s=0.001)
    h_prime = h.derivative()
    h_grid = f_mean(grid)
    h_bounds = np.array([f_mean(grid[0]), f_mean(grid[-1])])
    new_X_arr = X(h(grid))*h_prime(grid)*(1/alpha_mean)
    new_X = interp1d(h_grid, new_X_arr)

    opt_warp_partial_mean = collections.namedtuple('opt_warp_partial_mean', ['new_X', 'new_X_arr', 'h', 'h_prime', 'h_grid', 'h_bounds'])
    out = opt_warp_partial_mean(new_X, new_X_arr, h, h_prime, h_grid, h_bounds)
    return out


def estim_h_partial_mean_bis(X, init_ind, g_tab, alphabeta_tab, cd_tab, n_pts=400):

    grid = np.linspace(init_ind[0], init_ind[1], n_pts)
    N = len(g_tab)
    g_tab_func = np.empty((N), dtype=object)
    for i in range(N):
        n, n_inv  = compute_n(cd_tab[i][0], cd_tab[i][1]-cd_tab[i][0])
        n_gam = np.array([n(g_) for g_ in g_tab[i]])
        g_tab_func[i] = interp1d(np.array([n(t_) for t_ in np.linspace(0,1,int(n_pts/2))]), n_gam, fill_value=([n_gam[0]], [n_gam[-1]]), bounds_error=False)
    h_arr = mean_partial_samples(g_tab_func, cd_tab, grid)
    alpha_mean = np.mean(alphabeta_tab[:,0])
    beta_mean = np.mean(alphabeta_tab[:,1])
    f_mean, f_inv_mean = compute_f(alpha_mean, beta_mean)

    h = UnivariateSpline(grid, h_arr, k=4, s=0.01)
    h_prime = h.derivative()
    h_grid = f_inv_mean(grid)
    h_bounds = np.array([f_inv_mean(grid[0]), f_inv_mean(grid[-1])])
    new_X_arr = X(h(grid))*h_prime(grid)*(alpha_mean)
    new_X = interp1d(h_grid, new_X_arr)

    opt_warp_partial_mean = collections.namedtuple('opt_warp_partial_mean', ['new_X', 'new_X_arr', 'h', 'h_prime', 'h_grid', 'h_bounds'])
    out = opt_warp_partial_mean(new_X, new_X_arr, h, h_prime, h_grid, h_bounds)
    return out



def partial_mean_extrapolation(func_curve, tab_init_bounds, tab_mu, lbda, n_pts):
    N = func_curve.shape[0]
    g_inv_tab = np.empty((N,N), dtype=object)
    h_bounds = np.zeros((N,2))
    new_func = np.empty((N), dtype=object)
    h_tab = np.empty((N), dtype=object)
    h_prime_tab = np.empty((N), dtype=object)
    for i in range(N):
        for j in range(N):
            out_gam = estim_gam_from_mu(mu[j][i], func_curve[j], func_curve[i], lbda=lbda, n_disc_pts=int(n_pts/2))
            g_tilde, g_tilde_prime, g_tilde_inv = extrapolation_gam(out_gam.gam, out_gam.gam_inv, tab_mu[j][i], tab_init_bounds[i], n_pts=n_pts)
            g_inv_tab[i][j] = g_tilde_inv(np.linspace(tab_init_bounds[i][0], tab_init_bounds[i][-1], n_pts))
        out = estim_h_extrapolation(func_curve[i], tab_init_bounds[i], g_inv_tab[i], n_pts)
        new_func[i] = out.new_X
        h_bounds[i] = out.h_bounds
        h_tab[i] = out.h
        h_prime_tab[i] = out.h_prime

    x_mean = np.linspace(np.min(h_bounds), np.max(h_bounds), n_pts)
    y_mean = mean_partial_samples(new_func, h_bounds, x_mean)
    return x_mean, y_mean, h_tab, h_prime_tab



# def partial_mean_mean_warp(func_curve, tab_init_bounds, tab_mu, lbda, n_pts):
#     N = func_curve.shape[0]
#     g_inv_tab = np.empty((N,N), dtype=object)
#     h_bounds = np.zeros((N,2))
#     new_func = np.empty((N), dtype=object)
#     h_tab = np.empty((N), dtype=object)
#     for i in range(N):
#         for j in range(N):
#             out_gam = estim_gam_from_mu(mu[j][i], func_curve[j], func_curve[i], lbda=lbda, n_disc_pts=int(n_pts/2))
#             g_tilde, g_tilde_prime, g_tilde_inv = extrapolation_gam(out_gam.gam, out_gam.gam_inv, tab_mu[j][i], tab_init_bounds[i], n_pts=n_pts)
#             g_inv_tab[i][j] = g_tilde_inv(np.linspace(tab_init_bounds[i][0], tab_init_bounds[i][-1], n_pts))
#         out = estim_h_extrapolation(func_curve[i], tab_init_bounds[i], g_inv_tab[i], n_pts)
#         new_func[i] = out.new_X
#         h_bounds[i] = out.h_bounds
#         h_tab[i] = out.h
#         h_prime_tab[i] = out.h_prime
#
#     x_mean = np.linspace(np.min(h_bounds), np.max(h_bounds), n_pts)
#     y_mean = mean_partial_samples(new_func, h_bounds, x_mean)
#     return x_mean, y_mean, h_tab, h_prime_tab




def partial_alignement_v1(func_curve, tab_init_bounds, tab_param):

    N = func_curve.shape[0]
    res_align = np.empty((N,N), dtype=object)
    f_array = np.empty((N,N), dtype=object)
    grid = np.empty((N,N), dtype=object)

    g_inv = np.zeros((N,N,400))
    g_inv_mean = np.empty((N), dtype=object)
    g_mean = np.empty((N), dtype=object)
    g_prime_mean = np.empty((N), dtype=object)
    mean_grid = np.empty((N), dtype=object)
    mean_bounds = np.zeros((N,2))
    mean_func = np.empty((N), dtype=object)
    arr_mean_f = np.empty((N), dtype=object)

    for i in range(N):
        grid_i = np.linspace(0,tab_init_bounds[i][1],400)

        for j in range(N):
            res_align[i][j] = partial_align_1d_fromInt(func_curve[j], func_curve[i], tab_init_bounds[i], tab_param[i][j][:2], tab_param[i][j][2:], lbda=0.01)
            grid[i][j] = np.linspace(res_align[i][j].A, res_align[i][j].B, 400)
            f_array[i][j] = np.array([res_align[i][j].X2new(t) for t in grid[i][j]])

            a, b, c, d = tab_param[i][j][0], tab_param[i][j][1], tab_param[i][j][2], tab_param[i][j][3]
            alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
            f, f_inv = compute_f(alpha, beta)
            g_inv[i][j] = UnivariateSpline(grid_i, np.array([res_align[i][j].g_inv(f_inv(t_)) for t_ in grid_i]), k=4, s=0.001)(grid_i)

        g_inv_mean[i] = UnivariateSpline(grid_i, np.mean(g_inv[i], axis=0), k=4, s=0.001)
        g_mean[i] = UnivariateSpline(g_inv_mean[i](grid_i), grid_i, k=4, s=0.001)
        g_prime_mean[i] = g_mean[i].derivative()
        mean_grid[i] = np.linspace(g_inv_mean[i](grid_i[0]), g_inv_mean[i](grid_i[-1]),400)
        mean_bounds[i] = np.array([g_inv_mean[i](grid_i[0]), g_inv_mean[i](grid_i[-1])])
        # mean_func[i] = interp1d(mean_grid[i], func_curve[i](g_mean[i](mean_grid[i]))*g_prime_mean[i](mean_grid[i]))
        arr_mean_f[i] = func_curve[i](g_mean[i](mean_grid[i]))*g_prime_mean[i](mean_grid[i])
        mean_func[i] = interp1d(mean_grid[i], arr_mean_f[i])


    final_grid_mean = np.linspace(np.min(mean_bounds), np.max(mean_bounds), 400)
    partial_mean = mean_partial_samples(mean_func, mean_bounds, final_grid_mean)

    partial_mean_results = collections.namedtuple('partial_mean', ['res_align', 'func_align', 'f_align', 'grid_align', 'mean', 'grid_mean', 'h', 'h_prime'])
    out = partial_mean_results(res_align, mean_func, arr_mean_f, mean_grid, partial_mean, final_grid_mean, g_mean, g_prime_mean)
    return out



def partial_alignement_v2(func_curve, tab_init_bounds, tab_param):

    N = func_curve.shape[0]
    res_align = np.empty((N,N), dtype=object)

    alpha_tab = np.zeros(N)
    beta_tab = np.zeros(N)
    f_tab = np.empty((N), dtype=object)
    f_inv_tab = np.empty((N), dtype=object)
    gam_tab = np.empty((N,N), dtype=object)
    h_tab = np.empty((N), dtype=object)
    mean_grid = np.empty((N), dtype=object)
    mean_bounds = np.zeros((N,2))
    mean_func = np.empty((N), dtype=object)
    arr_mean_f = np.empty((N), dtype=object)
    g_mean = np.empty((N), dtype=object)
    g_prime_mean = np.empty((N), dtype=object)

    for i in range(N):
        grid_i = np.linspace(0,tab_init_bounds[i][1],400)
        for j in range(N):
            a, b, c, d = tab_param[j][i][0], tab_param[j][i][1], tab_param[j][i][2], tab_param[j][i][3]
            alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
            res_align[i][j] = partial_align_1d_v3(func_curve[i], func_curve[j], tab_init_bounds[j], a, D, alpha, beta, lbda=0.01)
            alpha_tab[i] += alpha
            beta_tab[i] += beta
            gam_tab[i][j] = res_align[i][j].gamma
        alpha_tab[i] = alpha_tab[i]/N
        beta_tab[i] = beta_tab[i]/N
        f_tab[i], f_inv_tab[i] = compute_f(alpha_tab[i], beta_tab[i])
        h_tab[i] = mean_partial_samples(gam_tab[i], np.array([tab_param[j][i][:2] for j in range(N)]), grid_i)

        g_mean[i] = UnivariateSpline(grid_i, h_tab[i], k=4, s=0.001)
        g_prime_mean[i] = g_mean[i].derivative()

        mean_grid[i] = f_tab[i](grid_i)
        mean_bounds[i] = np.array([f_tab[i](grid_i[0]), f_tab[i](grid_i[-1])])
        arr_mean_f[i] = func_curve[i](g_mean[i](grid_i))*g_prime_mean[i](grid_i)*(alpha_tab[i])
        mean_func[i] = interp1d(mean_grid[i], arr_mean_f[i])

    final_grid_mean = np.linspace(np.min(mean_bounds), np.max(mean_bounds), 400)
    partial_mean = mean_partial_samples(mean_func, mean_bounds, final_grid_mean)

    partial_mean_results = collections.namedtuple('partial_mean', ['res_align', 'func_align', 'f_align', 'grid_align', 'mean', 'grid_mean', 'h', 'h_prime', 'gam_tab'])
    out = partial_mean_results(res_align, mean_func, arr_mean_f, mean_grid, partial_mean, final_grid_mean, g_mean, g_prime_mean, gam_tab)
    return out



# ''' Gradient descent '''
#
# # def param_to_p(a, b, c, d, gamma_prime, int1):
# #     alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
# #     p = np.empty((5), dtype=object)
# #     p[0], p[1], p[2], p[3] = a, np.log(D/int1[1]), np.log(alpha), beta
# #     gam = np.sqrt(gamma_prime(np.linspace(0,1,200)))
# #     # p[4] = (gam - gam[0]) / (gam[-1] - gam[0])
# #     p[4] = gam
# #     return p
#
# def param_to_p(a, b, c, d, gam, int1):
#     alpha, beta, D = ((d-c)/(b-a)), c-((d-c)/(b-a))*a, b-a
#     p = np.empty((5), dtype=object)
#     p[0], p[1], p[2], p[3] = a, np.log(D/int1[1]), np.log(alpha), beta
#     psi = np.sqrt(np.gradient(gam))
#     # p[4] = (gam - gam[0]) / (gam[-1] - gam[0])
#     p[4] = psi
#     return p
#
# def p_to_param(p, init_ind1):
#     a, D, alpha, beta, psi = p[0], init_ind1[1]*np.exp(p[1]), np.exp(p[2]), p[3], p[4]
#     b = a+D
#     d = beta + alpha*b
#     c = d - alpha*D
#     return a, b, c, d
#
# def compose_Xk_p(p, Xk, indk, init_ind1):
#     ''' p = (a, delta, mu, beta, psi) '''
#     a, D, alpha, beta, psi = p[0], init_ind1[1]*np.exp(p[1]), np.exp(p[2]), p[3], p[4]
#     b = a+D
#     f, f_inv = compute_f(alpha, beta)
#     n, n_inv = compute_n(a, D)
#
#     time = np.linspace(0,1,200)
#     gam_prime = psi**2
#     # gam_prime = (gam_prime - gam_prime[0]) / (gam_prime[-1] - gam_prime[0])
#     gamma_prime = interp1d(time, gam_prime)
#     gam = cumtrapz(psi**2, time, initial=0)
#     gam = (gam - gam[0]) / (gam[-1] - gam[0])
#     gamma = interp1d(time, gam)
#
#     # step 4:
#     gamma_tilde = lambda s : n(gamma(n_inv(s)))
#     gamma_tilde_interp = UnivariateSpline(np.linspace(a,b,200), gamma_tilde(np.linspace(a,b,200)), k=3)
#     gamma_tilde_prime = gamma_tilde_interp.derivative()
#     # step 5:
#     def g(s):
#         if a<=s<=b:
#             return f(gamma_tilde(s))
#         else:
#             return f(s)
#     def g_prime(s):
#         if a<=s<=b:
#             return alpha*gamma_tilde_prime(s)
#         else:
#             return alpha
#     A, B = f_inv(indk[0]), f_inv(indk[1])
#     def Xknew(s):
#         return Xk(g(s))*g_prime(s)
#
#     compose_res = collections.namedtuple('compose_results', ['Xnew', 'A', 'B', 'g', 'g_prime', 'params'])
#     out = compose_res(Xknew, A, B, g, g_prime, [a, D, alpha, beta])
#     return out
#
#
# def compute_Ek(p, X1, X2, init_ind1, init_ind2, lbda, ratio):
#     out = compose_Xk_p(p, X2, init_ind2, init_ind1)
#     X2new = out.Xnew
#     A, B = out.A, out.B
#     # print(A, B)
#     cost = E(out.params[0], out.params[1], X1, X2new, init_ind1, [A,B], ratio)
#     Ek_results = collections.namedtuple('Ek_results', ['X2new', 'A', 'B', 'g', 'g_prime', 'cost'])
#     out = Ek_results(X2new, A, B, out.g, out.g_prime, cost)
#     return out
#
#
# def compute_Ek_bis(p, X1, X2, init_ind1, init_ind2, lbda, ratio):
#     a, D, alpha, beta, psi = p[0], init_ind1[1]*np.exp(p[1]), np.exp(p[2]), p[3], p[4]
#     out = partial_align_1d(X1, X2, init_ind2, a, D, alpha, beta)
#     X2new = out.X2new
#     A, B = out.A, out.B
#     # print(A, B)
#     cost = E(a, D, X1, X2new, init_ind1, [A,B], ratio)
#     Ek_results = collections.namedtuple('Ek_results', ['X2new', 'A', 'B', 'g', 'g_prime', 'cost'])
#     out = Ek_results(X2new, A, B, out.g, out.g_prime, cost)
#     return out
#
#
# def compute_dev(f, time, smooth):
#     if smooth:
#         spar = time.shape[0] * (.025 * fabs(np.array([f(t) for t in time])).max()) ** 2
#     else:
#         spar = 0
#     tmp_spline = UnivariateSpline(time, np.array([f(t) for t in time]), s=spar)
#     return tmp_spline.derivative()

# __________________________________________________________________________________________________________________________________________________________________________

# def compute_grad_Ek_id(X1, X2_k, init_ind1, int2_k, lbda):
#     L1 = init_ind1[1]
#     X2_k_dev = compute_dev(X2_k, np.linspace(int2_k[0], int2_k[1], int(abs(int2_k[1]-int2_k[0])/(1/1000))+1), False)
#     # X2_k_dev = lambda s: derivative(X2_k, s, dx=0.0001)
#     # print(X2_k_dev(int2_k[0]+0.0001))
#     x1_tilde = extend_X(X1, init_ind1)
#     x2_tilde = extend_X(X2_k, int2_k)
#     x2_dev_tilde = extend_X(X2_k_dev, int2_k)
#
#     DEk_a = (lbda-1)*abs(x1_tilde(0)-x2_tilde(0)) + (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1
#     DEk_delta = (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1
#
#     def f_xi(s):
#         return -np.sign(x1_tilde(s)-x2_tilde(s))*(s*x2_dev_tilde(s)+x2_tilde(s))
#     DEk_xi = quad(f_xi, 0, L1, epsabs=1e-4, limit=100)[0] + lbda*(quad(f_xi, -np.inf, 0, epsabs=1e-4, limit=100)[0] + quad(f_xi, L1, np.inf, epsabs=1e-4, limit=100)[0])
#
#     def f_beta(s):
#         return -np.sign(x1_tilde(s)-x2_tilde(s))*x2_dev_tilde(s)
#     DEk_beta = quad(f_beta, 0, L1, epsabs=1e-4, limit=100)[0] + lbda*(quad(f_beta, -np.inf, 0, epsabs=1e-4, limit=100)[0] + quad(f_beta, L1, np.inf, epsabs=1e-4, limit=100)[0])
#
#     print(DEk_a, DEk_delta, DEk_xi, DEk_beta)
#
#
#     def w_k(x):
#         return 2*L1*quad(f_beta, 0, L1*x, epsabs=1e-4, limit=100)[0] - 2*L1*np.sign(x1_tilde(L1*x)-x2_tilde(L1*x))*x2_tilde(L1*x)
#     # w_k = lambda x: 2*L1*quad(f_beta, 0, L1*x)[0] - 2*L1*np.sign(x1_tilde(L1*x)-x2_tilde(L1*x))*x2_tilde(L1*x)
#     int_w_k = quad(w_k, 0, 1, epsabs=1e-4, limit=100)[0]
#     print(int_w_k)
#     DEk_psi = lambda x: w_k(x) + int_w_k
#     scalprod_psi = quad(lambda s: DEk_psi(s)**2, 0, 1, epsabs=1e-4, limit=100)[0]
#
#     norm = np.sqrt(DEk_a**2 + DEk_delta**2 + DEk_xi**2 + DEk_beta**2 + scalprod_psi)
#
#     res = np.empty((5), dtype=object)
#     res[0], res[1], res[2], res[3] = DEk_a, DEk_delta, DEk_xi, DEk_beta
#     res[4] = DEk_psi
#     return res, norm, scalprod_psi

# __________________________________________________________________________________________________________________________________________________________________________
#
# def compute_grad_Ek_id(X1, X2_k, init_ind1, int2_k, lbda):
#     ratio = 1/1000
#     L1 = init_ind1[1]
#     min_int = np.min([init_ind1[0],int2_k[0]])
#     max_int = np.max([init_ind1[1],int2_k[1]])
#
#     X2_k_dev = compute_dev(X2_k, np.linspace(int2_k[0], int2_k[1], int(abs(int2_k[1]-int2_k[0])/ratio)+1), False)
#     x1_tilde = extend_X(X1, init_ind1)
#     x2_tilde = extend_X(X2_k, int2_k)
#     x2_dev_tilde = extend_X(X2_k_dev, int2_k)
#
#     DEk_a = (lbda-1)*abs(x1_tilde(0)-x2_tilde(0)) + (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1
#     DEk_delta = (1-lbda)*abs(x1_tilde(L1)-x2_tilde(L1))*L1
#
#     f_xi = lambda s: -np.sign(x1_tilde(s)-x2_tilde(s))*(s*x2_dev_tilde(s)+x2_tilde(s))
#     f_beta = lambda s: -np.sign(x1_tilde(s)-x2_tilde(s))*x2_dev_tilde(s)
#
#     grid2 = np.linspace(init_ind1[0], L1, int(abs(L1-init_ind1[0])/ratio)+1)
#     DEk_xi = np.trapz(y=np.array([f_xi(t) for t in grid2]), x=grid2)
#     DEk_beta = np.trapz(y=np.array([f_beta(t) for t in grid2]), x=grid2)
#     if min_int < init_ind1[0]:
#         grid1 = np.linspace(min_int, init_ind1[0], int(abs(init_ind1[0]-min_int)/ratio)+1)
#         DEk_xi += lbda*np.trapz(y=np.array([f_xi(t) for t in grid1]), x=grid1)
#         DEk_beta += lbda*np.trapz(y=np.array([f_beta(t) for t in grid1]), x=grid1)
#     if max_int > L1:
#         grid3 = np.linspace(L1, max_int, int(abs(max_int-L1)/ratio)+1)
#         DEk_xi += lbda*np.trapz(y=np.array([f_xi(t) for t in grid3]), x=grid3)
#         DEk_beta += lbda*np.trapz(y=np.array([f_beta(t) for t in grid3]), x=grid3)
#     # print(DEk_a, DEk_delta, DEk_xi, DEk_beta)
#
#     time = np.linspace(0, 1, 200)
#     grid = np.linspace(0, L1, 200)
#     y = np.array([f_beta(t) for t in grid])
#     int_x_wk = cumtrapz(y, grid, initial=0)
#     wk = np.array([(2*L1*int_x_wk[i] - 2*L1*np.sign(x1_tilde(L1*time[i])-x2_tilde(L1*time[i]))*x2_tilde(L1*time[i])) for i in range(len(time))])
#     int_w_k = np.trapz(wk, x=time)
#     DEk_psi = wk + int_w_k
#
#     norm = np.sqrt(DEk_a**2 + DEk_delta**2 + DEk_xi**2 + DEk_beta**2 + np.trapz(DEk_psi**2, x=time))
#
#     res = np.empty((5), dtype=object)
#     res[0], res[1], res[2], res[3] = DEk_a, DEk_delta, DEk_xi, DEk_beta
#     res[4] = DEk_psi
#     return res, norm
#
#
# def update_p_id(v, delta):
#     res = np.empty((5), dtype=object)
#     res[0], res[1], res[2], res[3] = -delta*v[0], -delta*v[1], -delta*v[2], -delta*v[3]
#     norm_v4 = np.sqrt(np.trapz(v[4]**2, np.linspace(0,1,200)))
#     res[4] = -delta*(np.sin(norm_v4)/norm_v4)*v[4] - delta*np.cos(norm_v4)
#     return res
#
# def group_action_psi(psi1, psi2):
#     time = np.linspace(0,1,200)
#     int_p2 = cumtrapz(psi2**2, time, initial=0)
#     int_p2 = (int_p2 - int_p2[0]) / (int_p2[-1] - int_p2[0])
#     p1_temp = np.interp((time[-1] - time[0]) * int_p2 + time[0], time, psi1)
#     return p1_temp*psi2
#
# def group_operation_P(p1, p2):
#     res = np.empty((5), dtype=object)
#     res[0], res[1], res[2], res[3] = p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]
#     # res[4] = lambda t: p1[4](quad(lambda s: p2[4](s)**2, 0, t, limit=100)[0])*p2[4](t)
#     res[4] = group_action_psi(p1[4], p2[4])
#     return res
#
# __________________________________________________________________________________________________________________________________________________________________________

# def gradient_descent_E(X1, X2_0, ind1, ind2_0, p0, beta, tau, step_size, tol, max_iter, lbda):
#     k = 0
#     p_id = np.empty((5), dtype=object)
#     p_id[0], p_id[1], p_id[2], p_id[3] = 0, 0, 0, 0
#     p_id[4] = np.ones((200))
#     print('pid', p_to_param(p_id, ind1))
#     X2k = X2_0
#     pk = p0
#     print('p0', p_to_param(pk, ind1))
#     ind2k = ind2_0
#     out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#     Ek_id = out_Ek.cost
#     print('Cost 0', Ek_id)
#     DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#     print('Norm grad 0', norm_k)
#     print("grad :", DEk_id[0], DEk_id[1], DEk_id[2], DEk_id[3])
#     while norm_k>tol and k<max_iter:
#         delta_k = step_size
#         p = update_p_id(DEk_id, delta_k)
#         print('update', p_to_param(p, ind1))
#         print('new p', p_to_param(group_operation_P(pk, p), ind1))
#         out_Ek_p = compute_Ek(p, X1, X2k, ind1, ind2k, lbda, 1/1000)
#         Ek_p = out_Ek_p.cost
#         print('Cost p ',k, Ek_p)
#         # psik_psi = np.array([pk[4](quad(lambda s: p[4](s)**2, 0, t, limit=100)[0])*p[4](t) for t in np.linspace(0,1,200)])
#         psik_psi = group_action_psi(pk[4], p[4])
#         while Ek_p > Ek_id - beta*step_size*norm_k or (psik_psi<0).any():
#             print(Ek_p, Ek_id - beta*step_size*norm_k, (psik_psi<0).any())
#             delta_k = tau*delta_k
#             p = update_p_id(DEk_id, delta_k)
#             print(p_to_param(p, ind1))
#             out_Ek_p = compute_Ek(p, X1, X2k, ind1, ind2k, lbda, 1/1000)
#             Ek_p = out_Ek_p.cost
#             psik_psi = group_action_psi(pk[4], p[4])
#
#         pk = group_operation_P(pk, p)
#         out_Xk = compose_Xk_p(p, X2k, ind2k, ind1)
#         X2k = out_Xk.Xnew
#         ind2k = [out_Xk.A, out_Xk.B]
#         out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#         Ek_id = out_Ek.cost
#         DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#         k = k+1
#         print(k)
#
#     grad_desc = collections.namedtuple('Gradient_descent_results', ['X2new', 'int2new', 'p_opt', 'g', 'g_prime', 'cost'])
#     out = grad_desc(X2k, ind2k, pk, out_Xk.g, out_Xk.g_prime, Ek_id)
#     return out

# def gradient_descent_E(X1, X2, ind1, ind2, p0, beta, tau, step_size, tol, max_iter, lbda):
#     k = 0
#     p_id = np.empty((5), dtype=object)
#     p_id[0], p_id[1], p_id[2], p_id[3] = 0, 0, 0, 0
#     p_id[4] = np.ones((200))
#     print('pid', p_to_param(p_id, ind1))
#     pk = p0
#     print('p0', p_to_param(pk, ind1))
#     out_Xk = compose_Xk_p(pk, X2, ind2, ind1)
#     X2k = out_Xk.Xnew
#     ind2k = [out_Xk.A, out_Xk.B]
#     out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#     Ek_id = out_Ek.cost
#     print('Cost 0', Ek_id)
#     DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#     print('Norm grad 0', norm_k)
#     print("grad :", DEk_id[0], DEk_id[1], DEk_id[2], DEk_id[3])
#     while norm_k>tol and k<max_iter:
#         delta_k = step_size
#         p = group_operation_P(pk, update_p_id(DEk_id, delta_k))
#         print('update', p_to_param(update_p_id(DEk_id, delta_k), ind1))
#         print('new p', p_to_param(group_operation_P(pk, update_p_id(DEk_id, delta_k)), ind1))
#         out_Ek_p = compute_Ek(p, X1, X2, ind1, ind2, lbda, 1/1000)
#         Ek_p = out_Ek_p.cost
#         print('Cost p ',k, Ek_p)
#         # psik_psi = np.array([pk[4](quad(lambda s: p[4](s)**2, 0, t, limit=100)[0])*p[4](t) for t in np.linspace(0,1,200)])
#         psik_psi = group_action_psi(pk[4], p[4])
#         while Ek_p > Ek_id - beta*step_size*norm_k or (psik_psi<0).any():
#             print(Ek_p, Ek_id - beta*step_size*norm_k, (psik_psi<0).any())
#             delta_k = tau*delta_k
#             # p = update_p_id(DEk_id, delta_k)
#             # print(p_to_param(p, ind1))
#             delta_p = update_p_id(DEk_id, delta_k)
#             p = group_operation_P(pk, delta_p)
#             print('update', p_to_param(delta_p, ind1))
#             print('new p', p_to_param(p, ind1))
#             out_Ek_p = compute_Ek(p, X1, X2, ind1, ind2, lbda, 1/1000)
#             Ek_p = out_Ek_p.cost
#             psik_psi = group_action_psi(pk[4], delta_p[4])
#
#         pk = p
#         out_Xk = compose_Xk_p(p, X2, ind2, ind1)
#         X2k = out_Xk.Xnew
#         ind2k = [out_Xk.A, out_Xk.B]
#         out_Ek = compute_Ek(p_id, X1, X2k, ind1, ind2k, lbda, 1/1000)
#         Ek_id = out_Ek.cost
#         DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#         k = k+1
#         print(k)
#
#     grad_desc = collections.namedtuple('Gradient_descent_results', ['X2new', 'int2new', 'p_opt', 'g', 'g_prime', 'cost'])
#     out = grad_desc(X2k, ind2k, pk, out_Xk.g, out_Xk.g_prime, Ek_id)
#     return out
# __________________________________________________________________________________________________________________________________________________________________________

# def gradient_descent_E(X1, X2, ind1, ind2, p0, beta, tau, step_size, tol, max_iter, lbda):
#     k = 0
#     p_id = np.empty((5), dtype=object)
#     p_id[0], p_id[1], p_id[2], p_id[3] = 0, 0, 0, 0
#     p_id[4] = np.ones((200))
#     print('pid', p_to_param(p_id, ind1))
#     pk = p0
#     print('p0', p_to_param(pk, ind1))
#     out_Ek = compute_Ek_bis(pk, X1, X2, ind1, ind2, lbda, 1/1000)
#     X2k = out_Ek.X2new
#     ind2k = [out_Ek.A, out_Ek.B]
#     Ek_id = out_Ek.cost
#     print('Cost 0', Ek_id)
#     DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#     print('Norm grad 0', norm_k)
#     print("grad :", DEk_id[0], DEk_id[1], DEk_id[2], DEk_id[3])
#     while norm_k>tol and k<max_iter:
#         delta_k = step_size
#         p = group_operation_P(pk, update_p_id(DEk_id, delta_k))
#         print('update', p_to_param(update_p_id(DEk_id, delta_k), ind1))
#         print('new p', p_to_param(group_operation_P(pk, update_p_id(DEk_id, delta_k)), ind1))
#         out_Ek_p = compute_Ek_bis(p, X1, X2, ind1, ind2, lbda, 1/1000)
#         Ek_p = out_Ek_p.cost
#         print('Cost p ',k, Ek_p)
#         # psik_psi = np.array([pk[4](quad(lambda s: p[4](s)**2, 0, t, limit=100)[0])*p[4](t) for t in np.linspace(0,1,200)])
#         psik_psi = group_action_psi(pk[4], p[4])
#         while Ek_p > Ek_id - beta*step_size*norm_k or (psik_psi<0).any():
#             print(Ek_p, Ek_id - beta*step_size*norm_k, (psik_psi<0).any())
#             delta_k = tau*delta_k
#             # p = update_p_id(DEk_id, delta_k)
#             # print(p_to_param(p, ind1))
#             delta_p = update_p_id(DEk_id, delta_k)
#             p = group_operation_P(pk, delta_p)
#             print('update', p_to_param(delta_p, ind1))
#             print('new p', p_to_param(p, ind1))
#             out_Ek_p = compute_Ek_bis(p, X1, X2, ind1, ind2, lbda, 1/1000)
#             Ek_p = out_Ek_p.cost
#             psik_psi = group_action_psi(pk[4], delta_p[4])
#
#         pk = p
#         out_Ek = compute_Ek_bis(p, X1, X2, ind1, ind2, lbda, 1/1000)
#         X2k = out_Ek.Xnew
#         ind2k = [out_Ek.A, out_Ek.B]
#         Ek_id = out_Ek.cost
#         DEk_id, norm_k = compute_grad_Ek_id(X1, X2k, ind1, ind2k, lbda)
#         k = k+1
#         print(k)
#
#     grad_desc = collections.namedtuple('Gradient_descent_results', ['X2new', 'int2new', 'p_opt', 'g', 'g_prime', 'cost'])
#     out = grad_desc(X2k, ind2k, pk, out_Xk.g, out_Xk.g_prime, Ek_id)
#     return out

# def n(a, D):
#     # f goes from [0,1] to [a,b]
#     # f_inv from [a,b] to [0,1]
#     f = lambda s: D*s + a
#     f_inv = lambda s: (s-a)/D
#     return f, f_inv
#
# def align_subparts_gam(a,b,c,d):
#     f = lambda s: ((c-d)/(a-b))*(s-a) + c
#     return f
#
# def align_subintervals(X1, X2, int1, int2, n):
#     '''
#     Function to align X2, restrict to the interval 'int2', to X1, restrict to the interval 'int1'
#     ...
#
#     Param:
#         X1: function object, 1d curve
#         X2: function object, 1d curve to be aligned to X1 on the respective intervals
#         int1: subinterval to which we restrict X1
#         int2: subinterval to which we restrict X2
#         n: number of discretization points to do the alignement
#
#     Return:
#         warp_func: warping function to align the subcurves of X2 to the one of X1
#         warp_func_inv: the inverse warping function
#         dist: distance between the align subcurves.
#
#     '''
#     f1, f1_inv = scale_gam(int1[0], int1[1])
#     f2, f2_inv = scale_gam(int2[0], int2[1])
#     t = np.linspace(0,1,n)
#     x1 = X1(f1(t))*(int1[1]-int1[0])
#     x2 = X2(f2(t))*(int2[1]-int2[0])
#     q1 = np.sqrt(x1)
#     q2 = np.sqrt(x2)
#     gam = fs.utility_functions.optimum_reparam(q1, t, q2)
#     gam_inv = fs.utility_functions.invertGamma(gam)
#     q2new = fs.utility_functions.warp_q_gamma(t, q2, gam)
#     dist = np.linalg.norm((x1-q2new**2),ord=1)
#     g = interp1d(t, gam)
#     g_inv = interp1d(t, gam_inv)
#     warp_func = lambda s: f2(g(f1_inv(s)))
#     warp_func_inv = lambda s: f1(g_inv(f2_inv(s)))
#     return warp_func, warp_func_inv, dist
#
#
# def extend_gam(g, g_inv, int_g, int_final):
#     '''
#     Function to extend the warping function found on the subintervals to the whole intervals.
#     ...
#
#     Param:
#         g: function object, warping function on the subinterval int_g
#         g_inv: function object, inverse wapring function
#         int_g: interval of definiton f the warping function g
#         int_final: final interval to which we extend g
#
#     Return:
#         gamma: extended warping function define on [A,B]
#         gamma_prime: first derivative of gamma
#         gamma_inv: inverse warping function define on int_final
#         [A,B]: interval of definiton of gamma
#
#     '''
#     a, b = int_g[0][0], int_g[0][1]
#     c, d = int_g[1][0], int_g[1][1]
#     g_interp = UnivariateSpline(np.linspace(a,b,200), g(np.linspace(a,b,200)), k=3)
#     gprime = g_interp.derivative()
#     gprime_a = gprime(a)
#     gprime_b = gprime(b)
#     A = (int_final[0] - g(a))/gprime_a + a
#     B = (int_final[1] - g(b))/gprime_b + b
#     def gamma(s):
#         if A<=s<a:
#             return gprime_a*(s-a) + g(a)
#         elif a<=s<=b:
#             return g(s)
#         elif b<s<=B:
#             return gprime_b*(s-b) + g(b)
#     def gamma_prime(s):
#         if A<=s<a:
#             return gprime_a
#         elif a<=s<=b:
#             return gprime(s)
#         elif b<s<=B:
#             return gprime_b
#     def gamma_inv(s):
#         if int_final[0]<=s<c:
#             return (s-c)/gprime_a + a
#         elif c<=s<=d:
#             return g_inv(s)
#         elif d<s<=int_final[1]:
#             return (s-d)/gprime_b + b
#     return gamma, gamma_prime, gamma_inv, [A,B]

# def partial_align_1d(X1, X2, init_ind1, init_ind2, sub_int1, sub_int2):
#     g, g_inv, dist = align_subintervals(X1, X2, sub_int1, sub_int2, 100)
#     ext_g, ext_g_prime, ext_g_inv, [A,B] = g2gamma(g, g_inv, sub_int1, init_ind2)
#     int_AB = np.linspace(A, B, 100)
#     def Xnew(s):
#         return X(ext_g(s))*ext_g_prime(s)
#     gam = np.array([np.round(ext_g(s), decimals=10) for s in int_AB])
#     gam_prime = np.array([ext_g_prime(s) for s in int_AB])
#     return X(gam)*gam_prime, Xnew

# def opt_grid(C1, C2, int1, int2, N):
    # grid1 = np.linspace(int1[0], int1[1], N)
    # grid2 = np.linspace(int2[0], int2[1], N)
    # xs, ys = np.meshgrid(grid1, grid1, sparse=False)
    # int_grid1 = np.unique(np.concatenate(np.stack((np.tril(xs),np.tril(ys)), axis=-1), axis=0), axis=0)
    # xs, ys = np.meshgrid(grid2, grid2, sparse=False)
    # int_grid2 = np.unique(np.concatenate(np.stack((np.tril(xs),np.tril(ys)), axis=-1), axis=0), axis=0)

# def opt_grid(C1, C2, int_grid1, int_grid2):
#     n1 = len(int_grid1)
#     n2 = len(int_grid2)
#     cost = np.zeros((n1,n2))
#     for i in range(n1):
#         for j in range(n2):
#             I1 = int_grid1[i]
#             I2 = int_grid2[j]
#             g, g_inv, dist = align_part(I1, I2, C1, C2, 100)
#             gamma, gamma_prime, gamma_inv, [A,B] = g2gamma(g, g_inv, I1, [0,1], True)
#             if A>B:
#                 cost[i,j] = None
#             else:
#                 c2new_grid, c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 # c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 cost[i,j] = cost_func(C1, c2new, [0,1], [A,B], 400)
#     return cost

# def opt_grid_bis(C1, C2, int_grid1, int_grid2, lbda):
#     n1 = len(int_grid1)
#     n2 = len(int_grid2)
#     cost_tot = np.zeros((n1,n2))
#     cost = np.zeros((n1,n2))
#     pen = np.zeros((n1,n2))
#     for i in range(n1):
#         for j in range(n2):
#             I1 = int_grid1[i]
#             I2 = int_grid2[j]
#             g, g_inv, dist = align_part(I1, I2, C1, C2, 100)
#             gamma, gamma_prime, gamma_inv, [A,B] = g2gamma(g, g_inv, I1, [0,1], True)
#             if A>B:
#                 cost_tot[i,j], cost[i,j], pen[i,j] = None, None, None
#             else:
#                 c2new_grid, c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 # c2new = final_warp(C2, gamma, gamma_prime, A, B)
#                 cost_tot[i,j], cost[i,j], pen[i,j] = cost_func_bis(C1, c2new, [0,1], [A,B], I1, 0.01, lbda)
#     return cost_tot, cost, pen

# def poly(order, gamma_spl, a):
#     if order==0:
#         poly = lambda s: gamma_spl(a) + 0*s
#         poly_dev = lambda s: 0*s
#     elif order==1:
#         poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative(1)(a)
#         poly_dev = lambda s: gamma_spl.derivative(1)(a) + 0*s
#     elif order==2:
#         poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative()(a) + ((s-a)**2)*gamma_spl.derivative(2)(a)/2
#         poly_dev = lambda s:  gamma_spl.derivative()(a) + (s-a)*gamma_spl.derivative(2)(a)
#     elif order==3:
#         poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative()(a) + ((s-a)**2)*gamma_spl.derivative(2)(a)/2 + ((s-a)**3)*gamma_spl.derivative(3)(a)/(2*3)
#         poly_dev = lambda s:  gamma_spl.derivative()(a) + (s-a)*gamma_spl.derivative(2)(a) + ((s-a)**2)*gamma_spl.derivative(3)(a)/(2)
#     elif order==4:
#         poly = lambda s: gamma_spl(a) + (s-a)*gamma_spl.derivative()(a) + ((s-a)**2)*gamma_spl.derivative(2)(a)/2 + ((s-a)**3)*gamma_spl.derivative(3)(a)/(2*3) + ((s-a)**4)*gamma_spl.derivative(4)(a)/(2*3*4)
#         poly_dev = lambda s:  gamma_spl.derivative()(a) + (s-a)*gamma_spl.derivative(2)(a) + ((s-a)**2)*gamma_spl.derivative(3)(a)/(2) + ((s-a)**3)*gamma_spl.derivative(4)(a)/(2*3)
#     else:
#         raise ValueError("order must be <= 4")
#     return poly, poly_dev
