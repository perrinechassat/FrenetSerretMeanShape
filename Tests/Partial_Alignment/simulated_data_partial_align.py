import sys
import os.path
sys.path.insert(1, '../../FrenetSerretMeanShape')
sys.path.insert(1, '../../../Persistence1D-master/python/')
# from pre_process_Mocaplab_data import *
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumtrapz
from visu_utils import *
from alignment_utils import *
from simu_utils import *
from pickle import *
import dill as pickle
from research_law_utils import *
import skfda
import fdasrsf as fs
from scipy.integrate import cumtrapz
from sympy import integrate, symbols
from scipy.optimize import minimize
from simu_utils import gamma, gamma_prime, omega, omega_prime

N = 3
int_h = [-10,20]
delta_h = int_h[1] - int_h[0]
knots_h = np.linspace(int_h[0], int_h[-1], 15)
basis_h = skfda.representation.basis.BSpline(knots=knots_h, order=3)
coef_h = np.random.random((basis_h.n_basis))*10
h = FDataBasis(basis_h, coef_h)
'on génère [A_i,B_i]'

coef_array = np.empty((N), dtype=object)
int_AB_array = np.zeros((N,2))
knots_array = np.empty((N), dtype=object)
for i in range(N):
    ' coef pour f_i '
    coef_array[i] = np.random.random((basis_h.n_basis))*10

    ' [A_i, B_i] '
    delta_i = 0.25*delta_h*np.random.random(1) + 0.5*delta_h
    min = (int_h[1]-delta_i-int_h[0])*np.random.random(1) + int_h[0]
    max = (int_h[1]-min-delta_i)*np.random.random(1) + min+delta_i
    knots_array[i] = knots_h[np.argmin(abs(knots_h-min)):np.argmin(abs(knots_h-max))]
    int_AB_array[i][0], int_AB_array[i][1] = knots_array[i][0], knots_array[i][-1]

perc = 70/100
int_ab_array = np.zeros((N,N,2))
for i in range(N):
    int_ab_array[i][i] = int_AB_array[i]
    for j in range(i+1,N):
        if i!=j:
            int_intersect = np.intersect1d(knots_array[i], knots_array[j])
            del_ij = (int_intersect[-1] - int_intersect[0])*perc
            m = (int_intersect[-1]-del_ij-int_intersect[0])*np.random.random(1) + int_intersect[0]
            M = m+del_ij
            knots_ij = int_intersect[np.argmin(abs(int_intersect-m)):np.argmin(abs(int_intersect-M))]
            int_ab_array[i][j] = np.array([knots_ij[0], knots_ij[-1]])
            int_ab_array[j][i] = np.array([knots_ij[0], knots_ij[-1]])
            ind = np.concatenate(np.array([np.where(knots_h==knots_ij[k])[0] for k in range(len(knots_ij))]))
            c_h = coef_h[ind]
            coef_array[i][ind] = c_h
            coef_array[j][ind] = c_h

f_array = np.empty((N), dtype=object)
f_eval_array = np.empty((N), dtype=object)
grid_array = np.empty((N), dtype=object)
for i in range(N):
    f_array[i] = FDataBasis(skfda.representation.basis.BSpline(knots=knots_h, order=3), coef_array[i])
    grid_array[i] = np.linspace(int_AB_array[i][0], int_AB_array[i][1], int((int_AB_array[i][1]-int_AB_array[i][0])*6))
    f_eval_array[i] = f_array[i].evaluate(grid_array[i]).squeeze()



sig = np.random.normal(loc=0.0, scale=0.6, size=(N,N))
interm_f_array = np.empty((N), dtype=object)
interm_grid_array = np.empty((N), dtype=object)
int_AB = int_AB_array
for i in range(N):
    for j in range(i+1,N):
        a_ij, b_ij = int_ab_array[i][j][0], int_ab_array[i][j][1]
        n, n_inv = compute_n(a_ij, b_ij-a_ij)
        def g(u):
            if a_ij <= u <= b_ij:
                return n(gamma(n_inv(u),sig[i,j]))
            elif u < a_ij:
                return gamma_prime(n_inv(a_ij),sig[i,j])*(u-a_ij) + a_ij
            else:
                return gamma_prime(n_inv(b_ij),sig[i,j])*(u-b_ij) + b_ij
        def g_inv(u):
            if a_ij <= u <= b_ij:
                return n(omega(n_inv(u),sig[i,j]))
            elif u < a_ij:
                return omega_prime(n_inv(a_ij),sig[i,j])*(u-a_ij) + a_ij
            else:
                return omega_prime(n_inv(b_ij),sig[i,j])*(u-b_ij) + b_ij
        def g_prime(u):
            if a_ij <= u <= b_ij:
                return gamma_prime(n_inv(u),sig[i,j])
            elif u < a_ij:
                return gamma_prime(n_inv(a_ij),sig[i,j])
            else:
                return gamma_prime(n_inv(b_ij),sig[i,j])
        int_AB[i][0], int_AB[i][1] = g_inv(int_AB[i][0]), g_inv(int_AB[i][1])
        s_i = np.linspace(int_AB[i][0], int_AB[i][1], int((int_AB[i][1]-int_AB[i][0]))*10)
        interm_grid_array[i] = s_i
        interm_f_array[i] = f_array[i].evaluate(np.array([g(s_) for s_ in s_i])).squeeze()*np.array([g_prime(s_) for s_ in s_i])

interm_grid_array[-1] = grid_array[-1]
interm_f_array[-1] = f_eval_array[-1]



L = np.array([(int_AB[i][1]-int_AB[i][0])*np.random.normal(loc=1, scale=0.25) for i in range(N)])
final_f_array = np.empty((N), dtype=object)
final_grid = np.empty((N), dtype=object)
for i in range(N):
    f = align_subparts_gam(int_AB[i][0],int_AB[i][1],0,L[i])
    final_grid[i] = f(interm_grid_array[i])
    f_prime = (int_AB[i][1]-int_AB[i][0])/L[i]
    final_f_array[i] = f_prime*interm_f_array[i]



func_curve = np.empty((N), dtype=object)
for i in range(N):
    func_curve[i] = interp1d(final_grid[i],final_f_array[i],fill_value=([final_f_array[i][0]], [final_f_array[i][-1]]), bounds_error=False)


res = np.empty((N,N), dtype=object)
# res i,j is the parameters to align curve i to curve j
for i in range(N):
    for j in range(i+1,N):
        if i==j:
            res[i][j] = [0,L[i],0,L[j]]
        else:
            print('Find opt param to partially align curve ', i, 'to curve ', j, '...')
            cost_func = cost_gridsearch(func_curve[j], func_curve[i], [0,L[j]], [0,L[i]], 0.01)
            param_list = make_grid(np.linspace(0,L[j],60), np.linspace(0,L[i],60), dist=2)
            print(param_list.shape[0])
            res[i][j] = gridsearch_optimisation(cost_func, param_list)
            print(res[i][j])

np.save('res_simu_partial_align.npy', res)
