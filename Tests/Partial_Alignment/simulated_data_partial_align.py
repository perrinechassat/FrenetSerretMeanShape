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
from simulate_data import *
import collections


# "_____ SIMU 10 knots _____"
#
# N = 3
# int_h = [-10,20]
# out_simu_1 = simu_partial_align_data(N, int_h, 10, 0.7)
# func_curve = out_simu_1.func_curve
# L = out_simu_1.L
#
# res = np.zeros((N,N,4))
# # res i,j is the parameters to align curve i to curve j
# for i in range(N):
#     for j in range(N):
#         if i==j:
#             res[i][j] = [0,L[i],0,L[i]]
#         else:
#             print('Find opt param to partially align curve ', i, 'to curve ', j, '...')
#             cost_func = cost_gridsearch(func_curve[j], func_curve[i], [0,L[j]], [0,L[i]], 0.01)
#             param_list = make_grid(np.linspace(0,L[j]/2,40), np.linspace(0,L[i]/2,40), dist=5)
#             res_grid_search = gridsearch_optimisation(cost_func, param_list)
#             res[i][j] = res_grid_search[0]
#             print(res[i][j])
#
#
# filename = "Results/simu_partial_align_10_knots"
# dic_error_indiv = {"res_grid_search" : res, "simu_data" : out_simu_1}
# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
# fil = open(filename,"xb")
# pickle.dump(dic_error_indiv,fil)
# fil.close()


"_____ SIMU 8 knots _____"

N = 3
int_h = [-10,20]
out_simu_1 = simu_partial_align_data(N, int_h, 8, 0.7)
func_curve = out_simu_1.func_curve
L = out_simu_1.L

res = np.zeros((N,N,4))
# res i,j is the parameters to align curve i to curve j
for i in range(N):
    for j in range(N):
        if i==j:
            res[i][j] = [0,L[i],0,L[i]]
        else:
            print('Find opt param to partially align curve ', i, 'to curve ', j, '...')
            cost_func = cost_gridsearch(func_curve[j], func_curve[i], [0,L[j]], [0,L[i]], 0.01)
            param_list = make_grid(np.linspace(0,L[j]/2,40), np.linspace(0,L[i]/2,40), dist=3)
            res_grid_search = gridsearch_optimisation(cost_func, param_list)
            res[i][j] = res_grid_search[0]
            print(res[i][j])


filename = "Results/simu_partial_align_8_knots"
dic_error_indiv = {"res_grid_search" : res, "simu_data" : out_simu_1}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic_error_indiv,fil)
fil.close()


"_____ SIMU 15 knots _____"

N = 3
int_h = [-10,20]
out_simu_1 = simu_partial_align_data(N, int_h, 15, 0.7)
func_curve = out_simu_1.func_curve
L = out_simu_1.L

res = np.zeros((N,N,4))
# res i,j is the parameters to align curve i to curve j
for i in range(N):
    for j in range(N):
        if i==j:
            res[i][j] = [0,L[i],0,L[i]]
        else:
            print('Find opt param to partially align curve ', i, 'to curve ', j, '...')
            cost_func = cost_gridsearch(func_curve[j], func_curve[i], [0,L[j]], [0,L[i]], 0.01)
            param_list = make_grid(np.linspace(0,L[j]/2,40), np.linspace(0,L[i]/2,40), dist=5)
            res_grid_search = gridsearch_optimisation(cost_func, param_list)
            res[i][j] = res_grid_search[0]
            print(res[i][j])


filename = "Results/simu_partial_align_15_knots"
dic_error_indiv = {"res_grid_search" : res, "simu_data" : out_simu_1}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic_error_indiv,fil)
fil.close()
