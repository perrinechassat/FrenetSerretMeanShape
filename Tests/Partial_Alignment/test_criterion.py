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


filename = "results_1simu_partial_align"
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()
res_grid_search, func_curve, grid_final_array, f_final_array, coef_h = dic["res_grid_search"], dic["func_curve"], dic["grid_final_array"], dic["f_final_array"], dic["coef_h"]
int_AB_array, int_ab_array, f_init_array, grid_init_array = dic["int_AB_array"], dic["int_ab_array"], dic["f_init_array"], dic["grid_init_array"]

res = np.empty((N,N), dtype=object)
# res i,j is the parameters to align curve i to curve j
for i in range(N):
    for j in range(N):
        if i==j:
            res[i][j] = [0,L[i],0,L[i]]
        else:
            print('Find opt param to partially align curve ', i, 'to curve ', j, '...')
            cost_func = cost_gridsearch(func_curve[j], func_curve[i], [0,L[j]], [0,L[i]], 0.01)
            param_list = make_grid(np.linspace(0,L[j]/2,30), np.linspace(0,L[i]/2,30), dist=6)
            print(param_list.shape[0])
            res[i][j] = gridsearch_optimisation(cost_func, param_list)
            print(res[i][j])


filename = "results_test_criterion"
dic_error_indiv = {"res_grid_search" : res}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic_error_indiv,fil)
fil.close()
