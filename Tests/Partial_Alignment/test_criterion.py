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


# filename = "results_1simu_partial_align"
# fil = open(filename,"rb")
# dic = pickle.load(fil)
# fil.close()
# res_grid_search, func_curve, grid_final_array, f_final_array, coef_h = dic["res_grid_search"], dic["func_curve"], dic["grid_final_array"], dic["f_final_array"], dic["coef_h"]
# int_AB_array, int_ab_array, f_init_array, grid_init_array = dic["int_AB_array"], dic["int_ab_array"], dic["f_init_array"], dic["grid_init_array"]
# N = len(func_curve)

t = np.linspace(0,1,200)
t1 = np.linspace(0,2,200)
t2 = np.linspace(0,3,200)
t3 = np.linspace(0,1,200)

curv = lambda s: 2*np.abs(np.sin(s*np.pi)) + 1
gam_3 = gamma(t, 0.5)
gam_1 = gamma(t, -0.8)
curv_warp_3 = interp1d(t3, warp_area_gamma(t, curv(t3)[np.newaxis,:], gam_3)[0])
curv_warp_1 = interp1d(t1, warp_area_gamma(t, curv(t1)[np.newaxis,:], gam_1)[0])

new_length = [3,2, 1.5]
c0 = interp1d(np.linspace(0,3,200), curv_warp_1(t1)*2/3)
c1 = interp1d(np.linspace(0,2,200), curv(t2)*3/2)
c2 = interp1d(np.linspace(0,1.5,200), curv_warp_3(t3)/1.5)
grid_final_array = np.array([np.linspace(0,3,200), np.linspace(0,2,200), np.linspace(0,1.5,200)])
func_curve = np.array([c0, c1, c2])
N = 3

res = np.zeros((N,N,4))
# res i,j is the parameters to align curve i to curve j
for i in range(N):
    for j in range(N):
        if i==j:
            res[i][j] = [0,grid_final_array[i][-1],0,grid_final_array[i][-1]]
        else:
            print('Find opt param to partially align curve ', i, 'to curve ', j, '...')
            L_i = grid_final_array[i][-1]
            L_j = grid_final_array[j][-1]
            cost_func = cost_bis(func_curve[j], func_curve[i], [0,L_j], [0,L_i], 0.01)
            param_list = make_grid(np.linspace(0,L_j,int(L_j/0.04)), np.linspace(0,L_i,int(L_i/0.04)), dist=0.5)
            res_grid_search = gridsearch_optimisation(cost_func, param_list)
            res[i][j] = res_grid_search[0]
            print(res[i][j])


filename = "results_test_simple_criterion_v3"
dic = {"res_grid_search" : res}
if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()
