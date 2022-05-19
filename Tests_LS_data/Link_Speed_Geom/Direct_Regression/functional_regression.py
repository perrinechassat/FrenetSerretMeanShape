import sys
import os.path
sys.path.insert(1, '../../../FrenetSerretMeanShape')
sys.path.insert(1, '../../')
sys.path.insert(1, '../../../../Persistence1D-master/python/')
import numpy as np
from scipy.interpolate import interp1d
from visu_utils import *
from pickle import *
import dill as pickle
import fdasrsf as fs
from sklearn.model_selection import train_test_split
from skfda.representation.basis import FDataBasis, BSpline, Constant, Tensor
from skfda.representation import FDataGrid
from skfda.ml.regression import MultivariateLinearRegression
from skfda.ml.regression import MultivariateAdditiveRegression
from functional_regression_models import *
from skfda.misc.regularization import TikhonovRegularization
from skfda.misc.operators import LinearDifferentialOperator
from skfda.preprocessing.smoothing import BasisSmoother


filename = "../data_link_speed_geom"
fil = open(filename,"rb")
dic = pickle.load(fil)
fil.close()
trajs, s, sdot, curv, tors, der_curv = dic["trajs"], dic["s"], dic["sdot"], dic["curv"], dic["tors"], dic["der_curv"]
der_tors, curv_L, tors_L, sdot_L, L, s_L = dic["der_tors"], dic["curv_L"], dic["tors_L"], dic["sdot_L"], dic["L"], dic["s_L"]
D2s, D3s, D4s, models, t, curv_t, tors_t = dic["D2s"], dic["D3s"], dic["D4s"], dic["models"], dic["t"], dic["curv_t"], dic["tors_t"]
fd_curv_t, fd_tors_t, fd_curv_der, fd_tors_der = dic["fd_curv_t"], dic["fd_tors_t"], dic["fd_curv_der"], dic["fd_tors_der"]
fd_curv_t_L, fd_tors_t_L = dic["fd_curv_t_L"], dic["fd_tors_t_L"]
N = len(trajs)
nT = len(t)
print(N, nT)

nb_samples = 0
for i in range(N):
    ni = len(trajs[i])
    nb_samples += ni



""" _______________________________________________ Linear Functional Regression _______________________________________________"""


""" -------------------- without log -------------------- """

param_lam = np.array([0.0, 0.01, 0.1, 1])
param_K = np.array([int(5), int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([param_K, param_K, param_K, param_K, param_lam])

X = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0]), BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s[i][j], grid_points=t)), fd_curv_t[i][j], fd_tors_t[i][j]]
        y[ind] = FDataGrid(data_matrix=sdot[i][j], grid_points=t)
        ind += 1

coef = [BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5)]
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateLinearRegression, 5, X, y, hyperparam_list, coef, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

reg_model = apply_lin_reg(X, y, t, coef, lam, save=False, filename='lin_reg_without_log')



""" -------------------- with log on sdot -------------------- """

param_lam = np.array([0.0, 0.01, 0.1, 1])
param_K = np.array([int(5), int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([param_K, param_K, param_K, param_K, param_lam])

X = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0]), BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s[i][j], grid_points=t)), fd_curv_t[i][j], fd_tors_t[i][j]]
        y[ind] = FDataGrid(data_matrix=np.log(sdot[i][j]), grid_points=t)
        ind += 1

coef = [BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5)]
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateLinearRegression, 5, X, y, hyperparam_list, coef, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

reg_model = apply_lin_reg(X, y, t, coef, lam, save=False, filename='lin_reg_with_log_on_sdot')


""" -------------------- with log -------------------- """

param_lam = np.array([0.0, 0.01, 0.1, 1])
param_K = np.array([int(5), int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([param_K, param_K, param_K, param_K, param_lam])

X = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0]), BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s[i][j], grid_points=t)), fd_curv_t[i][j], fd_tors_t[i][j]]
        y[ind] = FDataGrid(data_matrix=np.log(sdot[i][j]), grid_points=t)
        ind += 1

coef = [BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5)]
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateLinearRegression, 5, X, y, hyperparam_list, coef, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

reg_model = apply_lin_reg(X, y, t, coef, lam, save=False, filename='lin_reg_with_log', function_to_apply=lambda x: np.log(x + 1e-06))


""" -------------------- with log L -------------------- """

param_lam = np.array([0.0, 0.01, 0.1, 1])
param_K = np.array([int(5), int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([param_K, param_K, param_K, param_K, param_lam])

X = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0]), BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s_L[i][j], grid_points=t)), fd_curv_t_L[i][j], fd_tors_t_L[i][j]]
        y[ind] = FDataGrid(data_matrix=np.log(sdot_L[i][j]), grid_points=t)
        ind += 1

coef = [BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5), BSpline(domain_range=(0,1), n_basis=5)]
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateLinearRegression, 5, X, y, hyperparam_list, coef, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

reg_model = apply_lin_reg(X, y, t, coef, lam, save=False, filename='lin_reg_with_log_L', function_to_apply=lambda x: np.log(x + 1e-06))









""" _______________________________________________ GAM Functional _______________________________________________"""


""" -------------------- without log on sdot -------------------- """

hyperparam_s = np.array([0.1, 1, 10])
hyperparam_k = np.array([int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([hyperparam_k, hyperparam_k, hyperparam_k, hyperparam_s])

X = np.empty((nb_samples), dtype=object)
Z = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        # X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0])]
        X[ind] = []
        Z[ind] = [BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s[i][j], grid_points=t)), fd_curv_t[i][j], fd_tors_t[i][j]]
        y[ind] = FDataGrid(data_matrix=sdot[i][j], grid_points=t)
        ind += 1

coef = np.array([BSpline(domain_range=(np.min(np.concatenate(s)), np.max(np.concatenate(s))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(curv_t)), np.max(np.concatenate(curv_t))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(tors_t)), np.max(np.concatenate(tors_t))), n_basis=5)])
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateAdditiveRegression, 10, X, y, hyperparam_list, coef, Z=Z, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

res_model = apply_add_reg(X, Z, y, t, coef, lam, save=True, filename='gam_reg_without_log')


""" ---------------- without log on sdot L -------------------- """

hyperparam_s = np.array([0.1, 1, 10])
hyperparam_k = np.array([int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([hyperparam_k, hyperparam_k, hyperparam_k, hyperparam_s])

X = np.empty((nb_samples), dtype=object)
Z = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        # X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0])]
        X[ind] = []
        Z[ind] = [BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s_L[i][j], grid_points=t)), fd_curv_t_L[i][j], fd_tors_t_L[i][j]]
        y[ind] = FDataGrid(data_matrix=sdot_L[i][j], grid_points=t)
        ind += 1

coef = np.array([BSpline(domain_range=(np.min(np.concatenate(s_L)), np.max(np.concatenate(s_L))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(curv_L)), np.max(np.concatenate(curv_L))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(tors_L)), np.max(np.concatenate(tors_L))), n_basis=5)])
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateAdditiveRegression, 10, X, y, hyperparam_list, coef, Z=Z, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

res_model = apply_add_reg(X, Z, y, t, coef, lam, save=True, filename='gam_reg_without_log_on_sdot_L')


""" -------------------- with log on sdot -------------------- """

hyperparam_s = np.array([0.1, 1, 10])
hyperparam_k = np.array([int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([hyperparam_k, hyperparam_k, hyperparam_k, hyperparam_s])

X = np.empty((nb_samples), dtype=object)
Z = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        # X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0])]
        X[ind] = []
        Z[ind] = [BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s[i][j], grid_points=t)), fd_curv_t[i][j], fd_tors_t[i][j]]
        y[ind] = FDataGrid(data_matrix=np.log(sdot[i][j]), grid_points=t)
        ind += 1

coef = np.array([BSpline(domain_range=(np.min(np.concatenate(s)), np.max(np.concatenate(s))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(curv_t)), np.max(np.concatenate(curv_t))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(tors_t)), np.max(np.concatenate(tors_t))), n_basis=5)])
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateAdditiveRegression, 10, X, y, hyperparam_list, coef, Z=Z, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

res_model = apply_add_reg(X, Z, y, t, coef, lam, save=True, filename='gam_reg_with_log_on_sdot')



""" ---------------- with log on sdot L -------------------- """

hyperparam_s = np.array([0.1, 1, 10])
hyperparam_k = np.array([int(15), int(30), int(60), int(100)])
hyperparam_list = np.array([hyperparam_k, hyperparam_k, hyperparam_k, hyperparam_s])

X = np.empty((nb_samples), dtype=object)
Z = np.empty((nb_samples), dtype=object)
y = np.empty((nb_samples), dtype=object)

ind = 0
for i in range(N):
    ni = len(trajs[i])
    for j in range(ni):
        # X[ind] = [FDataBasis(Constant((0,1)), L[i][j][0])]
        X[ind] = []
        Z[ind] = [BasisSmoother(BSpline((0,1), n_basis=100), return_basis=True).fit_transform(FDataGrid(data_matrix=s_L[i][j], grid_points=t)), fd_curv_t_L[i][j], fd_tors_t_L[i][j]]
        y[ind] = FDataGrid(data_matrix=np.log(sdot_L[i][j]), grid_points=t)
        ind += 1

coef = np.array([BSpline(domain_range=(np.min(np.concatenate(s_L)), np.max(np.concatenate(s_L))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(curv_L)), np.max(np.concatenate(curv_L))), n_basis=5), BSpline(domain_range=(np.min(np.concatenate(tors_L)), np.max(np.concatenate(tors_L))), n_basis=5)])
coef, lam, tab_K = opti_smoothing_parameters_MFR(MultivariateAdditiveRegression, 10, X, y, hyperparam_list, coef, Z=Z, fit_intercept=False, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))

res_model = apply_add_reg(X, Z, y, t, coef, lam, save=True, filename='gam_reg_with_log_on_sdot_L')
