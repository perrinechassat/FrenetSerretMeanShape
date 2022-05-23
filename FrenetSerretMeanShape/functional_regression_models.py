import numpy as np
from skfda.ml.regression import MultivariateLinearRegression
from skfda.ml.regression import MultivariateAdditiveRegression
from skfda.representation import FDataGrid
from skfda.representation.basis import Constant
from skfda.misc.regularization import TikhonovRegularization
from skfda.misc.operators import LinearDifferentialOperator
from optimization_utils import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from pickle import *
import dill as pickle

# def cross_validation_score_MFR(Model, n_splits, X, y, K_coef_basis, type_coef_basis, domain_range, Z=None, fit_intercept=False, regularization=None, smoothing_parameter=None, function_to_apply = lambda x: x):
#
#     if len(K_coef_basis)!=len(type_coef_basis):
#         raise ValueError(
#                     "Precise number of basis for all basis.",
#                 )
#
#     coef_basis = []
#     for BasisType, k in zip(type_coef_basis, K_coef_basis):
#         if BasisType!=Constant:
#             coef_basis.append(BasisType(domain_range=domain_range, n_basis=int(k)))
#         else:
#             coef_basis.append(BasisType(domain_range))
#
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
#     N = len(X)
#
#     linear = Model(coef_basis=coef_basis, fit_intercept=fit_intercept, smoothing_parameter=smoothing_parameter, regularization=regularization)
#
#     # print(linear)
#     err = []
#     for train_index, test_index in kf.split(np.ones(N)):
#     # print('------- step ', k, ' cross validation --------')
#
#         try:
#             if isinstance(Model, MultivariateLinearRegression):
#                 _ = linear.fit(X[train_index].squeeze(), y[train_index].squeeze(), function_to_apply=function_to_apply)
#                 CV = 0
#                 for i in test_index:
#                     CV += np.mean((y[i].data_matrix.squeeze() - linear.predict(X[i], y[i].grid_points[0]))**2)
#                 err.append(CV/len(test_index))
#             elif isinstance(Model, MultivariateAdditiveRegression):
#                 _ = linear.fit(X[train_index].squeeze(), Z[train_index].squeeze(), y[train_index].squeeze(), function_to_apply=function_to_apply)
#                 CV = 0
#                 for i in test_index:
#                     CV += np.mean((y[i].data_matrix.squeeze() - linear.predict(X[i], Z[i], y[i].grid_points[0]))**2)
#                 err.append(CV/len(test_index))
#             else:
#                 raise ValueError(
#                             "Invalid model type",
#                         )
#         except:
#             err.append(np.inf)
#
#     print(err)
#     return np.mean(err)

def cross_validation_score_MFR(Model, n_splits, X, y, K_coef, coef_basis, Z=None, fit_intercept=False, regularization=None, smoothing_parameter=None, function_to_apply = lambda x: x):

    # if len(K_coef)!=len(coef_basis):
    #     raise ValueError(
    #                 "Precise number of basis for all basis.",
    #             )

    new_coef_basis = []
    for Basis, k in zip(coef_basis, K_coef):
        if type(Basis)!=Constant:
            new_coef_basis.append(type(Basis)(domain_range=Basis.domain_range, n_basis=int(k)))
        else:
            new_coef_basis.append(type(Basis)(domain_range=Basis.domain_range))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    N = len(X)

    linear = Model(coef_basis=new_coef_basis, fit_intercept=fit_intercept, smoothing_parameter=smoothing_parameter, regularization=regularization)

    # err = []
    # for train_index, test_index in kf.split(np.ones(N)):
    # # print('------- step ', k, ' cross validation --------')
    #
    #     try:
    #         if Model==MultivariateLinearRegression:
    #             _ = linear.fit(X[train_index].squeeze(), y[train_index].squeeze(), function_to_apply=function_to_apply)
    #             CV = 0
    #             for i in test_index:
    #                 # CV += np.mean((y[i].data_matrix.squeeze() - linear.predict(X[i], y[i].grid_points[0]))**2)
    #                 CV += np.sqrt(trapz((y[i].data_matrix.squeeze() - linear.predict(X[i], y[i].grid_points[0]))**2, y[i].grid_points[0]))
    #             err.append(CV/len(test_index))
    #         elif Model==MultivariateAdditiveRegression:
    #             _ = linear.fit(X[train_index].squeeze(), Z[train_index].squeeze(), y[train_index].squeeze(), function_to_apply=function_to_apply)
    #             CV = 0
    #             for i in test_index:
    #                 # CV += np.mean((y[i].data_matrix.squeeze() - linear.predict(X[i], Z[i], y[i].grid_points[0]))**2)
    #                 CV += np.sqrt(trapz((y[i].data_matrix.squeeze() - linear.predict(X[i], Z[i], y[i].grid_points[0]))**2, y[i].grid_points[0]))
    #             err.append(CV/len(test_index))
    #         else:
    #             raise ValueError(
    #                         "Invalid model type",
    #                     )
    #     except:
    #         # err.append(np.inf)
    #         pass

    def func_CV(train_index, test_index):
        try:
            if Model==MultivariateLinearRegression:
                _ = linear.fit(X[train_index].squeeze(), y[train_index].squeeze(), function_to_apply=function_to_apply)
                CV = 0
                for i in test_index:
                    # CV += np.mean((y[i].data_matrix.squeeze() - linear.predict(X[i], y[i].grid_points[0]))**2)
                    CV += np.sqrt(trapz((y[i].data_matrix.squeeze() - linear.predict(X[i], y[i].grid_points[0]))**2, y[i].grid_points[0]))
                e = CV/len(test_index)
                return e
            elif Model==MultivariateAdditiveRegression:
                _ = linear.fit(X[train_index].squeeze(), Z[train_index].squeeze(), y[train_index].squeeze(), function_to_apply=function_to_apply)
                CV = 0
                for i in test_index:
                    # CV += np.mean((y[i].data_matrix.squeeze() - linear.predict(X[i], Z[i], y[i].grid_points[0]))**2)
                    CV += np.sqrt(trapz((y[i].data_matrix.squeeze() - linear.predict(X[i], Z[i], y[i].grid_points[0]))**2, y[i].grid_points[0]))
                e = CV/len(test_index)
                return e
            else:
                raise ValueError(
                            "Invalid model type",
                        )
        except:
            # err.append(np.inf)
            pass

    err = Parallel(n_jobs=-1)(delayed(func_CV)(train_index, test_index) for train_index, test_index in kf.split(np.ones(N)))

    print(err)
    if np.array(err).all()==None:
        return np.inf
    else:
        return np.mean(np.array(err)[np.where([err[i]!=None for i in range(len(err))])])
    # if len(err)==0:
    #     return np.inf
    # else:
    #     return np.mean(err)


def opti_smoothing_parameters_MFR(Model, n_splits, X, y, hyperparam_list, coef_basis, Z=None, fit_intercept=False, regularization=None, function_to_apply=lambda x:x):

    # if hyperparam_bound_s is not None:
    #     hyperparam_bounds = hyperparam_bound_k + hyperparam_bound_s
    # else:
    #     hyperparam_bounds = hyperparam_bound_k
    # print(hyperparam_bounds)
    #
    # def Opt_fun(params):
    #     L = len(type_coef_basis)
    #     k = np.zeros(L)
    #     ind = np.where([i!=Constant for i in type_coef_basis])
    #     n_param = len(hyperparam_bound_k)
    #     if hyperparam_bound_s is not None:
    #         s = np.zeros(L)
    #         s[ind] = params[n_param:]
    #         k[ind] = params[:n_param]
    #     else:
    #         s = None
    #         k[ind] = params
    #
    #     return cross_validation_score_MFR(n_splits, X, y, k, type_coef_basis, domain_range, fit_intercept=fit_intercept, regularization=regularization, smoothing_parameter=s)

    # x = bayesian_optimisation(Opt_fun, n_calls, hyperparam_bounds)

    # if hyperparam_bound_s is not None:
    #     hyperparam_bounds = hyperparam_bound_k + hyperparam_bound_s
    # else:
    #     hyperparam_bounds = hyperparam_bound_k
    # print(hyperparam_bounds)

    def Opt_fun(params):
        # L = len(coef_basis)
        # k = np.zeros(L)
        # ind = np.where([i!=Constant for i in coef_basis])
        # n_param = len(hyperparam_list)
        # if regularization is not None:
        #     s = np.zeros(L)
        #     s[ind] = params[-1]
        #     k[ind] = params[:-1]
        # else:
        #     s = None
        #     k[ind] = params
        if regularization is not None:
            s = params[-1]
            k = params[:-1]
        else:
            s = None
            k = params
        return cross_validation_score_MFR(Model, n_splits, X, y, k, coef_basis, Z=Z, fit_intercept=fit_intercept, regularization=regularization, smoothing_parameter=s, function_to_apply=function_to_apply)

    x = gridsearch_optimisation(Opt_fun, create_hyperparam_grid_bis(hyperparam_list[0], hyperparam_list[1], hyperparam_list[2]))

    lam = x[-1]
    K_coef = x[:-1]

    new_coef_basis = []
    for Basis, k in zip(coef_basis, K_coef):
        if type(Basis)!=Constant:
            new_coef_basis.append(type(Basis)(domain_range=Basis.domain_range, n_basis=int(k)))
        else:
            new_coef_basis.append(type(Basis)(domain_range=Basis.domain_range))

    return new_coef_basis, lam, K_coef


def create_hyperparam_grid(hyperparam_list):

    n_param = len(hyperparam_list)
    grid = np.array(np.meshgrid(*hyperparam_list)).T.reshape(-1,n_param)

    return grid

def create_hyperparam_grid_bis(lam_param, K_param, n_K):

    n_plam = len(lam_param)
    n_pK = len(K_param)
    grid = np.zeros((n_pK*n_plam, n_K+1))
    ind = 0
    for i in range(n_plam):
        for j in range(n_pK):
            grid[ind] = np.concatenate((np.repeat(K_param[j], n_K), np.array([lam_param[i]])))
            ind += 1

    return grid


def compute_error(true, pred, t):
    n = len(pred)
    dist = []
    residuals = true - pred
    for i in range(n):
        dist.append(np.sqrt(np.trapz((pred[i] - true[i]) ** 2, t)))
    return residuals, dist, np.mean(dist), np.std(dist)



def apply_add_reg(X, Z, y, t, coef, lam, save=False, filename=''):
    nT = len(t)

    X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, test_size=0.2)

    model = MultivariateAdditiveRegression(coef_basis=coef, fit_intercept = False, smoothing_parameter=lam, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))
    _ = model.fit(X_train, Z_train, y_train)

    n_test = len(X_test)
    pred = np.zeros((n_test, nT))
    true = np.zeros((n_test, nT))
    for j in range(n_test):
        pred[j] = model.predict(X_test[j],Z_test[j],t)
        true[j] = y_test[j].data_matrix.squeeze()

    residuals, dist, m_dist, std_dist = compute_error(true, pred, t)

    dic = {"X" : X, "y" : y, "X_train" : X_train, "X_test" : X_test, "Z_train" : Z_train, "Z_test" : Z_test, "y_train": y_train, "y_test" : y_test, "coef" : coef, "lam" : lam,
    "pred" : pred, "true" : true, "residuals" : residuals, "dist" : dist, "m_dist" : m_dist, "std_dist" : std_dist, "model" : model}

    if save==True:
        filename = "Results/"+filename
        if os.path.isfile(filename):
            print("Le fichier ", filename, " existe déjà.")
            filename = filename+'_bis'
        fil = open(filename,"xb")
        pickle.dump(dic,fil)
        fil.close()

    return dic


def apply_lin_reg(X, y, t, coef, lam, save=False, filename='', function_to_apply=lambda x:x):
    nT = len(t)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_lin = MultivariateLinearRegression(coef_basis=coef, fit_intercept = False, smoothing_parameter=lam, regularization=TikhonovRegularization(LinearDifferentialOperator(2)))
    _ = model_lin.fit(X_train, y_train, function_to_apply=function_to_apply)

    n_test = len(X_test)
    pred = np.zeros((n_test, nT))
    true = np.zeros((n_test, nT))
    for j in range(n_test):
        pred[j] = model_lin.predict(X_test[j],t)
        true[j] = y_test[j].data_matrix.squeeze()

    residuals, dist, m_dist, std_dist = compute_error(true, pred, t)

    dic = {"X" : X, "y" : y, "X_train" : X_train, "X_test" : X_test, "y_train": y_train, "y_test" : y_test, "coef" : coef, "lam" : lam,
    "pred" : pred, "true" : true, "residuals" : residuals, "dist" : dist, "m_dist" : m_dist, "std_dist" : std_dist, "model" : model_lin}

    if save==True:
        filename = "Results/"+filename
        if os.path.isfile(filename):
            print("Le fichier ", filename, " existe déjà.")
            filename = filename+'_bis'
        fil = open(filename,"xb")
        pickle.dump(dic,fil)
        fil.close()

    return dic
