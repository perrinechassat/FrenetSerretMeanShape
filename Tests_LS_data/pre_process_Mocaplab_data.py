import sys
import os.path
sys.path.insert(1, '../FrenetSerretMeanShape')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frenet_path import *
from trajectory import *
from model_curvatures import *
from estimation_algo_utils import *
from maths_utils import *
from simu_utils import *
from visu_utils import *
from signal_utils import *
from pickle import *
import dill as pickle
from scipy.interpolate import UnivariateSpline

def barycenter_from_3ptsHand(pathFile, plot=True, hand='Right'):

    csvFile = pd.read_csv(pathFile, sep=';')
    # print(csvFile)
    objDic = {}
    if not csvFile.columns[0].startswith('Unnamed'):
        # print(csvFile.columns[0])
        for col in csvFile.columns:
            # print(csvFile[col][0])
            # print(col)
            # # print(csvFile[col])
            mkrtest = csvFile[col][0]
            # # print(mkrtest)
            axe = csvFile[col][1]
            # mkrtest = col
            # axe = csvFile[col][0]
            if isinstance(mkrtest, str):
                mkr = mkrtest
                objDic[mkr] = pd.DataFrame(columns=[axe])
            objDic[mkr][axe] = [float(i) for i in csvFile[col][2:]]
    else:
        for col in csvFile.columns:
            if col.startswith('Unnamed'):
                axe = csvFile[col][0]
                if axe!='y' and axe!='z':
                    objDic[axe] = pd.DataFrame(columns=[axe])
                else:
                    objDic[mkr][axe] = [float(i) for i in csvFile[col][1:]]
            else:
                mkr = col
                axe = csvFile[col][0]
                objDic[mkr] = pd.DataFrame(columns=[axe])
                objDic[mkr][axe] = [float(i) for i in csvFile[col][1:]]

    if hand=='Right' or hand=='right':
        a = objDic['Rwrist']
        b = objDic['Rindex0']
        c = objDic['Rring0']
    elif hand=='Left' or hand=='left':
        a = objDic['Lwrist']
        b = objDic['Lindex0']
        c = objDic['Lring0']
    else:
        print('The parameter "hand" needs to be "Right" or "Left". ')

    barycentre = a
    barycentre['x'] = (a['x'] + b['x'] + c['x'])/3
    barycentre['y'] = (a['y'] + b['y'] + c['y'])/3
    barycentre['z'] = (a['z'] + b['z'] + c['z'])/3

    if plot==True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1,  projection='3d')
        ax1.plot(barycentre['x'].values, barycentre['y'].values, barycentre['z'].values, color='blue')
        plt.show()

    return barycentre


def take_numpy_subset(barycentre, t_min, t_max):
    subset = barycentre[t_min:t_max]
    return subset.to_numpy()


def cut_at_peaks(data_traj, nb_peaks):
    t = np.linspace(0,1,len(data_traj))
    X = Trajectory(data_traj, t)
    X.loc_poly_estimation(X.t, 5, 0.01)
    X.compute_S(scale=True)
    # t_new = np.linspace(0,1,3000)
    Q_GS = X.TNB_GramSchmidt(t)

    # Estimate raw curvature and torsion
    h = 0.002
    Q_GS.compute_neighbors(h)
    mKappa, mTau, mS, mOmega, gam, ind_conv = compute_raw_curvatures(Q_GS, h, Q_GS, False)

    bornes_s = bornes_peaks(mS, mKappa, nb_peaks)
    n = len(bornes_s)
    s = X.S(t)
    parts = [data_traj[np.where(s<=bornes_s[0]),:][0]]
    for i in range(n-1):
        ind = np.intersect1d(np.where(s>=bornes_s[i]), np.where(s<=bornes_s[i+1]))
        parts.append(data_traj[ind,:])
    parts.append(data_traj[np.where(s>=bornes_s[-1]),:][0])
    return parts


def cut_regular_parts(data_traj, n_pts):
    n = data_traj.shape[0]
    # print(n)
    n_pts_inter = int(n_pts/4)
    # print(n_pts_inter)
    if n<n_pts or n_pts<4:
        return [data_traj]
    else:
        parts = []
        q = int(np.around(n/(n_pts-n_pts_inter),0))
        # print(q)
        for i in range(q-1):
            # print(i*(n_pts-n_pts_inter),i*(n_pts-n_pts_inter)+n_pts)
            parts.append(data_traj[i*(n_pts-n_pts_inter):i*(n_pts-n_pts_inter)+n_pts])
        # print((q-1)*(n_pts-n_pts_inter),n)
        parts.append(data_traj[(q-1)*(n_pts-n_pts_inter):])
        return parts


def join_parts(array_Q):
    n_parts = array_Q.shape[0]

    knots_curv = array_Q[0].curv._eval_args[0]
    coeffs_curv = array_Q[0].curv._eval_args[1]
    knots_tors = array_Q[0].tors._eval_args[0]
    coeffs_tors = array_Q[0].tors._eval_args[1]

    for i in range(1,n_parts):
        knots_curv = np.concatenate((knots_curv, array_Q[i].curv._eval_args[0]))
        coeffs_curv = np.concatenate((coeffs_curv, array_Q[i].curv._eval_args[1]))
        knots_tors = np.concatenate((knots_tors, array_Q[i].tors._eval_args[0]))
        coeffs_tors = np.concatenate((coeffs_tors, array_Q[i].tors._eval_args[1]))

    tck_curv = (knots_curv, coeffs_curv, 4)
    tck_tors = (knots_tors, coeffs_tors, 4)
    spl_curv = UnivariateSpline._from_tck(tck_curv)
    spl_tors = UnivariateSpline._from_tck(tck_tors)

    return spl_curv, spl_tors
