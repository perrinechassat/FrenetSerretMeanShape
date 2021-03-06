import numpy as np
from scipy.linalg import expm
from scipy import interpolate
from scipy.interpolate import UnivariateSpline, splev, splrep
from scipy.optimize import minimize
from skfda.representation.grid import FDataGrid
from skfda.representation import FDataBasis
from skfda.representation.basis import BSpline
from skfda import preprocessing
from skfda.misc.regularization import L2Regularization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic
from skfda.misc.regularization import TikhonovRegularization
from skfda.misc.operators import LinearDifferentialOperator


# class BasisSmoother:
#
#     """
#     A class used to define a Bspline smoother
#
#     ...
#
#     Attributes
#     ----------
#     basis : Bspline basis representation initialized with an order, a num of basis functions, a number of knots and the range of the domain
#     function : the smoothed function, initialized to 0
#     variances : array, weights for the smoothing
#     smoothing_parameter : float, parameter for the penalization
#     fd_basis : estimated basis after the smoothing
#     coefficients : estimated coefficients after the smoothing
#
#     Methods
#     -------
#     reinitialize():
#         put the argument function to 0.
#
#     smoothing(self, grid_pts, data_pts, weights, smoothing_parameter):
#         do a Bspline basis smoothing of the "data_pts".
#     """
#
#     def __init__(self, basis_type='bspline', domain_range=None, nb_basis=None, order=4, knots=None):
#         if basis_type=='bspline':
#             self.basis = BSpline(domain_range=domain_range, n_basis=nb_basis, order=order, knots=knots)
#             # self.basis = BSpline(domain_range=domain_range, order=order)
#         else:
#             raise ValueError("basis type does not exist")
#         def f_init(x): return 0*x
#         self.function = f_init
#
#     def reinitialize(self):
#         def f_init(x): return 0*x
#         self.function = lambda x: 0*x
#
#     def smoothing(self, grid_pts, data_pts, weights, smoothing_parameter):
#         self.variances = weights
#         self.smoothing_parameter = smoothing_parameter
#         fd = FDataGrid(data_matrix=data_pts, grid_points=grid_pts, extrapolation="bounds")
#         self.smoother = preprocessing.smoothing.BasisSmoother(self.basis, smoothing_parameter=self.smoothing_parameter, regularization=TikhonovRegularization(LinearDifferentialOperator(2)), weights=np.diag(self.variances), return_basis=True, method='cholesky')
#         self.fd_basis = self.smoother.fit_transform(fd)
#         self.coefficients = self.fd_basis.coefficients
#         def f(x): return np.squeeze(self.fd_basis.evaluate(x))
#         self.function = f
#         return np.squeeze(self.coefficients) #diff??rent de GPR peut ??tre changer ca



class BasisSmoother:

    """
    A class used to define a Bspline smoother

    ...

    Attributes
    ----------
    basis : Bspline basis representation initialized with an order, a num of basis functions, a number of knots and the range of the domain
    function : the smoothed function, initialized to 0
    variances : array, weights for the smoothing
    smoothing_parameter : float, parameter for the penalization
    fd_basis : estimated basis after the smoothing
    coefficients : estimated coefficients after the smoothing

    Methods
    -------
    reinitialize():
        put the argument function to 0.

    smoothing(self, grid_pts, data_pts, weights, smoothing_parameter):
        do a Bspline basis smoothing of the "data_pts".
    """

    def __init__(self, basis_type='bspline', domain_range=None, nb_basis=None, order=4, knots=None):
        self.basis_type = basis_type
        self.domain_range = domain_range
        self.nb_basis = nb_basis
        self.order = order
        self.knots = knots

        if basis_type!='bspline':
            raise ValueError("basis type does not exist")

        def f_init(x): return 0*x
        self.function = f_init

    def reinitialize(self): #, knots=None, nb_basis=None):
        def f_init(x): return 0*x
        self.function = f_init
        # self.knots = knots
        # self.nb_basis = nb_basis

    def smoothing(self, grid_pts, data_pts, weights, smoothing_parameter):
        self.variances = weights
        self.smoothing_parameter = smoothing_parameter

        if (self.knots is None) and (self.nb_basis is None):
            tck = splrep(x=grid_pts, y=data_pts, w=self.variances, k=self.order-1, s=self.smoothing_parameter)
            self.knots = tck[0][tck[2]:-tck[2]]

            basis = BSpline(domain_range=self.domain_range, order=self.order, knots=self.knots)
            self.nb_basis = basis.n_basis
        else:
            basis = BSpline(domain_range=self.domain_range, n_basis=self.nb_basis, order=self.order, knots=self.knots)
            self.nb_basis = basis.n_basis
            self.knots = basis.knots

        fd = FDataGrid(data_matrix=data_pts, grid_points=grid_pts, extrapolation="bounds")
        smoother = preprocessing.smoothing.BasisSmoother(basis, smoothing_parameter=self.smoothing_parameter, regularization=TikhonovRegularization(LinearDifferentialOperator(2)), weights=np.diag(self.variances), return_basis=True, method='cholesky')
        fd_basis = smoother.fit_transform(fd)
        self.coefficients = fd_basis.coefficients
        def f(x):
            return BSpline(domain_range=self.domain_range, n_basis=self.nb_basis, knots=self.knots, order=self.order)._to_scipy_bspline(self.coefficients.squeeze())(x)
            # fd_b = FDataBasis(basis=BSpline(domain_range=self.domain_range, n_basis=self.nb_basis, knots=self.knots, order=self.order), coefficients=self.coefficients)
            # return np.squeeze(fd_b.evaluate(x))
        self.function = f
        return np.squeeze(self.coefficients) #diff??rent de GPR peut ??tre changer ca



class BasisSmoother_scipy:

    """
    A class used to define a Bspline smoother

    ...

    Attributes
    ----------
    basis : Bspline basis representation initialized with an order, a num of basis functions, a number of knots and the range of the domain
    function : the smoothed function, initialized to 0
    variances : array, weights for the smoothing
    smoothing_parameter : float, parameter for the penalization
    fd_basis : estimated basis after the smoothing
    coefficients : estimated coefficients after the smoothing

    Methods
    -------
    reinitialize():
        put the argument function to 0.

    smoothing(self, grid_pts, data_pts, weights, smoothing_parameter):
        do a Bspline basis smoothing of the "data_pts".
    """

    def __init__(self, domain_range=None, nb_basis=None, order=4, knots=None):
        def f_init(x): return 0*x
        self.function = f_init
        self.order = order

    def reinitialize(self):
        def f_init(x): return 0*x
        self.function = f_init

    def smoothing(self, grid_pts, data_pts, weights, smoothing_parameter):
        self.variances = weights
        self.smoothing_parameter = smoothing_parameter
        # self.function = UnivariateSpline(grid_pts, data_pts, w=self.variances, k=self.order, s=self.smoothing_parameter, check_finite=True)
        # return self.function.get_coeffs() #diff??rent de GPR peut ??tre changer ca

        # self.tck = splrep(x=grid_pts, y=data_pts, w=self.variances, k=self.order, s=self.smoothing_parameter)
        # def f(x): return splev(x, self.tck)
        # self.function = f
        # return self.tck[1]

        tck = splrep(x=grid_pts, y=data_pts, w=self.variances, k=self.order-1, s=self.smoothing_parameter)
        self.tck = tck
        skfda_bspline = BSpline._from_scipy_bspline(interpolate.BSpline(tck[0], tck[1], tck[2]))
        self.fd_basis = FDataBasis(skfda_bspline[0], skfda_bspline[1][:-(tck[2]+1)])
        # def f(x): return np.squeeze(self.fd_basis.evaluate(x))
        def f(x): return splev(x, self.tck)
        self.function = f
        # skfda_bspline = BSpline._from_scipy_bspline(interpolate.BSpline(self.tck[0], self.tck[1], self.tck[2]))
        # self.fd_basis = FDataBasis(skfda_bspline[0], skfda_bspline[1][:-(self.tck[2]+1)])

        return np.squeeze(self.fd_basis.coefficients)




class WaveletSmoother:

    """
    A class used to define a Wavelet smoother

    ...

    Attributes
    ----------
    wavelet : Wavelet basis representation
    function : the smoothed function, initialized to 0
    threshold : float, parameter for the denoising
    coefficients : estimated coefficients after the smoothing

    Methods
    -------
    reinitialize():
        put the argument function to 0.

    smoothing(self, grid_pts, data_pts, threshold):
        do a Wavelet smoothing of the "data_pts".
    """

    def __init__(self, wavelet='db8', domain_range=None):
        self.wavelet = wavelet
        def f_init(x): return 0
        self.function = f_init

    def reinitialize(self):
        def f_init(x): return 0
        self.function = f_init

    def smoothing(self, grid_pts, data_pts, threshold):
        self.threshold = threshold*np.nanmax(data_pts)
        self.coefficients = pywt.wavedec(data_pts, self.wavelet, mode="per")
        self.coefficients[1:] = (pywt.threshold(i, value=self.threshold, mode="soft") for i in self.coefficients[1:])
        reconstructed_signal = pywt.waverec(self.coefficients, self.wavelet, mode="per")
        def f(x): return interpolate.interp1d(grid_pts, reconstructed_signal[:-1])
        self.function = f
        return np.squeeze(self.coefficients) #diff??rent de GPR peut ??tre changer ca

# def mykernel(x,y):
#     Kern = lambda x: (3/4)*(1-np.power(x,2))

class GaussianProcessSmoother:
    """
    A class used to define a Gaussian Process Regressor

    ...

    Attributes
    ----------
    gpr : Gaussian Process regressor
    kernel : kernel function
    function : the smoothed function, initialized to 0
    variances : array, weights for the smoothing

    Methods
    -------
    reinitialize():
        put the argument function to 0.

    smoothing(self, grid_pts, data_pts, weights, smoothing_parameter):
        do a Gaussian Process regression of the "data_pts".
    """

    def __init__(self, kernel, random_state=0):
        self.kernel = kernel
        self.gpr = GaussianProcessRegressor(random_state=0, optimizer=None)
        def f_init(x): return 0
        self.function = f_init

    def reinitialize(self):
        def f_init(x): return 0
        self.function = f_init


    def smoothing(self, grid_pts, data_pts, variances, smoothing_parameter):
        # self.variances = np.reciprocal(variances, where=(variances!=0))
        self.variances = variances
        param = {"alpha":self.variances, "kernel":self.kernel(smoothing_parameter)}
        self.gpr.set_params(**param)
        self.gpr = self.gpr.fit(grid_pts[:,np.newaxis], data_pts)
        def f(x):
            if isinstance(x,(list,np.ndarray)):
                return np.squeeze(self.gpr.predict(x[:,np.newaxis]))
            else:
                return np.squeeze(self.gpr.predict(np.expand_dims(x, axis=(0, 1))))
        self.function = f
        return np.squeeze(self.gpr.predict(grid_pts[:,np.newaxis]))



class Model:
    """
    A class used to represent a Model for curvature and torsion
    ...

    Attributes
    ----------
    curv : instance of BasisSmoother or GaussianProcessSmoother
    tors : instance of BasisSmoother or GaussianProcessSmoother

    """

    def __init__(self, curv_basis_smoother, tors_basis_smoother):
        self.curv = curv_basis_smoother
        self.tors = tors_basis_smoother

    def set_gam_functions(self, gam):
        self.gam = gam
