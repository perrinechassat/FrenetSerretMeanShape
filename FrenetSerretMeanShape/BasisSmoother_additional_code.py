from __future__ import annotations
import sys
import os.path
sys.path.insert(1, './.local/lib/python3.8/site-packages/skfda')
from enum import Enum
from typing import Union, Iterable
from typing import Callable, Optional
import numpy as np
from typing_extensions import Final, Literal
import scipy.linalg
from skfda._utils import _cartesian_product
# from skfda.misc.lstsq import LstsqMethod, solve_regularized_weighted_lstsq
from skfda.misc.regularization import TikhonovRegularization, compute_penalty_matrix
from skfda.representation import FData, FDataBasis, FDataGrid
from skfda.preprocessing.smoothing._basis import _Cholesky, _QR, _Matrix
# from skfda.representation._typing import GridPointsLike
from skfda.representation.basis import Basis


#
def solve_regularized_weighted_lstsq_double(
    coefs1: np.ndarray,
    result1: np.ndarray,
    coefs2: np.ndarray,
    result2: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    penalty_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve a regularized and weighted least squares problem.
    If the penalty matrix is not ``None`` and nonzero, there
    is a closed solution. Otherwise the problem can be reduced
    to a least squares problem.
    """

    print('shape data matrix', result1.shape)
    print(result1.shape)

    # Cholesky case (always used for the regularized case)
    if weights is None:
        left = coefs1.T @ coefs1 + coefs2.T @ coefs2
        right = coefs1.T @ result1 + coefs2.T @ result2
    else:
        left = coefs1.T @ weights @ coefs1 + coefs2.T @ weights @ coefs2
        right = coefs1.T @ weights @ result1 + coefs2.T @ weights @ result2

    if penalty_matrix is not None:
        left += penalty_matrix

    return scipy.linalg.solve(left, right, assume_a="pos")

#
#
# class BasisSmootherDouble:
#
#     def __init__(
#         self,
#         basis1: Basis,
#         basis2: Basis,
#         *,
#         smoothing_parameter: float = 1.0,
#         weights: Optional[np.ndarray] = None,
#         regularization: Optional[TikhonovRegularization[FDataGrid]] = None,
#         output_points: Optional[GridPointsLike] = None,
#         method: LstsqMethod = 'svd',
#         return_basis: bool = False,
#     ) -> None:
#         self.basis1 = basis1
#         self.basis2 = basis2
#         self.smoothing_parameter = smoothing_parameter
#         self.weights = weights
#         self.regularization = regularization
#         self.output_points = output_points
#         self.method = method
#         self.return_basis: Final = return_basis
#
#
#     def _coef_matrix(
#         self,
#         input_points1: GridPointsLike,
#         input_points2: GridPointsLike,
#         *,
#         data_matrix: Optional[np.ndarray] = None,
#     ) -> np.ndarray:
#         """Get the matrix that gives the coefficients."""
#
#         # basis_values_input1 = self.basis1.evaluate(
#         #     _cartesian_product(_to_grid_points(input_points1)),
#         # ).reshape((self.basis1.n_basis, -1)).T
#         #
#         # basis_values_input2 = self.basis2.evaluate(
#         #     _cartesian_product(_to_grid_points(input_points2)),
#         # ).reshape((self.basis2.n_basis, -1)).T
#         basis_values_input1 = self.basis1.evaluate(
#             _cartesian_product(input_points1)).reshape(
#             (self.basis1.n_basis, -1)).T
#         basis_values_input2 = self.basis2.evaluate(
#             _cartesian_product(input_points2)).reshape(
#             (self.basis2.n_basis, -1)).T
#
#         penalty_matrix = compute_penalty_matrix(
#             basis_iterable=(self.basis1,),
#             regularization_parameter=self.smoothing_parameter,
#             regularization=self.regularization,
#         )
#
#         # Get the matrix for computing the coefficients if no
#         # data_matrix is passed
#         if data_matrix1 is None:
#             data_matrix1 = np.eye(basis_values_input1.shape[0])
#         if data_matrix2 is None:
#             data_matrix2 = np.eye(basis_values_input2.shape[0])
#
#         return solve_regularized_weighted_lstsq_double(
#             coefs1=basis_values_input1,
#             result1=data_matrix1,
#             coefs2=basis_values_input2,
#             result2=data_matrix2,
#             weights=self.weights,
#             penalty_matrix=penalty_matrix,
#             lstsq_method=self.method,
#         )
#
#     def _hat_matrix(
#         self,
#         input_points1: GridPointsLike,
#         input_points2: GridPointsLike,
#         output_points: GridPointsLike,
#     ) -> np.ndarray:
#         basis_values_output = self.basis1.evaluate(
#             _cartesian_product(
#                 _to_grid_points(output_points),
#             ),
#         ).reshape((self.basis1.n_basis, -1)).T
#
#         return basis_values_output @ self._coef_matrix(input_points1, input_points2)
#
#     # def fit(
#     #     self,
#     #     X1: FDataGrid,
#     #     X2: FDataGrid,
#     #     y: None = None,
#     # ) -> BasisSmoother:
#     #     """Compute the hat matrix for the desired output points.
#     #
#     #     Args:
#     #         X: The data whose points are used to compute the matrix.
#     #         y: Ignored.
#     #
#     #     Returns:
#     #         self
#     #
#     #     """
#     #     self.input_points_1 = X1.grid_points
#     #     self.input_points_2 = X2.grid_points
#     #     self.output_points_ = (
#     #         _to_grid_points(self.output_points)
#     #         if self.output_points is not None
#     #         else self.input_points_1
#     #     )
#     #
#     #     if not self.return_basis:
#     #         self.hat_matrix_ = self._hat_matrix(self.input_points_1, self.input_points_2, self.output_points_)
#     #
#     #
#     #     return self
#
#
#     def transform(
#         self,
#         X1: FDataGrid,
#         X2: FDataGrid,
#         y: None = None,
#     ) -> FData:
#         """
#         Smooth the data.
#
#         Args:
#             X: The data to smooth.
#             y: Ignored
#
#         Returns:
#             Smoothed data.
#
#         """
#         assert all(
#             np.array_equal(i, s) for i, s in zip(
#                 self.input_points_,
#                 X1.grid_points,
#             )
#         )
#
#         coefficients = self._coef_matrix(
#             input_points1=X1.grid_points,
#             input_points2=X2.grid_points,
#             data_matrix1=X1.data_matrix.reshape((X1.n_samples, -1)).T,
#             data_matrix2=X2.data_matrix.reshape((X2.n_samples, -1)).T,
#         ).T
#
#         return FDataBasis(
#             basis=self.basis1,
#             coefficients=coefficients,
#             dataset_name=X1.dataset_name,
#             argument_names=X1.argument_names,
#             coordinate_names=X1.coordinate_names,
#             sample_names=X1.sample_names,
#         )
#
#
#


class BasisSmootherDouble:

    _required_parameters = ["basis"]

    class SolverMethod(Enum):
        cholesky = _Cholesky()
        qr = _QR()
        matrix = _Matrix()

    def __init__(self,
                 basis1,
                 basis2,
                 *,
                 smoothing_parameter: float = 1.,
                 weights=None,
                 regularization: Union[int, Iterable[float],
                                       'LinearDifferentialOperator'] = None,
                 output_points=None,
                 method='cholesky',
                 return_basis=False):
        self.basis1 = basis1
        self.basis2 = basis2
        self.smoothing_parameter = smoothing_parameter
        self.weights = weights
        self.regularization = regularization
        self.output_points = output_points
        self.method = method
        self.return_basis = return_basis

    def _method_function(self):
        """ Return the method function"""
        method_function = self.method
        if not isinstance(method_function, self.SolverMethod):
            method_function = self.SolverMethod[
                method_function.lower()]

        return method_function.value

    # def _coef_matrix(self, input_points1, input_points2):
    #     """Get the matrix that gives the coefficients"""
    #
    #     basis_values_input1 = self.basis1.evaluate(
    #         _cartesian_product(input_points1)).reshape(
    #         (self.basis1.n_basis, -1)).T
    #     basis_values_input2 = self.basis2.evaluate(
    #         _cartesian_product(input_points2)).reshape(
    #         (self.basis2.n_basis+1, -1)).T
    #
    #     # If no weight matrix is given all the weights are one
    #     if self.weights is not None:
    #         ols_matrix = (basis_values_input1.T @ self.weights @ basis_values_input1) + (basis_values_input2.T @ self.weights @ basis_values_input2)
    #     else:
    #         ols_matrix = basis_values_input1.T @ basis_values_input1 + basis_values_input2.T @ basis_values_input2
    #
    #     penalty_matrix = compute_penalty_matrix(
    #         basis_iterable=(self.basis1,),
    #         regularization_parameter=self.smoothing_parameter,
    #         regularization=self.regularization)
    #
    #     ols_matrix += penalty_matrix
    #
    #     right_side = basis_values_input1.T + basis_values_input2.T
    #     print('3')
    #     print(right_side.shape)
    #     print(basis_values_input1.T.shape)
    #     if self.weights is not None:
    #         right_side = right_side @ self.weights
    #         print('4')
    #         print(right_side.shape)
    #         print((basis_values_input1.T @ self.weights).shape)
    #
    #     return np.linalg.solve(
    #         ols_matrix, right_side)

    def _coef_matrix(
        self,
        input_points1: GridPointsLike,
        input_points2: GridPointsLike,
        *,
        data_matrix1: Optional[np.ndarray] = None,
        data_matrix2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get the matrix that gives the coefficients."""

        # basis_values_input1 = self.basis1.evaluate(
        #     _cartesian_product(_to_grid_points(input_points1)),
        # ).reshape((self.basis1.n_basis, -1)).T
        #
        # basis_values_input2 = self.basis2.evaluate(
        #     _cartesian_product(_to_grid_points(input_points2)),
        # ).reshape((self.basis2.n_basis, -1)).T
        basis_values_input1 = self.basis1.evaluate(
            _cartesian_product(input_points1)).reshape(
            (self.basis1.n_basis, -1)).T
        # print(self.basis1.evaluate(_cartesian_product(input_points1)).shape)
        # print(self.basis1.n_basis)
        # print(self.basis2.evaluate(_cartesian_product(input_points2)).shape)
        # print(self.basis2.n_basis)
        basis_values_input2 = self.basis2.evaluate(
            _cartesian_product(input_points2)).reshape(
            (self.basis2.n_basis+1, -1)).T

        penalty_matrix = compute_penalty_matrix(
            basis_iterable=(self.basis1,),
            regularization_parameter=self.smoothing_parameter,
            regularization=self.regularization,
        )

        # Get the matrix for computing the coefficients if no
        # data_matrix is passed
        if data_matrix1 is None:
            data_matrix1 = np.eye(basis_values_input1.shape[0])
        if data_matrix2 is None:
            data_matrix2 = np.eye(basis_values_input2.shape[0])

        return solve_regularized_weighted_lstsq_double(
            coefs1=basis_values_input1,
            result1=data_matrix1[0],
            coefs2=basis_values_input2,
            result2=data_matrix2[0],
            weights=self.weights,
            penalty_matrix=penalty_matrix
        )

    def _hat_matrix(self, input_points1, input_points2, output_points):
        basis_values_output = self.basis1.evaluate(_cartesian_product(
            output_points)).reshape(
            (self.basis1.n_basis, -1)).T

        return basis_values_output @ self._coef_matrix(input_points1, input_points2)


    def transform(
        self,
        X1: FDataGrid,
        X2: FDataGrid,
        y: None = None,
    ) -> FData:
        """
        Smooth the data.

        Args:
            X: The data to smooth.
            y: Ignored

        Returns:
            Smoothed data.

        """

        coefficients = self._coef_matrix(
            input_points1=X1.grid_points,
            input_points2=X2.grid_points,
            data_matrix1=X1.data_matrix,
            data_matrix2=X2.data_matrix).T

        print(coefficients.shape)

        return FDataBasis(
            basis=self.basis1,
            coefficients=coefficients,
            # dataset_name=X1.dataset_name,
            # argument_names=X1.argument_names,
            # coordinate_names=X1.coordinate_names,
            # sample_names=X1.sample_names,
        )
