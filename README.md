# FrenetSerretMeanShape
=====================

A python package for estimating the mean geometry and mean shape from a set of multi-dimensional curves. The mean curvature, the mean torsion, as well as the mean shape are defined by the notion of mean vector field, based on a geometric representation of the curves by the ordinary Frenet-Serret differential equations. This formulation of the mean for multidimensional curves allows the parameters of the shape characteristics to be integrated into the unified framework of functional data modelling. The functional parameter estimation problem is formulated in a penalised regression.  

### Requirements

"Cython",
"matplotlib",
"numpy",
"scipy",
"geomstats",
"skfda",
"fdasrsf",
"joblib",
"patsy",
"tqdm",
"six",
"numba",
"cffi>=1.0.0",
"pyparsing",
"torch",
"timeit",
"skopt",
"sklearn",
"plotly"

### Installation

To install this version
> `python setup.py install`

------------------------------------------------------------------------------
### License
This package use some codes (folder Sources) of the package fdasrsf that is licensed under the BSD 3-Clause License.

### References
Brunel, N. and J. Park (2019). The frenet-serret framework for aligning geometric curves.  InF. Nielsen and F. Barbaresco (Eds.),Geometric Science of Information, pp. 608–617. Springer.

Park, J. and Brunel, N. Mean curvature and mean shape for multivariate functional data under Frenet-Serret framework. arXiv:1910.12049, 2019.

Tucker, J. D., package "fdasrsf" [website](https://github.com/jdtuck/fdasrsf_python), references about it can be found at this [website](http://research.tetonedge.net)
