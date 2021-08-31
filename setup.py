import setuptools
import numpy
import sys, os
import platform
from distutils.core import setup
from distutils.core import Command
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import dist
from distutils.sysconfig import get_config_var, get_python_inc
from distutils.version import LooseVersion

sys.path.insert(1, 'Sources/')
# import dp_build_C

# Make sure I have the right Python version.
if sys.version_info[:2] < (3, 6):
    print(("FrenetSerretMeanShape requires Python 3.6 or newer. Python %d.%d detected" % sys.version_info[:2]))
    sys.exit(-1)


if (sys.platform == 'darwin'):
    mac_ver = str(LooseVersion(get_config_var('MACOSX_DEPLOYMENT_TARGET')))
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = mac_ver

extensions = [
	Extension(name="optimum_reparamN2_C",
	    sources=["Sources/optimum_reparamN2_C.pyx", "Sources/DynamicProgrammingQ2_C.c",
        "Sources/dp_grid_C.c", "Sources/dp_nbhd_C.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
	Extension(name="fpls_warp_C",
	    sources=["Sources/fpls_warp_C.pyx", "Sources/fpls_warp_grad_C.c", "Sources/misc_funcs_C.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
	Extension(name="mlogit_warp_C",
	    sources=["Sources/mlogit_warp_C.pyx", "Sources/mlogit_warp_grad_C.c", "Sources/misc_funcs_C.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
	Extension(name="ocmlogit_warp_C",
	    sources=["Sources/ocmlogit_warp_C.pyx", "Sources/ocmlogit_warp_grad_C.c", "Sources/misc_funcs_C.c"],
	    include_dirs=[numpy.get_include()],
	    language="c"
	),
    Extension(name="oclogit_warp_C",
        sources=["Sources/oclogit_warp_C.pyx", "Sources/oclogit_warp_grad_C.c", "Sources/misc_funcs_C.c"],
        include_dirs=[numpy.get_include()],
        language="c"
    ),
    # Extension(name="optimum_reparam_N_C",
    #     sources=["Sources/optimum_reparam_N_C.pyx", "Sources/DP_C.c"],
    #     include_dirs=[numpy.get_include()],
    #     language="c"
    # ),
    Extension(name="cbayesian_C",
        sources=["Sources/cbayesian_C.pyx", "Sources/bayesian_C.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++"
    ),
    # dp_build_C.ffibuilder.distutils_extension(),
]


setup(
    cmdclass={'build_ext': build_ext},
	ext_modules=extensions,
    name='FrenetSerretMeanShape',
    version='1.0',
    packages=['FrenetSerretMeanShape'],
    author='Perrine Chassat',
    description='Method for computing the mean of a set of Frenet paths',
    setup_requires=['Cython','numpy',"cffi>=1.0.0"],
    install_requires=[
        "Cython",
        "matplotlib",
        "numpy",
        "scipy",
        "geomstats",
        # "skfda",
        "fdasrsf",
        "joblib",
        "patsy",
        "tqdm",
        "six",
        "numba",
        "cffi>=1.0.0",
        "pyparsing",
        "torch",
        # "timeit",
        # "skopt",
        "sklearn",
        "plotly",
    ],
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ]
)
