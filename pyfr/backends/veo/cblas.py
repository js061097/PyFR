# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import ComputeKernel
from pyfr.backends.veo.provider import VeoKernelProvider


class VeoCBLASWrappers(object):
    ROW_MAJOR = 101
    COL_MAJOR = 102

    NO_TRANS = 111
    TRANS = 112
    CONJ_TRANS = 113

    def __init__(self, backend, libname):
        try:
            lib = backend.proc.load_library(libname)
        except RuntimeError:
            raise RuntimeError('Unable to load cblas')

        # cblas_dgemm
        self.cblas_dgemm = lib.find_function('cblas_dgemm')
        self.cblas_dgemm.ret_type('void')
        self.cblas_dgemm.args_type(
            'int', 'int', 'int', 'int', 'int', 'int',
            'double', 'double *', 'int', 'double *', 'int',
            'double', 'double *', 'int'
        )

        # cblas_sgemm
        self.cblas_sgemm = lib.find_function('cblas_sgemm')
        self.cblas_sgemm.ret_type('void')
        self.cblas_sgemm.args_type(
            'int', 'int', 'int', 'int', 'int', 'int',
            'float', 'float *', 'int', 'float *', 'int',
            'float', 'float *', 'int'
        )


class VeoCBLASKernels(VeoKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        libname = backend.cfg.getpath('backend-veo', 'vecblas')

        # Load and wrap blas
        self._wrappers = VeoCBLASWrappers(backend, libname)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        w = self._wrappers

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = a.nrow, b.ncol, a.ncol

        if a.dtype == np.float64:
            cblas_gemm = w.cblas_dgemm
        else:
            cblas_gemm = w.cblas_sgemm

        class MulKernel(ComputeKernel):
            def run(self, queue):
                queue.call(cblas_gemm, w.ROW_MAJOR, w.NO_TRANS, w.NO_TRANS,
                           m, n, k, alpha, a.data, a.leaddim, b.data,
                           b.leaddim, beta, out.data, out.leaddim)

        return MulKernel()
