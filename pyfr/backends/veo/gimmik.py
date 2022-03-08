# -*- coding: utf-8 -*-

from gimmik import generate_mm
import numpy as np

from pyfr.backends.base import Kernel, NotSuitableError
from pyfr.backends.veo.provider import VeoKernelProvider


class VeoGiMMiKKernels(VeoKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.max_nnz = backend.cfg.getint('backend-veo', 'gimmik-max-nnz',
                                          512)

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Check that A is reasonably sparse
        if np.count_nonzero(a.get()) > self.max_nnz:
            raise NotSuitableError('Matrix too dense for GiMMiK')

        # Generate the GiMMiK kernel
        src = generate_mm(a.get(), dtype=a.dtype, platform='c-omp',
                          alpha=alpha, beta=beta)


        src = src.replace('#pragma omp parallel for simd private(dotp)',
                          '#pragma omp parallel for private(dotp)\n    '
                          '#pragma _NEC ivdep')

        gimmik_mm = self._build_kernel('gimmik_mm', src,
                                       [np.int32] + [np.intp, np.int32]*2)

        class MulKernel(Kernel):
            def run(self, queue):
                queue.call(gimmik_mm, b.ncol, b.data, b.leaddim,
                           out.data, out.leaddim)

        return MulKernel()
