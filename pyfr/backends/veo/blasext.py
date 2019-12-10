# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.veo.provider import VeoKernelProvider
from pyfr.backends.base import ComputeKernel


class VeoBlasExtKernels(VeoKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ncol, ldim, dtype = arr[0].traits
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*3 + [np.intp]*nv + [dtype]*nv)

        class AxnpbyKernel(ComputeKernel):
            def run(self, queue, *consts):
                args = [x.data for x in arr] + list(consts)
                queue.call(kern, nrow, ncolb, ldim, *args)

        return AxnpbyKernel()

    def copy(self, dst, src):
        if dst.traits != src.traits:
            raise ValueError('Incompatible matrix types')

        if dst.nbytes >= 2**31:
            raise ValueError('Matrix too large for copy')

        # Render the kernel template
        ksrc = self.backend.lookup.get_template('par-memcpy').render()

        # Build the kernel
        kern = self._build_kernel('par_memcpy', ksrc,
                                  [np.intp, np.intp, np.int32])

        class CopyKernel(ComputeKernel):
            def run(self, queue):
                queue.call(kern, dst.data, src.data, dst.nbytes)

        return CopyKernel()

    def errest(self, x, y, z, *, norm):
        if x.traits != y.traits != z.traits:
            raise ValueError('Incompatible matrix types')

        nrow, ncol, ldim, dtype = x.traits
        ncola, ncolb = x.ioshape[1:]

        # Render the reduction kernel template
        src = self.backend.lookup.get_template('errest').render(norm=norm,
                                                                ncola=ncola)

        # Empty result buffer on host with nvars elements
        err_host = np.empty(ncola, dtype)

        # Paired device memory allocation
        err_dev = self.backend.proc.alloc_mem(err_host.nbytes)

        # Build
        rkern = self._build_kernel(
            'errest', src, [np.int32]*3 + [np.intp]*4 + [dtype]*2
        )

        class ErrestKernel(ComputeKernel):
            @property
            def retval(self):
                return err_host

            def run(self, queue, atol, rtol):
                queue.call(rkern, nrow, ncolb, ldim, err_dev, x.data, y.data,
                           z.data, atol, rtol)
                queue.memcpy_dtoh(err_host, err_dev, err_host.nbytes)

        return ErrestKernel()
