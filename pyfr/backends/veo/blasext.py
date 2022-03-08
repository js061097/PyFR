# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.veo.provider import VeoKernelProvider
from pyfr.backends.base import Kernel


class VeoBlasExtKernels(VeoKernelProvider):
    def axnpby(self, *arr, subdims=None):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        nrow, ncol, ldim, dtype = arr[0].traits[1:]
        ncola, ncolb = arr[0].ioshape[1:]

        # Render the kernel template
        src = self.backend.lookup.get_template('axnpby').render(
            subdims=subdims or range(ncola), ncola=ncola, nv=nv
        )

        # Build the kernel
        kern = self._build_kernel('axnpby', src,
                                  [np.int32]*3 + [np.intp]*nv + [dtype]*nv)

        class AxnpbyKernel(Kernel):
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

        class CopyKernel(Kernel):
            def run(self, queue):
                queue.call(kern, dst.data, src.data, dst.nbytes)

        return CopyKernel()

    def reduction(self, *rs, method, norm, dt_mat=None):
        if any(r.traits != rs[0].traits for r in rs[1:]):
            raise ValueError('Incompatible matrix types')

        nrow, ncol, ldim, dtype = rs[0].traits[1:]
        ncola, ncolb = rs[0].ioshape[1:]

        # Empty result buffer on host with nvars elements
        err_host = np.empty(ncola, dtype)

        # Paired device memory allocation
        err_dev = self.backend.proc.alloc_mem(err_host.nbytes)

        tplargs = dict(norm=norm, method=method)

        if method == 'resid':
            tplargs['dt_type'] = 'matrix' if dt_mat else 'scalar'

        # Get the kernel template
        src = self.backend.lookup.get_template('reduction').render(**tplargs)

        regs = list(rs) + [dt_mat] if dt_mat else rs

        # Argument types for reduction kernel
        if method == 'errest':
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]*2
        elif method == 'resid' and dt_mat:
            argt = [np.int32]*3 + [np.intp]*4 + [dtype]
        else:
            argt = [np.int32]*3 + [np.intp]*3 + [dtype]

        # Build the reduction kernel
        rkern = self._build_kernel('reduction', src, argt)

        # Norm type
        reducer = np.max if norm == 'uniform' else np.sum

        class ReductionKernel(Kernel):
            @property
            def retval(self):
                return reducer(reduced_host, axis=1)

            def run(self, queue, *facs):
                ptrs = [r.data for r in regs]

                queue.call(rkern, nrow, ncolb, ldim, reduced_dev, *ptrs, *facs)
                queue.memcpy_dtoh(err_host, err_dev, err_host.nbytes)

        return ReductionKernel()
