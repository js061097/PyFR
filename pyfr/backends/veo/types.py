# -*- coding: utf-8 -*-

from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class _VeoMatrixCommon(object):
    def __index__(self):
        return self.data.addr


class VeoMatrixBase(_VeoMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        from veo import VEMemPtr

        self.basedata = basedata
        self.data = VEMemPtr(basedata.proc, basedata.addr + offset,
                             self.nbytes)
        self.offset = offset

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.leaddim), dtype=self.dtype)

        # Copy
        self.data.proc.read_mem(buf, self.data, self.nbytes)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.data.proc.write_mem(self.data, buf, self.nbytes)


class VeoMatrix(VeoMatrixBase, base.Matrix):
    pass


class VeoMatrixSlice(_VeoMatrixCommon, base.MatrixSlice):
    @cached_property
    def data(self):
        if self.offset:
            from veo import VEMemPtr

            basedata = self.basedata
            nbytes = ((self.nrow - 1)*self.leaddim + self.ncol)*self.itemsize

            return VEMemPtr(basedata.proc, basedata.addr + self.offset, nbytes)
        else:
            return self.basedata


class VeoConstMatrix(VeoMatrixBase, base.ConstMatrix):
    pass


class VeoXchgMatrix(VeoMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # Allocate an empty buffer on the host for MPI to send/recv from
        self.hdata = np.empty((self.nrow, self.ncol), self.dtype)


class VeoXchgView(base.XchgView):
    pass


class VeoView(base.View):
    pass


class VeoQueue(base.Queue):
    def __init__(self, backend):
        super().__init__(backend)

        # VEO context thread
        self.veo_ctx = backend.ctx

        # Submitted request list
        self._veo_reqs = []

    def run(self, mpireqs=[]):
        # Start any MPI requests
        if mpireqs:
            self._startall(mpireqs)

        # Submit the kernels to the VEO
        for item, args, kwargs in self._items:
            item.run(self, *args, **kwargs)

        # If we started any MPI requests, first wait for them
        if mpireqs:
            self._waitall(mpireqs)

        # Then, wait for the kernels to finish
        for r in self._veo_reqs:
            r.wait_result()

        self._veo_reqs.clear()
        self._items.clear()

    def call(self, fn, *args):
        nargs = [getattr(arg, 'addr', arg) for arg in args]
        self._veo_reqs.append(fn(self.veo_ctx, *nargs))

    def memcpy_htod(self, dst, src, nbytes):
        self._veo_reqs.append(self.veo_ctx.async_write_mem(dst, src, nbytes))

    def memcpy_dtoh(self, dst, src, nbytes):
        self._veo_reqs.append(self.veo_ctx.async_read_mem(dst, src, nbytes))
