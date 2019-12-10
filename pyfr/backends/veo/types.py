# -*- coding: utf-8 -*-

import numpy as np

import pyfr.backends.base as base
from pyfr.util import lazyprop


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
        return self._unpack(buf[:, :self.ncol])

    def _set(self, ary):
        # Allocate a new buffer with suitable padding and pack it
        buf = np.zeros((self.nrow, self.leaddim), dtype=self.dtype)
        buf[:, :self.ncol] = self._pack(ary)

        # Copy
        self.data.proc.write_mem(self.data, buf, self.nbytes)


class VeoMatrix(VeoMatrixBase, base.Matrix):
    pass


class VeoMatrixSlice(_VeoMatrixCommon, base.MatrixSlice):
    def _init_data(self, mat):
        from veo import VEMemPtr

        start = self.ra*self.pitch + self.ca*self.itemsize
        nbytes = (self.nrow - 1)*self.pitch + self.ncol*self.itemsize

        return VEMemPtr(mat.data.proc, mat.data.addr + start, nbytes)


class VeoMatrixBank(base.MatrixBank):
    pass


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
        self.veo_ctx = backend.proc.open_context()

        # Submitted request list
        self.veo_reqs = []

    def call(self, fn, *args):
        nargs = [getattr(arg, 'addr', arg) for arg in args]
        self.veo_reqs.append(fn(self.veo_ctx, *nargs))

    def memcpy_htod(self, dst, src, nbytes):
        self.veo_reqs.append(self.veo_ctx.async_write_mem(dst, src, nbytes))

    def memcpy_dtoh(self, dst, src, nbytes):
        self.veo_reqs.append(self.veo_ctx.async_read_mem(dst, src, nbytes))

    def _wait(self):
        last = self._last

        if last and last.ktype == 'compute':
            for r in self.veo_reqs:
                r.wait_result()

            self.veo_reqs = []
        elif last and last.ktype == 'mpi':
            from mpi4py import MPI

            MPI.Prequest.Waitall(self.mpi_reqs)
            self.mpi_reqs = []

        self._last = None

    def _at_sequence_point(self, item):
        return self._last and self._last.ktype != item.ktype

    @staticmethod
    def runall(queues):
        # First run any items which will not result in an implicit wait
        for q in queues:
            q._exec_nowait()

        # So long as there are items remaining in the queues
        while any(queues):
            # Execute a (potentially) blocking item from each queue
            for q in filter(None, queues):
                q._exec_next()
                q._exec_nowait()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
