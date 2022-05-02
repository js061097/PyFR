# -*- coding: utf-8 -*-

from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class _OpenCLMatrixCommon:
    @cached_property
    def _as_parameter_(self):
        return int(self.data)


class OpenCLMatrixBase(_OpenCLMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.offset = offset

        # If necessary, slice the buffer
        if offset:
            self.data = basedata.slice(offset, self.nbytes)
        else:
            self.data = basedata

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.leaddim), dtype=self.dtype)

        # Copy
        self.backend.queue.barrier()
        self.backend.cl.memcpy(self.backend.queue, buf, self.data, self.nbytes,
                               blocking=True)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.backend.queue.barrier()
        self.backend.cl.memcpy(self.backend.queue, self.data, buf, self.nbytes,
                               blocking=True)


class OpenCLMatrixSlice(_OpenCLMatrixCommon, base.MatrixSlice):
    @cached_property
    def data(self):
        if self.offset:
            nbytes = ((self.nrow - 1)*self.leaddim + self.ncol)*self.itemsize
            return self.basedata.slice(self.offset, nbytes)
        else:
            return self.basedata


class OpenCLMatrix(OpenCLMatrixBase, base.Matrix): pass
class OpenCLConstMatrix(OpenCLMatrixBase, base.ConstMatrix): pass
class OpenCLView(base.View): pass
class OpenCLXchgView(base.XchgView): pass


class OpenCLXchgMatrix(OpenCLMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # Allocate an empty buffer on the host for MPI to send/recv from
        shape, dtype = (self.nrow, self.ncol), self.dtype
        self.hdata = backend.cl.pagelocked_empty(shape, dtype)


class OpenCLGraph(base.Graph):
    def _make_mpi_waitall_impl(self, reqs):
        super().make_mpi_waitall(reqs)

        kern = object()
        self.knodes[kern] = reqs

        return kern

    def commit(self):
        super().commit()

        # Map from kernels to event table locations
        evtidxs = {}

        # Kernel list complete with dependency information
        self.klist = klist = []

        # MPI wait list
        self.mpi_waits = mpi_waits = []

        for i, (k, v) in enumerate(self.knodes.items()):
            evtidxs[k] = i

            # See if the node corresponds to an MPI wait
            if v is not None:
                swreqs, reqs = [], v

                for r in reqs:
                    if (rdeps := self.mpi_ireq_deps[id(r)]):
                        swreqs.append((r, [evtidxs[dep] for dep in rdeps]))

                if reqs:
                    mpi_waits.append((swreqs, reqs, i))
            # Otherwise, it is a regular compute kernel
            else:
                # Resolve the event indices of kernels we depend on
                wait_evts = [evtidxs[dep] for dep in self.kdeps[k]] or None

                klist.append((i, k, wait_evts, k in self.depk))

    def run(self, queue):
        from mpi4py import MPI

        cl = self.backend.cl
        events = [None]*len(self.knodes)

        # Create any necessary user events
        for reqs, swreqs, ueidx in self.mpi_waits:
            events[ueidx] = cl.user_event()

        # Submit the kernels to the queue
        for i, k, wait_for, ret_evt in self.klist:
            if wait_for is not None:
                wait_for = [events[j] for j in wait_for]

            events[i] = k.run(queue, wait_for, ret_evt)

        # Start all dependency-free MPI requests
        MPI.Prequest.Startall(self.mpi_root_reqs)

        # Process any remaining requests as their dependencies are satisfied
        for swreqs, reqs, ueidx in self.mpi_waits:
            for req, wait_for in swreqs:
                cl.wait_for_events([events[j] for j in wait_for])
                req.Start()

            MPI.Prequest.Waitall(reqs)
            events[ueidx].complete()
