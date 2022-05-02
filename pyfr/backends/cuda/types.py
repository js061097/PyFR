# -*- coding: utf-8 -*-

from functools import cached_property
import threading

import numpy as np

import pyfr.backends.base as base


class _CUDAMatrixCommon:
    @cached_property
    def _as_parameter_(self):
        return self.data


class CUDAMatrixBase(_CUDAMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.data = int(self.basedata) + offset
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
        self.backend.cuda.memcpy(buf, self.data, self.nbytes)

        # Unpack
        return self._unpack(buf[None, :, :])

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.backend.cuda.memcpy(self.data, buf, self.nbytes)


class CUDAMatrixSlice(_CUDAMatrixCommon, base.MatrixSlice):
    @cached_property
    def data(self):
        return int(self.basedata) + self.offset


class CUDAMatrix(CUDAMatrixBase, base.Matrix): pass
class CUDAConstMatrix(CUDAMatrixBase, base.ConstMatrix): pass
class CUDAView(base.View): pass
class CUDAXchgView(base.XchgView): pass


class CUDAXchgMatrix(CUDAMatrix, base.XchgMatrix):
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        # Call the standard matrix constructor
        super().__init__(backend, ioshape, initval, extent, aliases, tags)

        # If MPI is CUDA-aware then simply annotate our device buffer
        if backend.mpitype == 'cuda-aware':
            class HostData:
                __array_interface__ = {
                    'version': 3,
                    'typestr': np.dtype(self.dtype).str,
                    'data': (self.data, False),
                    'shape': (self.nrow, self.ncol)
                }

            self.hdata = np.array(HostData(), copy=False)
        # Otherwise, allocate a buffer on the host for MPI to send/recv from
        else:
            shape, dtype = (self.nrow, self.ncol), self.dtype
            self.hdata = backend.cuda.pagelocked_empty(shape, dtype)


class CUDAGraph(base.Graph):
    def __init__(self, backend):
        super().__init__(backend)

        self.graph = backend.cuda.create_graph()
        self.stale_kparams = {}
        self.mpi_events = []
        self.mpi_waits = []
        self.last_mpi_wait_gnode = None

    def add_mpi_req(self, req, deps=[]):
        super().add_mpi_req(req, deps)

        if deps:
            kdeps = [self.knodes[d] for d in deps]
            event = self.backend.cuda.create_event()
            gnode = self.graph.add_event_record(event, kdeps)

            self.mpi_events.append((event, req, gnode))

    @staticmethod
    def _host_wait_callback(event):
        event.wait()
        event.clear()

    def _make_mpi_waitall_impl(self, reqs):
        # See what requests need to start before we can wait
        deps, ereqs = [], []
        for event, req, gnode in self.mpi_events:
            if req in reqs:
                deps.append(gnode)
                ereqs.append((event, req))

        # Ensure we are only ever have one host-side callback in flight
        if self.last_mpi_wait_gnode:
            deps.append(self.last_mpi_wait_gnode)

        # Add our host-side callback function to the graph
        func, arg = self._host_wait_callback, threading.Event()
        gnode = self.graph.add_host_func(func, arg=arg, deps=deps)

        # Group the relevant events and requests together
        self.mpi_waits.append((ereqs, reqs, arg))

        handle = object()
        self.knodes[handle] = self.last_mpi_wait_gnode = gnode

        return handle

    def commit(self):
        super().commit()

        self.exc_graph = self.graph.instantiate()

    def run(self, stream):
        from mpi4py import MPI

        # Ensure our kernel parameters are up to date
        for node, params in self.stale_kparams.items():
            self.exc_graph.set_kernel_node_params(node, params)

        stream.synchronize()
        self.exc_graph.launch(stream)
        self.stale_kparams.clear()

        # Start all dependency-free MPI requests
        MPI.Prequest.Startall(self.mpi_root_reqs)

        # Process any remaining requests as their dependencies are satisfied
        for ereqs, reqs, cvar in self.mpi_waits:
            for event, req in ereqs:
                event.synchronize()
                req.Start()

            MPI.Prequest.Waitall(reqs)
            cvar.set()
