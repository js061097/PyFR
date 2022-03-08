# -*- coding: utf-8 -*-

from pyfr.backends.base import Kernel, NullKernel
from pyfr.backends.veo.provider import VeoKernelProvider


class VeoPackingKernels(VeoKernelProvider):
    def pack(self, mv):
        # An exchange view is simply a regular view plus an exchange matrix
        m, v = mv.xchgmat, mv.view

        # Render the kernel template
        src = self.backend.lookup.get_template('pack').render()

        # Build
        kern = self._build_kernel('pack_view', src, 'iiiPPPP')

        class PackXchgViewKernel(Kernel):
            def run(self, queue):
                vbd = v.basedata
                vmd = v.mapping.data
                vrd = v.rstrides.data if v.rstrides else 0
                xmd = m.data

                # Pack
                queue.call(kern, v.n, v.nvrow, v.nvcol, vbd, vmd, vrd, xmd)

                # Copy the packed buffer to the host
                queue.memcpy_dtoh(m.hdata, m.data, m.nbytes)

        return PackXchgViewKernel()

    def unpack(self, mv):
        class UnpackXchgMatrixKernel(Kernel):
            def run(self, queue):
                queue.memcpy_htod(mv.data, mv.hdata, mv.nbytes)

        return UnpackXchgMatrixKernel()

