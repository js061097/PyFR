# -*- coding: utf-8 -*-

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, ComputeKernel)
from pyfr.backends.veo.compiler import SourceModule
import pyfr.backends.veo.generator as generator
from pyfr.util import memoize


class VeoKernelProvider(BaseKernelProvider):
    @memoize
    def _build_kernel(self, name, src, argtypes, restype=None):
        mod = SourceModule(src, self.backend.proc, self.backend.cfg)
        return mod.function(name, restype, argtypes)


class VeoPointwiseKernelProvider(VeoKernelProvider,
                                 BasePointwiseKernelProvider):
    kernel_generator_cls = generator.VeoKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst):
        class PointwiseKernel(ComputeKernel):
            def run(self, queue, **kwargs):
                narglst = []
                for arg in arglst:
                    if isinstance(arg, str):
                        narglst.append(float(kwargs[arg]))
                    elif hasattr(arg, 'data'):
                        narglst.append(arg.data)
                    else:
                        narglst.append(arg)

                queue.call(fun, *narglst)

        return PointwiseKernel()
