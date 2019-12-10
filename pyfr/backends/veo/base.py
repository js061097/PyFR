# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import BaseBackend
from pyfr.mpiutil import get_local_rank


class VeoBackend(BaseBackend):
    name = 'veo'

    def __init__(self, cfg):
        super().__init__(cfg)

        from veo import VeoProc

        # Get the desired VEO device
        devid = cfg.get('backend-veo', 'device-id', 'local-rank')
        devid = get_local_rank() if devid == 'local-rank' else int(devid)

        # Create a VEO process on this device
        self.proc = VeoProc(int(devid))

        # Take the default alignment requirement to be 64-bytes
        self.alignb = 64

        # Compute the SoA size
        self.soasz = self.alignb // np.dtype(self.fpdtype).itemsize

        from pyfr.backends.veo import (blasext, cblas, gimmik, packing,
                                       provider, types)

        # Register our data types
        self.base_matrix_cls = types.VeoMatrixBase
        self.const_matrix_cls = types.VeoConstMatrix
        self.matrix_cls = types.VeoMatrix
        self.matrix_bank_cls = types.VeoMatrixBank
        self.matrix_slice_cls = types.VeoMatrixSlice
        self.queue_cls = types.VeoQueue
        self.view_cls = types.VeoView
        self.xchg_matrix_cls = types.VeoXchgMatrix
        self.xchg_view_cls = types.VeoXchgView

        # Instantiate mandatory kernel provider classes
        kprovcls = [provider.VeoPointwiseKernelProvider,
                    blasext.VeoBlasExtKernels,
                    packing.VeoPackingKernels,
                    gimmik.VeoGiMMiKKernels]

        self._providers = [k(self) for k in kprovcls]

        # Instantiate optional classes
        try:
            self._providers.append(cblas.VeoCBLASKernels(self))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            pass

        # Pointwise kernels
        self.pointwise = self._providers[0]

    def _malloc_impl(self, nbytes):
        # Allocate
        data = self.proc.alloc_mem(nbytes)

        # Zero
        self.proc.write_mem(data, np.zeros(nbytes, dtype=np.uint8), nbytes)

        return data
