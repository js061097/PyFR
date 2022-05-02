# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem
from pyfr.util import memoize


class BaseAdvectionSystem(BaseSystem):
    @memoize
    def _rhs_graph(self, uinbank, foutbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g = self.backend.graph()
        g.add_mpi_reqs(m['scal_fpts_recv'])

        # Interpolate the solution to the flux points
        g.add_all(k['eles/disu'])

        # Pack and send these interpolated solutions to our neighbours
        g.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'], k['mpiint/scal_fpts_pack']):
            g.add_mpi_req(send, deps=[pack])

        # Compute the common normal flux at our internal/boundary interfaces
        g.add_all(k['iint/comm_flux'],
                  deps=k['eles/disu'] + k['mpiint/scal_fpts_pack'])
        g.add_all(k['bcint/comm_flux'], deps=k['eles/disu'])

        # Make a copy of the solution (if used by source terms)
        g.add_all(k['eles/copy_soln'])

        # Interpolate the solution to the quadrature points
        g.add_all(k['eles/qptsu'])

        # Compute the transformed flux
        for l in k['eles/tdisf_curved'] + k['eles/tdisf_linear']:
            g.add(l, deps=deps(l, 'eles/qptsu'))

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/tdivtpcorf']:
            ldeps = deps(l, 'eles/tdisf_curved', 'eles/tdisf_linear',
                         'eles/copy_soln', 'eles/disu')
            g.add(l, deps=ldeps + k['mpiint/scal_fpts_pack'])

        # Unpack interpolated solutions from our neighbours (may be a no-op)
        mwait = g.make_mpi_wait_deps(m['scal_fpts_recv'] + m['scal_fpts_send'])
        g.add_all(k['mpiint/scal_fpts_unpack'], deps=mwait)

        # Compute the common normal flux at our MPI interfaces
        for l in k['mpiint/comm_flux']:
            g.add(l, deps=deps(l, 'mpiint/scal_fpts_unpack') or mwait)

        # Compute the transformed divergence of the corrected flux
        for l in k['eles/tdivtconf']:
            ldeps = deps(l, 'eles/tdivtpcorf')
            g.add(l, deps=ldeps + k['mpiint/comm_flux'] + k['iint/comm_flux'])

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g.add(l, deps=deps(l, 'eles/tdivtconf'))

        g.commit()
        return g
