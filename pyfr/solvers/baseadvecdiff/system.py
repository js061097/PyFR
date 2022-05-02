# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvec import BaseAdvectionSystem
from pyfr.util import memoize


class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    @memoize
    def _rhs_graph(self, uinbank, foutbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, foutbank)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g = self.backend.graph()
        g.add_mpi_reqs(m['scal_fpts_recv'])
        g.add_mpi_reqs(m['artvisc_fpts_recv'])
        g.add_mpi_reqs(m['vect_fpts_recv'])

        # Interpolate the solution to the flux points
        g.add_all(k['eles/disu'])

        # Pack and send these interpolated solutions to our neighbours
        g.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'], k['mpiint/scal_fpts_pack']):
            g.add_mpi_req(send, deps=[pack])

        # Make a copy of the solution (if used by source terms)
        g.add_all(k['eles/copy_soln'])

        # Compute the common solution at our internal/boundary interfaces
        for l in k['eles/copy_fpts']:
            g.add(l, deps=deps(l, 'eles/disu'))
        kdeps = k['eles/copy_fpts'] or k['eles/disu']
        g.add_all(k['iint/con_u'], deps=kdeps + k['mpiint/scal_fpts_pack'])
        g.add_all(k['bcint/con_u'], deps=kdeps)

        # Run the shock sensor (if enabled)
        g.add_all(k['eles/shocksensor'])
        g.add_all(k['mpiint/artvisc_fpts_pack'], deps=k['eles/shocksensor'])

        # Compute the transformed gradient of the partially corrected solution
        g.add_all(k['eles/tgradpcoru_upts'], deps=k['mpiint/scal_fpts_pack'])

        # Unpack interpolated solutions from our neighbours (may be a no-op)
        mwait = g.make_mpi_wait_deps(m['scal_fpts_recv'] + m['scal_fpts_send'])
        g.add_all(k['mpiint/scal_fpts_unpack'], deps=mwait)

        # Compute the common solution at our MPI interfaces
        for l in k['mpiint/con_u']:
            ldeps = deps(l, 'mpiint/scal_fpts_unpack') or mwait
            g.add(l, deps=ldeps + k['eles/copy_fpts'])

        # Compute the transformed gradient of the corrected solution
        kdeps = k['bcint/con_u'] + k['iint/con_u'] + k['mpiint/con_u']
        for l in k['eles/tgradcoru_upts']:
            g.add(l, deps=deps(l, 'eles/tgradpcoru_upts') + kdeps)

        # Obtain the physical gradients at the solution points
        for l in k['eles/gradcoru_upts_curved']:
            g.add(l, deps=deps(l, 'eles/tgradcoru_upts'))
        for l in k['eles/gradcoru_upts_linear']:
            g.add(l, deps=deps(l, 'eles/tgradcoru_upts'))

        # Interpolate these gradients to the flux points
        for l in k['eles/gradcoru_fpts']:
            ldeps = deps(l, 'eles/gradcoru_upts_curved',
                         'eles/gradcoru_upts_linear')
            g.add(l, deps=ldeps)

        # Pack and send these interpolated gradients to our neighbours
        g.add_all(k['mpiint/vect_fpts_pack'], deps=k['eles/gradcoru_fpts'])
        for send, pack in zip(m['vect_fpts_send'], k['mpiint/vect_fpts_pack']):
            g.add_mpi_req(send, deps=[pack])

        # Compute the common normal flux at our internal/boundary interfaces
        g.add_all(k['iint/comm_flux'],
                   deps=k['eles/gradcoru_fpts'] + k['mpiint/vect_fpts_pack'])
        g.add_all(k['bcint/comm_flux'], deps=k['eles/gradcoru_fpts'])

        # Interpolate the gradients to the quadrature points
        for l in k['eles/gradcoru_qpts']:
            ldeps = deps(l, 'eles/gradcoru_upts_curved',
                         'eles/gradcoru_upts_linear')
            g.add(l, deps=ldeps + k['mpiint/vect_fpts_pack'])

        # Interpolate the solution to the quadrature points
        g.add_all(k['eles/qptsu'])

        # Compute the transformed flux
        for l in k['eles/tdisf_curved'] + k['eles/tdisf_linear']:
            if k['eles/qptsu']:
                ldeps = deps(l, 'eles/gradcoru_qpts', 'eles/qptsu')
            else:
                ldeps = deps(l, 'eles/gradcoru_fpts')
            g.add(l, deps=ldeps)

        # Compute the transformed divergence of the partially corrected flux
        for l in k['eles/tdivtpcorf']:
            g.add(l, deps=deps(l, 'eles/tdisf_curved', 'eles/tdisf_linear'))

        # Unpack interpolated gradients from our neighbours (may be a no-op)
        mwait = g.make_mpi_wait_deps(
            m['artvisc_fpts_recv'] + m['artvisc_fpts_send'] +
            m['vect_fpts_recv'] + m['vect_fpts_send']
        )
        g.add_all(k['mpiint/artvisc_fpts_unpack'], deps=mwait)
        g.add_all(k['mpiint/vect_fpts_unpack'], deps=mwait)

        # Compute the common normal flux at our MPI interfaces
        for l in k['mpiint/comm_flux']:
            ldeps = deps(l, 'mpiint/artvisc_fpts_unpack',
                         'mpiint/vect_fpts_unpack')
            g.add(l, deps=ldeps or mwait)

        # Compute the transformed divergence of the corrected flux
        kdeps = (k['bcint/comm_flux'] + k['iint/comm_flux'] +
                 k['mpiint/comm_flux'])
        for l in k['eles/tdivtconf']:
            g.add(l, deps=deps(l, 'eles/tdivtpcorf') + kdeps)

        # Obtain the physical divergence of the corrected flux
        for l in k['eles/negdivconf']:
            g.add(l, deps=deps(l, 'eles/tdivtconf'))

        g.commit()
        return g

    @memoize
    def _compute_grads_graph(self, uinbank):
        m = self._mpireqs
        k, _ = self._get_kernels(uinbank, None)

        def deps(dk, *names): return self._kdeps(k, dk, *names)

        g = self.backend.graph()
        g.add_mpi_reqs(m['scal_fpts_recv'])

        # Interpolate the solution to the flux points
        g.add_all(k['eles/disu'])

        # Pack and send these interpolated solutions to our neighbours
        g.add_all(k['mpiint/scal_fpts_pack'], deps=k['eles/disu'])
        for send, pack in zip(m['scal_fpts_send'], k['mpiint/scal_fpts_pack']):
            g.add_mpi_req(send, deps=[pack])

        # Compute the common solution at our internal/boundary interfaces
        for l in k['eles/copy_fpts']:
            g.add(l, deps=deps(l, 'eles/disu'))
        kdeps = k['eles/copy_fpts'] or k['eles/disu']
        g.add_all(k['iint/con_u'], deps=kdeps + k['mpiint/scal_fpts_pack'])
        g.add_all(k['bcint/con_u'], deps=kdeps)

        # Compute the transformed gradient of the partially corrected solution
        g.add_all(k['eles/tgradpcoru_upts'])

        # Unpack interpolated solutions from our neighbours (may be a no-op)
        mwait = g.make_mpi_wait_deps(m['scal_fpts_recv'] + m['scal_fpts_send'])
        g.add_all(k['mpiint/scal_fpts_unpack'], deps=mwait)

        # Compute the common solution at our MPI interfaces
        for l in k['mpiint/con_u']:
            ldeps = deps(l, 'mpiint/scal_fpts_unpack') or mwait
            g.add(l, deps=ldeps + k['eles/copy_fpts'])

        # Compute the transformed gradient of the corrected solution
        kdeps = k['bcint/con_u'] + k['iint/con_u'] + k['mpiint/con_u']
        for l in k['eles/tgradcoru_upts']:
            g.add(l, deps=deps(l, 'eles/tgradpcoru_upts') + kdeps)

        # Obtain the physical gradients at the solution points
        for l in k['eles/gradcoru_upts_curved']:
            g.add(l, deps=deps(l, 'eles/tgradcoru_upts'))
        for l in k['eles/gradcoru_upts_linear']:
            g.add(l, deps=deps(l, 'eles/tgradcoru_upts'))

        g.commit()
        return g
