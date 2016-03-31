# -*- coding: utf-8 -*-

import numpy as np

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements


class MHDElements(BaseAdvectionDiffusionElements):
    privarmap = {2: ['rho', 'u', 'v', 'w', 'Bx', 'By', 'Bz', 'p'],
                 3: ['rho', 'u', 'v', 'w', 'Bx', 'By', 'Bz', 'p']}

    convarmap = {2: ['rho', 'rhou', 'rhov', 'rhow', 'Bx', 'By', 'Bz', 'E'],
                 3: ['rho', 'rhou', 'rhov', 'rhow', 'Bx', 'By', 'Bz', 'E']}

    visvarmap = {
        2: {'density': ['rho'],
            'velocity': ['u', 'v', 'w'],
            'B': ['Bx', 'By', 'Bz'],
            'pressure': ['p']},
        3: {'density': ['rho'],
            'velocity': ['u', 'v', 'w'],
            'B': ['Bx', 'By', 'Bz'],
            'pressure': ['p']}
    }

    shockvar = 'rho'

    @staticmethod
    def pri_to_con(pris, cfg):
        # Density, pressure, velocity field, and magnetic field
        rho, p = pris[0], pris[-1]
        vf, Bf = list(pris[1:4]), list(pris[4:7])

        # Multiply velocity components by rho
        rhovf = [rho*v for v in vf]

        # Squared velocity and magnetic fields
        vf2 = sum(v*v for v in vf)
        bf2 = sum(b*b for b in Bf)

        # Compute the energy
        gamma = cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*vf2 + 0.5*bf2

        return [rho] + rhovf + Bf + [E]

    @staticmethod
    def con_to_pri(cons, cfg):
        # Density, energy, momentum field, and magnetic field
        rho, E = cons[0], cons[-1]
        rhovf, Bf = list(cons[1:4]), list(cons[4:7])

        # Divide momentum components by rho
        vf = [rhov/rho for rhov in rhovf]

        # Squared velocity and magnetic fields
        vf2 = sum(v*v for v in vf)
        bf2 = sum(b*b for b in Bf)

        # Compute the pressure
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*vf2 - 0.5*bf2)

        return [rho] + vf + Bf + [p]

    @property
    def _soln_in_src_exprs(self):
        return True

    def set_backend(self, backend, nscalupts, nonce):
        super().set_backend(backend, nscalupts, nonce)

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias
        divfluxaa = 'div-flux' in self.antialias

        # What the source term expressions (if any) are a function of
        plocsrc = self._ploc_in_src_exprs

        # Shock capturing scheme (if any)
        shock_capturing = self.cfg.get('solver', 'shock-capturing')

        # Dimensions
        ndims, neles = self.ndims, self.neles
        nupts, nqpts = self.nupts, self.nqpts

        # Convenience routine for matrix allocation
        alloc = lambda n: backend.matrix(n, tags={'align'})

        # Allocate additional storage for B and its divergence
        b = alloc((ndims, nqpts if fluxaa else nupts, 1, neles))
        divb = alloc((nqpts if divfluxaa else nupts, 1, neles))

        # Generate the operator matrix to compute ∇·B from B
        if fluxaa and divfluxaa:
            divbop = self.opmat('M7*M1*M10')
        elif fluxaa:
            divbop = self.opmat('M1*M10')
        elif divfluxaa:
            divbop = self.opmat('M7*M1')
        else:
            divbop = self.opmat('M1')

        # Transformed flux kernel
        backend.pointwise.register('pyfr.solvers.mhd.kernels.tflux')
        fluxtplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'c': self.cfg.items_as('constants', float),
            'shock_capturing': shock_capturing
        }

        if fluxaa:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=fluxtplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=self.smat_at('qpts'),
                f=self._vect_qpts, b=b, artvisc=self.artvisc
            )
        else:
            self.kernels['tdisf'] = lambda: backend.kernel(
                'tflux', tplargs=fluxtplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=self.smat_at('upts'),
                f=self._vect_upts, b=b, artvisc=self.artvisc
            )

        # Divergence kernel
        self.kernels['divb'] = lambda: backend.kernel(
            'mul', divbop, b, out=divb
        )

        # Transformed to physical divergence kernel + Powell source term
        backend.pointwise.register(
            'pyfr.solvers.mhd.kernels.negdivconf_powell'
        )
        srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'srcex': self._src_exprs
        }

        if divfluxaa:
            plocqpts = self.ploc_at('qpts') if plocsrc else None

            self.kernels['copy_soln'] = lambda: backend.kernel(
                'copy', self._scal_qpts_cpy, self._scal_qpts
            )
            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf_powell', tplargs=srctplargs,
                dims=[self.nqpts, self.neles], tdivtconf=self._scal_qpts,
                rcpdjac=self.rcpdjac_at('qpts'), ploc=plocqpts,
                u=self._scal_qpts_cpy, divb=divb
            )
        else:
            plocupts = self.ploc_at('upts') if plocsrc else None
            solnupts = self._scal_upts_cpy

            self.kernels['copy_soln'] = lambda: backend.kernel(
                'copy', self._scal_upts_cpy, self.scal_upts_inb
            )
            self.kernels['negdivconf'] = lambda: backend.kernel(
                'negdivconf_powell', tplargs=srctplargs,
                dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
                rcpdjac=self.rcpdjac_at('upts'), ploc=plocupts,
                u=self._scal_upts_cpy, divb=divb
            )
