# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.mhd.kernels.flux'/>

<%pyfr:kernel name='tflux' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              artvisc='in broadcast fpdtype_t'
              f='out fpdtype_t[${str(ndims)}][${str(nvars)}]'
              b='out fpdtype_t[${str(ndims)}][1]'>
    // Compute the flux
    fpdtype_t ftemp[${ndims}][${nvars}];
    fpdtype_t p, v[3];
    ${pyfr.expand('ideal_flux', 'u', 'ftemp', 'p', 'v')};
    ${pyfr.expand('artificial_viscosity_add', 'f', 'ftemp', 'artvisc')};

    // Transform the fluxes
% for i, j in pyfr.ndrange(ndims, nvars):
    f[${i}][${j}] = ${' + '.join('smats[{0}][{1}]*ftemp[{1}][{2}]'
                                 .format(i, k, j)
                                 for k in range(ndims))};
% endfor

    // Transform the magnetic field
% for i in range(ndims):
    b[${i}][0] = ${' + '.join('smats[{0}][{1}]*u[{2}]'
                              .format(i, k, k + 4)
                              for k in range(ndims))};
% endfor
</%pyfr:kernel>
