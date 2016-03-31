# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.mhd.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[3], vr[3];
    fpdtype_t pl, pr;

    ${pyfr.expand('ideal_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('ideal_flux', 'ur', 'fr', 'pr', 'vr')};

    // Estimate the sound speed squared
    fpdtype_t c2 = ${c['gamma']}*(pl + pr)/(ul[0] + ur[0]);

    // Average the normal velocities
    fpdtype_t mnv = 0.5*${pyfr.dot('n[{i}]', 'vl[{i}] + vr[{i}]', i=ndims)};

    // Average the magnetic field
    fpdtype_t mB[3];
% for i in range(3):
    mB[${i}] = 0.5*(ul[${i + 4}] + ur[${i + 4}]);
% endfor

    // Take the dot product
    fpdtype_t mBB2 = ${pyfr.dot('mB[{i}]', 'mB[{i}]', i=3)};

    // Compute the average normal magnetic field
    fpdtype_t mBn = ${pyfr.dot('n[{i}]', 'mB[{i}]', i=ndims)};

    // Sum the squared sound speed and the squared B field
    fpdtype_t c2pmBB2 = c2 + mBB2;

    // Estimate the maximum eigenvalue of the quasi-linear system
    fpdtype_t lambda = fabs(mnv)
                     + sqrt(0.5*(c2pmBB2 + sqrt(c2pmBB2*c2pmBB2 - 4*c2*mBn*mBn)));

    // Output
% for i in range(nvars):
    nf[${i}] = 0.5*(${' + '.join('n[{j}]*(fl[{j}][{i}] + fr[{j}][{i}])'
                                 .format(i=i, j=j) for j in range(ndims))})
             + 0.5*lambda*(ul[${i}] - ur[${i}]);
% endfor
</%pyfr:macro>
