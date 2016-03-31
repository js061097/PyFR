# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='negdivconf_powell' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              divb='in fpdtype_t'
              rcpdjac='in fpdtype_t'>
    // Density, energy
    fpdtype_t rho = u[0];

    // Velocity and magnetic fields
    fpdtype_t v[] = {(1.0/rho)*u[1], (1.0/rho)*u[2], (1.0/rho)*u[3]};
    fpdtype_t B[] = {u[4], u[5], u[6]};

    // Compute B·v
    fpdtype_t Bdotv = ${pyfr.dot('v[{i}]', 'B[{i}]', i=3)};

    // Untransform the divergences and apply the source terms
    tdivtconf[0] = -rcpdjac*tdivtconf[0] + ${srcex[0]};
    tdivtconf[1] = -rcpdjac*(divb*B[0] + tdivtconf[1]) + ${srcex[1]};
    tdivtconf[2] = -rcpdjac*(divb*B[1] + tdivtconf[2]) + ${srcex[2]};
    tdivtconf[3] = -rcpdjac*(divb*B[2] + tdivtconf[3]) + ${srcex[3]};
    tdivtconf[4] = -rcpdjac*(divb*v[0] + tdivtconf[4]) + ${srcex[4]};
    tdivtconf[5] = -rcpdjac*(divb*v[1] + tdivtconf[5]) + ${srcex[5]};
    tdivtconf[6] = -rcpdjac*(divb*v[2] + tdivtconf[6]) + ${srcex[6]};
    tdivtconf[7] = -rcpdjac*(divb*Bdotv + tdivtconf[7]) + ${srcex[7]};
</%pyfr:kernel>
