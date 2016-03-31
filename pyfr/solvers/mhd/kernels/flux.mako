# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='ideal_flux' params='s, f, p, v'>
    // Extract the variables
    fpdtype_t rho = s[0];
    fpdtype_t rhou = s[1], rhov = s[2], rhow = s[3];
    fpdtype_t Bx = s[4], By = s[5], Bz = s[6];
    fpdtype_t E = s[7];

    // Reciprocal of rho
    fpdtype_t invrho = 1.0/rho;

    // Compute the velocities
    v[0] = invrho*rhou;
    v[1] = invrho*rhov;
    v[2] = invrho*rhow;

    // Compute B·v and B·B/2
    fpdtype_t Bdotv = Bx*v[0] + By*v[1] + Bz*v[2];
    fpdtype_t BdotB2 = 0.5*(Bx*Bx + By*By + Bz*Bz);

    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*invrho*(rhou*rhou + rhov*rhov + rhow*rhow)
                             - BdotB2);

    // F_x
    f[0][0] = rhou;
    f[0][1] = rhou*v[0] + p + BdotB2 - Bx*Bx;
    f[0][2] = rhov*v[0] - By*Bx;
    f[0][3] = rhow*v[0] - Bz*Bx;
    f[0][4] = 0;
    f[0][5] = By*v[0] - Bx*v[1];
    f[0][6] = Bz*v[0] - Bx*v[2];
    f[0][7] = v[0]*(E + p + BdotB2) - Bx*Bdotv;

    // F_y
    f[1][0] = rhov;
    f[1][1] = rhou*v[1] - Bx*By;
    f[1][2] = rhov*v[1] + p + BdotB2 - By*By;
    f[1][3] = rhow*v[1] - Bz*By;
    f[1][4] = Bx*v[1] - By*v[0];
    f[1][5] = 0;
    f[1][6] = Bz*v[1] - By*v[2];
    f[1][7] = v[1]*(E + p + BdotB2) - By*Bdotv;

% if ndims == 3:
    // F_z
    f[2][0] = rhow;
    f[2][1] = rhou*v[2] - Bx*Bz;
    f[2][2] = rhov*v[2] - By*Bz;
    f[2][3] = rhow*v[2] + p + BdotB2 - Bz*Bz;
    f[2][4] = Bx*v[2] - Bz*v[0];
    f[2][5] = By*v[2] - Bz*v[1];
    f[2][6] = 0;
    f[2][7] = v[2]*(E + p + BdotB2) - Bz*Bdotv;
% endif
</%pyfr:macro>
