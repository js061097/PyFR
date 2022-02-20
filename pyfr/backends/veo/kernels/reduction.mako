# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
reduction(int nrow, int ncolb, int ldim, fpdtype_t *__restrict__ reduced,
          fpdtype_t *__restrict__ rcurr, fpdtype_t *__restrict__ rold,
% if method == 'errest':
          fpdtype_t *__restrict__ rerr, fpdtype_t atol, fpdtype_t rtol)
% elif method == 'resid' and dt_type == 'matrix':
          fpdtype_t *__restrict__ dt_mat, fpdtype_t dt_fac)
% elif method == 'resid':
          fpdtype_t dt_fac)
% endif
{
    #define X_IDX_AOSOA(v, nv) ((ci/SOA_SZ*(nv) + (v))*SOA_SZ + cj)

    // Initalise the reduction variables
    fpdtype_t ${','.join(f'red{i} = 0.0' for i in range(ncola))};

% if norm == 'uniform':
    #pragma omp parallel reduction(max : ${','.join(f'red{i}' for i in range(ncola))})
% else:
    #pragma omp parallel reduction(+ : ${','.join(f'red{i}' for i in range(ncola))})
% endif
    {
        int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
        int rb, re, cb, ce;
        loop_sched_2d(nrow, ncolb, align, &rb, &re, &cb, &ce);
        int nci = ((ce - cb) / SOA_SZ)*SOA_SZ;

        for (int r = rb; r < re; r++)
        {
            for (int ci = cb; ci < cb + nci; ci += SOA_SZ)
            {
                for (int cj = 0; cj < SOA_SZ; cj++)
                {
                    int idx;
                    fpdtype_t temp;

                % for i in range(ncola):
                    idx = r*ldim + X_IDX_AOSOA(${i}, ${ncola});

                % if method == 'errest':
                    temp = rerr[idx]/(atol + rtol*max(fabs(rcurr[idx]), fabs(rold[idx])));
                % elif method == 'resid':
                    temp = (rcurr[idx] - rold[idx])/(1.0e-8 + dt_fac${'*dt_mat[idx]' if dt_type == 'matrix' else ''});
                % endif

                % if norm == 'uniform':
                    red${i} = max(red${i}, temp*temp);
                % else:
                    red${i} += temp*temp;
                % endif
                % endfor
                }
            }

            for (int ci = cb + nci, cj = 0; cj < ce - ci; cj++)
            {
                int idx;
                fpdtype_t temp;

            % for i in range(ncola):
                idx = r*ldim + X_IDX_AOSOA(${i}, ${ncola});

            % if method == 'errest':
                temp = rerr[idx]/(atol + rtol*max(fabs(rcurr[idx]), fabs(rold[idx])));
            % elif method == 'resid':
                temp = (rcurr[idx] - rold[idx])/(1.0e-8 + dt_fac${'*dt_mat[idx]' if dt_type == 'matrix' else ''});
            % endif

            % if norm == 'uniform':
                red${i} = max(red${i}, temp*temp);
            % else:
                red${i} += temp*temp;
            % endif
            % endfor
            }
        }
    }

    // Copy
% for i in range(ncola):
    reduced[${i}] = red${i};
% endfor
}
