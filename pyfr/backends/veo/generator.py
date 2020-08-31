# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class VeoKernelGenerator(BaseKernelGenerator):
    def render(self):
        if self.ndim == 1:
            inner = '''
                    int cb, ce;
                    loop_sched_1d(_nx, align, &cb, &ce);
                    #pragma _NEC ivdep
                    for (int _xi = cb; _xi < ce; _xi++)
                    {{
                        {body}
                    }}'''.format(body=self.body)
        else:
            inner = '''
                    int rb, re, cb, ce;
                    loop_sched_2d(_ny, _nx, align, &rb, &re, &cb, &ce);
                    for (int _y = rb; _y < re; _y++)
                    {{
                        #pragma _NEC ivdep
                        for (int _xi = cb; _xi < ce; _xi++)
                        {{
                            {body}
                        }}
                    }}'''.format(body=self.body)

        return '''{spec}
               {{
                   #define X_IDX (_xi)
                   #define X_IDX_AOSOA(v, nv)\
                       ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xi % SOA_SZ)
                   int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
                   #pragma omp parallel
                   {{
                       {inner}
                   }}
                   #undef X_IDX
                   #undef X_IDX_AOSOA
               }}'''.format(spec=self._render_spec(), inner=inner)

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = ['int ' + d for d in self._dims]

        # Now add any scalar arguments
        kargs.extend('{0.dtype} {0.name}'.format(sa) for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append('{0.dtype}* restrict {0.name}_v'.format(va))
                kargs.append('const int* restrict {0.name}_vix'
                             .format(va))

                if va.ncdim == 2:
                    kargs.append('const int* restrict {0.name}_vrstri'
                                 .format(va))
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append('{0} {1.dtype}* restrict {1.name}_v'
                             .format(const, va).strip())

                if self.needs_ldim(va):
                    kargs.append('int ld{0.name}'.format(va))

        return 'void {0}({1})'.format(self.name, ', '.join(kargs))
