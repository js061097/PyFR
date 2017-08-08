# -*- coding: utf-8 -*-

import itertools as it

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin
from pyfr.writers.native import DistributedWriter


class CheckpointPlugin(BasePlugin):
    name = 'checkpoint'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        from mpi4py import MPI

        super().__init__(intg, cfgsect, suffix)

        # Backend data type and MPI rank to physical rank map
        dtype = intg.backend.fpdtype
        mprankmap = intg.rallocs.mprankmap

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Output directory and name
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        comm, rank, root = get_comm_rank_root()

        # Collect the host names of the ranks
        hostn = MPI.Get_processor_name()
        hosts = comm.gather(hostn, root=root)

        # Find send/recv partners for each rank
        if rank == root:
            rsend, rrecv = [None]*len(hosts), [None]*len(hosts)

            for i, ih in enumerate(hosts):
                for j, jh in enumerate(it.chain(hosts[i + 1:], hosts[:i])):
                    j = (j + i + 1) % len(hosts)
                    if jh != ih and not rrecv[j]:
                        rsend[i], rrecv[j] = j, i
                        break
                else:
                    raise RuntimeError('Unable to construct a buddy scheme')

            fname = self.cfg.get(cfgsect, 'buddy-file')
            fname = fname if fname.endswith('.csv') else fname + '.csv'

            # Output the buddy list to a CSV file
            with open(fname, 'w') as f:
                if self.cfg.getbool(cfgsect, 'header', True):
                    print('rank,prank,host,rsend,rrecv', file=f)

                rows = zip(range(len(hosts)), mprankmap, hosts, rsend, rrecv)
                print('\n'.join(','.join(str(c) for c in r) for r in rows),
                      file=f)
        else:
            rsend = rrecv = None

        # Distribute rank data
        rsend = comm.scatter(rsend, root=root)
        rrecv = comm.scatter(rrecv, root=root)

        # Exchange element info with our partners
        eisend = list(zip(intg.system.ele_types, intg.system.ele_shapes))
        eirecv = comm.sendrecv(eisend, rsend)

        # Allocate send buffers and MPI requests
        self.sbufs = [np.empty(shape, dtype=dtype) for etype, shape in eisend]
        self.sreqs = [comm.Send_init(buf, rsend, tag)
                      for tag, buf in enumerate(self.sbufs)]

        # Allocate recv buffers and MPI requests
        self.rbufs = [np.empty(shape, dtype=dtype) for etype, shape in eirecv]
        self.rreqs = [comm.Recv_init(buf, rrecv, tag)
                      for tag, buf in enumerate(self.rbufs)]

        # Determine physical rank of the received solution
        rprank = intg.rallocs.mprankmap[rrecv]

        # Prepare the local and received solution writers
        self.lwriter = DistributedWriter(
            intg, self.nvars, basedir, basename, prefix='soln'
        )
        self.rwriter = DistributedWriter(
            intg, self.nvars, basedir, basename, prefix='soln',
            prank=rprank, etypes=[etype for etype, shape in eirecv]
        )

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps == 0 or intg.nacptsteps % self.nsteps:
            return

        from mpi4py import MPI

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        # Prepare the metadata
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        # Copy our solution to the send buffers
        for i, buf in enumerate(intg.soln):
            self.sbufs[i][:] = buf

        # Start the MPI requests
        MPI.Prequest.Startall(self.sreqs + self.rreqs)

        # Write out our solution
        self.lwriter.write(self.sbufs, metadata, intg.tcurr)

        # Wait for the MPI requests to finish
        MPI.Prequest.Waitall(self.sreqs + self.rreqs)

        # Write out the received solution
        self.rwriter.write(self.rbufs, metadata, intg.tcurr)
