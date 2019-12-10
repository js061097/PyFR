# -*- coding: utf-8 -*-

import itertools as it
import os
import shlex
import tempfile
import uuid

from appdirs import user_cache_dir
import numpy as np
from pytools.prefork import call_capture_output

from pyfr.ctypesutil import platform_libname
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import digest, lazyprop, mv, rm


class SourceModule(object):
    _dir_seq = it.count()

    def __init__(self, src, proc, cfg):
        self.proc = proc

        # Find the NEC compiler
        self.cc = cfg.getpath('backend-veo', 'cc', 'ncc')

        # User specified compiler flags
        self.cflags = shlex.split(cfg.get('backend-veo', 'cflags', ''))

        # Get the compiler version string
        version = call_capture_output([self.cc, '--version'])

        # Get the base compiler command strig
        cmd = self.cc_cmd(None, None)

        # Compute a digest of the current version, compiler, and source
        self.digest = digest(version, cmd, src)

        # Attempt to load the library from the cache
        self.mod = self._cache_loadlib()

        # Otherwise, we need to compile the kernel
        if not self.mod:
            # Create a scratch directory
            tmpidx = next(self._dir_seq)
            tmpdir = tempfile.mkdtemp(prefix='pyfr-{0}-'.format(tmpidx))

            try:
                # Compile and link the source into a shared library
                cname, lname = 'tmp.c', platform_libname('tmp')

                # Write the source code out
                with open(os.path.join(tmpdir, cname), 'w') as f:
                    f.write(src)

                # Invoke the compiler
                call_capture_output(self.cc_cmd(cname, lname), cwd=tmpdir)

                # Determine the fully qualified library name
                lpath = os.path.join(tmpdir, lname)

                # Add it to the cache and load
                self.mod = self._cache_set_and_loadlib(lpath)
            finally:
                # Unless we're debugging delete the scratch directory
                if 'PYFR_DEBUG_VEO_KEEP_LIBS' not in os.environ:
                    rm(tmpdir)

    def cc_cmd(self, srcname, libname):
        cmd = [
            self.cc,                # Compiler name
            '-shared',              # Create a shared library
            '-std=c99',             # Enable C99 support
            '-O4',                  # Optimise, incl. -ffast-math
            '-fopenmp',             # Enable OpenMP support
            '-fPIC',                # Generate position-independent code
            '-o', libname, srcname, # Library and source file names
        ]

        # Append any user-provided arguments and return
        return cmd + self.cflags

    @lazyprop
    def cachedir(self):
        return os.environ.get('PYFR_VEO_CACHE_DIR',
                              user_cache_dir('pyfr', 'pyfr'))

    def _cache_loadlib(self):
        # If caching is disabled then return
        if 'PYFR_DEBUG_VEO_DISABLE_CACHE' in os.environ:
            return
        # Otherwise, check the cache
        else:
            # Determine the cached library path
            clpath = os.path.join(self.cachedir, platform_libname(self.digest))

            # Attempt to load the library onto the VE
            try:
                return self.proc.load_library(clpath)
            except RuntimeError:
                return

    def _cache_set_and_loadlib(self, lpath):
        # If caching is disabled then just load the library as-is
        if 'PYFR_DEBUG_VEO_DISABLE_CACHE' in os.environ:
            return self.proc.load_library(lpath)
        # Otherwise, move the library into the cache and load
        else:
            # Determine the cached library name and path
            clname = platform_libname(self.digest)
            clpath = os.path.join(self.cachedir, clname)
            ctpath = os.path.join(self.cachedir, str(uuid.uuid4()))

            try:
                # Ensure the cache directory exists
                os.makedirs(self.cachedir, exist_ok=True)

                # Perform a two-phase move to get the library in place
                mv(lpath, ctpath)
                mv(ctpath, clpath)
            # If an exception is raised, load from the original path
            except OSError:
                return self.proc.load_library(lpath)
            # Otherwise, load from the cache dir
            else:
                return self.proc.load_library(clpath)

    def function(self, name, restype, argtypes):
        type_map = {
            np.int32: 'int', np.int64: 'long', np.uint64: 'unsigned long',
            np.float32: 'float', np.float64: 'double'
        }

        # Get the function
        fun = self.mod.find_function(name)
        fun.args_type(*[type_map[np.dtype(t).type] for t in argtypes])
        fun.ret_type(type_map[np.dtype(restype).type] if restype else 'void')

        return fun
