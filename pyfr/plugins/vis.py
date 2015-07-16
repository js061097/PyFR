# -*- coding: utf-8 -*-

from ctypes import CDLL, POINTER, Structure, c_int, c_void_p

import numpy as np

from pyfr.ctypesutil import load_library
from pyfr.plugins.base import BasePlugin
from pyfr.shapes import BaseShape
from pyfr.util import proxylist, subclass_where


class Piece(Structure):
    _fields_ = [
        ('na', c_int),
        ('nb', c_int),

        ('verts', c_void_p),

        ('ldim', c_int),
        ('lsdim', c_int),
        ('soln', c_void_p),

        ('nel', c_int),

        ('con', c_void_p),
        ('off', c_void_p),
        ('type', c_void_p)
    ]


class VisPlugin(BasePlugin):
    name = 'vis'
    systems = ['euler', 'navier-stokes']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.backend = backend = intg.backend
        self.mesh = intg.system.mesh

        # Amount of subdivision to perform
        self.divisor = self.cfg.getint(cfgsect, 'divisor', 3)

        # Allocate a queue on the backend
        self._queue = backend.queue()

        # Solution arrays
        self.eles_scal_upts_inb = inb = intg.system.eles_scal_upts_inb

        # Load the vis library
        self._load_vislib(self.cfg.getpath(cfgsect, 'libvis', abs=False))

        # Prepare the VTU pieces and interpolation kernels
        pieces, kerns = [], []
        for etype, solnmat in zip(intg.system.ele_types, inb):
            p, solnop = self._prepare_vtu(etype, intg.rallocs.prank)

            # Allocate on the backend
            vismat = backend.matrix((p.na, self.nvars, p.nb),
                                    tags={'align'})
            solnop = backend.const_matrix(solnop)
            backend.commit()

            # Populate the soln field and dimension info
            p.soln = vismat.data
            p.ldim = vismat.leaddim
            p.lsdim = vismat.leadsubdim

            # Prepare the matrix multiplication kernel
            k = backend.kernel('mul', solnop, solnmat, out=vismat)

            # Append
            pieces.append(p)
            kerns.append(k)

        # Save the pieces
        self._pieces = (Piece*len(pieces))(*pieces)

        # Wrap the kernels in a proxy list
        self._interpolate_upts = proxylist(kerns)

        # Finally, initialise the vis library
        self._vptr = self._lib.vis_init(len(pieces), self._pieces)

    def _prepare_vtu(self, etype, part):
        from pyfr.writers.paraview import BaseShapeSubDiv

        mesh = self.mesh['spt_{0}_p{1}'.format(etype, part)]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)
        subdvcls = subclass_where(BaseShapeSubDiv, name=etype)

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = shapecls.std_ele(self.divisor)
        nsvpts = len(svpts)

        # Shape
        soln_b = shapecls(nspts, self.cfg)

        # Generate the operator matrices
        mesh_vtu_op = soln_b.sbasis.nodal_basis_at(svpts)
        soln_vtu_op = soln_b.ubasis.nodal_basis_at(svpts)

        # Calculate node locations of vtu elements
        vpts = np.dot(mesh_vtu_op, mesh.reshape(nspts, -1))
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Reorder and cast
        vpts = vpts.swapaxes(1, 2).astype(self.backend.fpdtype)

        # Perform the sub division
        nodes = subdvcls.subnodes(self.divisor)

        # Prepare vtu cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]
        vtu_con = vtu_con.astype(np.int32)

        # Generate offset into the connectivity array
        vtu_off = np.tile(subdvcls.subcelloffs(self.divisor), (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]
        vtu_off = vtu_off.astype(np.int32)

        # Tile vtu cell type numbers
        vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)
        vtu_typ = vtu_typ.astype(np.uint8)

        # Construct the piece
        piece = Piece(na=nsvpts, nb=neles, verts=vpts.ctypes.data,
                      nel=len(vtu_typ), con=vtu_con.ctypes.data,
                      off=vtu_off.ctypes.data, type=vtu_typ.ctypes.data)

        # Retain the underlying NumPy objects
        piece._vpts = vpts
        piece._vtu_con = vtu_con
        piece._vtu_off = vtu_off
        piece._vtu_typ = vtu_typ

        return piece, soln_vtu_op

    def _load_vislib(self, lpath):
        self._lib = lib = CDLL(lpath)

        # vis_init
        lib.vis_init.argtypes = [c_int, POINTER(Piece)]
        lib.vis_init.restype = c_void_p

        # vis_step
        lib.vis_step.argtypes = [c_void_p]
        lib.vis_step.restype = None

        # vis_del
        lib.vis_del.argtypes = [c_void_p]
        lib.vis_step.restype = None

    def __del__(self):
        self._lib.vis_del(self._vptr)

    def __call__(self, intg):
        # Configure the input bank
        self.eles_scal_upts_inb.active = intg._idxcurr

        # Interpolate to the vis points
        self._queue % self._interpolate_upts()

        # Call out to the library method
        self._lib.vis_step(self._vptr)
