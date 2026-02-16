from firedrake import *
import numpy as numpy
import sympy as sp

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import Point, TensorProductCell, UFCQuadrilateral, UFCHexahedron

interval = UFCInterval()
triangle = UFCTriangle()
quadrilateral = UFCQuadrilateral()
hexahedron = UFCHexahedron()
tetrahedron = UFCTetrahedron()
interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)

# UFL symbolics (point on a mesh domain)
mesh_1d = UnitIntervalMesh(10)
mesh_2d = UnitSquareMesh(10, 10)
mesh_3d = UnitCubeMesh(10, 10, 10)

x_1d = SpatialCoordinate(mesh_1d)
x_2d = SpatialCoordinate(mesh_2d)
x_3d = SpatialCoordinate(mesh_3d)

# SymPy symbolics (free symbol)
x1, x2, x3 = sp.symbols("x1 x2 x3")
p = [x1, x2]
q = [x1, x2, x3]

bary_coords_interval_sympy = interval.compute_barycentric_coordinates([x1])
bary_coords_interval_ufl = interval.compute_barycentric_coordinates(x_1d)

bary_coords_tri_sympy = triangle.compute_barycentric_coordinates(p)
bary_coords_tri_ufl = triangle.compute_barycentric_coordinates(x_2d)

bary_coords_quad_sympy = quadrilateral.compute_barycentric_coordinates(p)
bary_coords_quad_ufl = quadrilateral.compute_barycentric_coordinates(x_2d)

bary_coords_interval_x_interval_sympy = interval_x_interval.compute_axis_barycentric_coordinates(p)
bary_coords_interval_x_interval_ufl = interval_x_interval.compute_axis_barycentric_coordinates(x_2d)

bary_coords_tri_x_interval_sympy = triangle_x_interval.compute_axis_barycentric_coordinates(q)
bary_coords_tri_x_interval_ufl = triangle_x_interval.compute_axis_barycentric_coordinates(x_3d)

breakpoint()