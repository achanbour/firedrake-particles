import numpy as np
from firedrake import UnitSquareMesh, PeriodicUnitSquareMesh
from particle_tracking.topology import find_next_cell

"""
The series of unit tests below assume a triangular mesh.
"""
def test_find_next_cell_is_pure():
    """Test that `find_next_cell` is a pure function (no side effects)."""
    mesh = UnitSquareMesh(2, 2)

    for c in range(mesh.num_cells()):
        for lf in range(3):
            a = find_next_cell(mesh, c, lf)
            b = find_next_cell(mesh, c, lf)
            assert a == b

def test_find_next_cell_boundary_vs_interior():
    """Test that `find_next_cell` correctly identifies boundary vs interior facets."""
    # A 2x2 mesh has 4 squares each split into 2 triangles -> 8 cells.
    mesh = UnitSquareMesh(2, 2) 
    facet_info = mesh.cell_to_facets.data_ro

    for c in range(mesh.num_cells()):
        for lf in range(3):
            is_interior = facet_info[c][lf][0]
            nxt = find_next_cell(mesh, c, lf)

            if not is_interior:
                # Boundary facet
                assert nxt is None
            
            if nxt is not None:
                # Interior facet
                assert nxt is not None
                assert nxt != c
                assert 0 <= nxt < mesh.num_cells()

def test_find_next_cell_unique_neighbour():
    """Test that `find_next_cell` returns a unique neighbouring cell 
    across interior facets."""
    mesh = UnitSquareMesh(2, 2)
    facet_info = mesh.cell_to_facets.data_ro

    for c in range(mesh.num_cells()):
        for lf in range(3):
            if facet_info[c][lf][0] != 1:
                continue # Skip boundary facets

            nxt = find_next_cell(mesh, c, lf)

            matches = [
                lf_ for lf_ in range(3)
                if find_next_cell(mesh, nxt, lf_) == c
            ]
            assert len(matches) == 1

def test_find_next_cell_adjacency_symmetry():
    """Test that `find_next_cell` is symmetric across interior facets."""
    mesh = UnitSquareMesh(2, 2)
    facet_info = mesh.cell_to_facets.data_ro

    for c in range(mesh.num_cells()):
        for lf in range(3):
            is_interior = facet_info[c][lf][0]
            if not is_interior:
                continue # Skip boundary facets

            nxt = find_next_cell(mesh, c, lf)
            assert nxt is not None
            
            # The neighbouring cell must have a facet pointing back to the original cell
            found_backlink = False
            for lf_ in range(3):
                nxt_nxt = find_next_cell(mesh, nxt, lf_)
                if nxt_nxt == c:
                    found_backlink = True
                    break
            assert found_backlink, (
                f"Adjacency symmetry violated: cell {c} facet {lf} -> {nxt}, "
                f"but no facet of {nxt} points back to {c}"
            )

def test_find_next_cell_matches_interior_facets():
    """Test that `find_next_cell` matches Firedrake's internal interior 
    facet connectivity."""
    mesh = UnitSquareMesh(2, 2)

    facet_cells = mesh.interior_facets.facet_cell
    local_facet_numbers = mesh.interior_facets.local_facet_dat.data_ro

    for f in range(facet_cells.shape[0]):
        c0, c1 = facet_cells[f]
        lf0, lf1 = local_facet_numbers[f]

        # c0 -> c1
        assert find_next_cell(mesh, c0, lf0) == c1

        # c1 -> c0
        assert find_next_cell(mesh, c1, lf1) == c0

def _cell_centroids(mesh):
    coords = mesh.coordinates.dat.data_ro
    cell_nodes = mesh.coordinates.function_space().cell_node_list # maps each mesh cell to function space nodes
    cell_centroids = np.zeros((mesh.num_cells(), mesh.geometric_dimension))
    for c in range(mesh.num_cells()):
        cell_centroids[c] = coords[cell_nodes[c]].mean(axis=0)
    return cell_centroids

def test_find_next_cell_periodic_mesh_geometry_wrap():
    """Test that `find_next_cell` correctly identifies neighbouring cells
    in a periodic mesh, including across periodic boundaries."""
    mesh = PeriodicUnitSquareMesh(3, 3)
    cell_centroids = _cell_centroids(mesh)

    def neighbours(c):
        return [find_next_cell(mesh, c, lf) for lf in range(3)]
    
    for c in range(mesh.num_cells()):
        xc, yc = cell_centroids[c]
        nbs = neighbours(c)
        assert all(nb is not None for nb in nbs)
        
        nbs_centroids = [cell_centroids[nb] for nb in nbs]

        # If close to left edge, expect a neighbour on the right edge
        if xc < 0.1:
            assert any(xn > 0.9 for xn, yn in nbs_centroids)
        
        # If close to right edge, expect a neighbour on the left edge
        if xc > 0.9:
            assert any(xn < 0.1 for xn, yn in nbs_centroids)

        # If close to bottom edge, expect a neighbour on the top edge
        if yc < 0.1:
            assert any(yn > 0.9 for xn, yn in nbs_centroids)
         
        # If close to top edge, expect a neighbour on the bottom edge
        if yc > 0.9:
            assert any(yn < 0.1 for xn, yn in nbs_centroids)

