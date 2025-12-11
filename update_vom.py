import numpy as np

class RebuildVertexOnlyMesh:
    """
    A VertexOnlyMesh that rebuilds itself when coordinates change exactly how Firedrake
    builds it internally: using _pic_swarm_in_mesh -> VertexOnlyMeshTopology ->
    make_vom_from_vom_topology.

    This produces a brand-new fully consistent VOM.
    """

    def __init__(self, vom, parent_mesh, redundant=True, tolerance=None):
        self.vom = vom                 
        self.parent_mesh = parent_mesh
        self.redundant = redundant
        self.tolerance = tolerance or parent_mesh.tolerance
    
    def _build_new_vom(self, new_coords):
        """Rebuild the internal VOM from new coordinates."""
        from firedrake.mesh import _pic_swarm_in_mesh, VertexOnlyMeshTopology, make_vom_from_vom_topology
        
        # Step 1: recompute the PIC swarm 
        swarm, input_ordering_swarm, n_missing = _pic_swarm_in_mesh(
            self.parent_mesh,
            np.asarray(new_coords, float),
            tolerance=self.tolerance,
            redundant=self.redundant,
            exclude_halos=False,
        )

        # Step 2: build the new VOM topology
        topology = VertexOnlyMeshTopology(
            swarm,
            self.parent_mesh.topology,
            name=swarm.getName(),
            reorder=False,
            input_ordering_swarm=input_ordering_swarm,
        )

        # Step 3: Build the VOM from the new topology
        new_vom = make_vom_from_vom_topology(
            topology,
            name=self.vom.name,      
            tolerance=self.tolerance,
        )
        new_vom._parent_mesh = self.parent_mesh

        return new_vom

    def update(self, new_coords):
        # Build a new VOM
        new_vom = self._build_new_vom(new_coords)

        # Swap the reference to the new VOM
        self.vom = new_vom
        return new_vom

    def __getattr__(self, name):
        # Forward all attributes to underlying VOM
        if hasattr(self.vom, name):
            return getattr(self.vom, name)
        raise AttributeError(name)

# class DynamicVertexOnlyMesh:
#     """
#     A VertexOnlyMesh that rebuilds itself when coordinates change. (same as above)
#     """
#     def __init__(self, parent_mesh, coords, *, redundant=True, tolerance=None):
#         self.parent_mesh = parent_mesh
#         self.coords = np.asarray(coords, float)
#         self.redundant = redundant
#         self.tolerance = tolerance

#         self._vom = None
#         self._rebuild()

#     @property
#     def vom(self):
#         """Return the underlying VOM"""
#         return self._vom
    
#     # @property
#     # def coordinates(self):
#     #     """Return the coordinate field of the underlying VOM"""
#     #     return self._vom.coordinates
    
#     # Update coordinates + rebuild everything
#     def update(self, new_coords):
#         self.coords = np.asarray(new_coords, float)
#         self._rebuild()

#     # Internal VOM rebuild
#     def _rebuild(self):
#         """Recompute parent mesh embedding and topoology"""
#         new_vom = VertexOnlyMesh(
#             self.parent_mesh,
#             self.coords,
#             tolerance=self.tolerance,
#             redundant=self.redundant,
#             missing_points_behaviour="error",
#         )
#         # Overwrite all mesh internals with the newly created VOM
#         # The old VOM gets garbage-collected as long as nothing else holds reference to it (for example old function spaces)
#         self.__dict__["_vom"] = new_vom # equivalent to `self._vom = new_vom` but bypasses `__getattr__` by writing directly into attribute dict.


#     def __getattr__(self, name):
#         # Forward all attributes to underlying VOM
#         if hasattr(self._vom, name):
#             return getattr(self._vom, name)
#         raise AttributeError(name)

class VertexOnlyMeshUpdater:
    """
    An in-place updater of a VertexOnlyMesh

    VertexOnlyMesh internally builds the following objects:
	- a PETSc DMSwarm
	- a VertexOnlyMeshTopology (wrapper for DMSwarm + container for topological properties)
	- a MeshGeometry object (defining coordinate fields)

    Therefore, to avoid recreating a new VOM, we mutate the DMSwarm object, invalidate cached topological properties,
    and update the coordinate fields. 
    This means that the updater keeps the same topology, and just modifies the coordinates/parent cells/reference fields.

    The following two assumptions are made:
    - The number of particles is fixed.
    - Only coordinates (and their parent embedding) change.
    """

    def __init__(self, vom, parent_mesh):
        self.vom = vom  # existing VOM
        self.parent_mesh = parent_mesh # parent mesh

    def update(self, new_coords_fn, tolerance=None, redundant=True, exclude_halos=False):
        # new_coords = np.asarray(new_coords, float)

        """
       `new_coords_fn` is a Firedrake function storing the updated coordinates in VOM ordering
        we need to feed these coordinates in input ordering to `_parent_mesh_embedding`.
   
        NOTE: The `input_ordering` VOM does not currently get updated. It is cached when creating 
        the initial VOM, and our update function does not currently recreate it.
        This means that `input_ordering.coordinates` Function has DoFs at outdated points.
        However, Functions and Function Spaces have no notion of coordinates; instead they depend on the
        Geometry. So, interpolation into the input ordering VOM works as long as the points don't get reordered.
        """
        from firedrake import assemble, interpolate
        
        vom_coords_io = self.vom.input_ordering.coordinates
        new_coords_fn_io = assemble(interpolate(new_coords_fn, vom_coords_io.function_space()))
        new_coords= new_coords_fn_io.dat.data_ro

        tolerance = tolerance or self.parent_mesh.tolerance

        # Step 1: recompute the VOM embedding
        embedding = self._recompute_embedding(new_coords, tolerance, redundant, exclude_halos)

        # Step 2: update the DMSwarm fields
        self._update_dmswarm_fields(embedding)

        # Step 3: Invalidate topological properties in VertexOnlyMeshTopology
        self.invalidate_topology_properties()

        # Step 4: Update the coordinates and reference coordinates Functions
        self._update_coordinates(embedding)
    
    def _recompute_embedding(self, new_coords, tolerance, redundant, exclude_halos):
        from firedrake.mesh import _parent_mesh_embedding
        (
            coords_embedded,
            global_idxs,
            reference_coords,
            parent_cell_nums,
            owned_ranks,
            input_ranks,
            input_coords_idxs,
            missing_global_idxs,
        ) = _parent_mesh_embedding(
            self.parent_mesh,
            new_coords,
            tolerance,
            redundant,
            exclude_halos,
            remove_missing_points=False,
        )

        # `_parent_mesh_embedding` returns point data in input order

        return dict(
        coords=coords_embedded,
        parent_cells=parent_cell_nums,
        refcoords=reference_coords,
        global_indices=global_idxs,
        owned_ranks=owned_ranks,
        input_ranks=input_ranks,
        input_indices=input_coords_idxs,
        missing_global_indices=missing_global_idxs,
    )

    # def _compute_order(self, embedding):
    #     swarm = self.vom.topology_dm
    #     # Fixed particle IDs in the existing VOM ?
    #     swarm_gid = swarm.getField("globalindex").reshape(-1)
    #     swarm.restoreField("globalindex")
    #     emb_gid = embedding["global_indices"]
    #     inv = np.empty_like(emb_gid)
    #     inv[emb_gid] = np.arange(len(emb_gid))

    #     return inv[swarm_gid]
    
    def _update_dmswarm_fields(self, embedding):
        swarm = self.vom.topology_dm
        # print(swarm.fields)

        # We cannot update the swarm fields by blindly assuming that the rows of embedding match the rows of the DMSwarm fields
        # In particular, _parent_mesh_embedding and DMSwarm store point data in different order so we need a way to match them.

        # swarm_gid = swarm.getField("globalindex").copy() # particle ID in the original VOM
        # swarm.restoreField("globalindex") 
        # emb_input = embedding["input_indices"] # map embedding ordering -> input ordering
        # inv = np.empty_like(emb_input)
        # inv[emb_input] = np.arange(len(emb_input))
        # order = inv[swarm_gid]

        # NOTE: Fields currently not updated:
        # - DMSwarm_rank
        # - globalindex (unique ID for each DMSwarm point)
        # - inputrank (MPI rank at which the input point coordinates were supplied)
        # - inputindex (index of each point in the input coordinates array after it has been redistributed to the correct rank)

        # Physical coordinates
        arr = swarm.getField("DMSwarmPIC_coor")
        arr_field = embedding["coords"]
        arr_field = arr_field.reshape(len(arr_field), -1)
        arr[:, :] = arr_field
        swarm.restoreField("DMSwarmPIC_coor")

        # Firedrake parent cell ID
        arr = swarm.getField("parentcellnum")
        arr_field = embedding["parent_cells"]
        arr_field = embedding["parent_cells"].reshape(len(arr_field), -1)
        # print("old parentcellnum: ", arr)
        # print("new parentcellnum: ", arr_field)
        arr[:] = arr_field
        swarm.restoreField("parentcellnum")

        # The above field is different to swarm.getCellDMActive().getCellID()
        # which stores the parent cell numbers in DMSwarm numbering
        # I think this is set to plex_parent_cell_nums
        # so it must be recomputed?

        # parent_mesh.topology.cell_closure maps cell numbers -> plex numbers
        # plex_parent_cell_nums = self.parent_mesh.topology.cell_closure[
        #     embedding["parent_cells"], -1
        # ]
        # plex_parent_cell_nums = plex_parent_cell_nums.reshape(len(plex_parent_cell_nums), -1)
        # arr = swarm.getField(swarm.getCellDMActive().getCellID())
        # arr[:, :] = plex_parent_cell_nums
        # swarm.restoreField(swarm.getCellDMActive().getCellID())

        # Reference coordinates (on reference cell)
        arr = swarm.getField("refcoord")
        arr_field = embedding["refcoords"]
        arr_field = arr_field.reshape(len(arr_field), -1)
        arr[:, :] = arr_field
        swarm.restoreField("refcoord")

        # TODO: build SF between primary swarm and updated swarm
    
    def invalidate_topology_properties(self):
        # There's a bunch of topological attributes using `topology_dm` that get computed in
        # `AbstractMeshTopology __init__()` - do they need to be invalidated now that the DMSwarm fields have changed?
        # There's a field called `self._dm_renumbering` which is computed by `_renumber_entities`
        # called from VertexOnlyMeshTopology which uses the plex cell numbering
        # I don't think this is an issue as long as VOM has been created with `reorder=None` or `False``

        # Delete cached attributes so they are lazily recomputed on next access using the updated swarm fields
        topology = self.vom.topology
        for name in (
            "exterior_facets"
            "interior_facets",
            "cell_to_facets"
            "cell_closure",
            "cell_set",
            "cell_parent_cell_list",
            "cell_parent_cell_map",
            "cell_parent_base_cell_list",
            "cell_parent_base_cell_map",
            "cell_parent_extrusion_height_list",
            "cell_parent_extrusion_height_map",
            "cell_global_index",
            "input_ordering",
            "input_ordering_sf",
            "input_ordering_without_halos_sf",
        ):
            if name in topology.__dict__:
                del topology.__dict__[name]

            if name in self.vom.__dict__:
                del self.vom.__dict__[name]


    def _update_coordinates(self, embedding):
        coords_embedded = embedding["coords"]
        refcoords_embedded = embedding["refcoords"]

        # This is wrong since the coords are reordered before being passed to the coordinate functions

        # Update physical coordinates (DG0 on vom)
        # coords_func = self.vom.coordinates
        # coords_func.dat.data[:] = coords_embedded

        # # Update reference coordinates (this is used by interpolation from parent mesh)
        # ref_func = self.vom.reference_coordinates
        # ref_func.dat.data[:] = refcoords_embedded

        import firedrake.cython.dmcommon as dmcommon

        topology = self.vom.topology
        parent_tdim = self.parent_mesh.topological_dimension

        # we don't need the embedding dict anymore since the data is already stored in `topology_dm`

        # NOTE: `reordered_coords` does not change the ordering of points in the DMSwarm or in the VOM
        # it merely reorders the coords. data to match the FS layout

        # Update ref. coords.
        ref_coords_func = self.vom.reference_coordinates
        ref_coords_fs = ref_coords_func.function_space()
        ref_coords_data = dmcommon.reordered_coords(topology.topology_dm, ref_coords_fs.dm.getDefaultSection(),
                                                    (topology.num_vertices(), parent_tdim),
                                                    reference_coord=True)
        ref_coords_func.dat.data[:] = ref_coords_data

        # Update physical coords.
        coords_func = self.vom.coordinates
        coords_fs = coords_func.function_space()
        gdim = self.vom.geometric_dimension
        coords_data = dmcommon.reordered_coords(topology.topology_dm, coords_fs.dm.getDefaultSection(),
                                                (topology.num_vertices(), gdim))
        coords_func.dat.data[:] = coords_data

        # Reset mesh-geometry properties so they rebuild from the updated coordinates
        self.vom._spatial_index = None
        self.vom._bounding_box_coords = None
        self.vom._saved_coordinate_dat_version = self.vom.coordinates.dat.dat_version

