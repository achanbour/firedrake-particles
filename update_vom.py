import numpy as np

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
        self._invalidate_topology_properties()

        # Step 4: Update the coordinates and reference coordinates Functions
        self._update_coordinates(embedding)
    
    def update_ref_view(self, next_parent_cells, new_refcoords):
        """A reference-only update method:
        Updates the parent cell ownership and reference coordinates only assuming that the new pair 
        (parent cell, ref coords) is geometrically correct.
        """
        swarm = self.vom.topology_dm
        vom_to_swarm = self.vom.cell_closure[:, -1] # VOM cell ID -> DMSwarm point ID

        # NOTE: Swarm fields are in DMSwarm ordering but `next_parent_cells` and `new_refcoords` are in VOM ordering.

        next_parent_cells = np.asarray(next_parent_cells, dtype=int).reshape((-1, 1))
        new_refcoords = np.asarray(new_refcoords, dtype=float)

        # Update Firedrake parent cell numbers
        arr = swarm.getField("parentcellnum")
        arr[vom_to_swarm, 0] = next_parent_cells[:, 0]
        swarm.restoreField("parentcellnum")

        # Update DMSwarm parent cell numbers (uses plex numbering)
        cell_id_name = swarm.getCellDMActive().getCellID()
        arr = swarm.getField(cell_id_name)

        plex_ids = self.parent_mesh.topology.cell_closure[
            next_parent_cells.reshape(-1), -1
        ].reshape((-1, 1))

        arr[vom_to_swarm, :] = plex_ids
        swarm.restoreField(cell_id_name)

        # Update reference coordinates
        arr = swarm.getField("refcoord")
        arr[vom_to_swarm, :] = new_refcoords
        swarm.restoreField("refcoord")
        
        # 4) Invalidate cached topology
        # NOTE: Invalidating caches causes them to be recomputed on next access,
        # but we haven't updated all the fields at this stage yet.
        # Instead of invalidating all properties by calling `self.invalidate_topology_properties()`
        # we only delete the cached properties that depend on parent cell ownership.
        topology = self.vom.topology
        for name in (
            "cell_parent_cell_list",
            "cell_parent_cell_map",
            "cell_parent_base_cell_list",
            "cell_parent_base_cell_map",
            "cell_parent_extrusion_height_list",
            "cell_parent_extrusion_height_map",
        ):
            if name in topology.__dict__:
                del topology.__dict__[name]

            if name in self.vom.__dict__:
                del self.vom.__dict__[name]

        # 5) Update reference coordinates Function
        # NOTE: Since new_ref_coords is already in VOM ordering, AND assuming the VOM does not get reodered,
        # we can bypass the `dmcommon.reordered_coords` and directly update the Function dat array.
        ref_coords_func = self.vom.reference_coordinates
        ref_coords_func.dat.data[:] = new_refcoords
    
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

        # `_parent_mesh_embedding` expects data in input order

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

        # NOTE" Fields currently not updated:
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
    
    def _invalidate_topology_properties(self):
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

        # Reset mesh-geometry properties so they are rebuilt from the updated coordinates
        self.vom._spatial_index = None
        self.vom._bounding_box_coords = None
        self.vom._saved_coordinate_dat_version = self.vom.coordinates.dat.dat_version


