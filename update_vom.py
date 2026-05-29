import numpy as np

class EmptyVOMError(Exception):
    """Raised when all particles have been absorbed and the VertexOnlyMesh is empty."""
    pass

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
        """
        Perform a full VOM update by mutating the appropriate fields.

       `new_coords_fn` is a Firedrake function storing the updated coordinates in VOM ordering
        we need to feed these coordinates in input ordering to `_parent_mesh_embedding`.
   
        NOTE: The `input_ordering` VOM does not currently get updated. It is cached when creating 
        the initial VOM, and our update function does not currently recreate it.
        This means that `input_ordering.coordinates` Function has DoFs at outdated points.
        However, Functions and Function Spaces do not depend on coordinate field, instead they depend on the
        Geometry of the mesh. So, interpolation into the input ordering VOM works as long as the points don't get reordered.
        """
        from firedrake import assemble, interpolate
        
        vom_coords_io = self.vom.input_ordering.coordinates
        new_coords_fn_io = assemble(interpolate(new_coords_fn, vom_coords_io.function_space()))
        new_coords = new_coords_fn_io.dat.data_ro

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
        """Perform a reference-only VOM update.

        Updates the parent cell ownership and reference coordinates under the assumption that
        (next_parent_cells, new_refcoords) are geometrically consistent 
        i.e., ref_coords correctly represents each point's reference coordinates in its new parent cell.
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
            # "cell_parent_cell_list",
            # "cell_parent_cell_map",
            "cell_parent_base_cell_list",
            "cell_parent_base_cell_map",
            "cell_parent_extrusion_height_list",
            "cell_parent_extrusion_height_map",
        ):
            if name in topology.__dict__:
                del topology.__dict__[name]

            if name in self.vom.__dict__:
                del self.vom.__dict__[name]
        
        # Mutate the map instead of invalidating it
        if "cell_parent_cell_list" in topology.__dict__:
            topology.__dict__["cell_parent_cell_list"][:] = next_parent_cells[:]

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


    def _update_dmswarm_fields(self, embedding):
        swarm = self.vom.topology_dm
        # print(swarm.fields)

        # NOTE: embedding and DMSwarm store points in different order.
        # `_parent_mesh_embedding` returns particles in global index order (which corresponds to input order when redundant=True) 
        # while swarm fields list particle data in swarm ordering

        # Remap the data between the two different orderings
        # NOTE: This only works when `redundant=True` (serial or broadcast input)
        swarm_gid = swarm.getField("globalindex").ravel() # for each swarm point: its original input index
        emb_input = embedding["input_indices"] # for each embedding row: its original input index
        inv =  np.empty(len(emb_input), dtype=int)
        inv[emb_input] = np.arange(len(emb_input)) # for each input index: its embedding row
        order = inv[swarm_gid] # for each swarm point k: its embedding row
        swarm.restoreField("globalindex")

        # Physical coordinates
        arr = swarm.getField("DMSwarmPIC_coor")
        arr_field = embedding["coords"]
        arr_field = arr_field.reshape(len(arr_field), -1)
        arr[:, :] = arr_field[order]
        swarm.restoreField("DMSwarmPIC_coor")

        # Parent cells
        arr = swarm.getField("parentcellnum")
        arr_field = embedding["parent_cells"]
        arr_field = embedding["parent_cells"].reshape(len(arr_field), -1)
        arr[:] = arr_field[order]
        swarm.restoreField("parentcellnum")

        # The above field is different to swarm.getCellDMActive().getCellID()
        # which stores the parent cell numbers in DMSwarm numbering
        # I believe this is set to the `plex_parent_cell_nums` swarm field 

        # The `cell_closure` maps Firedrake cell numbers (new ordering) -> plex cell numbers (old ordering)
        plex_parent_cell_nums = self.parent_mesh.topology.cell_closure[
            arr_field[order], -1
        ]
        plex_parent_cell_nums = plex_parent_cell_nums.reshape(len(plex_parent_cell_nums), -1)
        arr = swarm.getField(swarm.getCellDMActive().getCellID())
        arr[:, :] = plex_parent_cell_nums
        swarm.restoreField(swarm.getCellDMActive().getCellID())

        # Reference coordinates
        arr = swarm.getField("refcoord")
        arr_field = embedding["refcoords"]
        arr_field = arr_field.reshape(len(arr_field), -1)
        arr[:, :] = arr_field[order]
        swarm.restoreField("refcoord")

        # NOTE: Fields currently not updated:
        # - DMSwarm_rank
        # - globalindex (unique ID for each DMSwarm point)
        # - inputrank (MPI rank at which the input point coordinates were supplied)
        # - inputindex (index of each point in the input coordinates array after it has been redistributed to the correct rank)


    def _invalidate_topology_properties(self):
        # Delete cached attributes so they are lazily recomputed on next access using the updated swarm fields
        topology = self.vom.topology
        for name in (
            "exterior_facets",
            "interior_facets",
            "cell_to_facets",
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

        import firedrake.cython.dmcommon as dmcommon

        topology = self.vom.topology
        parent_tdim = self.parent_mesh.topological_dimension

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


    def rebuild_vom(self, absorbed_vom_indices, new_coords=None):
        from firedrake.mesh import (
            _parent_mesh_embedding,
            _swarm_original_ordering_preserve,
        )
        from firedrake.function import Function
        from firedrake.utils import IntType
        import firedrake.cython.dmcommon as dmcommon
        from pyop2.mpi import MPI
        
        comm = self.parent_mesh.comm
        swarm = self.vom.topology_dm
        topology = self.vom.topology
        gdim = self.vom.geometric_dimension
        tdim = self.parent_mesh.topological_dimension
            
        if new_coords is not None:
            if not isinstance(new_coords, Function):
                raise TypeError("Expected new particle coordinates to be supplied as a Function")

            new_coords = new_coords.dat.data_ro

            # Re-embedd all points using their updated coordinates (the resulting arrays are rank-local)
            tolerance = self.parent_mesh.tolerance # default
            (
                coords_local,
                global_idxs_local,
                reference_coords_local,
                parent_cell_nums_local,
                owned_ranks_local,
                input_ranks_local,
                input_coords_idxs_local,
                missing_global_idxs,
            ) = _parent_mesh_embedding(
                parent_mesh=self.parent_mesh,
                coords=new_coords,
                tolerance=tolerance,
                redundant=False, # each rank embedds its own local coordinates
                exclude_halos=False,
                remove_missing_points=False
            )
        else:
            # n_local = topology.num_vertices() # this returns owned + halo particles.
            n_local = topology.cell_set.size # returns owned particles only
            offset = comm.scan(n_local, op=MPI.SUM) - n_local # offset for current rank r (sums n_local across all ranks from 0 through to r)

            # Read data from the current VOM, ensuring that it is given to us in VOM order
            coords_local = self.vom.coordinates.dat.data_ro.copy()
            global_idxs_local = np.arange(offset, offset + n_local, dtype=IntType)
            input_coords_idxs_local = np.arange(n_local, dtype=IntType)
            reference_coords_local = self.vom.reference_coordinates.dat.data_ro.copy()
            parent_cell_nums_local =topology.cell_parent_cell_list[:n_local].copy()

            # Read data from the current VOM's swarm, then reoder to match current VOM's current
            # perm = topology._dm_renumbering.getIndices()[:n_local] # perm[cell_j] = swarm point k
            # owned_ranks_swarm = swarm.getField("DMSwarm_rank").ravel().copy()
            # swarm.restoreField("DMSwarm_rank")
            # owned_ranks_local = owned_ranks_swarm[perm]
            owned_ranks_local = np.full(n_local, comm.rank)

            # input_ranks_swarm = swarm.getField("inputrank").ravel().copy()
            # swarm.restoreField("inputrank")
            # input_ranks_local = input_ranks_swarm[perm]
            input_ranks_local = np.full(n_local, comm.rank)


        plex_parent_cell_nums_local = self.parent_mesh.topology.cell_closure[parent_cell_nums_local, -1]

        if len(absorbed_vom_indices) > 0:
            is_absorbed = np.isin(input_coords_idxs_local, absorbed_vom_indices)
            n_visible_local = int((parent_cell_nums_local != -1).sum()) - int(is_absorbed.sum())
        else:
            n_visible_local = int((parent_cell_nums_local != -1).sum())

        n_visible_global = comm.allreduce(n_visible_local, op=MPI.SUM)
        if n_visible_global == 0:
            raise EmptyVOMError("All particles have left the domain (no points remaining in the VertexOnlyMesh).")


        # Build the IO swarm capturing the current VOM state
        # NOTE: If we rewrite the IO VOM at every rebuild, then how do we maintain a ref. to the very first IO VOM?
        new_io_swarm = _swarm_original_ordering_preserve(
            comm=comm,
            swarm=swarm,
            original_ordering_coords_local=new_coords if new_coords is not None else coords_local,
            plex_parent_cell_nums_local=plex_parent_cell_nums_local,
            global_idxs_local=global_idxs_local,
            reference_coords_local=reference_coords_local,
            parent_cell_nums_local=parent_cell_nums_local,
            ranks_local=owned_ranks_local,
            input_ranks_local=input_ranks_local,
            input_idxs_local=input_coords_idxs_local,
            extruded=self.parent_mesh.extruded,
            layers=getattr(self.parent_mesh, "layers", None)
        )
        # Remove halos from the IO swarm pointSF (describes parallel distribution over ranks)
        # roots = locally owned points, leaves = ghost/halo points that are copies of roots owned by other ranks
        # This produces a directed graph from leaves -> roots across ranks
        # no halos ensure that the pair (inputrank, inputindex) uniquely identifies every point (every point is supplied by one rank only)

        sf = new_io_swarm.getPointSF()
        sf.setGraph(new_io_swarm.getLocalSize(), None, [])
        new_io_swarm.setPointSF(sf)

        topology.input_ordering_swarm = new_io_swarm

        # Mask any asborbed points as supplied by the user
        # NOTE: absorbed_vom_indices must be rank-local when running in parallel
        if len(absorbed_vom_indices):
            is_absorbed = np.isin(input_coords_idxs_local, absorbed_vom_indices)
            parent_cell_nums_local[is_absorbed] = -1 # overwrite as missing
        
        # This gives us points that have been absorbed and that have been unsuccessfully located in the local mesh partition
        visible = parent_cell_nums_local != -1
        n_visible = int(visible.sum())

        # Mutate the current swarm in-place
        # Compact the swarm fields to include visible points only
        # This calls `realloc` behind the hood which preserves the existing n_visible entries
        # so we can move the data to the first n_visible entries and compact
        swarm.setLocalSizes(n_visible, 0)

        # Now write back the visible entries into the compacted fields
        swarm_coords = swarm.getField("DMSwarmPIC_coor").reshape((n_visible, gdim))
        swarm_coords[...] = coords_local[visible]
        swarm.restoreField("DMSwarmPIC_coor")

        # TODO: When parent mesh is extruded, compute `base_parent_cell_nums`, `extrusion_heights`
        # and `plex_parent_cell_nums` based on `base_parent_cell_nums`

        plex_parent_cell_nums = np.full_like(parent_cell_nums_local, -1)
        plex_parent_cell_nums[visible] = self.parent_mesh.topology.cell_closure[
            parent_cell_nums_local[visible], -1
        ]

        cell_id_name = swarm.getCellDMActive().getCellID()
        swarm_parent_cell_nums = swarm.getField(cell_id_name).ravel()
        swarm_parent_cell_nums[...] = plex_parent_cell_nums[visible]
        swarm.restoreField(cell_id_name)

        field_global_index = swarm.getField("globalindex").ravel()
        field_global_index[...] = global_idxs_local[visible]
        swarm.restoreField("globalindex")

        field_reference_coords = swarm.getField("refcoord").reshape((n_visible, tdim))
        field_reference_coords[...] = reference_coords_local[visible]
        swarm.restoreField("refcoord")

        field_parent_cell_nums = swarm.getField("parentcellnum").ravel()
        field_parent_cell_nums[...] = parent_cell_nums_local[visible]
        swarm.restoreField("parentcellnum")

        field_rank = swarm.getField("DMSwarm_rank").ravel()
        field_rank[...] = owned_ranks_local[visible]
        swarm.restoreField("DMSwarm_rank")

        field_input_rank = swarm.getField("inputrank").ravel()
        field_input_rank[...] = input_ranks_local[visible]
        swarm.restoreField("inputrank")

        field_input_index = swarm.getField("inputindex").ravel()
        field_input_index[...] = input_coords_idxs_local[visible]
        swarm.restoreField("inputindex")
        
        # TODO:
        # if self.parent_mesh.extruded:
        #     field_base_parent_cell_nums = swarm.getField("parentcellbasenum").ravel()
        #     field_extrusion_heights = swarm.getField("parentcellextrusionheight").ravel()
        #     field_base_parent_cell_nums[...] = base_parent_cell_nums[visible]
        #     field_extrusion_heights[...] = extrusion_heights[visible]
        #     swarm.restoreField("parentcellbasenum")
        #     swarm.restoreField("parentcellextrusionheight")
        
        if new_coords is not None and comm.size > 1:
            # Parallel: redistribute particles accross ranks because they have now ended up in different parent cells.
            swarm.migrate(remove_sent_points=True)
            
            # Update the rank-local number of visible particles (post inter-rank exchange)
            n_visible = swarm.getLocalSize()

            # Update nroots in the pointSF
            sf = swarm.getPointSF()
            sf.setGraph(n_visible, None, [])
            swarm.setPointSF(sf)

        # Rebuild entity renumbering on the existing topology
        topology._dm_renumbering = topology._renumber_entities(reorder=True) # PETSc IS which maps Firedrake cell j to swarm point perm[j]

        # Clear stale entity class labels from the previous swarm size before re-marking
        for _label_name in ("pyop2_core", "pyop2_owned", "pyop2_ghost"):
            if swarm.hasLabel(_label_name):
                swarm.clearLabelStratum(_label_name, 1)

        # Refresh the entity classes (before calling create_section which uses the DM's entity class labels)
        dmcommon.mark_entity_classes_using_cell_dm(swarm) # rewrite the class labels on the swarm points based on the plex cell classes of each point's parent cell
        topology._entity_classes = dmcommon.get_entity_classes(swarm) # read those labels and store counts on the vom's topology

        # Rebuild _cell_numbering and _vertex_numbering from new _dm_renumbering
        entity_dofs = np.array([1], dtype=IntType)  # 1 DoF per point
        topology._cell_numbering, _ = topology.create_section(entity_dofs)
        topology._vertex_numbering = topology._cell_numbering

        # The PETSc Section, describing the FS of the coordinate field is built out of the IS
        # which essentially maps each swarm point to its DoF offset.
        # Function Spaces defined on the mesh borrow a reference to this Section when computing quantities
        # such as global numbering (these are constructed once, cached on the mesh and shared among all Function Spaces with the same entity dofs)
        # For the VOM coordinate fields, the FSs (and their sections) were created when the VOM was first built.
        
        # Clear the FS caches on the VOM
        topology._shared_data_cache.clear()

        # Invalidate cached topological properties
        # Amongst other things, this triggers a recomputation of the IO SF on next access
        self._invalidate_topology_properties()

        # Increment the VOM topology version
        # In parallel, execute as a collective operation
        # topology._topology_version += 1
        topology._topology_version = comm.allreduce(topology._topology_version + 1, op=MPI.MAX)

        # Store the one step SF
        # maps version k (new) -> version k-1 (old) stored under the key k
        topology._topology_step_sfs[topology._topology_version] = topology.input_ordering_sf

        # Refresh coordinate FSs eagerly
        coords_fs = self.vom._coordinates.function_space()
        ref_coords_fs = self.vom.reference_coordinates.function_space().topological
        for fs in [coords_fs, ref_coords_fs]:
            fs._refresh_shared_data()

        # Rebuild coordinate data using fresh sections
        coords_data = dmcommon.reordered_coords(swarm, coords_fs.dm.getDefaultSection(),
                                                (topology.num_vertices(), gdim))
        # Resize the CoordinatelessFunction dat buffer and assign new values
        self.vom._coordinates.dat = coords_fs.make_dat(val=coords_data, name=self.vom._coordinates.name()) # returns a new op2.Dat

        # The coordinate field has changed so we need to reset the rtree
        self.vom.clear_rtree()

        # To be sure, clear the geometry caches ensuring the rtree is properly rebuilt
        if "bounding_box_coords" in self.__dict__:
            del self.__dict__["bounding_box_coords"]

        parent_tdim = self.parent_mesh.ufl_cell().topological_dimension
        if parent_tdim > 0:
            ref_coords_data = dmcommon.reordered_coords(swarm, ref_coords_fs.dm.getDefaultSection(),
                                                        (topology.num_vertices(), parent_tdim), reference_coord=True)
            # Resize the CoordinatelessFunction dat buffer and assign new values
            # NOTE: Since reference_coordinates is a Function, ._data accesses the CoordinatelessFunction it wraps where the data buffer lives
            self.vom.reference_coordinates._data.dat = ref_coords_fs.make_dat(val=ref_coords_data, name=self.vom.reference_coordinates.name()) # returns a new op2.Dat
        else:
            # This should have been already set to None when the VOM was first constructed
            self.vom.reference_coordinates = None

