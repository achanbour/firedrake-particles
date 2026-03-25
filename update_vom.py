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

    def _rebuild_topology(self, absorbed_vom_indices, new_coords=None, tolerance=None, redundant=True, exclude_halos=False):
        """
        Rebuild the VOM topology after removing some particles (replicating broadly the steps in `_pic_swarm_in_mesh`).

        1) Embedd the coordinates of the current VOM and mark absorbed points as 'missing'.
        Extended to allow the embedding of updated coordinates if provided. 
        `new_coords` is assumed to contain the new coordinates of all points including the absorbed ones.

        2) Rebuild a new DMSwarm with only visible points.

        3) Define the current VOM as the input ordering VOM for the new VOM.

        4) Construct an SF between the current (old) VOM and the new VOM orderings.
        """
        from firedrake.mesh import (
            _parent_mesh_embedding,
            _dmswarm_create,
            _swarm_original_ordering_preserve,
            VertexOnlyMeshTopology
        )
        comm = self.parent_mesh.comm
        if comm.size != 1:
            raise NotImplementedError("Rebuilding the VOM topology is currently only supported for serial runs.")
        
        if not redundant:
            raise NotImplementedError("Rebuilding the VOM topology uses globalindex as persistent point IDs, so `redundant=True` is required.")
        
        absorbed_vom_indices = np.asarray(absorbed_vom_indices, dtype=int) # assumed to index into current VOM ordering

        coords_to_embedd = new_coords.dat.data_ro if new_coords is not None else self.vom.coordinates.dat.data_ro # coordinates to embedd (in current VOM)

        # 1. Embed current VOM coordinates
        tolerance = tolerance or self.parent_mesh.tolerance
        (
            coords_local,
            global_idxs_local,
            reference_coords_local,
            parent_cell_nums_local,
            owned_ranks_local,
            input_ranks_local,
            input_coords_idxs_local,
            missing_global_idxs_local
        ) = _parent_mesh_embedding(
            self.parent_mesh,
            coords_to_embedd,
            tolerance,
            redundant,
            exclude_halos,
            remove_missing_points=False
        )

        # Treat absorbed_vom_indices as missing points by overriding the outputs of _parent_mesh_embedding
        # Assuming serial + redundant input_coords_idxs_local is in the range 0,...,N_old-1
        # it corresponds to the indices of points in the coordinates array that was embedded (assumed to be the full array of points, including the missing ones)
        if absorbed_vom_indices.size:
            is_absorbed = np.isin(input_coords_idxs_local, absorbed_vom_indices)
            parent_cell_nums_local[is_absorbed] = -1  # mark as missing
            reference_coords_local[is_absorbed, :] = np.nan 
            owned_ranks_local[is_absorbed] = comm.size + 1  # set an invalid rank

        visible = parent_cell_nums_local != -1

        if self.parent_mesh.extruded:
            raise NotImplementedError("Rebuilding the VOM topology currently only supports non-extruded parent meshes.")
        
        # 2. Create a new DMSwarm with only visible points
        plex_parent_cell_nums_local = np.full_like(parent_cell_nums_local, -1)
        # For visible points, map parent cell nums to plex numbering using `cell_closure` (maps Firedrake's cell number to the corresponding PETSc plex cell number)
        plex_parent_cell_nums_local[visible] = self.parent_mesh.topology.cell_closure[parent_cell_nums_local[visible], -1]

        # NOTE:
        # - `coords_idxs` is written into the DMSwarm field `globalindex` as a unique ID for each swarm point.
        #       This is set to the index of each point in the input coordinates array when `redundant=True`, otherwise it's an index in rank order.
        # - `input_coords_idxs` is written into the DMSwarm field `inputindex`.
        #       This is set to the index of each point in the input coordinates array on the rank that supplied it (paired with `inputrank`).
        #       The pair (inputindex, inputrank) is then used to build the input-ordering SF (map the "point i"->"input-order point j").

        new_swarm = _dmswarm_create(
            fields=None,
            comm=comm,
            plex=self.parent_mesh.topology.topology_dm,
            coords=coords_local[visible],
            plex_parent_cell_nums=plex_parent_cell_nums_local[visible],
            coords_idxs=global_idxs_local[visible],
            reference_coords=reference_coords_local[visible],
            parent_cell_nums=parent_cell_nums_local[visible],
            ranks=owned_ranks_local[visible],
            input_ranks=input_ranks_local[visible],
            input_coords_idxs=input_coords_idxs_local[visible],
            base_parent_cell_nums=None,
            extrusion_heights=None,
            extruded=False,
            tdim=self.parent_mesh.topological_dimension,
            gdim=self.parent_mesh.geometric_dimension,
        )
        
        # 3. Create the input ordering swarm for the new DMSwarm (input ordering is given by VOM_0)
        new_input_ordering_swarm = _swarm_original_ordering_preserve(
            comm,
            new_swarm,
            coords_to_embedd, # coordinates in old VOM ordering instead of user input ordering
            plex_parent_cell_nums_local,
            global_idxs_local,
            reference_coords_local,
            parent_cell_nums_local,
            owned_ranks_local,
            input_ranks_local,
            input_coords_idxs_local,
            extruded=False,
            layers=getattr(self.parent_mesh, "layers", None),
        )

        # NOTE:
        # We created a new swarm with only visible points but then passed the full array of points when constructing the IO swarm.
        # The IO swarm faithfully represents the input state i.e., VOM_0
        # The IO SF between VOM_0 and VOM_1 will only map visible points

        # Ensure no halos on the IO swarm pointSF.
        # It must provide a pure index set so that each swarm point can be consistently identified in the input.
        # Get PETSc pointSF attached to the IO DMSwarm
        # roots = points "owned" on a rank, leaves = halos/ghost copies that reference roots on other ranks
        sf = new_input_ordering_swarm.getPointSF() # PETSc pointSF attached to the IO DMSwarm
        nroots = new_input_ordering_swarm.getLocalSize()
        sf.setGraph(nroots, None, []) # remove all leaves -> no halos
        new_input_ordering_swarm.setPointSF(sf)

        # --Create new VOM topology around the new DMSwarm--
        old_topology = self.vom.topology

        new_topology = VertexOnlyMeshTopology(
            new_swarm,
            self.parent_mesh.topology,
            name=new_swarm.getName() if new_swarm.getName() else "vom_topology_rebuild",
            reorder=False,
            input_ordering_swarm=new_input_ordering_swarm,
        )

        # NOTE: Old SF mapping
        # Build SF mapping from the new VOM (version k) back to the previous VOM (version k-1)
        # using the persistent DMSwarm "globalindex" IDs as the invariant key.
        # This SF will be used by Function._match_mesh_topology_version to migrate function data correctly
        # when particles are removed (so the new index i maps to the old index pid_to_old[pid_i]).
        # from firedrake.petsc import PETSc
        # from firedrake.utils import IntType

        # # Old VOM: pid (input index) per VOM vertex
        # old_swarm = old_topology.topology_dm
        # old_pids = old_swarm.getField("globalindex").ravel()
        # old_swarm.restoreField("globalindex")
        # old_vom_to_swarm = old_topology.cell_closure[:, -1]
        # pids_old_order = old_pids[old_vom_to_swarm]

        # # New VOM: pid per vertex
        # new_swarm = new_topology.topology_dm
        # new_pids = new_swarm.getField("globalindex").ravel()
        # new_swarm.restoreField("globalindex")
        # new_vom_to_swarm = new_topology.cell_closure[:, -1]
        # pids_new_order = new_pids[new_vom_to_swarm]

        # pid_to_old = {pid: i for i, pid in enumerate(pids_old_order)}
        # roots = np.array([pid_to_old[pid] for pid in pids_new_order], dtype=IntType)
        # nleaves = len(pids_new_order)
        # remote = np.empty(2 * nleaves, dtype=IntType)
        # remote[0::2] = 0 
        # remote[1::2] = roots

        # sf_new_to_old = PETSc.SF().create(comm=self.parent_mesh.comm)
        # sf_new_to_old.setGraph(old_topology.num_vertices(), None, remote)
        # sf_new_to_old.setUp()

        # NOTE: Old point mapping
        # --Build an explicit old VOM->new VOM point mapping--
        # using a persistent point ID that does not depend on the VOM ordering
        # `globalindex` works only if `redundant=True` since otherwise the globalindex is rank-dependent.
        # new_vom_to_swarm = new_topology.cell_closure[:, -1] # map new VOM index -> new swarm point ID
    
        # Extract persistent point IDs from old and new swarms
        # new_pids = new_swarm.getField("globalindex").ravel()
        # pids_in_new_vom_order = new_pids[new_vom_to_swarm]
        # new_swarm.restoreField("globalindex")

        # Build mapping: point ID (pid) -> new VOM vertex index
        # pid_to_new_vom = {pid: j for j, pid in enumerate(pids_in_new_vom_order)}

        # Build mapping: old VOM vertex index -> new VOM vertex index based on pid
        # N_old = old_topology.num_vertices()
        # old_to_new_point_mapping = np.full(N_old, -1, dtype=np.int32)
        # for i_old, pid in enumerate(pids_in_old_vom_order):
        #     if i_old in absorbed_vom_indices:
        #         continue
        #     old_to_new_point_mapping[i_old] = pid_to_new_vom.get(pid, -1)

        return new_topology

    def rebuild_vom(self, absorbed_vom_indices, new_coords=None):
        """Rebuild the VOM around a new topology.
        
        1) Recreate the underlying MeshGeometry object.

        2) Swap all attributes of the existing VOM and clear cached properties.

        3) Increment VOM version number to indicate that the VOM has changed.
        """
        # if absorbed_vom_indices is None or len(absorbed_vom_indices) == 0:
        #     self.update(new_coords)
        #     return

        import firedrake.cython.dmcommon as dmcommon
        import firedrake.functionspace as functionspace
        import firedrake.functionspaceimpl as functionspaceimpl
        import firedrake.function as function
        from firedrake.mesh import (
            make_vom_from_vom_topology,
            _generate_default_mesh_reference_coordinates_name,
        )
        import weakref

        new_vom_topology = self._rebuild_topology(absorbed_vom_indices, new_coords)
        
        # Stash one-step lineage
        # If using point mappings, we need to store a history of mappings
        """
        Suppose mesh versions evolve as follow:
	        - version 0 -> rebuild -> version 1 (stash mapping 0->1)
            - version 1 -> rebuild -> version 2 (stash mapping 1-> 2, overwriting the old stash)

	    a Function created at version 1 can migrate to version 2 using the current stash
	    a Function created at version 0 and first accessed at version 2 cannot migrate, because the stash no longer contains 0->1 and 0->2 is not available)
        """
        # self.vom._topology_lineage = {
        #     "from_version": old_version,
        #     "old_to_new_point_mapping": old_to_new_point_mapping,
        # }

        # --Build a new MeshGeometry object around the new topology--
        tolerance = getattr(self.vom, "_tolerance", self.parent_mesh.tolerance)
        new_mesh_geometry = make_vom_from_vom_topology(new_vom_topology, self.vom.name, tolerance)

        # NOTE: `_parent_mesh` is a property of a MeshGeometry object defined with a setter method, so we can set it explicitly.
        # MeshGeometry inherits from ufl.Mesh. 
        # UFL objects are meant to be immutable symbolic objects with a stable identity (used for compilation, subexpression elimination etc.)
        # Hence, the mutable properties of a MeshGeometry object are stored in a ufl_cargo() which is intended to store non-symbolic states without breaking UFL's expectations about the mesh object.
        new_mesh_geometry._parent_mesh = self.parent_mesh

        # --Transfer the new topology into the existing mesh (MeshGeometry) object--
        self.vom.topology = new_vom_topology
        self.vom._parent_mesh = self.parent_mesh
        self.vom._tolerance = tolerance

        # --Transfer the new coordinates into the VOM--
        # NOTE: the mesh coordinate field defines the geometry of the mesh. Therefore, it is defined as a CoordinatelessFunction (as opposed to a Function WithGeometry)
        # This means that its function space is built entirely from the mesh topology and the element type, and is not bound to a geometry object.
        # A `CoordinatelessFunction(V, ..)` lives on a function space `V` whose DM/Section specifies how many DoFs there are and how they're arranged. It does not carry a reference to a mesh (`MeshGeometry` object).
        # So `_coordinates` is just a vector of numbers laid out in the FS DoF layout. It can exist independently of the mesh.
        # When we access the `mesh.coordinates` field, Firedrake wraps that coordinateless fucnction in a `WithGeometry` function which binds it to the mesh geometry. This allows UFL to treat it as a spatial coordinate field.

        self.vom._coordinates = new_mesh_geometry._coordinates
        self.vom._coordinates_function = new_mesh_geometry._coordinates_function

        # Remove cached `coordinates` so they are rebuilt from the new topology.
        if "coordinates" in self.vom.__dict__:
            del self.vom.__dict__["coordinates"]

        # --Recreate reference coordinates as a WithGeometry Function--
        # First clear cached reference coordinates 
        if "reference_coordinates" in self.vom.__dict__:
            del self.vom.__dict__["reference_coordinates"]
        if "_reference_coordinates" in self.vom.__dict__:
            del self.vom.__dict__["_reference_coordinates"]

        parent_tdim = self.parent_mesh.topological_dimension
        ref_coords_fs = functionspace.VectorFunctionSpace(new_vom_topology, "DG", 0, dim=parent_tdim,)
        ref_coords_data = dmcommon.reordered_coords(
            new_vom_topology.topology_dm,
            ref_coords_fs.dm.getDefaultSection(),
            (new_vom_topology.num_vertices(), parent_tdim),
            reference_coord=True,
        )
        ref_coords_top = function.CoordinatelessFunction(
            ref_coords_fs,
            val=ref_coords_data,
            name=_generate_default_mesh_reference_coordinates_name(self.vom.name),
        )
        refV = functionspaceimpl.WithGeometry(ref_coords_fs, self.vom)
        self.vom.reference_coordinates = function.Function(refV,val=ref_coords_top)

        # --Clear cached topology-derived attributes on both the topology object and the mesh object--
        self._invalidate_topology_properties()
        
        # --Clear geometry caches so they are rebuilt from the new coordinate field--
        self.vom._spatial_index = None
        self.vom._bounding_box_coords = None
        self.vom._saved_coordinate_dat_version = self.vom.coordinates.dat.dat_version

        # --Increment VOM version number--
        # TODO: this has to be a collective operation
        if not hasattr(self.vom, "_topology_version"):
            self.vom._topology_version = 0
        self.vom._topology_version += 1

        # Initialize the dictionary that stores the one-step SFs 
        # This is done lazily, i.e., the first time the VOM gets rebuilt.
        if not hasattr(self.vom, "_topology_step_sfs"):
            self.vom._topology_step_sfs = {}

        # One-step SF maps version k (new) -> version k-1 (old) stored under the key k
        self.vom._topology_step_sfs[self.vom._topology_version] = self.vom.input_ordering_sf  

        return self.vom