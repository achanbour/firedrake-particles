import numpy as np

# TODO: append particle IDs all the time?
def write_particle_vtk(fname, vom, functions, coordinates=None):
    if coordinates is None:
        coordinates = vom.coordinates.dat.data_ro
    coordinates = np.pad(coordinates, ((0,0), (0,1)))
    num_points = coordinates.shape[0]

    # Accept a single function or a list of functions
    if not isinstance(functions, (list, tuple)):
        functions = [functions]
    
    # Build (name, padded-array) pairs
    fields = []
    for i, fn in enumerate(functions):
        arr = fn.dat.data_ro
        if fn.name().lower().endswith(("id", "ids")):
            arr = arr.astype(np.int64)
        arr = np.ascontiguousarray(fn.dat.data_ro)
        if arr.ndim > 1:
            arr = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])))   # -> width 3
        fields.append((fn.name() or f"field_{i}", np.ascontiguousarray(arr)))

    def write_array_descriptor(f, name, arr, offset):
        nbytes = 0
        # ncmp = {0: "", 1: "3"}[len(arr.shape[1:])]
        ncmp = 1 if arr.ndim == 1 else arr.shape[1]
        typ = {np.dtype("float64"): "Float64",}[arr.dtype]
        offset += nbytes
        nbytes += (4 + arr.nbytes)
        f.write(f"""<DataArray Name="{name}" type="{typ}" NumberOfComponents="{ncmp}" format="appended" offset="{offset}" />""".encode())
        return nbytes

    def write_array(f, arr):
        np.uint32(arr.nbytes).tofile(f)
        arr.tofile(f)

    with open(fname, "wb") as f:
        offset = 0
        f.write(f"""<?xml version="1.0" ?>
<VTKFile type="UnstructuredGrid" version="2.2" byte_order="LittleEndian" header_type="UInt32">
<UnstructuredGrid>
<Piece NumberOfPoints="{num_points}" NumberOfCells="0">
<Cells>
<DataArray type="Int32" Name="connectivity" format="ascii"></DataArray>
<DataArray type="Int32" Name="offsets" format="ascii"></DataArray>
<DataArray type="UInt8" Name="types" format="ascii"></DataArray>
</Cells>
<Points>
""".encode())
        offset += write_array_descriptor(f, "Coordinates", coordinates, offset)
        f.write(b"</Points>")

        # Use vector field as default vector attribute used for glyphs + vector operations
        vectors_attr = next(
        (name for name, arr in fields if arr.ndim > 1 and arr.shape[1] == 3),
            ""
        )
        f.write(f"""<PointData Vectors="{vectors_attr}">""".encode())
        for name, arr in fields:
            offset += write_array_descriptor(f, name, arr, offset)
        f.write(b"""</PointData>
</Piece>
</UnstructuredGrid>
<AppendedData encoding="raw">
_""")
        write_array(f, coordinates)
        for _, arr in fields:
            write_array(f, arr)
        f.write(b"""</AppendedData>
</VTKFile>""")


def write_pvd(fname, tmpl, count):
    with open(fname, "w") as f:
        f.write("""<?xml version="1.0" ?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection>
""")
        for i in range(count):
            f.write(f"""<DataSet timestep="{i}" file="{tmpl.format(i)}" />\n""")

        f.write("</Collection></VTKFile>")