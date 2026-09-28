# bl_reader

`bl_reader` provides readers and analysis helpers for Athena++ simulation
outputs, with a focus on boundary-layer calculations.

## Getting started

The low-level readers are useful when working directly with Athena++ files:

- `athena_read` reads history, table, VTK, HDF5, and input files.
- `delayed_read.athdf` loads HDF5 fields lazily and can materialize all fields
  with `load_all()`.

The `simBLclass` package builds higher-level simulation and data-wrapper
objects on top of those readers. The API reference currently emphasizes the
reader functions, `BLsim`, and the core file-wrapper classes.

## Documentation status

This site is being expanded incrementally. The legacy analysis modules contain
many specialized plotting and diagnostic methods; those will be documented in
focused passes as their input, output, and file-side effects are verified.
