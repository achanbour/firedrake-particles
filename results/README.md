# Experiment results

Generated plots, VTK files, movies, and other experiment artifacts live here.
Each experiment writes to its own subdirectory so that common names such as
`output.pvd` and `particles.pvd` do not overwrite another run's files.

New scripts can use the same convention with:

```python
from output_paths import experiment_results_dir

output_dir = experiment_results_dir(__file__)
```

The generated contents of this directory are ignored by Git.
