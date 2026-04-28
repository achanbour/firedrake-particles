import numpy as np

class ParticleLogger:
    """
    Logger exposing methods for pretty-printing data computed in the particle trajectory loop
    """

    LEVELS = {
        "silent": 0,
        "summary": 1,
        "info": 2,
        "debug": 3
    }
    
    def __init__(self, level="info", float_fmt=".4f"):
        self.level = self.LEVELS.get(level, None) # verbosity level
        if self.level is None:
            valid = ", ".join(self.LEVELS.keys())
            raise ValueError(
                f"Unrecognised log level '{level}': must be one of {valid}."
            )
        self.float_fmt = float_fmt # precision for displaying floats

    # Print
    def log(self, *args, level="info", **kwargs):
        # Print if the current verbosity level allows it
        if self.LEVELS.get(level, 0) <= self.level:
            print(*args, **kwargs)

    def section(self, title, level="summary"):
        if self.LEVELS.get(level, 0) <= self.level:
            print()
            print("="*60)
            print(f"    {title}")
            print("="*60)

    def subsection(self, title, level="info"):
        if self.LEVELS.get(level, 0) <= self.level:
            print(f"\n--- {title} ---")

    
    # Inspect
    def inspect(self, label, value, level="debug", indent=0):
        if self.LEVELS[level] > self.level:
            return
        pad = "    "*indent
        if isinstance(value, np.ndarray):
            formatted = np.array2string(value, precision=4, suppress_small=True)
        elif isinstance(value, (float, np.floating)):
            formatted = f"{value:{self.float_fmt}}"
        else:
            formatted = repr(value)
        print(f"{pad}{label}: {formatted}")


    def inspect_particles(self, label, data, level="debug"):
        if self.LEVELS[level] > self.level:
            return
        
        arrays = {k: np.atleast_1d(np.asarray(v)) for k, v in data.items()}
        N = next(iter(arrays.values())).shape[0]
        
        # For each field k, fetch the value(s) of each particle
        rows = {}
        for k, a in arrays.items():
            rows[k] = []
            for i in range(N):
                entry = a[i]
                values = np.atleast_1d(entry)
                parts = []
                for v in values:
                    if isinstance(v, (float, np.floating)):
                        parts.append(f"{v:{self.float_fmt}}")
                    else:
                        parts.append(repr(v))
                rows[k].append("   ".join(parts))

        # Compute the column width of each field
        widths = {}
        for k in arrays:
            widths[k] = max(len(k), max(len(r) for r in rows[k]))
        
        # Print data
        # One row for each particle
        header = "  " + " | ".join(f"{k:<{widths[k]}}" for k in arrays)
        print(f"\n [{label}]")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for i in range(N):
            row = "  " + "  ".join(f"{rows[k][i]:<{widths[k]}}" for k in arrays)
            print(row)
    
    def print_particles(self, label, data, indices=None, level="debug"):
        if self.LEVELS[level] > self.level:
            return

        arrays = {k: np.atleast_1d(np.asarray(v)) for k, v in data.items()}
        N = next(iter(arrays.values())).shape[0]

        # For each field k, fetch the value(s) of each particle
        rows = {}
        for k, a in arrays.items():
            rows[k] = []
            for i in range(N):
                entry = a[i]
                values = np.atleast_1d(entry)
                parts = []
                for v in values:
                    if isinstance(v, (float, np.floating)):
                        parts.append(f"{v:{self.float_fmt}}")
                    else:
                        parts.append(repr(v))
                rows[k].append("   ".join(parts))

        # Compute the column width of each field
        widths = {}
        for k in arrays:
            widths[k] = max(len(k), max(len(r) for r in rows[k]))

        pid_width = max(len(str(indices[i] if indices is not None else i)) for i in range(N))
        pid_width = max(pid_width, len("pid"))

        # Print data
        # One row for each particle
        header = f"| {'pid':<{pid_width}} | " + " | ".join(f"{k:<{widths[k]}}" for k in arrays) + " |"
        div = "-" * max(len(header), len(label) + 4)

        print(f"\n{div}\n {label}\n{div}")
        print(header)
        for i in range(N):
            block_label = indices[i] if indices is not None else i
            print(f"| {block_label:<{pid_width}} | " + " | ".join(f"{rows[k][i]:<{widths[k]}}" for k in arrays) + " |")
        print(div)


    # Summarise loops
    def outer_loop(self, iteration, t, dt, T, N):
        if self.LEVELS["summary"] <= self.level:
            self.section(
                f"Outer loop {iteration}  |  t = {t:{self.float_fmt}} -> "
                f"{min(t + dt, T):{self.float_fmt}}  |  N = {N}",
                level="summary"
            )

    def inner_loop(self, iteration, active, passed, failed):
        if self.LEVELS["info"] <= self.level:
            self.subsection(f"Inner loop iteration {iteration}", level="info")
            self.inspect("active", active,  level="info")
            self.inspect("passed", passed, level="info")
            self.inspect("failed", failed,  level="info")

    def outer_summary(self, iteration, inner_iters, active_iters, boundary, ref_positions, phys_positions):
        if self.LEVELS["summary"] <= self.level:
            self.section(f"End of time step {iteration} — summary", level="summary")
            self.inspect("Inner iterations", inner_iters, level="summary", indent=1)
            self.inspect("Active iters/particle", active_iters, level="summary", indent=1)
            self.inspect("Boundary particles", boundary, level="summary", indent=1)
            self.inspect("New reference positions", ref_positions,  level="summary", indent=1)
            self.inspect("New physical positions",  phys_positions, level="summary", indent=1)
    
