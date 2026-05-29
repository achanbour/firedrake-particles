from typing import Protocol
import os
import matplotlib.pyplot as plt
from firedrake.pyplot import triplot, scatter

class ParticlePlotterProtocol(Protocol):
    def setup(self, particle_vom, parent_mesh):
        """Creates the figure and plots the particles embedded in their parent mesh."""
        pass

    def update(self, particle_vom):
        """Updates the particles' plot"""
        pass
    
    def close(self):
        """Closes the figure on which the plots are drawn."""
        pass

class ParticlePlotter(Protocol):
    """A concrete class implementing the ParticlePlotterProtocol, providing default matplotlib-based plotting"""

    def __init__(self, x_lim=(0,1), y_lim=(0, 1), output_dir=None, dpi=150):
        self.x_lim = x_lim
        self.y_lim = y_lim
        if output_dir is None:
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.dpi = dpi

    def setup(self, particle_vom, parent_mesh):
        self._fig, axes = plt.subplots()
        triplot(parent_mesh, axes=axes)
        self._sc = scatter(particle_vom, axes=axes)
        # Fix axess limits and aspect
        axes.set_xlim(self.x_lim)
        axes.set_ylim(self.y_lim)
        axes.set_aspect("equal")
        # Initialise the frame
        self._frame = 0
    
    def update(self, particle_vom):
        if self._sc is None:
            raise RuntimeError("")
        self._sc.set_offsets(particle_vom.coordinates.dat.data_ro)
        plt.savefig(f"{self.output_dir}/frame_{self._frame:04d}.png", dpi=self.dpi)
        self._frame += 1
    
    def close(self):
        plt.close(self._fig)


# Generate a movie of moving particles
# ffmpeg -r 15 -i output/frame_%04d.png -c:v libx264 -pix_fmt yuv420p particles.mp4
# use `-crf 18` for higher quality, less compression
# Use `ls -1A | wc -l` to count generated frames in output directory

# An alternative plotting method makes use of matplotlib's FuncAnimation
# see https://github.com/firedrakeproject/firedrake/blob/release/docs/notebooks/04-burgers.ipynb
