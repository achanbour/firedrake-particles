import matplotlib.pyplot as plt
from firedrake.pyplot import triplot


def plot_particles_snapshot(mesh, vom, frame=0):
    particle_pos = vom.coordinates.dat.data_ro
    fig, axes = plt.subplots(figsize=(6, 6))
    triplot(mesh, axes=axes)
    axes.scatter(particle_pos[:, 0], particle_pos[:, 1], color="red", zorder=5, label="particles")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    # axes.legend(loc="upper left")
    plt.savefig(f"output/frame_{frame:04d}.png", dpi=150)
    plt.close(fig)

# Generate a video of moving particles
# ffmpeg -r 15 -i output/frame_%04d.png -c:v libx264 -pix_fmt yuv420p particles.mp4
# use `-crf 18` for higher quality, less compression
# Use `ls -1A | wc -l` to count generated frames in output directory