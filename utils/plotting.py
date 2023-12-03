
def plot_2d_cmap(xvec, z, ax, vmin = -1, vmax = 1, title = ''):
    map = ax.pcolormesh(
        xvec,
        xvec,
        z,
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_title(label=title)
    ax.set_aspect("equal", "box")
    return map