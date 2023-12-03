import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from utils_fitting import gaussian_fit, gaussian, get_2d_guassian_sigma, gaussian_2d


def plot_2d_cmap(xvec, yvec, z, ax, vmin, vmax, title):
    map = ax.pcolormesh(
        xvec,
        yvec,
        (z),
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_title(label=title)
    ax.set_aspect("equal", "box")
    return map


def plot_double_2d_cmap(xvecs, yvecs, zs, vmin, vmax, titles):
    fig, axes = plt.subplots(1, 2)
    map1 = plot_2d_cmap(
        ax=axes[0],
        xvec=xvecs[0],
        yvec=yvecs[0],
        z=zs[0],
        vmin=vmin,
        vmax=vmax,
        title=titles[0],
    )

    map2 = plot_2d_cmap(
        ax=axes[1],
        xvec=xvecs[1],
        yvec=yvecs[1],
        z=zs[1],
        vmin=vmin,
        vmax=vmax,
        title=titles[1],
    )

    return fig, axes


def plot_double_1d_graphs(fig, axes, xvecs, yvecs, label, do_fits):
    if not fig:
        fig, axes = plt.subplots(1, 2)
    opt_results = [None, None]
    for i in range(len(axes)):
        if do_fits[i]:
            axes[i].scatter(xvecs[i], yvecs[i], label=label)
            opt_results[i] = gaussian_fit[i](yvecs[i], xvecs[i])
            axes[i].plot(xvecs[i], [gaussian(j, *opt_results) for j in xvecs[i]])
        else:
            axes[i].plot(xvecs[i], yvecs[i], label=label)
            opt_results[i] = None
    return fig, axes, opt_results


def do_fitting(xvec, yvec, z_values, axes, plot=True):
    (sig1, sig2, v1, v2, opt) = get_2d_guassian_sigma(x=xvec, y=yvec, z=z_values)
    V = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
    origin = np.array([[0, 0], [0, 0]])  # origin point

    x_mesh, y_mesh = np.meshgrid(xvec, xvec)
    xdata = np.c_[x_mesh.flatten(), y_mesh.flatten()]
    gaussian_fit = gaussian_2d(xdata, *opt).reshape(len(xvec), len(xvec))

    if plot:
        axes[0].quiver(
            *origin,
            V[:, 0],
            V[:, 1],
            color=["green", "black"],
            scale=1,
        )
        axes[0].contour(
            xvec, xvec, gaussian_fit, levels=[-0.9, -0.5, -0.1, 0.1, 0.5, 0.9]
        )
    return sig1, sig2, v1, v2, opt

def plot_wigner(
    state,
    contour=False,
    fig=None,
    ax=None,
    max_alpha=3,
    cbar=False,
    npts=81,
):
    
    xvec = np.linspace(-max_alpha, max_alpha, npts)
    W = qt.wigner(qt.ptrace(state,1) ,xvec , xvec, g = 2 )
    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    if ax is None:
        ax = fig.subplots()
    if contour:
        levels = np.linspace(-1.1, 1.1, 102)
        im = ax.contourf(
            xvec, xvec, W, cmap="seismic", vmin=-1, vmax=+1, levels=levels,
        )
    else:
        im = ax.pcolormesh(
            xvec, xvec, W, cmap="seismic", vmin=-1, vmax=+1
        )
    
    ax.set_xlabel(r"Re$(\alpha)$")
    ax.set_ylabel(r"Im$(\alpha)$")
    ax.grid()
    # ax.set_title(title)

    fig.tight_layout()
    if cbar:
        fig.subplots_adjust(right=0.8, hspace=0.25, wspace=0.25)
        # todo: ensure colorbar even with plot...
        # todo: fix this colorbar

        cbar_ax = fig.add_axes([0.85, 0.225, 0.025, 0.65])
        ticks = np.linspace(-1, 1, 5)
        fig.colorbar(im, cax=cbar_ax, ticks=ticks)
        cbar_ax.set_title(r"$\frac{\pi}{2} W(\alpha)$", pad=10)
    ax.set_aspect("equal", adjustable="box")