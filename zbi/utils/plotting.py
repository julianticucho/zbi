from typing import Any, Optional
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import colors as mcolors
from getdist import MCSamples, plots

plt.style.use(['science', 'bright'])
plt.rcParams['figure.dpi'] = 300


def plot_ppc(
    all_samples: list[np.ndarray],
    param_names: list[str],
    true_parameter: Optional[list[float]] = None,
    param_labels: Optional[list[str]] = None,
    sample_labels: Optional[list[str]] = None,
    sample_colors: Optional[list[str]] = None,
    filled: bool | list[bool] = True,
    linestyles: Optional[list[str]] = None,
    title: Optional[str] = None,
    limits: Optional[list[tuple[float, float]]] = None,
    gdist_fine_bins: int = 1500,
    gdist_scaling_factor: float = 0.1,
    bounds_list: Optional[list[dict]] = None,
    legend_fontsize: Optional[float] = None,
    figsize: Optional[tuple[float, float]] = None,
    linewidth: Optional[float] = None,
    output_path: Optional[str] = None,
) -> Any:
    if sample_colors is None:
        gdist_colors = plots.get_subplot_plotter().settings.solid_colors
        sample_colors = list(gdist_colors[:len(all_samples)])
    if param_labels is None:
        param_labels = [name.replace("_", "\\_") for name in param_names]

    gdist_samples = []
    for i, sample in enumerate(all_samples):
        label = (
            sample_labels[i]
            if sample_labels and i < len(sample_labels)
            else f"Run {i+1}"
        )
        gdist = MCSamples(
            samples=np.array(sample, copy=True),
            names=param_names,
            labels=param_labels,
            label=label,
        )
        gdist.fine_bins = gdist.fine_bins_2D = gdist_fine_bins
        gdist_samples.append(gdist)

    g = plots.get_subplot_plotter()
    g.settings.scaling_factor = gdist_scaling_factor
    g.settings.solid_colors = sample_colors
    if legend_fontsize is not None:
        g.settings.legend_fontsize = legend_fontsize
    if linewidth is not None:
        g.settings.linewidth = linewidth
        g.settings.linewidth_contour = linewidth

    limits_dict = (
        {param_names[i]: limits[i] for i in range(len(param_names))}
        if limits
        else {}
    )
    markers_dict = None
    if true_parameter:
        markers_dict = {
            param_names[i]: true_parameter[i] for i in range(len(param_names))
        }

    g.triangle_plot(
        gdist_samples,
        params=param_names,
        filled=filled,
        param_limits=limits_dict,
        markers=markers_dict,
        legend_labels=sample_labels,
        contour_colors=sample_colors,
        contour_ls=linestyles,
        legend_loc="upper right",
    )

    if figsize is not None:
        g.fig.set_size_inches(*figsize)

    if title:
        plt.suptitle(title)

    if bounds_list:
        from matplotlib.patches import Rectangle
        n = len(param_names)
        bound_proxies = []
        bound_labels = []

        for b in bounds_list:
            lo, hi = b["low"], b["high"]
            ec = b.get("edgecolor", "gray")
            fc = b.get("color", "none")
            alpha = b.get("alpha", 0.3)
            ls = b.get("linestyle", "-")
            lw = b.get("linewidth", 0.8)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ax = g.subplots[i, j]
                    if ax is None:
                        continue
                    w = hi[j] - lo[j]
                    h = hi[i] - lo[i]
                    ax.add_patch(Rectangle(
                        (lo[j], lo[i]), w, h,
                        facecolor=fc, edgecolor=ec,
                        alpha=alpha, linestyle=ls, linewidth=lw, zorder=4,
                    ))

            for i in range(n):
                ax = g.subplots[i, i]
                if ax is None:
                    continue
                ax.axvline(lo[i], color=ec, linestyle=ls, linewidth=lw, alpha=alpha)
                ax.axvline(hi[i], color=ec, linestyle=ls, linewidth=lw, alpha=alpha)

            proxy = Rectangle(
                (0, 0), 1, 1,
                facecolor=fc, edgecolor=ec,
                alpha=alpha, linestyle=ls, linewidth=lw,
            )
            bound_proxies.append(proxy)
            bound_labels.append(b.get("label", ""))

        fig = plt.gcf()
        handles, labels = [], []
        if fig.legends:
            leg = fig.legends[0]
            handles = list(leg.legend_handles or [])
            labels = [t.get_text() for t in (leg.texts or [])]
            leg.remove()
        handles.extend(bound_proxies)
        labels.extend(bound_labels)
        leg_kw = dict(loc="upper right")
        if legend_fontsize is not None:
            leg_kw["prop"] = {"size": legend_fontsize}
        fig.legend(handles, labels, **leg_kw)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        g.export(output_path)

    return g


def plot_kl_matrix(
    kl: np.ndarray,
    sample_labels: Optional[list[str]] = None,
    title: Optional[str] = None,
    log_scale: bool = True,
    annotate: bool = True,
    annotate_fmt: str = ".2g",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    linthresh: Optional[float] = None,
    figsize: Optional[tuple[float, float]] = None,
    dim_theta: Optional[int] = None,
    output_path: Optional[str] = None,
):
    kl = np.asarray(kl, dtype=float)
    if kl.ndim != 2 or kl.shape[0] != kl.shape[1]:
        raise ValueError(f"kl must be a square matrix, shape={kl.shape}")
    m = kl.shape[0]

    if sample_labels is None:
        sample_labels = [f"model {i + 1}" for i in range(m)]
    if len(sample_labels) != m:
        raise ValueError("sample_labels must have one label per ensemble member")

    if figsize is None:
        figsize = (max(4.5, m * 1.6), max(4.0, m * 1.3))
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=0.9)

    offdiag = kl[~np.eye(m, dtype=bool)]
    finite = offdiag[np.isfinite(offdiag)]
    amax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if amax <= 0:
        amax = 1.0

    if log_scale:
        if linthresh is None:
            linthresh = amax * 1e-3
        norm = mcolors.SymLogNorm(
            linthresh=max(float(linthresh), np.finfo(float).tiny),
            vmin=vmin if vmin is not None else -amax,
            vmax=vmax if vmax is not None else amax,
            base=10,
        )
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax is not None else amax)

    im = ax.imshow(kl, cmap=cmap, norm=norm, aspect="auto")
    fig.colorbar(im, ax=ax, label="KL divergence")

    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels(sample_labels, rotation=45, ha="right")
    ax.set_yticklabels(sample_labels)
    ax.set_xlabel(r"$q_j$")
    ax.set_ylabel(r"$K_{ij} = \mathrm{KL}(q_i \| q_j)$")

    if annotate:
        for i in range(m):
            for j in range(m):
                v = kl[i, j]
                if not np.isfinite(v):
                    txt = "inf" if v > 0 else "nan"
                elif v == 0:
                    txt = "0"
                else:
                    txt = f"{v:{annotate_fmt}}"
                try:
                    nv = norm(v)
                except Exception:
                    nv = 0.5
                tc = "white" if nv < 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=tc)

    mean_kl = float(np.mean(finite)) if finite.size else np.nan
    max_kl = float(np.max(finite)) if finite.size else np.nan
    summary_parts = [f"mean = {mean_kl:.3g}", f"max = {max_kl:.3g}"]
    if dim_theta:
        summary_parts.append(f"max/n = {max_kl / dim_theta:.3g}")
    summary = ", ".join(summary_parts)
    ax.set_title(f"{title}\n{summary}" if title else summary)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, ax
