import matplotlib as mpl
import matplotlib.pyplot as plt


def apply():
    # ── Fonts ──────────────────────────────────────────────
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "serif"]
    mpl.rcParams["mathtext.fontset"] = "dejavuserif"
    mpl.rcParams["text.usetex"] = False

    # ── Font sizes ─────────────────────────────────────────
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.titlesize"] = 15
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["legend.title_fontsize"] = 13

    # ── Figure / savefig ───────────────────────────────────
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.dpi"] = 600
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.format"] = "pdf"

    # ── Axes / spines ─────────────────────────────────────
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 1.2

    # ── Color cycle ────────────────────────────────────────
    N_colores = 8
    cmap = mpl.colormaps["rainbow"]
    colores = [cmap(i / (N_colores - 1)) for i in range(N_colores)]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colores)

    # ── Grid ───────────────────────────────────────────────
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.color"] = "0.85"
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.linewidth"] = 0.6
    mpl.rcParams["grid.alpha"] = 0.7

    # ── Ticks ──────────────────────────────────────────────
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 5
    mpl.rcParams["ytick.major.size"] = 5
    mpl.rcParams["xtick.minor.size"] = 3
    mpl.rcParams["ytick.minor.size"] = 3
    mpl.rcParams["xtick.major.width"] = 1.0
    mpl.rcParams["ytick.major.width"] = 1.0
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True

    # ── Lines ──────────────────────────────────────────────
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["lines.markersize"] = 6

    # ── Legend ──────────────────────────────────────────────
    mpl.rcParams["legend.frameon"] = True
    mpl.rcParams["legend.framealpha"] = 0.85
    mpl.rcParams["legend.edgecolor"] = "0.8"
    mpl.rcParams["legend.fancybox"] = False

    # ── Colormap ───────────────────────────────────────────
    mpl.rcParams["image.cmap"] = "rainbow"
