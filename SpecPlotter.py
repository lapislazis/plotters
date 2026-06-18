import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np
from matplotlib import colors as mcolors

# set seaborn style to match TauPlotter
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
})


def plot_spectra(paths,
                 normalize=True,
                 xlabel=r'$\lambda$ (nm)',
                 ylabel='Intensity (a.u.)',
                 labels=None,
                 title=None,
                 outfile=None,        # nur Dateiname ohne Endung
                 filetype="png",     # "png" oder "pdf"
                 xlim=None,           # z.B. (300, 800) oder None
                 ylim=None,           # z.B. (0, 1) or None
                 annotate_max=False,
                 outdir="Plot_Spectra",
                 palette=None,
                 skip_first_color=True,
                 use_seaborn=False,
                 linestyles=None,
                 linewidth=1.5,
                 value=None,
                 value_label=None,
                 files_per_palette=3,
                 palette_cycle=False,
                 linestyle_cycle=False,
                 linestyles_cycle_list=None,
                 first_declared_on_top=True,
                 clean_plot=False
                 ):
    """
    Liest beliebig viele Spektrendateien ein (2 Spalten), plottet sie zusammen
    und speichert optional als PNG oder PDF. Unterstützt seaborn/pyplot Farben.
    
        Parameters:
        - palette: Seaborn palette name (str) for normal color assignment,
            or list of palette names for palette cycling.
        - palette_cycle (bool): If True, cycle palette names every files_per_palette files.
            No palette cycling occurs unless this is explicitly True.
        - use_seaborn (bool): Use seaborn color palettes.
        - files_per_palette (int): Number of files to use each palette for (default: 3).
        - linestyle_cycle (bool): If True, cycle through linestyles independently of palette behavior.
    - linestyles_cycle_list (list): List of linestyles to cycle through. 
      Default: ["-", "--", "-.", ":"]
        - first_declared_on_top (bool): If True, files listed first are drawn on top.
                - clean_plot (bool): If True, generate a graphics-clean plot with no legend,
                        no peak annotations, no title, and no axis label text. Axis tick numbers
                        remain visible.
    """

    def _load_file(path):
        """
        Load spectrum data file, skipping headers and non-numeric lines.
        Accepts any line with 2+ comma/whitespace-separated numeric values.
        """
        x_vals = []
        y_vals = []
        
        with open(path, 'r', encoding="latin1") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comment lines
                if not line or line.startswith('#'):
                    continue
                
                # Try to parse the line as numeric values
                try:
                    # Split by comma or whitespace
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        x_vals.append(x)
                        y_vals.append(y)
                except (ValueError, IndexError):
                    # Skip lines that don't have valid numeric pairs
                    continue
        
        if not x_vals or not y_vals:
            raise ValueError(f"No valid data found in {path}")
        
        return np.array(x_vals), np.array(y_vals)

    def _drop_brightest_rgb_list(rgb_list):
        """Return a new list with the brightest (highest luminance) color removed."""
        if not rgb_list:
            return rgb_list
        lums = [0.2126 * r + 0.7152 * g + 0.0722 * b for (r, g, b) in rgb_list]
        idx = int(np.argmax(lums))
        return [c for i, c in enumerate(rgb_list) if i != idx]

    # Accept single path or list
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = [Path(p) for p in paths]

    n = len(path_list)

    # Allow annotate_max to be specified per spectrum
    if isinstance(annotate_max, (list, tuple)):
        if len(annotate_max) != n:
            raise ValueError("'annotate_max' list must have the same length as 'paths'.")
        annotate_max_flags = [bool(v) for v in annotate_max]
    else:
        annotate_max_flags = [bool(annotate_max)] * n

    # Determine palette mode (explicit palette cycling only)
    palette_list = None
    if palette_cycle:
        if palette is None:
            palette_list = ["Blues", "Reds", "Greens", "Purples", "Oranges",
                            "YlOrBr", "BuPu", "GnBu", "PuBu", "coolwarm"]
        elif isinstance(palette, (list, tuple)):
            palette_list = list(palette)
        else:
            raise ValueError("When palette_cycle=True, 'palette' must be a list/tuple of palette names or None.")
        if files_per_palette < 1:
            raise ValueError("'files_per_palette' must be >= 1.")
        use_seaborn = True  # Palette cycling requires seaborn palettes

    # Default list of linestyles for cycling
    if linestyles_cycle_list is None:
        linestyles_cycle_list = ["-", "--", "-.", ":"]

    # Farben
    if use_seaborn:
        if palette_cycle:
            colors = None
        else:
            if isinstance(palette, (list, tuple)):
                raise ValueError("When palette_cycle=False, 'palette' must be a single palette name (str) or None.")
            if palette is None:
                palette = "muted"
            # Allow skipping the visually brightest color in the palette if requested
            if skip_first_color:
                pal_all = sns.color_palette(palette, n_colors=max(n, 2) + 1)
                colors = _drop_brightest_rgb_list(pal_all)
            else:
                colors = sns.color_palette(palette, n_colors=max(n, 2))
    else:
        # Matplotlib default cycle returns hex strings; convert to RGB tuples
        mpl_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        mpl_rgb = [mcolors.to_rgb(c) for c in mpl_colors]
        if skip_first_color and len(mpl_rgb) > 1:
            mpl_rgb = _drop_brightest_rgb_list(mpl_rgb)
        colors = mpl_rgb

    # linestyles default (non-cycling fallback)
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"]

    # Dateien laden and optionally normalize
    data = []
    for p in path_list:
        x, y = _load_file(p)
        data.append((Path(p), x, y))

    if normalize:
        for i, (p, x, y) in enumerate(data):
            if getattr(y, "size", 0) > 0 and not np.all(np.isnan(y)):
                # If xlim is declared, compute the normalization factor only
                # from the data points inside that x-range. Otherwise use
                # the full range.
                if xlim is not None:
                    try:
                        xmin, xmax = xlim
                        mask = (x >= xmin) & (x <= xmax)
                        masked_y = y[mask]
                        if masked_y.size > 0 and not np.all(np.isnan(masked_y)):
                            maxval = np.nanmax(masked_y)
                        else:
                            maxval = np.nanmax(y)
                    except Exception:
                        maxval = np.nanmax(y)
                else:
                    maxval = np.nanmax(y)

                if maxval != 0:
                    newy = y / maxval
                    # For normalized plots, ensure no values fall below 0
                    newy = np.clip(newy, a_min=0, a_max=None)
                    data[i] = (p, x, newy)

    # Plot
    plt.figure(figsize=(8, 5))

    legend_labels = []
    for i, (p, x, y) in enumerate(data):
        linestyle_idx = i
        if palette_cycle:
            palette_idx = i // files_per_palette
            position_in_palette = i % files_per_palette
            current_palette = palette_list[palette_idx % len(palette_list)]
            # Generate palette colors, optionally skipping the visually brightest one
            if skip_first_color:
                pal_all = sns.color_palette(current_palette, n_colors=max(files_per_palette, 2) + 1)
                palette_colors = _drop_brightest_rgb_list(pal_all)
            else:
                palette_colors = sns.color_palette(current_palette, n_colors=max(files_per_palette, 2))
            color = palette_colors[position_in_palette]
            linestyle_idx = position_in_palette
        else:
            color = colors[i] if use_seaborn else colors[i % len(colors)]

        # Handle linestyle independently of palette behavior
        if linestyle_cycle:
            linestyle = linestyles_cycle_list[linestyle_idx % len(linestyles_cycle_list)]
        else:
            linestyle = linestyles[i % len(linestyles)]
        # determine label
        if labels is not None:
            if len(labels) != n:
                raise ValueError("'labels' must have same length as 'paths'")
            lab = labels[i]
        else:
            lab = p.stem

        zorder = (n - i) if first_declared_on_top else (i + 1)
        plt.plot(x, y, label=lab, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)

        # annotate peak wavelength directly on the trace (same visual style as StreakPlotter)
        if annotate_max_flags[i] and not clean_plot:
            try:
                if getattr(y, "size", 0) > 0 and not np.all(np.isnan(y)):
                    # If xlim is declared, find the peak only within that range.
                    if xlim is not None:
                        try:
                            xmin, xmax = xlim
                            mask = (x >= xmin) & (x <= xmax)
                            indices = np.where(mask)[0]
                            if indices.size > 0:
                                sub_y = y[indices]
                                if not np.all(np.isnan(sub_y)):
                                    sub_idx = int(np.nanargmax(sub_y))
                                    idx = int(indices[sub_idx])
                                else:
                                    # No valid data in range; skip annotation
                                    idx = None
                            else:
                                idx = None
                        except Exception:
                            idx = int(np.nanargmax(y))
                    else:
                        idx = int(np.nanargmax(y))

                    if idx is not None:
                        peak_x = float(x[idx])
                        peak_y = float(y[idx])
                        plt.annotate(f"{peak_x:.1f} nm",
                                     xy=(peak_x, peak_y),
                                     xytext=(5, 5),
                                     textcoords='offset points',
                                     fontsize=9,
                                     fontweight='bold',
                                     color=color)
            except Exception:
                pass

        legend_labels.append(lab)

    # Achsentexte
    if xlabel is not None and not clean_plot:
        plt.xlabel(xlabel)
    if ylabel is not None and not clean_plot:
        plt.ylabel(ylabel)
    if title is not None and not clean_plot:
        plt.title(title)

    # show legend if any label content
    if (any(legend_labels) or value is not None) and not clean_plot:
        legend_title = None
        if value is not None:
            if value_label is None:
                value_label = "Value"
            legend_title = f"{value_label}: {value}"
        plt.legend(loc='upper right', title=legend_title)

    # Achsenbereich setzen
    if xlim is not None:
        plt.xlim(xlim)

    # Y-axis limits: use provided `ylim`, otherwise autoscale tightly
    if ylim is not None:
        plt.ylim(ylim)
    else:
        # compute a sensible ymax from plotted data (after normalization/clipping)
        y_max = 0.0
        for (_p, _x, _y) in data:
            if getattr(_y, "size", 0) > 0 and not np.all(np.isnan(_y)):
                try:
                    cur = float(np.nanmax(_y))
                    if cur > y_max:
                        y_max = cur
                except Exception:
                    continue
        if y_max > 0:
            ymax = y_max * 1.05
        else:
            ymax = 1.0
        plt.ylim(0, ymax)

    # Kein Abstand am Rand der x- und y-Achse
    plt.margins(x=0, y=0)

    # Speichern
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    if outfile is None:
        if n == 1:
            outfile = path_list[0].stem
        else:
            stems = "+".join([p.stem for p in path_list])
            outfile = stems + "_spectra"

    savepath = outdir / f"{outfile}.{filetype}"
    should_save = True
    if savepath.exists():
        resp = input(f"File exists: {savepath.resolve()}\nOverwrite? Type Y to confirm: ")
        if resp.strip() != "Y":
            should_save = False
            print(f"Skipped saving (overwrite not confirmed): {savepath.resolve()}")

    if should_save:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Plot gespeichert unter: {savepath.resolve()}")

    plt.show()

'''plot_spectra(paths=[
    "DMAC-s2PFL-PN\\260302Abs\\260302_DMAC-2sp_1E-4M_UVAbs.txt",
],
    labels=["1E-4 Abs"],
    title="Normalised Absorption of DMAC-s2PFL-PN in Toluene",
    xlim=(250, 350),
    annotate_max=True,
    outfile="0302_DMAC_sol_abs",
    use_seaborn=True,
    ylabel="Normalised Intensity (a.u.)",
    palette="Blues",
    #files_per_palette=1,
    #value = 37.9,
    #value_label = "PLQY (290nm)",
    #linestyle_cycle=True,
    #linestyles_cycle_list=["-.", "--", "-"]
)'''


#Triple plot setup
plot_spectra(paths=[
    "DMAC-s2PFL-PN/20260610flu/s2pfl_degassed_exc290.txt",
    "DMAC-s2PFL-PN/20260608flu/s2pfl_exc290_1.txt"
],
    labels=["Degassed PL Flu.", "Aerated PL Flu."],
    title="PL intensity comparison of Degassed/Aerated DMAC-s2PFL-PN in Toluene",
    xlim=(370, 600),
    #ylim=(0, 1.10),
    annotate_max=[False, False],
    outfile="0610_DMAC_sol_intcomp",
    use_seaborn=True,
    ylabel="Intensity",
    #palette_cycle=True,
    palette="Blues_r",
    #files_per_palette=3,
    #value = 6.5,
    #value_label = "PLQY exc. 290nm (%)",
    linestyle_cycle=True,
    normalize=False,
    #clean_plot=True
    skip_first_color=True
)