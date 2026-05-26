import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np

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
                 ylim=None,           # z.B. (0, 1) oder None
                 annotate_max=False,
                 outdir="Plot_Spectra",
                 palette=None,
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
        - linestyles (str or list): Line style(s) to use when not cycling. If a list is provided,
            the first element is used for all lines unless linestyle_cycle=True.
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

    # Accept single path or list
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = [Path(p) for p in paths]

    n = len(path_list)

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

    # Default linestyle for non-cycling mode
    if linestyles is None:
        linestyles = "-"

    # Farben
    if use_seaborn:
        if palette_cycle:
            colors = None
        else:
            if isinstance(palette, (list, tuple)):
                raise ValueError("When palette_cycle=False, 'palette' must be a single palette name (str) or None.")
            if palette is None:
                palette = "muted"
            colors = sns.color_palette(palette, n_colors=max(n, 2))
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Dateien laden (defer normalization until xlim is known)
    data = []
    for p in path_list:
        x, y = _load_file(p)
        data.append((Path(p), x, y))

    # Normalize based on xlim window if specified, otherwise global bounds
    if normalize:
        for i, (p, x, y) in enumerate(data):
            if getattr(y, "size", 0) > 0:
                if xlim is not None:
                    # Normalize to max/min within xlim window
                    mask = (x >= xlim[0]) & (x <= xlim[1])
                    if np.any(mask):
                        y_window = y[mask]
                        y_min = np.nanmin(y_window)
                        y_max = np.nanmax(y_window)
                        if y_max > y_min:
                            y_norm = (y - y_min) / (y_max - y_min)
                        else:
                            y_norm = y / np.nanmax(np.abs(y)) if np.nanmax(np.abs(y)) != 0 else y
                    else:
                        y_norm = y / np.nanmax(np.abs(y)) if np.nanmax(np.abs(y)) != 0 else y
                else:
                    # Normalize to global max/min
                    y_min = np.nanmin(y)
                    y_max = np.nanmax(y)
                    if y_max > y_min:
                        y_norm = (y - y_min) / (y_max - y_min)
                    else:
                        y_norm = y / np.nanmax(np.abs(y)) if np.nanmax(np.abs(y)) != 0 else y
                data[i] = (p, x, y_norm)

    # Plot
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    annotation_boxes = []

    def _bbox_overlap(b1, b2):
        return not (
            b1[0] + b1[2] < b2[0] or
            b1[0] > b2[0] + b2[2] or
            b1[1] + b1[3] < b2[1] or
            b1[1] > b2[1] + b2[3]
        )

    legend_labels = []
    for i, (p, x, y) in enumerate(data):
        linestyle_idx = i
        if palette_cycle:
            palette_idx = i // files_per_palette
            position_in_palette = i % files_per_palette
            current_palette = palette_list[palette_idx % len(palette_list)]
            palette_colors = sns.color_palette(current_palette, n_colors=max(files_per_palette, 2))
            color = palette_colors[position_in_palette]
            linestyle_idx = position_in_palette
        else:
            color = colors[i] if use_seaborn else colors[i % len(colors)]

        # Handle linestyle independently of palette behavior
        if linestyle_cycle:
            linestyle = linestyles_cycle_list[linestyle_idx % len(linestyles_cycle_list)]
        else:
            if isinstance(linestyles, (list, tuple)):
                linestyle = linestyles[0]
            else:
                linestyle = linestyles
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
        if annotate_max and not clean_plot:
            try:
                if getattr(y, "size", 0) > 0 and not np.all(np.isnan(y)):
                    # Find peak within xlim window if specified, else global
                    if xlim is not None:
                        mask = (x >= xlim[0]) & (x <= xlim[1])
                        if np.any(mask):
                            y_window = y[mask]
                            idx_window = int(np.nanargmax(y_window))
                            idx = np.where(mask)[0][idx_window]
                        else:
                            idx = None
                    else:
                        idx = int(np.nanargmax(y))
                    
                    if idx is not None:
                        peak_x = float(x[idx])
                        peak_y = float(y[idx])
                        text = f"{peak_x:.1f} nm"
                        offsets = [(5, 5), (5, 20), (5, 35), (5, -20), (5, -35), (5, 50), (5, -50), (5, 65), (5, -65)]
                        dpi = ax.figure.dpi
                        pt2px = dpi / 72.0
                        text_width = max(40, len(text) * 6)
                        text_height = 14 * 1.2
                        x_disp, y_disp = ax.transData.transform((peak_x, peak_y))
                        chosen_offset = None
                        for dx, dy in offsets:
                            dx_disp = dx * pt2px
                            dy_disp = dy * pt2px
                            bbox = (x_disp + dx_disp, y_disp + dy_disp, text_width * pt2px, text_height * pt2px)
                            if not any(_bbox_overlap(bbox, existing) for existing in annotation_boxes):
                                chosen_offset = (dx, dy)
                                annotation_boxes.append(bbox)
                                break
                        if chosen_offset is None:
                            chosen_offset = (5, 5)
                            dx_disp = chosen_offset[0] * pt2px
                            dy_disp = chosen_offset[1] * pt2px
                            annotation_boxes.append((x_disp + dx_disp, y_disp + dy_disp, text_width * pt2px, text_height * pt2px))
                        ax.annotate(text,
                                     xy=(peak_x, peak_y),
                                     xytext=chosen_offset,
                                     textcoords='offset points',
                                     fontsize=12,
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
    if ylim is not None:
        plt.ylim(ylim)

    # Kein Abstand am Rand der x-Achse
    plt.margins(x=0)

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
    "3 - DCNP/Photophysical/PL/20260224flu/260224-DCNP-Flu1.txt",
    "3 - DCNP/Photophysical/UV/20260224abs/26.02.24.DCNPAbs1.txt"
],
    labels=["Sol. PL", "Sol. Absorption"],
    title="Normalised PL/Abs of DCNP in Toluene",
    xlim=(380, 600),
    ylim=(0, 1.1),
    annotate_max=True,
    outfile="0224_DCNP_sol_absemphos",
    use_seaborn=True,
    ylabel="Normalised Intensity (a.u.)",
    palette_cycle=False,
    palette="Blues_r",
    #files_per_palette=1,
    #value = 83.5,
    #value_label = "PLQY (460nm)",
    linestyle_cycle=True,
    normalize=True
    #clean_plot=True
)