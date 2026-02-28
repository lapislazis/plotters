import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import numpy as np

# set seaborn style to match TauPlotter
sns.set_style("whitegrid")


def plot_spectra(paths,
                 normalize=True,
                 xlabel=r'$\lambda$ (nm)',
                 ylabel='Intensity (a.u.)',
                 labels=None,
                 title=None,
                 outfile=None,        # nur Dateiname ohne Endung
                 filetype="png",     # "png" oder "pdf"
                 xlim=None,           # z.B. (300, 800) oder None
                 annotate_max=False,
                 outdir="Sasaki/Plot_Spectra",
                 palette=None,
                 use_seaborn=False,
                 same_color=False,
                 single_color=None,
                 linestyles=None,
                 linewidth=1.5,
                 value=None,
                 value_label=None
                 ):
    """
    Liest beliebig viele Spektrendateien ein (2 Spalten), plottet sie zusammen
    und speichert optional als PNG oder PDF. Unterstützt seaborn/pyplot Farben.
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

    # Farben
    if use_seaborn:
        if palette is None:
            palette = "muted"
        colors = sns.color_palette(palette, n_colors=max(n, 2))
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # linestyles default
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"]

    # Dateien laden and optionally normalize
    data = []
    for p in path_list:
        x, y = _load_file(p)
        data.append((Path(p), x, y))

    if normalize:
        for i, (p, x, y) in enumerate(data):
            if getattr(y, "size", 0) > 0 and np.nanmax(y) != 0:
                data[i] = (p, x, y / np.nanmax(y))

    # Plot
    plt.figure(figsize=(8, 5))

    legend_labels = []
    for i, (p, x, y) in enumerate(data):
        if same_color:
            # choose a single color for all lines
            if single_color is not None:
                color = single_color
            else:
                color = colors[0] if use_seaborn else colors[0]
            linestyle = linestyles[i % len(linestyles)]
        else:
            color = colors[i] if use_seaborn else colors[i % len(colors)]
            linestyle = "-"
        # determine label
        if labels is not None:
            if len(labels) != n:
                raise ValueError("'labels' must have same length as 'paths'")
            lab = labels[i]
        else:
            lab = p.stem

        # annotate max if requested
        if annotate_max:
            try:
                if getattr(y, "size", 0) > 0 and not np.all(np.isnan(y)):
                    idx = int(np.nanargmax(y))
                    max_x = float(x[idx]) if getattr(x, "size", 0) > idx else None
                else:
                    max_x = None
            except Exception:
                max_x = None
            if max_x is not None:
                lab = f"{lab} (max \u03bb={max_x:.1f} nm)"

        plt.plot(x, y, label=lab, color=color, linewidth=linewidth, linestyle=linestyle)
        legend_labels.append(lab)

    # Achsentexte
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    # show legend if any label content
    if any(legend_labels) or value is not None:
        legend_title = None
        if value is not None:
            if value_label is None:
                value_label = "Value"
            legend_title = f"{value_label}: {value}"
        plt.legend(loc='upper right', title=legend_title)

    # Achsenbereich setzen
    if xlim is not None:
        plt.xlim(xlim)

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
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Plot gespeichert unter: {savepath.resolve()}")

    plt.show()


plot_spectra(paths=[
    "Sasaki/DMAC-s2PFL-PN/260219 opt DMAC-s2PFL-PN 1e-5M toluene/260227 PL/15wt_DMAC-s2PFL-PN_DPEPO_film.txt",
    "Sasaki/DMAC-s2PFL-PN/260219 opt DMAC-s2PFL-PN 1e-5M toluene/260227Abs/sasaki_260227-01_15wt DMAC-s2PFL-PN_DPEPO_film.txt",
    "Sasaki/DMAC-s2PFL-PN/260219 opt DMAC-s2PFL-PN 1e-5M toluene/260227 PL/PHOS_15wt_DMAC-s2PFL-PN_DPEPO_film.txt"
],
    labels=["Flu.", "Abs.", "Phos."],
    title="Normalised Abs/Em of 15wt% DMAC-s2PFL-PN in DPEPO doped film",
    xlim=(250, 700),
    #normalize=False,
    #annotate_max=True,
    outfile="0227_DMAC_film_absemphos",
    use_seaborn=True,
    same_color=True,
    ylabel="Normalised Intensity (a.u.)",
    #palette="viridis"
    value = 37.9,
    value_label = "PLQY (290nm)"
)