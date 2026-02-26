import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

def plot_two_spectra(path1, # absorption
                     path2, # emission
                     normalize=True,
                     xlabel=r'$\lambda$ (nm)',
                     ylabel='Intensity (a.u.)',
                     label1=None,
                     label2=None,
                     title=None,
                     outfile=None,        # nur Dateiname ohne Endung
                     filetype="png",      # "png" oder "pdf"
                     xlim=None,            # z.B. (300, 800) oder None
                     max=False,
                     outdir="Sasaki/Plot_Double"
                     ):
    """
    Liest zwei Spektrendateien ein (2 Spalten), plottet sie zusammen
    und speichert optional als PNG oder PDF.
    """

    # ---------------------------------------
    # Helferfunktion zum Laden der Datei
    # ---------------------------------------
    def _load_file(path):
        df = pd.read_csv(
            path,
            encoding="latin1",
            comment="#",
            sep=None,
            engine="python"
        )

        # Zahlen erzwingen
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce")

        # ungültige Zeilen entfernen
        mask = ~(x.isna() | y.isna())
        return x[mask].to_numpy(), y[mask].to_numpy()

    # Dateien laden
    x1, y1 = _load_file(path1)
    x2, y2 = _load_file(path2)

    # Normierung (optional)
    if normalize:
        if y1.max() != 0:
            y1 = y1 / y1.max()
        if y2.max() != 0:
            y2 = y2 / y2.max()

    # ---------------------------------------
    # Plot
    # ---------------------------------------
    plt.figure(figsize=(8,5))

    # Kurven
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)

    # Achsentexte
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    # Legende nur wenn labels angegeben wurden
    if (label1 is not None) or (label2 is not None):
        plt.legend(loc='upper right')

    # Achsenbereich setzen
    if xlim is not None:
        plt.xlim(xlim)

    # Kein Abstand am Rand der x-Achse
    plt.margins(x=0)

    if max is True:
        try:

            # x at which y is maximum (handle empty arrays and NaNs)
            if getattr(y1, "size", 0) > 0 and not np.all(np.isnan(y1)):
                idx1 = int(np.nanargmax(y1))
                max_x1 = float(x1[idx1]) if getattr(x1, "size", 0) > idx1 else None
            else:
                max_x1 = None

            if getattr(y2, "size", 0) > 0 and not np.all(np.isnan(y2)):
                idx2 = int(np.nanargmax(y2))
                max_x2 = float(x2[idx2]) if getattr(x2, "size", 0) > idx2 else None
            else:
                max_x2 = None
        except Exception:
            max_x1 = None
            max_x2 = None

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()

        # Ensure we have two labels/handles to annotate
        if len(labels) < 2:
            labels = [label1 or "Spectrum 1", label2 or "Spectrum 2"]
            handles = list(ax.lines[:2])

        if max_x1 is not None:
            labels[0] = f"{labels[0]} (max λ={max_x1:.1f} nm)"
        if max_x2 is not None:
            labels[1] = f"{labels[1]} (max λ={max_x2:.1f} nm)"

        ax.legend(handles[:2], labels[:2], loc='upper right')

    # ---------------------------------------
    # Speichern
    # ---------------------------------------
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    if outfile is None:
        path2 = Path(path2)
        outfile = path2.stem + "_abs_em"

    savepath = outdir / f"{outfile}.{filetype}"
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Plot gespeichert unter: {savepath.resolve()}")



    plt.show()

plot_two_spectra(path1 = "Sasaki/DMAC-s2PFL-PN/260219 opt DMAC-s2PFL-PN 1e-5M toluene/260219 PL/Em_Ex300_DMAC-s2PFL-PN_1e-5M_toluene.txt", 
                 path2 = "Sasaki/DMAC-s2PFL-PN/260219 opt DMAC-s2PFL-PN 1e-5M toluene/260219 Abs/sasaki_260219-06_DMAC-s2PFL-PN_1e-5M_toluene.txt", 
                 label1 = "Abs", label2 = "Em", 
                 title = "Normalised Absorption/Emission of DMAC-s2PFL-PN in Toluene", 
                 xlim=(250, 700),
                 normalize=True,
                 max=True,
                 outfile="0219_DMAC_abs_em")