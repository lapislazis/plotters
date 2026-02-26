import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_spectrum(path,
                  normalize=True,
                  xlabel=r'$\lambda$ (nm)',
                  ylabel='Intensity (a.u.)',
                  title=None,
                  outfile=None,
                  filetype="png",
                  xlim=None,
                  max=False,
                  label=None,
                  outdir="Sasaki/Plot_Single"
                  ):
    """
    Liest eine Spektrendatei ein (2 Spalten), plottet sie
    und speichert optional als PNG oder PDF.
    """

    def _load_file(path):
        df = pd.read_csv(
            path,
            encoding="latin1",
            comment="#",
            sep=None,
            engine="python"
        )
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        mask = ~(x.isna() | y.isna())
        return x[mask].to_numpy(), y[mask].to_numpy()

    x, y = _load_file(path)

    if normalize:
        if y.max() != 0:
            y = y / y.max()

    plt.figure(figsize=(8, 5))
    
    # Prepare label with max wavelength if requested
    plot_label = label if label is not None else ""
    
    if max:
        try:
            if getattr(y, "size", 0) > 0 and not np.all(np.isnan(y)):
                idx = int(np.nanargmax(y))
                max_x = float(x[idx]) if getattr(x, "size", 0) > idx else None
            else:
                max_x = None
        except Exception:
            max_x = None
        
        if max_x is not None:
            if plot_label:
                plot_label = f"{plot_label} (max λ={max_x:.1f} nm)"
            else:
                plot_label = f"max λ={max_x:.1f} nm"
    
    plt.plot(x, y, linewidth=1.5, label=plot_label if plot_label else None)  # Standard-Matplotlib-Blau

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)

    plt.margins(x=0)
    
    # Add legend if there's content to display
    if plot_label:
        plt.legend()

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    if outfile is None:
        outfile = Path(path).stem

    savepath = outdir / f"{outfile}.{filetype}"
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Plot gespeichert unter: {savepath.resolve()}")

    plt.show()

plot_spectrum(path="Sasaki/DMAC-s2PFL-PN/260219 opt DMAC-s2PFL-PN 1e-5M toluene/260219 PL/Em_Ex300_DMAC-s2PFL-PN_1e-5M_toluene.txt",
              title="Normalised emission of DMAC-s2PFL-PN in Toluene (19th)",
              xlim=(350, 650),
              normalize=True,
              max=True)
