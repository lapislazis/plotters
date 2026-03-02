import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path

# Set seaborn style
sns.set_style("whitegrid")

def plot_lifetime(
    filepaths,
    log_x=False,
    log_y=False,
    title=None,
    xlabel=None,
    ylabel="counts",
    fit=False,
    fit_range=(5.0, 12.0),      # (x_min, x_max) fürs Fitting; can be a list per-file
    fit_range2=None,            # Optional: second fit range (x_min, x_max) für Dual-Fitting; can be list
    dual_fit=False,             # Enable dual fitting mode with two separate x ranges; can be list
    interpolate=True,
    interp_to_y=1000,            # bis zu welchem y-Wert extrapolieren (Standard: 1 = 10^0)
    outfile=None,               # Name ohne Endung; wenn None -> aus Dateiname abgeleitet
    filetype="png",             # "png" oder "pdf"
    outdir="Sasaki/Plot_Lifetime",
    xlim=None,                  # optional: (xmin, xmax) für Darstellung
    ylim=None,                  # optional: (ymin, ymax) für Darstellung
    labels=None,                # optional: list of labels for each dataset
    time_unit="ns",             # time unit: "ns" (nanosecond), "us" (microsecond), or "ms" (millisecond)
    palette=None,            # Palette: seaborn sequential palette ("Blues", "Greens", "Reds", etc.) or "pyplot" for standard matplotlib colors
    use_seaborn=False,            # Use seaborn palette (True) or standard pyplot colors (False)
    specalpha=1.0
):
    """
    Plottet eine Zeitreihe aus 'filepath', optional mit monoexponentiellem Fit
    (C fest = 0) und Extrapolation bis zu einem bestimmten y-Wert.

    Datendatei: 2 Spalten, Komma-separiert, ohne Header.
    Assumes input data is in nanoseconds.
    """

    # Accept single filepath or list/tuple of filepaths
    if isinstance(filepaths, (str, Path)):
        file_list = [Path(filepaths)]
    else:
        file_list = [Path(p) for p in filepaths]
    n_files = len(file_list)
    
    # --- Validate input parameters ---
    if interp_to_y is not None and (not isinstance(interp_to_y, (int, float)) or interp_to_y <= 0):
        raise ValueError("interp_to_y must be a positive number or None")

    # Normalize parameters that may be scalar or lists
    def _broadcast_param(param, name, default=None):
        if isinstance(param, (list, tuple)):
            if len(param) != n_files:
                raise ValueError(f"Parameter '{name}' must have length equal to number of filepaths ({n_files})")
            return list(param)
        else:
            return [param] * n_files
    
    # Special handling for fit_range and fit_range2: distinguish between a single range tuple and a list of ranges
    def _broadcast_range(param, name, default=None):
        if param is None:
            return [None] * n_files
        if isinstance(param, (list, tuple)):
            # Check possibilities:
            # 1) A list of ranges, one per file: [(x_min,x_max), ...] with len == n_files
            # 2) A single (x_min, x_max) tuple which should be broadcast to all files
            # 3) A single-element list containing a (x_min, x_max) tuple -> broadcast it
            if len(param) == n_files and all(isinstance(p, (tuple, list)) for p in param):
                return list(param)
            # Single tuple like (x_min, x_max)
            if len(param) == 2 and all(isinstance(p, (int, float)) for p in param):
                return [param] * n_files
            # Single-element list containing a tuple: [(x_min,x_max)]
            if len(param) == 1 and isinstance(param[0], (tuple, list)) and len(param[0]) == 2 and all(isinstance(p, (int, float)) for p in param[0]):
                return [tuple(param[0])] * n_files
            raise ValueError(f"Parameter '{name}' must be either (x_min, x_max) or a list of (x_min, x_max) tuples")
        else:
            return [param] * n_files

    fit_list = _broadcast_param(fit, "fit", False)
    dual_fit_list = _broadcast_param(dual_fit, "dual_fit", False)
    fit_range_list = _broadcast_range(fit_range, "fit_range", None)
    fit_range2_list = _broadcast_range(fit_range2, "fit_range2", None)
    
    # --- Validate and convert time units ---
    valid_units = {"ns": 1.0, "us": 1000.0, "ms": 1e6}
    if time_unit not in valid_units:
        raise ValueError(f"Invalid time_unit '{time_unit}'. Must be one of: {list(valid_units.keys())}")
    
    conversion_factor = valid_units[time_unit]

    # --- Daten laden und plotten (mehrere Dateien möglich) ---
    # --- Set xlabel if not provided ---
    if xlabel is None:
        unit_label = "μs" if time_unit == "us" else time_unit
        xlabel = f"time ({unit_label})"

    # --- Figure/Axes anlegen ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use seaborn palette or standard pyplot colors
    if use_seaborn:
        colors = sns.color_palette(palette, n_colors=max(n_files, 2))
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    taus_out = []

    for idx, filepath in enumerate(file_list):
        # Load data
        df = pd.read_csv(filepath, sep=",", header=None, comment=";")
        x = df.iloc[:, 0].to_numpy() / conversion_factor  # Convert from ns to desired unit
        y = df.iloc[:, 1].to_numpy()

        # Label handling
        if labels is None:
            label_data = filepath.stem
        else:
            label_list = labels if isinstance(labels, (list, tuple)) else [labels]
            if len(label_list) != n_files:
                raise ValueError("'labels' must be a list with the same length as filepaths")
            label_data = label_list[idx]

        # Plot data
        if use_seaborn:
            color = colors[idx]
        else:
            color = colors[idx % len(colors)]
        ax.plot(x, y, label=label_data, linewidth=1.0, color=color, alpha=specalpha, zorder=2)

        tau = None
        tau2 = None

        # Determine per-file fit parameters
        fit_i = fit_list[idx]
        dual_fit_i = dual_fit_list[idx]
        frange_i = fit_range_list[idx]
        frange2_i = fit_range2_list[idx]

        # --- Validate dual_fit parameters for this dataset (only when fitting is enabled) ---
        if fit_i and dual_fit_i:
            if frange2_i is None:
                raise ValueError(f"dual_fit=True for {filepath} requires fit_range2 to be specified")
            if not isinstance(frange2_i, (tuple, list)) or len(frange2_i) != 2:
                raise ValueError(f"fit_range2 must be a tuple/list of (x_min, x_max), got {frange2_i}")
            if frange2_i[0] >= frange2_i[1]:
                raise ValueError(f"fit_range2: x_min must be less than x_max for {filepath}")
            # Warn if ranges overlap
            if frange_i is not None:
                if not (frange_i[1] < frange2_i[0] or frange2_i[1] < frange_i[0]):
                    print(f"Warning for {filepath}: fit_range and fit_range2 overlap. Consider using separate, non-overlapping ranges.")
        if fit_i:
            def monoexp_C0(xv, A, tau):
                return A * np.exp(-xv / tau)

            # Validate fit ranges for this dataset
            if dual_fit_i:
                x_min1, x_max1 = frange_i
                x_min2, x_max2 = frange2_i

                # First fit
                mask1 = (x >= x_min1) & (x <= x_max1)
                x_fit1 = x[mask1]
                y_fit1 = y[mask1]
                if len(x_fit1) < 2:
                    raise ValueError(f"Insufficient data points in fit_range ({x_min1}, {x_max1}) for {filepath}")

                # Second fit
                mask2 = (x >= x_min2) & (x <= x_max2)
                x_fit2 = x[mask2]
                y_fit2 = y[mask2]
                if len(x_fit2) < 2:
                    raise ValueError(f"Insufficient data points in fit_range2 ({x_min2}, {x_max2}) for {filepath}")

                A0_1 = y_fit1.max()
                tau0_1 = (x_max1 - x_min1) / 2.0 if (x_max1 > x_min1) else 1.0
                p0_1 = [A0_1, tau0_1]
                params1, cov1 = curve_fit(monoexp_C0, x_fit1, y_fit1, p0=p0_1, maxfev=50000)
                A1, tau = params1
                print(f"{filepath.name} - fit1: A={A1:.3f}, tau={tau:.6f} ({time_unit})")

                A0_2 = y_fit2.max()
                tau0_2 = (x_max2 - x_min2) / 2.0 if (x_max2 > x_min2) else 1.0
                p0_2 = [A0_2, tau0_2]
                params2, cov2 = curve_fit(monoexp_C0, x_fit2, y_fit2, p0=p0_2, maxfev=50000)
                A2, tau2 = params2
                print(f"{filepath.name} - fit2: A={A2:.3f}, tau={tau2:.6f} ({time_unit})")

                # Plot fits for this dataset using dataset color
                unit_label = "μs" if time_unit == "us" else time_unit
                x_smooth1 = np.linspace(x_min1, x_max1, 400)
                y_smooth1 = monoexp_C0(x_smooth1, A1, tau)
                ax.plot(x_smooth1, y_smooth1, color=color, linestyle="-", linewidth=2.5, label=f"{label_data} fit1 (τ={tau:.3f} {unit_label})", zorder=10)

                x_smooth2 = np.linspace(x_min2, x_max2, 400)
                y_smooth2 = monoexp_C0(x_smooth2, A2, tau2)
                ax.plot(x_smooth2, y_smooth2, color=color, linestyle="-", linewidth=2.5, label=f"{label_data} fit2 (τ={tau2:.3f} {unit_label})", zorder=10)

                # Extrapolations
                if interpolate and interp_to_y is not None and interp_to_y > 0:
                    if A1 > interp_to_y:
                        x_target1 = tau * np.log(A1 / interp_to_y)
                        if x_target1 > x_max1:
                            x_ext1 = np.linspace(x_max1, x_target1, 400)
                            y_ext1 = monoexp_C0(x_ext1, A1, tau)
                            ax.plot(x_ext1, y_ext1, color=color, linestyle="--", linewidth=1.5, zorder=8)
                    if A2 > interp_to_y:
                        x_target2 = tau2 * np.log(A2 / interp_to_y)
                        if x_target2 > x_max2:
                            x_ext2 = np.linspace(x_max2, x_target2, 400)
                            y_ext2 = monoexp_C0(x_ext2, A2, tau2)
                            ax.plot(x_ext2, y_ext2, color=color, linestyle=(0, (1,1)), linewidth=1.5, zorder=8)

                taus_out.append((tau, tau2))

            else:
                if frange_i is None:
                    raise ValueError("fit_range must be specified for datasets with fit=True")
                x_min, x_max = frange_i
                mask = (x >= x_min) & (x <= x_max)
                x_fit = x[mask]
                y_fit = y[mask]
                if len(x_fit) < 2:
                    raise ValueError(f"Insufficient data points in fit_range ({x_min}, {x_max}) for {filepath}")

                A0 = y_fit.max()
                tau0 = (x_max - x_min) / 2.0 if (x_max > x_min) else 1.0
                p0 = [A0, tau0]
                params, cov = curve_fit(monoexp_C0, x_fit, y_fit, p0=p0, maxfev=50000)
                A, tau = params
                print(f"{filepath.name} - fit: A={A:.3f}, tau={tau:.6f} ({time_unit})")

                x_smooth = np.linspace(x_min, x_max, 400)
                y_smooth = monoexp_C0(x_smooth, A, tau)
                unit_label = "μs" if time_unit == "us" else time_unit
                ax.plot(x_smooth, y_smooth, color=color, linestyle="-", linewidth=2.5, label=f"{label_data} fit (τ={tau:.3f} {unit_label})", zorder=10)

                if interpolate and interp_to_y is not None and interp_to_y > 0:
                    if A > interp_to_y:
                        x_target = tau * np.log(A / interp_to_y)
                        if x_target > x_max:
                            x_ext = np.linspace(x_max, x_target, 400)
                            y_ext = monoexp_C0(x_ext, A, tau)
                            ax.plot(x_ext, y_ext, color=color, linestyle="--", linewidth=1.5, zorder=8)

                taus_out.append(tau)

    # --- Achsenskalierung ---
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # --- Grid for logarithmic axes ---
    if log_x or log_y:
        # Show minor gridlines for better logarithmic representation
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.7, color="gray")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3, color="gray")
    
    # --- Labels / Titel ---
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # --- Darstellungsbereiche ---
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Kein Abstand links/rechts
    ax.margins(x=0)

    ax.legend()

    # --- Speichern ---
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    if outfile is None:
        # Derive outfile from concatenated stems
        stems = "+".join([p.stem for p in file_list])
        outfile = stems + "_lifetime"

    savepath = outdir / f"{outfile}.{filetype}"
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Plot gespeichert unter: {savepath.resolve()}")

    plt.show()

    # Return taus per dataset (None for datasets without fit)
    return taus_out

plot_lifetime(
    filepaths=[
        "Sasaki/DMAC-s2PFL-PN/260219 opt DMAC-s2PFL-PN 1e-5M toluene/260219 tau/DMAC-s2PFL-PN_1e-5M_toluene/200ns_Ex340.dat",
    ],
    log_y=True,
    #log_x=True,
    title="DMAC-s2PFL-PN in 1E-5 Toluene 200ns",
    fit=[True],
    dual_fit=[True],
    fit_range=[(20, 33)],
    fit_range2=[(65, 100)],
    labels=["DMAC-s2PFL-PN"],
    outfile="0219_DMAC_sol_200ns",
    interp_to_y=1,
    xlim=(15.6, 200),
    ylim=(250, 6200),
    time_unit="ns",
    use_seaborn=True,
    palette="muted",
    specalpha=0.8
)

# Temperature dependent template
'''plot_lifetime(
    filepaths=[
        "Sasaki/DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film including 300K-2/250K.dat",
        "Sasaki/DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film including 300K-2/260K.dat",
        "Sasaki/DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film including 300K-2/270K.dat",
        "Sasaki/DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film including 300K-2/280K.dat",
        "Sasaki/DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film including 300K-2/290K.dat",
        "Sasaki/DMAC-s2PFL-PN/260228tau/15wt%DMAC-s2PFL-PN DPEPO film including 300K-2/300K-2.dat",
    ],
    log_y=True,
    title="Temperature Dependent Lifetime of 15%wt DMAC-s2PFL-PN in DPEPO Film",
    fit=[True, True, True, True, True, True],
    #dual_fit=[True, False],
    fit_range=[(0.9, 2), (0.9, 2), (0.9, 2), (0.9, 2), (0.9, 2), (0.9, 2)],
    #fit_range2=[(0.25, 0.75)],
    labels=["250K", "260K", "270K", "280K", "290K", "300K"],
    outfile="0228_DMAC_temp_tau",
    interp_to_y=1,
    xlim=(0.71, 6),
    #ylim=(1, 6000),
    time_unit="us",
    use_seaborn=True,
    palette="icefire",
    specalpha=0.6
)'''

# Palette note: deep is good for qualitative, like comparisons, "Blues", "Reds" are sequentially darker, good for
# Temp dependent etc, as well as Viridis. All sequential can be reverse with "*_r" suffix
# icefire is nice for cold-hot
